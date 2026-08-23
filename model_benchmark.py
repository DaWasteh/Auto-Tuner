"""Real llama-server performance tuning for one or more GGUF profiles.

The engine is deliberately UI-independent. It starts a private loopback
llama-server for every candidate, waits for the exact server identity, runs a
long deterministic prompt + decode workload, and records llama.cpp's native
prompt-processing and n_decode timings. Winner ranking uses the measured
end-to-end inference time instead of a geometric mean that could exaggerate a
large prompt-only gain as a similarly large real-world speedup.

Context, KV precision, GPU placement, Flash Attention, sampling, and the chosen
performance target stay fixed inside one sweep. Runtime axes (CPU threads,
batch threads, batch, and ubatch) are searched for every target independently;
an optional final stage also measures MTP/draft rollback depths.
"""

from __future__ import annotations

import copy
import http.client
import json
import math
import os
import socket
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from scanner import ModelEntry
from server_process import ServerProcess
from settings_loader import ModelProfile
from tuner import (
    TunedConfig,
    build_command,
    check_profile_build,
    prepare_command_for_binary,
    probe_binary_build_number,
    resolve_draft_n_max,
)


class BenchmarkCancelled(RuntimeError):
    """Raised internally when the user cancels a benchmark sweep."""


class BenchmarkFailure(RuntimeError):
    """Raised when the baseline or benchmark infrastructure cannot run."""


@dataclass(frozen=True)
class BenchmarkLimits:
    """Hard bounds that keep a thorough tuning run finite."""

    max_candidates: int = 14
    confirmation_runs: int = 2
    samples_per_candidate: int = 2
    startup_timeout_s: float = 300.0
    request_timeout_s: float = 900.0
    total_timeout_s: float = 7200.0
    generated_tokens: int = 256
    min_prompt_tokens: int = 4096
    # ``None`` deliberately means uncapped. The standard 12.5% workload keeps
    # the 65,536-token guard; a user-selected custom percentage is allowed to
    # exercise the full requested fraction of context.
    max_prompt_tokens: Optional[int] = 65536
    prompt_context_fraction: float = 0.125
    # Speculative depth is explored sequentially until decode speed regresses.
    # This high safety ceiling replaces the old hard stop at n-max 8; trained
    # fixed-block drafters (DFlash/DFlash2) tighten it from GGUF metadata.
    max_draft_tokens: int = 64
    min_improvement: float = 0.03
    max_sample_spread: float = 0.35


@dataclass(frozen=True)
class BenchmarkCandidate:
    """The non-quality settings varied for one fresh server launch."""

    id: str
    label: str
    threads: int
    batch_threads: int
    batch: int
    ubatch: int
    draft_n_max: int = 0

    def apply(self, base: TunedConfig) -> TunedConfig:
        cfg = copy.copy(base)
        cfg.threads = max(1, int(self.threads))
        cfg.batch_threads = max(1, int(self.batch_threads))
        cfg.batch = max(1, int(self.batch))
        cfg.ubatch = max(1, min(int(self.ubatch), cfg.batch))
        cfg.draft_n_max = max(0, int(self.draft_n_max))
        # One slot makes the requested context exact and keeps every candidate
        # a single-user latency/throughput comparison instead of a concurrency
        # benchmark. All memory/quality choices from compute_config stay fixed.
        cfg.n_parallel = 1
        cfg.n_parallel_forced = True
        return cfg

    def settings(self) -> Dict[str, int]:
        return {
            "threads": self.threads,
            "batch_threads": self.batch_threads,
            "batch": self.batch,
            "ubatch": self.ubatch,
            "draft_n_max": self.draft_n_max,
        }


@dataclass(frozen=True)
class BenchmarkSample:
    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float

    @property
    def inference_s(self) -> float:
        """Native llama.cpp prompt + decode time reconstructed from timings."""
        return (self.prompt_tokens / self.prompt_tps) + (
            self.generated_tokens / self.generation_tps
        )

    @property
    def overall_tps(self) -> float:
        elapsed = self.inference_s
        if elapsed <= 0:
            return 0.0
        return (self.prompt_tokens + self.generated_tokens) / elapsed


@dataclass
class CandidateResult:
    candidate: BenchmarkCandidate
    samples: List[BenchmarkSample] = field(default_factory=list)
    error: str = ""
    log_tail: List[str] = field(default_factory=list)
    confirmations: int = 0

    @property
    def valid(self) -> bool:
        return bool(self.samples) and not self.error

    @property
    def prompt_tps(self) -> float:
        return statistics.median(s.prompt_tps for s in self.samples)

    @property
    def generation_tps(self) -> float:
        return statistics.median(s.generation_tps for s in self.samples)

    @property
    def overall_tps(self) -> float:
        """Median end-to-end native inference throughput for winner ranking."""
        return statistics.median(s.overall_tps for s in self.samples)

    @property
    def inference_s(self) -> float:
        return statistics.median(s.inference_s for s in self.samples)

    def sample_spread(self) -> float:
        """Worst relative min/max spread across prompt and decode samples."""
        spreads: List[float] = []
        for values in (
            [s.prompt_tps for s in self.samples],
            [s.generation_tps for s in self.samples],
        ):
            if not values or max(values) <= 0:
                return math.inf
            spreads.append((max(values) - min(values)) / max(values))
        return max(spreads, default=math.inf)


@dataclass
class BenchmarkResult:
    desired_context: int
    baseline_id: str
    winner_id: str
    candidates: List[CandidateResult]
    elapsed_s: float
    runtime_binary: str
    runtime_build: Optional[int]
    reason: str

    def by_id(self, candidate_id: str) -> CandidateResult:
        for result in self.candidates:
            if result.candidate.id == candidate_id:
                return result
        raise KeyError(candidate_id)

    @property
    def baseline(self) -> CandidateResult:
        return self.by_id(self.baseline_id)

    @property
    def winner(self) -> CandidateResult:
        return self.by_id(self.winner_id)

    def score(self, result: CandidateResult) -> float:
        """Measured end-to-end speed ratio for this exact benchmark workload."""
        base = self.baseline
        if not result.valid or not base.valid or base.overall_tps <= 0:
            return 0.0
        return result.overall_tps / base.overall_tps

    def winning_config(self, base: TunedConfig) -> TunedConfig:
        return self.winner.candidate.apply(base)

    def to_record(self, *, model_path: str, model_size: int, model_mtime_ns: int) -> dict:
        """Return bounded JSON evidence; prompts, raw argv, and full logs stay out."""
        return {
            "schema": 2,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_path": model_path,
            "model_size": int(model_size),
            "model_mtime_ns": int(model_mtime_ns),
            "desired_context": int(self.desired_context),
            "runtime_binary": self.runtime_binary,
            "runtime_build": self.runtime_build,
            "elapsed_s": round(self.elapsed_s, 3),
            "winner_id": self.winner_id,
            "reason": self.reason,
            "winner_score": round(self.score(self.winner), 6),
            "winner_settings": self.winner.candidate.settings(),
            "baseline_settings": self.baseline.candidate.settings(),
            "candidates": [
                {
                    "id": item.candidate.id,
                    "label": item.candidate.label,
                    "settings": item.candidate.settings(),
                    "prompt_tps": round(item.prompt_tps, 4) if item.valid else None,
                    "generation_tps": (
                        round(item.generation_tps, 4) if item.valid else None
                    ),
                    "overall_tps": round(item.overall_tps, 4) if item.valid else None,
                    "inference_s": round(item.inference_s, 4) if item.valid else None,
                    "score": round(self.score(item), 6) if item.valid else None,
                    "samples": [asdict(sample) for sample in item.samples],
                    "confirmations": item.confirmations,
                    "error": item.error,
                    "log_tail": item.log_tail[-20:],
                }
                for item in self.candidates
            ],
        }


ProgressCallback = Callable[[int, int, str], None]


def _candidate_key(candidate: BenchmarkCandidate) -> Tuple[int, int, int, int, int]:
    return (
        candidate.threads,
        candidate.batch_threads,
        candidate.batch,
        candidate.ubatch,
        candidate.draft_n_max,
    )


def baseline_candidate(
    cfg: TunedConfig, *, effective_draft_n_max: Optional[int] = None
) -> BenchmarkCandidate:
    draft_n_max = (
        max(0, int(effective_draft_n_max))
        if effective_draft_n_max is not None
        else max(0, int(cfg.draft_n_max or 0))
    )
    return BenchmarkCandidate(
        id="baseline",
        label="Auto baseline",
        threads=max(1, cfg.threads),
        batch_threads=max(1, cfg.batch_threads),
        batch=max(1, cfg.batch),
        ubatch=max(1, min(cfg.ubatch, cfg.batch)),
        draft_n_max=draft_n_max,
    )


def thread_candidates(
    seed: BenchmarkCandidate, physical_cores: int, logical_cores: int
) -> List[BenchmarkCandidate]:
    """Return deterministic, unique thread alternatives around ``seed``."""
    physical = max(1, int(physical_cores or seed.threads))
    logical = max(physical, int(logical_cores or physical))
    values = [physical, max(1, physical // 2), logical]
    out: List[BenchmarkCandidate] = []
    seen = {_candidate_key(seed)}
    for threads in values:
        batch_threads = min(logical, max(1, threads))
        item = BenchmarkCandidate(
            id=f"threads-{threads}-{batch_threads}",
            label=f"Threads {threads} / batch threads {batch_threads}",
            threads=threads,
            batch_threads=batch_threads,
            batch=seed.batch,
            ubatch=seed.ubatch,
            draft_n_max=seed.draft_n_max,
        )
        if _candidate_key(item) not in seen:
            seen.add(_candidate_key(item))
            out.append(item)
    return out


def batch_candidates(seed: BenchmarkCandidate) -> List[BenchmarkCandidate]:
    """Return bounded power-of-two batch/ubatch alternatives."""
    pairs = [
        (512, 256),
        (1024, 512),
        (1024, 1024),
        (2048, 512),
        (2048, 1024),
    ]
    out: List[BenchmarkCandidate] = []
    seen = {_candidate_key(seed)}
    for batch, ubatch in pairs:
        item = BenchmarkCandidate(
            id=f"batch-{batch}-{ubatch}-t{seed.threads}",
            label=f"Batch {batch} / ubatch {ubatch}",
            threads=seed.threads,
            batch_threads=seed.batch_threads,
            batch=batch,
            ubatch=min(ubatch, batch),
            draft_n_max=seed.draft_n_max,
        )
        if _candidate_key(item) not in seen:
            seen.add(_candidate_key(item))
            out.append(item)
    return out


def draft_candidates(
    seed: BenchmarkCandidate, *, maximum: int, minimum: int = 1
) -> List[BenchmarkCandidate]:
    """Return every increasing draft depth in the requested safe interval."""
    lower = max(1, int(minimum))
    upper = max(lower, int(maximum))
    out: List[BenchmarkCandidate] = []
    seen = {_candidate_key(seed)}
    for draft_n_max in range(lower, upper + 1):
        item = BenchmarkCandidate(
            id=f"draft-{draft_n_max}-t{seed.threads}-b{seed.batch}-{seed.ubatch}",
            label=f"MTP/draft n-max {draft_n_max}",
            threads=seed.threads,
            batch_threads=seed.batch_threads,
            batch=seed.batch,
            ubatch=seed.ubatch,
            draft_n_max=draft_n_max,
        )
        if _candidate_key(item) not in seen:
            seen.add(_candidate_key(item))
            out.append(item)
    return out


def parse_timing_payload(
    payload: dict,
    elapsed_s: float,
    *,
    min_prompt_tokens: int = 64,
    min_generated_tokens: int = 16,
) -> BenchmarkSample:
    """Parse current llama.cpp native-completion timing fields defensively."""
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        raise BenchmarkFailure("llama-server response has no timings object")

    def _integer(*names: str) -> int:
        for name in names:
            value = timings.get(name, payload.get(name))
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                return parsed
        return 0

    def _positive(*names: str) -> float:
        for name in names:
            value = timings.get(name)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed > 0:
                return parsed
        return 0.0

    prompt_n = _integer("prompt_n", "tokens_evaluated")
    predicted_n = _integer("predicted_n", "tokens_predicted")
    prompt_tps = _positive("prompt_per_second")
    generation_tps = _positive("predicted_per_second")

    prompt_ms = _positive("prompt_ms")
    predicted_ms = _positive("predicted_ms")
    if prompt_tps <= 0 and prompt_n > 0 and prompt_ms > 0:
        prompt_tps = prompt_n / (prompt_ms / 1000.0)
    if generation_tps <= 0 and predicted_n > 0 and predicted_ms > 0:
        generation_tps = predicted_n / (predicted_ms / 1000.0)

    if prompt_n < min_prompt_tokens:
        raise BenchmarkFailure(
            f"measurement prompt was too short ({prompt_n} tokens)"
        )
    if predicted_n < min_generated_tokens:
        raise BenchmarkFailure(
            f"generation ended too early ({predicted_n} tokens)"
        )
    if prompt_tps <= 0 or generation_tps <= 0:
        raise BenchmarkFailure("llama-server returned invalid throughput timings")
    return BenchmarkSample(
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        prompt_tokens=prompt_n,
        generated_tokens=predicted_n,
        elapsed_s=max(0.0, float(elapsed_s)),
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fixed_prompt(target_tokens: int) -> str:
    """Build a deterministic document/code mix without trivial exact repeats."""
    # Numbered records keep tokenisation stable across BPE/SPM families while
    # avoiding the old 24-word loop that made n-gram speculation unrealistically
    # easy. The response's prompt_n remains the authoritative measured count.
    # A single record is the practical floor for very small custom percentages;
    # do not silently inflate those runs back to the old 4k-token minimum.
    lines: List[str] = []
    estimated = 0
    index = 0
    while estimated < max(1, int(target_tokens)):
        checksum = (index * 2654435761) & 0xFFFFFFFF
        lines.append(
            f"Record {index:05d}: validate worker_{index % 97}(input_{index % 53}) "
            f"against policy_{index % 31}; checksum=0x{checksum:08x}. "
            "If validation succeeds, preserve ordering, explain edge cases, "
            "and return the normalized result."
        )
        estimated += 32
        index += 1
    return "\n".join(lines)


class BenchmarkRunner:
    """Run one bounded staged search against fresh private llama servers."""

    def __init__(
        self,
        *,
        model: ModelEntry,
        profile: ModelProfile,
        base_config: TunedConfig,
        runtime_binary: str,
        physical_cores: int,
        logical_cores: int,
        draft_model: Optional[ModelEntry] = None,
        use_thinking: bool = False,
        enable_speculative: bool = False,
        enable_ngram: bool = False,
        tune_draft_n_max: bool = False,
        enable_prompt_cache: bool = False,
        prompt_cache_ram_mib: int = 0,
        limits: Optional[BenchmarkLimits] = None,
        progress: Optional[ProgressCallback] = None,
        process_factory=ServerProcess,
    ) -> None:
        self.model = copy.copy(model)
        self.profile = profile
        self.base_config = copy.copy(base_config)
        self.runtime_binary = str(runtime_binary)
        self.physical_cores = max(1, int(physical_cores or 1))
        self.logical_cores = max(self.physical_cores, int(logical_cores or 1))
        self.draft_model = draft_model
        self.use_thinking = bool(use_thinking)
        self.enable_speculative = bool(enable_speculative)
        self.enable_ngram = bool(enable_ngram)
        self.tune_draft_n_max = bool(tune_draft_n_max and enable_speculative)
        self.enable_prompt_cache = bool(enable_prompt_cache)
        self.prompt_cache_ram_mib = int(prompt_cache_ram_mib)
        self.limits = limits or BenchmarkLimits()
        self.progress = progress or (lambda _done, _total, _message: None)
        self.process_factory = process_factory
        self._cancel = threading.Event()
        self._active_connection: Optional[http.client.HTTPConnection] = None
        self._connection_lock = threading.Lock()
        self._completed_runs = 0
        self._calibrated_prompt: Optional[str] = None
        self._calibrated_prompt_target = 0
        self._calibrated_prompt_tokens = 0
        self._draft_depth_min, self._draft_depth_max = self._draft_depth_bounds()
        draft_runs = (
            self._draft_depth_max - self._draft_depth_min + 1
            if self.tune_draft_n_max
            else 0
        )
        self._total_runs = (
            self.limits.max_candidates
            + self.limits.confirmation_runs
            + max(0, draft_runs)
        )
        self._deadline = 0.0

    def _draft_depth_bounds(self) -> Tuple[int, int]:
        """Return safe sequential sweep bounds for this drafter format."""
        lower = 1
        upper = max(1, int(self.limits.max_draft_tokens))
        draft = self.draft_model
        if draft is not None and draft.drafter_spec_type == "dflash":
            try:
                block_size = int(
                    (draft.metadata or {}).get("dflash.block_size", 0) or 0
                )
            except (TypeError, ValueError):
                block_size = 0
            if block_size > 1:
                # The anchor occupies one position in the trained block.
                upper = min(upper, block_size - 1)
            if draft.is_dflash2_drafter:
                # PR #27342's Qwen3.8 lattice is valid/measured at 5, 6, 7;
                # smaller values hit an early loader/shape path and cannot
                # answer the user's requested 5 -> 6 -> 7 progression.
                lower = min(upper, 5)
        return lower, max(lower, upper)

    def cancel(self) -> None:
        self._cancel.set()
        with self._connection_lock:
            connection = self._active_connection
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise BenchmarkCancelled("Performance tuning cancelled")
        if self._deadline and time.monotonic() >= self._deadline:
            raise BenchmarkFailure("Performance tuning reached its total time limit")

    def _emit(self, message: str) -> None:
        self.progress(self._completed_runs, self._total_runs, message)

    def _request_json(
        self,
        port: int,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        *,
        timeout: Optional[float] = None,
        accept_status: Sequence[int] = (200,),
    ) -> Tuple[int, dict]:
        self._check_cancelled()
        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=timeout or self.limits.request_timeout_s
        )
        with self._connection_lock:
            self._active_connection = connection
        try:
            body = None if payload is None else json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"} if body is not None else {}
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            if response.status not in accept_status:
                detail = raw.decode("utf-8", errors="replace")[-500:]
                raise BenchmarkFailure(
                    f"HTTP {response.status} from {path}: {detail}"
                )
            if not raw:
                return response.status, {}
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise BenchmarkFailure(f"non-object JSON from {path}")
            return response.status, parsed
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            if self._cancel.is_set():
                raise BenchmarkCancelled("Performance tuning cancelled") from exc
            raise BenchmarkFailure(f"HTTP request to {path} failed: {exc}") from exc
        finally:
            with self._connection_lock:
                if self._active_connection is connection:
                    self._active_connection = None
            try:
                connection.close()
            except OSError:
                pass

    def _wait_ready(self, process: ServerProcess, port: int, alias: str) -> None:
        timeout_at = min(
            self._deadline, time.monotonic() + self.limits.startup_timeout_s
        )
        last_error = "server still loading"
        while time.monotonic() < timeout_at:
            self._check_cancelled()
            if process.proc is not None and process.proc.poll() is not None:
                logs = "".join(process.get_logs())[-2000:]
                raise BenchmarkFailure(
                    f"llama-server exited during startup ({process.proc.returncode}): {logs}"
                )
            try:
                status, _body = self._request_json(
                    port,
                    "GET",
                    "/health",
                    timeout=2.0,
                    accept_status=(200, 503),
                )
                if status == 200:
                    _status, models = self._request_json(
                        port, "GET", "/v1/models", timeout=5.0
                    )
                    data = models.get("data")
                    ids = {
                        str(item.get("id"))
                        for item in data or []
                        if isinstance(item, dict) and item.get("id")
                    }
                    if alias not in ids:
                        raise BenchmarkFailure(
                            "the selected port answered, but not with this benchmark model"
                        )
                    return
            except BenchmarkCancelled:
                raise
            except BenchmarkFailure as exc:
                last_error = str(exc)
            self._cancel.wait(0.25)
        raise BenchmarkFailure(
            f"llama-server did not become ready within "
            f"{self.limits.startup_timeout_s:.0f}s ({last_error})"
        )

    def _target_prompt_tokens(self, context: int) -> int:
        """Use the selected context fraction and only its applicable bounds."""
        generated = max(1, int(self.limits.generated_tokens))
        available = max(1, int(context) - generated - 128)
        proportional = max(
            1, int(max(0, context) * self.limits.prompt_context_fraction)
        )
        requested = max(1, int(self.limits.min_prompt_tokens), proportional)
        cap = self.limits.max_prompt_tokens
        if cap is not None and int(cap) > 0:
            requested = min(requested, int(cap))
        return max(1, min(requested, available))

    def _prompt_for_server(self, port: int, target_tokens: int) -> str:
        """Calibrate deterministic text against this model's real tokenizer.

        The old fixed-text heuristic underestimated Qwen token counts by about
        1.6x, so a nominal 65k cap could actually submit >100k tokens and a
        Custom 100% run could overflow context. ``/tokenize`` is excluded from
        timings and needed only once per runner; every later candidate reuses
        the exact same calibrated prompt.
        """
        target = max(1, int(target_tokens))
        if (
            self._calibrated_prompt is not None
            and self._calibrated_prompt_target == target
        ):
            return self._calibrated_prompt

        estimate = target
        best_prompt: Optional[str] = None
        best_count = 0
        last_prompt = _fixed_prompt(estimate)
        try:
            for _attempt in range(5):
                last_prompt = _fixed_prompt(estimate)
                _status, tokenized = self._request_json(
                    port,
                    "POST",
                    "/tokenize",
                    {
                        "content": last_prompt,
                        "add_special": False,
                        "parse_special": True,
                        "with_pieces": False,
                    },
                )
                tokens = tokenized.get("tokens")
                if not isinstance(tokens, list) or not tokens:
                    raise BenchmarkFailure("llama-server /tokenize returned no tokens")
                count = len(tokens)
                if count <= target and count > best_count:
                    best_prompt, best_count = last_prompt, count
                if count <= target and count >= max(1, int(target * 0.98)):
                    best_prompt, best_count = last_prompt, count
                    break
                scaled = max(1, int(estimate * target / max(1, count) * 0.99))
                if scaled == estimate:
                    scaled = max(1, estimate - 1 if count > target else estimate + 1)
                estimate = scaled
        except BenchmarkFailure as exc:
            # Very old/forked servers may not expose /tokenize. Keep the test
            # runnable with a conservative fallback rather than restoring the
            # known 1.6x overshoot.
            self._emit(f"Prompt tokenizer calibration unavailable ({exc}); using fallback")
            best_prompt = _fixed_prompt(max(1, int(target * 0.55)))
            best_count = 0

        if best_prompt is None:
            # Tiny percentages can be below one practical record. Use that
            # minimum rather than failing preparation; the measured prompt_n
            # remains authoritative in the saved result.
            best_prompt = last_prompt
        self._calibrated_prompt = best_prompt
        self._calibrated_prompt_target = target
        self._calibrated_prompt_tokens = best_count
        return best_prompt

    def _measurement_payload(self, context: int, *, port: Optional[int] = None) -> dict:
        target_prompt = self._target_prompt_tokens(context)
        prompt = (
            self._prompt_for_server(port, target_prompt)
            if port is not None
            else _fixed_prompt(target_prompt)
        )
        return {
            "prompt": prompt,
            "n_predict": self.limits.generated_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "seed": 424242,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": False,
        }

    def _benchmark_candidate(self, candidate: BenchmarkCandidate) -> CandidateResult:
        self._check_cancelled()
        cfg = candidate.apply(self.base_config)
        port = _free_loopback_port()
        alias = f"autotuner-bench-{os.getpid()}-{port}"
        cmd = build_command(
            model=self.model,
            config=cfg,
            profile=self.profile,
            draft_model=self.draft_model if self.enable_speculative else None,
            server_binary=self.runtime_binary,
            host="127.0.0.1",
            port=port,
            extra_args=["-a", alias],
            use_thinking=self.use_thinking,
            enable_speculative=self.enable_speculative,
            enable_ngram=self.enable_ngram,
            enable_prompt_cache=self.enable_prompt_cache,
            prompt_cache_ram_mib=self.prompt_cache_ram_mib,
            enable_metrics=False,
            enable_slots_api=False,
        )
        allowed, message, _build = check_profile_build(self.profile, cmd[0])
        if not allowed:
            raise BenchmarkFailure(message or "selected llama.cpp build is too old")
        cmd, _removed = prepare_command_for_binary(cmd)
        process = self.process_factory(cmd, env_overrides=cfg.env_overrides)
        result = CandidateResult(candidate=candidate)
        try:
            process.start()
            self._wait_ready(process, port, alias)
            # Excluded warm-up: initializes kernels/JIT without favoring later
            # candidates through server-side prompt caching.
            self._request_json(
                port,
                "POST",
                "/completion",
                {
                    "prompt": "Warm-up: answer with a short deterministic sequence.",
                    "n_predict": 8,
                    "temperature": 0.0,
                    "top_k": 1,
                    "seed": 424242,
                    "ignore_eos": True,
                    "cache_prompt": False,
                    "stream": False,
                },
            )
            target_prompt_tokens = self._target_prompt_tokens(cfg.ctx)
            payload = self._measurement_payload(cfg.ctx, port=port)
            for _index in range(self.limits.samples_per_candidate):
                self._check_cancelled()
                started = time.monotonic()
                _status, response = self._request_json(
                    port, "POST", "/completion", payload
                )
                elapsed = time.monotonic() - started
                result.samples.append(
                    parse_timing_payload(
                        response,
                        elapsed,
                        min_prompt_tokens=max(1, int(target_prompt_tokens * 0.50)),
                        min_generated_tokens=max(
                            32, int(self.limits.generated_tokens * 0.80)
                        ),
                    )
                )
            if result.sample_spread() > self.limits.max_sample_spread:
                raise BenchmarkFailure(
                    "measurements were too noisy for a deterministic decision"
                )
            return result
        finally:
            try:
                result.log_tail = [
                    line.rstrip() for line in process.get_logs()[-20:] if line.strip()
                ]
            except Exception:
                pass
            try:
                process.stop()
            finally:
                # Give drivers a short bounded moment to release model/KV memory.
                self._cancel.wait(0.75)

    def _run_candidate(
        self, candidate: BenchmarkCandidate, *, confirmation: bool = False
    ) -> CandidateResult:
        phase = "Confirming" if confirmation else "Testing"
        self._emit(f"{phase}: {candidate.label}")
        try:
            result = self._benchmark_candidate(candidate)
        except (BenchmarkCancelled, BenchmarkFailure):
            raise
        except Exception as exc:
            raise BenchmarkFailure(str(exc)) from exc
        finally:
            self._completed_runs += 1
        self._emit(f"Measured: {candidate.label}")
        return result

    @staticmethod
    def _risk_distance(
        candidate: BenchmarkCandidate, baseline: BenchmarkCandidate
    ) -> float:
        return sum(
            abs(a - b) / max(1, b)
            for a, b in zip(_candidate_key(candidate), _candidate_key(baseline))
        )

    def _score(
        self, item: CandidateResult, baseline: CandidateResult
    ) -> float:
        if not item.valid or not baseline.valid or baseline.overall_tps <= 0:
            return 0.0
        return item.overall_tps / baseline.overall_tps

    def _rank(
        self,
        results: Sequence[CandidateResult],
        baseline: CandidateResult,
    ) -> List[CandidateResult]:
        valid = [item for item in results if item.valid]
        return sorted(
            valid,
            key=lambda item: (
                -self._score(item, baseline),
                self._risk_distance(item.candidate, baseline.candidate),
                item.candidate.id,
            ),
        )

    def run(self) -> BenchmarkResult:
        started = time.monotonic()
        self._deadline = started + self.limits.total_timeout_s
        if self.base_config.ctx <= 0:
            raise BenchmarkFailure("desired context must be positive")
        if self.limits.max_candidates < 1:
            raise BenchmarkFailure("at least one benchmark candidate is required")

        effective_draft_n_max: Optional[int] = None
        if self.enable_speculative:
            effective_draft_n_max = resolve_draft_n_max(
                self.profile,
                self.draft_model,
                int(self.base_config.draft_n_max or 0) or None,
            )
        baseline_spec = baseline_candidate(
            self.base_config, effective_draft_n_max=effective_draft_n_max
        )
        results: List[CandidateResult] = []
        by_id: Dict[str, CandidateResult] = {}
        seen = {_candidate_key(baseline_spec)}

        try:
            baseline = self._run_candidate(baseline_spec)
        except BenchmarkCancelled:
            raise
        except BenchmarkFailure as exc:
            raise BenchmarkFailure(f"baseline failed: {exc}") from exc
        results.append(baseline)
        by_id[baseline_spec.id] = baseline

        # Stage 1: find the best CPU thread count while every other axis stays
        # identical. Individual failures are evidence, not a reason to abort.
        for candidate in thread_candidates(
            baseline_spec, self.physical_cores, self.logical_cores
        ):
            if len(results) >= self.limits.max_candidates:
                break
            self._check_cancelled()
            try:
                measured = self._run_candidate(candidate)
            except BenchmarkCancelled:
                raise
            except BenchmarkFailure as exc:
                measured = CandidateResult(candidate=candidate, error=str(exc))
            results.append(measured)
            by_id[candidate.id] = measured
            seen.add(_candidate_key(candidate))

        best_thread = self._rank(results, baseline)[0].candidate

        # Stage 2: combine the winning thread setting with bounded batch pairs.
        for candidate in batch_candidates(best_thread):
            if len(results) >= self.limits.max_candidates:
                break
            if _candidate_key(candidate) in seen:
                continue
            self._check_cancelled()
            try:
                measured = self._run_candidate(candidate)
            except BenchmarkCancelled:
                raise
            except BenchmarkFailure as exc:
                measured = CandidateResult(candidate=candidate, error=str(exc))
            results.append(measured)
            by_id[candidate.id] = measured
            seen.add(_candidate_key(candidate))

        # Stage 3 (optional): retain the best runtime axes and increase draft
        # depth one token at a time until decode speed regresses. The old fixed
        # [1, 3, 4, 6, 8] list could stop at 6 even when 7 was still faster.
        # Draft exploration now has its own bounded budget instead of competing
        # with CPU/batch candidates for ``max_candidates`` slots.
        if self.tune_draft_n_max:
            best_runtime_result = self._rank(results, baseline)[0]
            best_runtime = best_runtime_result.candidate
            depth_candidates = {
                item.draft_n_max: item
                for item in draft_candidates(
                    best_runtime,
                    minimum=self._draft_depth_min,
                    maximum=self._draft_depth_max,
                )
            }
            previous_depth_result: Optional[CandidateResult] = None
            for depth in range(self._draft_depth_min, self._draft_depth_max + 1):
                self._check_cancelled()
                if depth == best_runtime.draft_n_max:
                    measured = best_runtime_result
                else:
                    candidate = depth_candidates[depth]
                    if _candidate_key(candidate) in seen:
                        measured = next(
                            item
                            for item in results
                            if _candidate_key(item.candidate)
                            == _candidate_key(candidate)
                        )
                    else:
                        try:
                            measured = self._run_candidate(candidate)
                        except BenchmarkCancelled:
                            raise
                        except BenchmarkFailure as exc:
                            measured = CandidateResult(
                                candidate=candidate, error=str(exc)
                            )
                        results.append(measured)
                        by_id[candidate.id] = measured
                        seen.add(_candidate_key(candidate))

                if not measured.valid:
                    # A depth the backend cannot run is a hard boundary; larger
                    # depths are not useful/safe to probe automatically.
                    break
                if (
                    previous_depth_result is not None
                    and measured.generation_tps
                    < previous_depth_result.generation_tps
                ):
                    self._emit(
                        "Draft-depth regression: "
                        f"n-max {depth} ({measured.generation_tps:.2f} tok/s) < "
                        f"n-max {previous_depth_result.candidate.draft_n_max} "
                        f"({previous_depth_result.generation_tps:.2f} tok/s); stopping"
                    )
                    break
                previous_depth_result = measured

        ranked = self._rank(results, baseline)
        finalists = ranked[: min(2, self.limits.confirmation_runs, len(ranked))]
        confirmed_ids: set[str] = set()
        # Reverse order avoids always giving the exploratory winner the same
        # thermal/order advantage during confirmation.
        for finalist in reversed(finalists):
            self._check_cancelled()
            try:
                confirmation = self._run_candidate(
                    finalist.candidate, confirmation=True
                )
            except BenchmarkCancelled:
                raise
            except BenchmarkFailure as exc:
                finalist.error = f"confirmation failed: {exc}"
                continue
            finalist.samples.extend(confirmation.samples)
            finalist.confirmations += 1
            finalist.log_tail = confirmation.log_tail
            if finalist.sample_spread() <= self.limits.max_sample_spread:
                confirmed_ids.add(finalist.candidate.id)
            else:
                finalist.error = "confirmation measurements were too noisy"

        final_ranked = self._rank(results, baseline)
        if not final_ranked:
            raise BenchmarkFailure("all benchmark candidates failed")
        winner = final_ranked[0]
        winner_score = self._score(winner, baseline)
        reason = "measured winner"
        if winner.candidate.id != baseline_spec.id and (
            winner_score < 1.0 + self.limits.min_improvement
            or (
                confirmed_ids
                and winner.candidate.id not in confirmed_ids
                and baseline_spec.id in confirmed_ids
            )
        ):
            winner = baseline
            reason = (
                "Auto baseline kept because no confirmed candidate improved the "
                f"end-to-end workload by {self.limits.min_improvement * 100:.0f}%"
            )
        elif winner.candidate.id == baseline_spec.id:
            reason = "Auto baseline remained fastest within the noise threshold"

        self._completed_runs = self._total_runs
        self._emit("Performance tuning complete")
        return BenchmarkResult(
            desired_context=int(self.base_config.ctx),
            baseline_id=baseline_spec.id,
            winner_id=winner.candidate.id,
            candidates=results,
            elapsed_s=time.monotonic() - started,
            runtime_binary=self.runtime_binary,
            runtime_build=probe_binary_build_number(self.runtime_binary),
            reason=reason,
        )


@dataclass
class BenchmarkSuiteJob:
    """One model/performance-target sweep in a sequential benchmark suite."""

    key: str
    label: str
    performance_target: str
    runner: BenchmarkRunner
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteJobResult:
    job: BenchmarkSuiteJob
    result: Optional[BenchmarkResult] = None
    error: str = ""

    @property
    def valid(self) -> bool:
        return self.result is not None and not self.error


@dataclass
class BenchmarkSuiteResult:
    jobs: List[BenchmarkSuiteJobResult]
    elapsed_s: float
    planned_jobs: int = 0
    stop_reason: str = ""

    @property
    def stopped_early(self) -> bool:
        return bool(self.stop_reason)

    @property
    def successful(self) -> List[BenchmarkSuiteJobResult]:
        return [item for item in self.jobs if item.valid]

    @property
    def failed(self) -> List[BenchmarkSuiteJobResult]:
        return [item for item in self.jobs if not item.valid]


class BenchmarkSuiteRunner:
    """Run model/target sweeps sequentially so RAM, VRAM, and thermals stay sane."""

    def __init__(
        self,
        jobs: Sequence[BenchmarkSuiteJob],
        *,
        progress: Optional[ProgressCallback] = None,
        checkpoint: Optional[Callable[[BenchmarkSuiteJobResult], None]] = None,
    ) -> None:
        self.jobs = list(jobs)
        self.progress = progress or (lambda _done, _total, _message: None)
        self.checkpoint = checkpoint
        self._cancel = threading.Event()
        self._stop_after_mode = threading.Event()
        self._stop_after_model = threading.Event()
        self._active: Optional[BenchmarkRunner] = None

    def cancel(self) -> None:
        self._cancel.set()
        if self._active is not None:
            self._active.cancel()

    def stop_after_performance_mode(self) -> None:
        """Finish the active model/mode job, then return partial results."""
        self._stop_after_mode.set()

    def stop_after_model(self) -> None:
        """Finish every remaining mode for the active model, then stop."""
        self._stop_after_model.set()

    @staticmethod
    def _job_model_key(job: BenchmarkSuiteJob) -> str:
        value = job.metadata.get("model_key") or job.metadata.get("model_path")
        if value:
            return str(value)
        # Backward-compatible fallback for callers that predate model metadata.
        return str(job.key).rsplit("::", 1)[0]

    def run(self) -> BenchmarkSuiteResult:
        if not self.jobs:
            raise BenchmarkFailure("no benchmark jobs were prepared")
        started = time.monotonic()
        total = sum(max(1, job.runner._total_runs) for job in self.jobs)
        completed = 0
        outcomes: List[BenchmarkSuiteJobResult] = []
        stop_reason = ""
        for index, job in enumerate(self.jobs, start=1):
            if self._cancel.is_set():
                raise BenchmarkCancelled("Performance tuning cancelled")
            if outcomes and self._stop_after_mode.is_set():
                stop_reason = "Stopped after the completed performance mode."
                break
            if (
                outcomes
                and self._stop_after_model.is_set()
                and self._job_model_key(job)
                != self._job_model_key(outcomes[-1].job)
            ):
                stop_reason = "Stopped after all modes for the completed model."
                break
            allocation = max(1, job.runner._total_runs)
            prefix = f"[{index}/{len(self.jobs)}] {job.label}"

            def relay(done: int, _job_total: int, message: str) -> None:
                self.progress(
                    min(total, completed + max(0, done)),
                    total,
                    f"{prefix}: {message}",
                )

            job.runner.progress = relay
            self.progress(completed, total, f"{prefix}: preparing")
            self._active = job.runner
            outcome: Optional[BenchmarkSuiteJobResult] = None
            try:
                result = job.runner.run()
                outcome = BenchmarkSuiteJobResult(job=job, result=result)
                outcomes.append(outcome)
            except BenchmarkCancelled:
                raise
            except BenchmarkFailure as exc:
                outcome = BenchmarkSuiteJobResult(job=job, error=str(exc))
                outcomes.append(outcome)
            finally:
                self._active = None
                completed += allocation
                self.progress(
                    min(total, completed),
                    total,
                    f"{prefix}: complete",
                )
            # Persist synchronously before another model/mode can start. This
            # is the crash/power-loss boundary promised by the GUI.
            if outcome is not None and self.checkpoint is not None:
                self.checkpoint(outcome)
        return BenchmarkSuiteResult(
            jobs=outcomes,
            elapsed_s=time.monotonic() - started,
            planned_jobs=len(self.jobs),
            stop_reason=stop_reason,
        )
