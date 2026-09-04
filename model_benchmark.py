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
    check_model_build,
    check_profile_build,
    prepare_command_for_binary,
    probe_binary_build_number,
    resolve_draft_n_max,
)


class BenchmarkCancelled(RuntimeError):
    """Raised internally when the user cancels a benchmark sweep."""


class BenchmarkFailure(RuntimeError):
    """Raised when the baseline or benchmark infrastructure cannot run."""


# Increment when persisted evidence or the search/decision procedure changes
# in a way that makes an older winner unsafe to reuse automatically.
BENCHMARK_RECORD_SCHEMA = 4
BENCHMARK_SEARCH_SCHEMA = 3


@dataclass(frozen=True)
class BenchmarkLimits:
    """Hard bounds that keep a thorough tuning run finite."""

    # v5.3.9 raised the Standard budget from 14 to 18 so the batch stage can
    # fund the classic family for two thread finalists AND the larger
    # micro-batch probes plus a hill-climb refinement round.
    max_candidates: int = 18
    confirmation_runs: int = 2
    samples_per_candidate: int = 2
    startup_timeout_s: float = 300.0
    request_timeout_s: float = 900.0
    # Zero or a negative value disables only the overall sweep deadline;
    # startup and per-request timeouts plus explicit cancellation stay active.
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
    # A single tiny decode dip is measurement noise, not proof that larger MTP
    # depths are slower. Require two consecutive regressions that exceed this
    # threshold before ending the one-token sweep; an unsupported depth remains
    # an immediate hard boundary.
    draft_regression_threshold: float = 0.03
    draft_regression_patience: int = 2


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
    draft_tokens: int = 0
    draft_tokens_accepted: int = 0

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

    @property
    def draft_acceptance(self) -> float:
        if self.draft_tokens <= 0:
            return 0.0
        return max(
            0.0,
            min(1.0, self.draft_tokens_accepted / self.draft_tokens),
        )


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

    @property
    def draft_tokens(self) -> int:
        return sum(max(0, int(sample.draft_tokens)) for sample in self.samples)

    @property
    def draft_tokens_accepted(self) -> int:
        return sum(max(0, int(sample.draft_tokens_accepted)) for sample in self.samples)

    @property
    def draft_acceptance(self) -> float:
        drafted = self.draft_tokens
        if drafted <= 0:
            return 0.0
        return max(0.0, min(1.0, self.draft_tokens_accepted / drafted))

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

    def paired_ratio_bounds(
        self, baseline: "CandidateResult", metric: str = "overall"
    ) -> Tuple[float, float]:
        """Return conservative paired sample-ratio bounds versus ``baseline``.

        Every candidate uses the same deterministic prompt variants in the same
        order, so pairwise ratios preserve workload shape better than comparing
        two independent medians. The lower bound gates winner promotion; the
        upper generation bound confirms that an MTP-depth decline is real.
        """
        count = min(len(self.samples), len(baseline.samples))
        if count <= 0:
            return 0.0, math.inf

        def value(sample: BenchmarkSample) -> float:
            if metric == "generation":
                return sample.generation_tps
            if metric == "prompt":
                return sample.prompt_tps
            return sample.overall_tps

        ratios: List[float] = []
        for index in range(count):
            base_value = value(baseline.samples[index])
            item_value = value(self.samples[index])
            if base_value <= 0 or item_value <= 0:
                continue
            ratios.append(item_value / base_value)
        if not ratios:
            return 0.0, math.inf
        return min(ratios), max(ratios)


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

    def conservative_score(self, result: CandidateResult) -> float:
        """Worst paired end-to-end ratio across deterministic sample variants."""
        if not result.valid or not self.baseline.valid:
            return 0.0
        return result.paired_ratio_bounds(self.baseline, "overall")[0]

    def winning_config(self, base: TunedConfig) -> TunedConfig:
        return self.winner.candidate.apply(base)

    def to_record(
        self, *, model_path: str, model_size: int, model_mtime_ns: int
    ) -> dict:
        """Return bounded JSON evidence; prompts, raw argv, and full logs stay out."""
        return {
            "schema": BENCHMARK_RECORD_SCHEMA,
            "search_schema": BENCHMARK_SEARCH_SCHEMA,
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
            "winner_conservative_score": round(self.conservative_score(self.winner), 6),
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
                    "draft_tokens": item.draft_tokens if item.valid else None,
                    "draft_tokens_accepted": (
                        item.draft_tokens_accepted if item.valid else None
                    ),
                    "draft_acceptance": (
                        round(item.draft_acceptance, 6) if item.valid else None
                    ),
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
    seed: BenchmarkCandidate,
    physical_cores: int,
    logical_cores: int,
    *,
    full_offload: bool = False,
) -> List[BenchmarkCandidate]:
    """Return deterministic, unique thread alternatives around ``seed``.

    The classic probes (all physical cores, half of them, every logical
    core) stay first. Two data-driven probes follow:

    * decoupled generation/prompt threads (``threads`` = physical cores,
      ``batch_threads`` = logical cores) because prompt processing scales
      with SMT threads while decode usually prefers physical cores; and
    * a deliberately small thread count when every layer already lives on
      the GPU (``full_offload``): the CPU only feeds the backend there, and
      oversubscribing it with spin-waiting workers measurably slows decode
      on many desktop CPUs.
    """
    physical = max(1, int(physical_cores or seed.threads))
    logical = max(physical, int(logical_cores or physical))
    probes: List[Tuple[int, int]] = [
        (physical, min(logical, physical)),
        (max(1, physical // 2), max(1, physical // 2)),
        (logical, logical),
    ]
    if logical > physical:
        probes.append((physical, logical))
    if full_offload and physical > 6:
        low = max(2, min(physical, 6))
        probes.append((low, low))
    out: List[BenchmarkCandidate] = []
    seen = {_candidate_key(seed)}
    for threads, batch_threads in probes:
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


_MIN_BATCH = 128
_MAX_BATCH = 8192


def _batch_pair(batch: int, ubatch: int) -> Tuple[int, int]:
    """Clamp one batch/ubatch proposal to llama.cpp's useful power-of-two range."""
    batch = max(_MIN_BATCH, min(_MAX_BATCH, int(batch)))
    ubatch = max(_MIN_BATCH // 2, min(batch, int(ubatch)))
    return batch, ubatch


def batch_candidates(seed: BenchmarkCandidate) -> List[BenchmarkCandidate]:
    """Return bounded power-of-two batch/ubatch alternatives around ``seed``.

    The classic anchors (512/256 ... 2048/1024) remain so every run still
    covers the historically validated region. They are followed by a
    geometric neighbourhood of the *seed* itself: half, equal, and double
    steps of both axes plus a larger 4096/8192 prompt-batch probe. On 24-32
    GB GPUs the larger micro-batches are frequently the real prompt
    processing optimum, and the old fixed list could never reach them.
    """
    anchors = [
        (512, 256),
        (1024, 512),
        (1024, 1024),
        (2048, 512),
        (2048, 1024),
    ]
    neighbourhood = [
        _batch_pair(seed.batch // 2, seed.ubatch // 2),
        _batch_pair(seed.batch, seed.ubatch // 2),
        _batch_pair(seed.batch, seed.ubatch * 2),
        _batch_pair(seed.batch * 2, seed.ubatch),
        _batch_pair(seed.batch * 2, seed.ubatch * 2),
        _batch_pair(2048, 2048),
        _batch_pair(4096, 1024),
        _batch_pair(4096, 2048),
    ]
    out: List[BenchmarkCandidate] = []
    seen = {_candidate_key(seed)}
    for batch, ubatch in [*anchors, *neighbourhood]:
        item = _batch_candidate(seed, batch, ubatch)
        if _candidate_key(item) not in seen:
            seen.add(_candidate_key(item))
            out.append(item)
    return out


def _batch_candidate(
    seed: BenchmarkCandidate, batch: int, ubatch: int
) -> BenchmarkCandidate:
    batch, ubatch = _batch_pair(batch, ubatch)
    return BenchmarkCandidate(
        id=f"batch-{batch}-{ubatch}-t{seed.threads}",
        label=f"Batch {batch} / ubatch {ubatch}",
        threads=seed.threads,
        batch_threads=seed.batch_threads,
        batch=batch,
        ubatch=ubatch,
        draft_n_max=seed.draft_n_max,
    )


def refine_batch_candidates(
    best: BenchmarkCandidate, tried: Sequence[BenchmarkCandidate]
) -> List[BenchmarkCandidate]:
    """Return the next untried hill-climb steps beyond the current best pair.

    The staged search measures a bounded batch/ubatch family. When the best
    pair sits on the edge of that family, the true optimum may lie one step
    further out. Each refinement proposes the immediate neighbours in the
    direction that was not yet measured (double/halve ubatch, double batch),
    so the sweep keeps climbing until a step regresses instead of stopping
    at a fixed list boundary.
    """
    seen = {(item.batch, item.ubatch) for item in tried}
    seen.add((best.batch, best.ubatch))
    proposals = [
        _batch_pair(best.batch, best.ubatch * 2),
        _batch_pair(best.batch * 2, best.ubatch * 2),
        _batch_pair(best.batch * 2, best.ubatch),
        _batch_pair(best.batch, best.ubatch // 2),
    ]
    out: List[BenchmarkCandidate] = []
    for batch, ubatch in proposals:
        if (batch, ubatch) in seen:
            continue
        seen.add((batch, ubatch))
        out.append(_batch_candidate(best, batch, ubatch))
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


def shortlist_candidates_from_record(
    record: dict,
    seed: BenchmarkCandidate,
    *,
    maximum: int = 6,
    max_spread: float = 0.20,
) -> List[BenchmarkCandidate]:
    """Build an opt-in conservative long-run shortlist from stable short evidence.

    Baseline plus the best overall candidates are retained, together with a
    representative alternative thread count and batch family. Incomplete,
    failed, single-sample, or noisy short runs return an empty list so callers
    automatically fall back to the full search.
    """
    raw_candidates = record.get("candidates") if isinstance(record, dict) else None
    if not isinstance(raw_candidates, list):
        return []
    stable: List[Tuple[float, dict]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict) or raw.get("error"):
            continue
        settings = raw.get("settings")
        samples = raw.get("samples")
        if not isinstance(settings, dict) or not isinstance(samples, list):
            continue
        valid_samples = [sample for sample in samples if isinstance(sample, dict)]
        if len(valid_samples) < 2:
            continue
        spreads: List[float] = []
        for metric_field in ("prompt_tps", "generation_tps"):
            values: List[float] = []
            for sample in valid_samples:
                try:
                    value = float(sample.get(metric_field, 0.0) or 0.0)
                except (TypeError, ValueError):
                    value = 0.0
                if value > 0.0 and math.isfinite(value):
                    values.append(value)
            if len(values) < 2 or max(values) <= 0.0:
                spreads.append(math.inf)
            else:
                spreads.append((max(values) - min(values)) / max(values))
        if max(spreads, default=math.inf) > max_spread:
            continue
        try:
            score = float(raw.get("overall_tps", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > 0.0 and math.isfinite(score):
            stable.append((score, raw))
    if len(stable) < 3:
        return []
    stable.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))

    chosen: List[dict] = []
    baseline = next(
        (raw for _score, raw in stable if str(raw.get("id", "")) == "baseline"),
        None,
    )
    if baseline is not None:
        chosen.append(baseline)
    chosen.extend(raw for _score, raw in stable[:3])

    best = stable[0][1].get("settings") or {}
    best_threads = int(best.get("threads", seed.threads) or seed.threads)
    best_batch = (
        int(best.get("batch", seed.batch) or seed.batch),
        int(best.get("ubatch", seed.ubatch) or seed.ubatch),
    )
    different_thread = next(
        (
            raw
            for _score, raw in stable
            if int((raw.get("settings") or {}).get("threads", best_threads))
            != best_threads
        ),
        None,
    )
    different_batch = next(
        (
            raw
            for _score, raw in stable
            if (
                int((raw.get("settings") or {}).get("batch", best_batch[0])),
                int((raw.get("settings") or {}).get("ubatch", best_batch[1])),
            )
            != best_batch
        ),
        None,
    )
    if different_thread is not None:
        chosen.append(different_thread)
    if different_batch is not None:
        chosen.append(different_batch)

    out: List[BenchmarkCandidate] = []
    seen: set[Tuple[int, int, int, int, int]] = set()
    for raw in chosen:
        settings = raw.get("settings") or {}
        try:
            candidate = BenchmarkCandidate(
                id=f"shortlist-{str(raw.get('id', 'candidate'))}",
                label=f"Short-pass finalist: {str(raw.get('label', raw.get('id', 'candidate')))}",
                threads=max(1, int(settings.get("threads", seed.threads))),
                batch_threads=max(
                    1, int(settings.get("batch_threads", seed.batch_threads))
                ),
                batch=max(1, int(settings.get("batch", seed.batch))),
                ubatch=max(1, int(settings.get("ubatch", seed.ubatch))),
                draft_n_max=max(0, int(settings.get("draft_n_max", seed.draft_n_max))),
            )
        except (TypeError, ValueError):
            continue
        candidate = BenchmarkCandidate(
            id=candidate.id,
            label=candidate.label,
            threads=candidate.threads,
            batch_threads=candidate.batch_threads,
            batch=candidate.batch,
            ubatch=min(candidate.ubatch, candidate.batch),
            draft_n_max=candidate.draft_n_max,
        )
        key = _candidate_key(candidate)
        if key in seen or key == _candidate_key(seed):
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= max(1, int(maximum)):
            break
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
    draft_n = _integer("draft_n", "tokens_drafted")
    draft_n_accepted = min(
        draft_n,
        _integer("draft_n_accepted", "tokens_drafted_accepted"),
    )

    prompt_ms = _positive("prompt_ms")
    predicted_ms = _positive("predicted_ms")
    if prompt_tps <= 0 and prompt_n > 0 and prompt_ms > 0:
        prompt_tps = prompt_n / (prompt_ms / 1000.0)
    if generation_tps <= 0 and predicted_n > 0 and predicted_ms > 0:
        generation_tps = predicted_n / (predicted_ms / 1000.0)

    if prompt_n < min_prompt_tokens:
        raise BenchmarkFailure(f"measurement prompt was too short ({prompt_n} tokens)")
    if predicted_n < min_generated_tokens:
        raise BenchmarkFailure(f"generation ended too early ({predicted_n} tokens)")
    if prompt_tps <= 0 or generation_tps <= 0:
        raise BenchmarkFailure("llama-server returned invalid throughput timings")
    return BenchmarkSample(
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        prompt_tokens=prompt_n,
        generated_tokens=predicted_n,
        elapsed_s=max(0.0, float(elapsed_s)),
        draft_tokens=draft_n,
        draft_tokens_accepted=draft_n_accepted,
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fixed_prompt(target_tokens: int, *, variant: int = 0) -> str:
    """Build a deterministic document/code mix without trivial exact repeats."""
    # Numbered records keep tokenisation stable across BPE/SPM families while
    # avoiding the old 24-word loop that made n-gram speculation unrealistically
    # easy. The response's prompt_n remains the authoritative measured count.
    # A single record is the practical floor for very small custom percentages;
    # do not silently inflate those runs back to the old 4k-token minimum.
    lines: List[str] = []
    estimated = 0
    index = 0
    variant_id = max(0, int(variant))
    offset = variant_id * 100_003
    operations = (
        "validate",
        "classify",
        "normalize",
        "audit",
        "refactor",
        "compare",
        "summarize",
        "prioritize",
    )
    operation = operations[variant_id % len(operations)]
    while estimated < max(1, int(target_tokens)):
        item_index = index + offset
        checksum = ((item_index + 1) * 2654435761) & 0xFFFFFFFF
        lines.append(
            f"Variant {variant_id:02d} record {item_index:07d}: {operation} "
            f"worker_{item_index % 97}(input_{item_index % 53}) against "
            f"policy_{item_index % 31}; checksum=0x{checksum:08x}. "
            f"If the {operation} step succeeds, preserve ordering, identify "
            "boundary conditions, and return a concise normalized result."
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
        candidate_plan: Optional[Sequence[BenchmarkCandidate]] = None,
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
        self.candidate_plan = list(candidate_plan or [])
        self._cancel = threading.Event()
        self._active_connection: Optional[http.client.HTTPConnection] = None
        self._connection_lock = threading.Lock()
        self._completed_runs = 0
        self._calibrated_prompts: Dict[int, str] = {}
        self._calibrated_prompt_targets: Dict[int, int] = {}
        self._calibrated_prompt_tokens: Dict[int, int] = {}
        self._draft_depth_min, self._draft_depth_max = self._draft_depth_bounds()
        draft_runs = (
            self._draft_depth_max - self._draft_depth_min + 1
            if self.tune_draft_n_max
            else 0
        )
        candidate_runs = (
            min(
                self.limits.max_candidates,
                1 + len({_candidate_key(item) for item in self.candidate_plan}),
            )
            if self.candidate_plan
            else self.limits.max_candidates
        )
        self._total_runs = (
            candidate_runs + self.limits.confirmation_runs + max(0, draft_runs)
        )
        self._deadline = 0.0

    def _sample_spread_limit(self) -> float:
        # Speculative acceptance legitimately varies across the deliberately
        # different sample corpora. Treating that workload sensitivity as
        # infrastructure noise made valid embedded/DFlash runs fail, while
        # non-speculative runs keep the stricter generic threshold.
        return max(
            float(self.limits.max_sample_spread),
            0.65 if self.enable_speculative else 0.0,
        )

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
        if self._deadline > 0.0 and time.monotonic() >= self._deadline:
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
        request_timeout = float(timeout or self.limits.request_timeout_s)
        if self._deadline > 0.0:
            request_timeout = min(
                request_timeout, max(0.1, self._deadline - time.monotonic())
            )
        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=request_timeout
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
                raise BenchmarkFailure(f"HTTP {response.status} from {path}: {detail}")
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
        startup_deadline = time.monotonic() + self.limits.startup_timeout_s
        timeout_at = (
            min(self._deadline, startup_deadline)
            if self._deadline > 0.0
            else startup_deadline
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

    def _prompt_for_server(
        self, port: int, target_tokens: int, *, variant: int = 0
    ) -> str:
        """Calibrate deterministic text against this model's real tokenizer.

        The old fixed-text heuristic underestimated Qwen token counts by about
        1.6x, so a nominal 65k cap could actually submit >100k tokens and a
        Custom 100% run could overflow context. ``/tokenize`` is excluded from
        timings and needed only once per runner; every later candidate reuses
        the exact same calibrated prompt.
        """
        target = max(1, int(target_tokens))
        variant_id = max(0, int(variant))
        if self._calibrated_prompt_targets.get(variant_id) == target:
            cached = self._calibrated_prompts.get(variant_id)
            if cached is not None:
                return cached

        estimate = target
        best_prompt: Optional[str] = None
        best_count = 0
        last_prompt = _fixed_prompt(estimate, variant=variant_id)
        try:
            for _attempt in range(5):
                last_prompt = _fixed_prompt(estimate, variant=variant_id)
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
            self._emit(
                f"Prompt tokenizer calibration unavailable ({exc}); using fallback"
            )
            best_prompt = _fixed_prompt(max(1, int(target * 0.55)), variant=variant_id)
            best_count = 0

        if best_prompt is None:
            # Tiny percentages can be below one practical record. Use that
            # minimum rather than failing preparation; the measured prompt_n
            # remains authoritative in the saved result.
            best_prompt = last_prompt
        self._calibrated_prompts[variant_id] = best_prompt
        self._calibrated_prompt_targets[variant_id] = target
        self._calibrated_prompt_tokens[variant_id] = best_count
        return best_prompt

    def _measurement_payload(
        self,
        context: int,
        *,
        port: Optional[int] = None,
        variant: int = 0,
    ) -> dict:
        target_prompt = self._target_prompt_tokens(context)
        prompt = (
            self._prompt_for_server(port, target_prompt, variant=variant)
            if port is not None
            else _fixed_prompt(target_prompt, variant=variant)
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
        allowed, message, _build = check_model_build(self.model, cmd[0])
        if not allowed:
            raise BenchmarkFailure(
                message or "selected llama.cpp build is incompatible"
            )
        cmd, adjustments = prepare_command_for_binary(cmd)
        mtp_disabled = next(
            (
                adjustment
                for adjustment in adjustments
                if adjustment.startswith("draft-mtp disabled")
            ),
            None,
        )
        if mtp_disabled is not None:
            raise BenchmarkFailure(
                "Cannot benchmark the requested MTP variant because " + mtp_disabled
            )
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
            for sample_index in range(self.limits.samples_per_candidate):
                self._check_cancelled()
                payload = self._measurement_payload(
                    cfg.ctx, port=port, variant=sample_index
                )
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
            if result.sample_spread() > self._sample_spread_limit():
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

    def _score(self, item: CandidateResult, baseline: CandidateResult) -> float:
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
        total_timeout_s = float(self.limits.total_timeout_s)
        self._deadline = started + total_timeout_s if total_timeout_s > 0.0 else 0.0
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

        if self.candidate_plan:
            # Opt-in target validation of stable short-pass finalists. An empty
            # or low-confidence shortlist is never passed by the caller, so the
            # normal full search remains the automatic fallback.
            for candidate in self.candidate_plan:
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
        else:
            # Stage 1: find the best CPU thread count while every other axis
            # stays identical. Individual failures remain visible evidence.
            for candidate in thread_candidates(
                baseline_spec,
                self.physical_cores,
                self.logical_cores,
                full_offload=bool(getattr(self.base_config, "full_offload", False)),
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

            # Stage 2: batch/ubatch can interact strongly with CPU thread count.
            # Standard's 18-candidate budget fits bounded batch families for the
            # top two distinct thread finalists; testing only the single Stage-1
            # winner could miss the global combination while leaving budget idle.
            # Finalists are distinct generation-thread counts (the best
            # batch-thread variant of each), so the decoupled 8/16 probe never
            # crowds out a genuinely different CPU configuration.
            thread_finalists: List[BenchmarkCandidate] = []
            seen_threads: set[int] = set()
            for item in self._rank(list(results), baseline):
                if item.candidate.threads in seen_threads:
                    continue
                seen_threads.add(item.candidate.threads)
                thread_finalists.append(item.candidate)
                if len(thread_finalists) >= 2:
                    break

            # Split the remaining budget: the leading finalist explores the
            # classic family plus the larger micro-batch probes, the runner-up
            # still receives the validated classic family, and any surplus
            # funds the refinement stage below.
            remaining = max(0, self.limits.max_candidates - len(results))
            quotas: List[int] = []
            if thread_finalists:
                first_quota = (remaining + 1) // 2
                quotas.append(first_quota)
                if len(thread_finalists) > 1:
                    quotas.append(min(5, remaining - first_quota))
            for thread_seed, quota in zip(thread_finalists, quotas):
                launched = 0
                for candidate in batch_candidates(thread_seed):
                    if len(results) >= self.limits.max_candidates or launched >= quota:
                        break
                    if _candidate_key(candidate) in seen:
                        continue
                    launched += 1
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
                if len(results) >= self.limits.max_candidates:
                    break

            # Stage 2b: hill-climb beyond the measured batch family. While the
            # candidate budget lasts, keep stepping the current best pair
            # outward (larger micro-batches first) and stop as soon as one
            # refinement fails to beat the incumbent by the noise threshold.
            # This is what lets the sweep discover a 4096/2048 prompt optimum
            # on large-VRAM GPUs instead of stopping at the list boundary.
            refine_rounds = 0
            while len(results) < self.limits.max_candidates and refine_rounds < 4:
                incumbent = self._rank(list(results), baseline)[0]
                proposals = refine_batch_candidates(
                    incumbent.candidate, [item.candidate for item in results]
                )
                proposals = [
                    item for item in proposals if _candidate_key(item) not in seen
                ]
                if not proposals:
                    break
                refine_rounds += 1
                improved = False
                for candidate in proposals:
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
                    if measured.valid and self._score(measured, baseline) > (
                        self._score(incumbent, baseline)
                        * (1.0 + float(self.limits.min_improvement))
                    ):
                        self._emit(
                            "Batch refinement improved the incumbent: "
                            f"{candidate.label} ({measured.overall_tps:.2f} tok/s "
                            f"vs {incumbent.overall_tps:.2f} tok/s)"
                        )
                        improved = True
                        break
                if not improved:
                    break

        # Stage 3 (optional): retain the best runtime axes and increase draft
        # depth one token at a time until two meaningful decode regressions are
        # confirmed. The old fixed [1, 3, 4, 6, 8] list could stop at 6 even
        # when 7 was still faster; a single noisy dip could do the same.
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
            best_depth_result: Optional[CandidateResult] = None
            consecutive_regressions = 0
            regression_threshold = max(
                0.0, float(self.limits.draft_regression_threshold)
            )
            regression_patience = max(1, int(self.limits.draft_regression_patience))
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
                    best_depth_result is None
                    or measured.generation_tps > best_depth_result.generation_tps
                ):
                    best_depth_result = measured
                    consecutive_regressions = 0
                    continue

                _lower_ratio, upper_ratio = measured.paired_ratio_bounds(
                    best_depth_result, "generation"
                )
                meaningful_regression = upper_ratio < 1.0 - regression_threshold
                if meaningful_regression:
                    consecutive_regressions += 1
                    self._emit(
                        "Draft-depth regression evidence: "
                        f"n-max {depth} ({measured.generation_tps:.2f} tok/s) vs "
                        f"best n-max {best_depth_result.candidate.draft_n_max} "
                        f"({best_depth_result.generation_tps:.2f} tok/s), "
                        f"{consecutive_regressions}/{regression_patience}"
                    )
                    if consecutive_regressions >= regression_patience:
                        self._emit(
                            "Draft-depth regression confirmed; stopping the "
                            "one-token sweep"
                        )
                        break
                else:
                    # A sub-threshold dip is allowed one or more chances to
                    # recover; it must not hide a faster following depth.
                    consecutive_regressions = 0

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
            if finalist.sample_spread() <= self._sample_spread_limit():
                confirmed_ids.add(finalist.candidate.id)
            else:
                finalist.error = "confirmation measurements were too noisy"

        final_ranked = self._rank(results, baseline)
        if not final_ranked:
            raise BenchmarkFailure("all benchmark candidates failed")
        winner = final_ranked[0]
        winner_score = self._score(winner, baseline)
        conservative_score = winner.paired_ratio_bounds(baseline, "overall")[0]
        reason = "measured winner"
        if winner.candidate.id != baseline_spec.id and (
            winner_score < 1.0 + self.limits.min_improvement
            or conservative_score < 1.0 + self.limits.min_improvement
            or (
                confirmed_ids
                and winner.candidate.id not in confirmed_ids
                and baseline_spec.id in confirmed_ids
            )
        ):
            winner = baseline
            reason = (
                "Auto baseline kept because no uncertainty-safe candidate "
                f"improved every paired workload sample by "
                f"{self.limits.min_improvement * 100:.0f}%"
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
                and self._job_model_key(job) != self._job_model_key(outcomes[-1].job)
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
            # Close the cancellation handoff window between the loop-level
            # check and publishing the active runner. If cancellation arrived
            # in that interval, propagate it into the now-visible runner before
            # it can launch a private server.
            if self._cancel.is_set():
                self._active.cancel()
                self._active = None
                raise BenchmarkCancelled("Performance tuning cancelled")
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
