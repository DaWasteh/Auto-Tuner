"""Scan a models folder for GGUF files, pair them with mmproj projectors,
and pull a few useful fields from GGUF metadata when available.

GGUF format reference (v3): https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
We read only the header (KV pairs), never tensor data, so this is fast even
for 100+ GB files.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# GGUF metadata reader (minimal, no external deps)

_GGUF_MAGIC = b"GGUF"

# GGUF value type IDs
_GT_UINT8, _GT_INT8 = 0, 1
_GT_UINT16, _GT_INT16 = 2, 3
_GT_UINT32, _GT_INT32 = 4, 5
_GT_FLOAT32 = 6
_GT_BOOL = 7
_GT_STRING = 8
_GT_ARRAY = 9
_GT_UINT64, _GT_INT64 = 10, 11
_GT_FLOAT64 = 12

_SCALAR_FMT = {
    _GT_UINT8:  ("<B", 1),
    _GT_INT8:   ("<b", 1),
    _GT_UINT16: ("<H", 2),
    _GT_INT16:  ("<h", 2),
    _GT_UINT32: ("<I", 4),
    _GT_INT32:  ("<i", 4),
    _GT_FLOAT32:("<f", 4),
    _GT_BOOL:   ("<?", 1),
    _GT_UINT64: ("<Q", 8),
    _GT_INT64:  ("<q", 8),
    _GT_FLOAT64:("<d", 8),
}


def _read_value(f, vtype: int, want_array_elements: bool = True) -> Any:
    """Read one GGUF value of given type. Skips array contents to save
    memory if `want_array_elements` is False."""
    if vtype in _SCALAR_FMT:
        fmt, size = _SCALAR_FMT[vtype]
        data = f.read(size)
        if len(data) < size:
            raise EOFError("Unexpected EOF in GGUF value")
        return struct.unpack(fmt, data)[0]
    if vtype == _GT_STRING:
        ln = struct.unpack("<Q", f.read(8))[0]
        return f.read(ln).decode("utf-8", errors="replace")
    if vtype == _GT_ARRAY:
        atype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        # Token vocab arrays can be huge — skip them silently.
        if not want_array_elements or n > 256:
            for _ in range(n):
                _read_value(f, atype, want_array_elements=False)
            return None
        return [_read_value(f, atype, True) for _ in range(n)]
    raise ValueError(f"Unknown GGUF value type {vtype}")


def read_gguf_metadata(path: Path) -> Dict[str, Any]:
    """Read GGUF header KV pairs. Returns {} on any failure."""
    try:
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                return {}
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                return {}  # v1 layout differed; not worth supporting
            _n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            md: Dict[str, Any] = {}
            for _ in range(n_kv):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                md[key] = _read_value(f, vtype)
            return md
    except (OSError, struct.error, EOFError, ValueError, UnicodeDecodeError):
        return {}


def metadata_layer_count(md: Dict[str, Any]) -> int:
    """Find architecture's `block_count` (number of transformer layers)."""
    if not md:
        return 0
    arch = md.get("general.architecture")
    if arch:
        key = f"{arch}.block_count"
        if key in md:
            try:
                return int(md[key])
            except (TypeError, ValueError):
                pass
    # Fallback: scan all keys for *.block_count
    for k, v in md.items():
        if k.endswith(".block_count"):
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return 0


def metadata_native_context(md: Dict[str, Any]) -> int:
    """Find architecture's training context length."""
    if not md:
        return 0
    arch = md.get("general.architecture")
    if arch:
        key = f"{arch}.context_length"
        if key in md:
            try:
                return int(md[key])
            except (TypeError, ValueError):
                pass
    for k, v in md.items():
        if k.endswith(".context_length"):
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return 0


# ---------------------------------------------------------------------------
# Model entries + scanner

# Strip quant + extension when normalizing for mmproj pairing
_QUANT_PATTERN = re.compile(
    r"[-.]"
    r"(?:UD-)?"
    r"(?:i\d+-)?"           # i1- prefix (imatrix variants)
    r"(?:Q\d+(?:_[A-Z0-9]+)*"
    r"|IQ\d+(?:_[A-Z0-9]+)*"
    r"|BF16|F16|F32"
    r")"
    r"(?:[-.][0-9.]+bpw)?"
    r"(?:[-.](?:bf16|f16|f32))?"
    r"\.gguf$",
    re.IGNORECASE,
)


@dataclass
class ModelEntry:
    path: Path
    name: str          # display name (filename stem)
    group: str         # parent folder relative to scan root (e.g. "Alibaba/Qwen3.6")
    size_bytes: int
    mmproj: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def has_vision(self) -> bool:
        return self.mmproj is not None

    @property
    def n_layers(self) -> int:
        return metadata_layer_count(self.metadata)

    @property
    def native_context(self) -> int:
        return metadata_native_context(self.metadata)

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "") or "")


def _strip_quant(filename: str) -> str:
    if filename.lower().endswith(".gguf"):
        return _QUANT_PATTERN.sub("", filename).rstrip(".-_")
    return filename


def _normalize_model(filename: str) -> str:
    return _strip_quant(filename).lower()


def _normalize_mmproj(filename: str) -> str:
    base = _strip_quant(filename)
    low = base.lower()
    if low.startswith("mmproj-"):
        base = base[len("mmproj-"):]
    elif low.startswith("mmproj_"):
        base = base[len("mmproj_"):]
    return base.lower().rstrip(".-_")


def _find_mmproj(model: Path, candidates: List[Path]) -> Optional[Path]:
    """Pick the most-specific mmproj that matches the given model.

    A candidate matches if its normalized base is a prefix of the model's
    normalized name (same directory only). The longest matching prefix wins.
    """
    model_norm = _normalize_model(model.name)
    best: Optional[Path] = None
    best_len = 0
    for c in candidates:
        if c.parent != model.parent:
            continue
        c_norm = _normalize_mmproj(c.name)
        if not c_norm:
            continue
        if model_norm.startswith(c_norm) and len(c_norm) > best_len:
            best = c
            best_len = len(c_norm)
    return best


def scan_models(
    root: Path,
    read_metadata: bool = True,
) -> List[ModelEntry]:
    """Walk `root` recursively and return all loadable GGUF models,
    each paired with its mmproj if one is present in the same folder."""
    if not root.exists() or not root.is_dir():
        return []

    all_gguf = list(root.rglob("*.gguf"))
    mmprojs: List[Path] = []
    models: List[Path] = []
    for f in all_gguf:
        nm = f.name.lower()
        if nm.startswith("mmproj-") or nm.startswith("mmproj_"):
            mmprojs.append(f)
        else:
            models.append(f)

    entries: List[ModelEntry] = []
    for m in sorted(models):
        try:
            size = m.stat().st_size
        except OSError:
            continue
        try:
            rel = m.relative_to(root)
            parts = rel.parts
            group = "/".join(parts[:-1]) if len(parts) > 1 else "."
        except ValueError:
            group = str(m.parent)
        md = read_gguf_metadata(m) if read_metadata else {}
        entries.append(ModelEntry(
            path=m,
            name=m.stem,
            group=group,
            size_bytes=size,
            mmproj=_find_mmproj(m, mmprojs),
            metadata=md,
        ))
    return entries


def group_entries(entries: List[ModelEntry]) -> Dict[str, List[ModelEntry]]:
    """Group entries by their `group` field, preserving discovery order."""
    out: Dict[str, List[ModelEntry]] = {}
    for e in entries:
        out.setdefault(e.group, []).append(e)
    return out
