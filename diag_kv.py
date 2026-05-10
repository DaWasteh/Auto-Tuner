"""KV-cache diagnostic — prints the metadata-derived per-token KV size
for every GGUF in a folder, side-by-side, so you can spot architectural
oddities or bad metadata.

Usage:
    python diag_kv.py D:/models                       # scan a tree
    python diag_kv.py D:/models/Gemma-4-26B-A4B...    # one specific file

What it shows for each model (the inputs to the KV formula):
    block_count          — total transformer layers
    attention_layer_count — KV-bearing layers (= block_count for pure
                            Transformer, smaller for hybrids)
    head_count           — query heads
    head_count_kv        — KV heads (after GQA reduction)
    embedding_length     — model dim
    key_length / value_length — head dim (defaults to embd/n_heads when 0)
    --
    bytes/token (f16)    — full f16 KV size per token
    MB/token (q4_0)      — what the AutoTuner would actually allocate
                           with the default K=q4_0 V=q4_0 cache quant
    GB at 32k / 64k ctx  — quick capacity sanity check

If two MoE models that look architecturally similar produce wildly
different MB/token values, look at which input column is responsible.
The two most common outliers are:
  * head_count_kv is much smaller on one (aggressive GQA / MLA)
  * block_count is much smaller on one (compact MoE / hybrid)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the scanner's GGUF reader — no new parsing logic.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scanner import (  # noqa: E402
    read_gguf_metadata,
    metadata_layer_count,
    metadata_attention_layer_count,
    metadata_is_hybrid_architecture,
)
from tuner import kv_per_token_mb_from_metadata, kv_quant_factor  # noqa: E402


def _md_int(md, key):
    v = md.get(key, 0)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def diagnose(path: Path) -> None:
    md = read_gguf_metadata(path)
    if not md:
        print(f"[!] {path.name}: could not read GGUF metadata")
        return

    arch = md.get("general.architecture", "?")
    block_count = metadata_layer_count(md)
    att_layers = metadata_attention_layer_count(md)
    is_hybrid = metadata_is_hybrid_architecture(md)
    n_heads = _md_int(md, f"{arch}.attention.head_count")
    n_kv = _md_int(md, f"{arch}.attention.head_count_kv")
    embd = _md_int(md, f"{arch}.embedding_length")
    kl = _md_int(md, f"{arch}.attention.key_length")
    vl = _md_int(md, f"{arch}.attention.value_length")
    ctx_native = _md_int(md, f"{arch}.context_length")
    expert_count = _md_int(md, f"{arch}.expert_count")
    # If head dims missing, scanner falls back to embd / n_heads
    if (kl <= 0 or vl <= 0) and n_heads > 0 and embd > 0:
        kl_eff = vl_eff = max(1, embd // n_heads)
        head_dim_note = " (derived from embd/n_heads)"
    else:
        kl_eff, vl_eff = kl, vl
        head_dim_note = ""

    f16_per_tok = kv_per_token_mb_from_metadata(md)
    q4_per_tok = f16_per_tok * kv_quant_factor("q4_0")

    # Also list every key that *contains* "ssm" — flags accidental
    # hybrid-detection.
    ssm_keys = [k for k in md.keys() if ".ssm." in k]

    print(f"━━━ {path.name} ━━━")
    print(f"  architecture          : {arch}")
    print(f"  block_count           : {block_count}")
    print(f"  attention_layer_count : {att_layers}"
          + (" (HYBRID detected!)" if is_hybrid else ""))
    print(f"  head_count            : {n_heads}")
    print(f"  head_count_kv         : {n_kv}"
          + (f"   (GQA ratio {n_heads/n_kv:.1f}:1)"
             if n_kv > 0 and n_heads > 0 else ""))
    print(f"  embedding_length      : {embd}")
    print(f"  key_length            : {kl_eff}{head_dim_note}")
    print(f"  value_length          : {vl_eff}{head_dim_note}")
    print(f"  context_length        : {ctx_native:,}")
    print(f"  expert_count (md)     : {expert_count}")
    if ssm_keys:
        print(f"  ssm keys present      : {ssm_keys[:3]}"
              + ("…" if len(ssm_keys) > 3 else ""))
    print(f"  ── KV size per token ──")
    print(f"  f16 base              : {f16_per_tok:.4f} MB/token")
    print(f"  q4_0 actual           : {q4_per_tok:.4f} MB/token")
    print(f"  → 32k ctx q4_0        : {32768 * q4_per_tok / 1024:6.2f} GB")
    print(f"  → 64k ctx q4_0        : {65536 * q4_per_tok / 1024:6.2f} GB")
    print(f"  → 128k ctx q4_0       : {131072 * q4_per_tok / 1024:6.2f} GB")
    print()


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = Path(argv[1])
    if target.is_file():
        diagnose(target)
        return
    if not target.is_dir():
        print(f"[!] {target} is neither a file nor a directory")
        sys.exit(2)
    for gguf in sorted(target.rglob("*.gguf")):
        n = gguf.name.lower()
        # Skip drafters / mmprojs
        if n.startswith("mmproj-") or n.startswith("mmproj_"):
            continue
        if "-assistant" in n or "-draft" in n:
            continue
        diagnose(gguf)


if __name__ == "__main__":
    main(sys.argv)