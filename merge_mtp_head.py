"""Merge a separate NextN/MTP head GGUF into its target model.

Some community conversions ship the MTP block of a model as a trunk-less
``mtp-*.gguf`` head next to a target GGUF that was exported without it.
llama.cpp can attach such a head with ``-md`` for several architectures, but
not for all: for Qwen3.8 Flash Next (``qwen4exp``) b11371 loads target and
head and then aborts with ``GGML_ASSERT(buffer)``. The embedded form — one
GGUF whose last block is the NextN block — is the path upstream implements
and tests (PR #29761, b11330+).

This tool writes that embedded form: all target tensors (every shard of a
split GGUF), then the head's ``blk.<N>.*`` tensors, with

  * ``<arch>.block_count`` raised by the head's NextN layer count,
  * ``<arch>.nextn_predict_layers`` taken from the head,
  * per-block metadata arrays taken from the head where the head carries the
    longer (trunk + NextN) version.

The head's own ``token_embd`` / ``output`` copies are dropped; the embedded
graph shares the target's. Nothing is re-quantized and the inputs are never
modified. Only pair files that were converted from the same checkpoint.

Needs the ``gguf`` Python package of a llama.cpp checkout (gguf-py) and
numpy. The gguf-py path is taken from ``--gguf-py``, ``LLAMA_CPP_DIR`` or
the fork selected in the AutoTuner settings.

Usage:
    python merge_mtp_head.py <target.gguf> <mtp-head.gguf> <output.gguf>
    python merge_mtp_head.py --gguf-py /path/to/llama.cpp/gguf-py <...>

For a split target pass its first shard (``...-00001-of-0000N.gguf``).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from fix_gemma_assistant_arch import _find_gguf_py

_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


def split_parts(first: Path) -> list[Path]:
    """Return every shard of a split GGUF, or ``[first]`` for a single file."""
    match = _SPLIT_RE.match(first.name)
    if not match:
        return [first]
    count = int(match.group("count"))
    parts = [
        first.with_name(f"{match.group('stem')}-{no:05d}-of-{count:05d}.gguf")
        for no in range(1, count + 1)
    ]
    missing = [part.name for part in parts if not part.is_file()]
    if missing:
        raise ValueError("Missing shard(s): " + ", ".join(missing))
    return parts


def plan_merge(target_meta: dict, head_meta: dict, head_tensor_names: list[str]):
    """Validate the pair and return ``(arch, metadata_updates, block_names)``.

    ``target_meta`` / ``head_meta`` map metadata keys to plain values. Raises
    ``ValueError`` when the files do not form a trunk + NextN pair.
    """
    arch = str(target_meta.get("general.architecture") or "")
    if not arch or head_meta.get("general.architecture") != arch:
        raise ValueError(
            "Target and head must share general.architecture "
            f"(target {arch!r}, head {head_meta.get('general.architecture')!r})"
        )
    trunk_blocks = int(target_meta.get(f"{arch}.block_count") or 0)
    if int(target_meta.get(f"{arch}.nextn_predict_layers") or 0) > 0:
        raise ValueError("The target already declares an embedded NextN/MTP block")
    nextn = int(head_meta.get(f"{arch}.nextn_predict_layers") or 0)
    head_blocks = int(head_meta.get(f"{arch}.block_count") or 0)
    if trunk_blocks <= 0 or nextn <= 0 or head_blocks != trunk_blocks + nextn:
        raise ValueError(
            f"Block counts do not line up: target {trunk_blocks}, head "
            f"{head_blocks} with {nextn} NextN layer(s)"
        )
    wanted = {f"blk.{index}." for index in range(trunk_blocks, head_blocks)}
    block_names = [
        name for name in head_tensor_names if any(name.startswith(p) for p in wanted)
    ]
    if not any(".nextn." in name for name in block_names):
        raise ValueError("The head carries no blk.<N>.nextn.* tensors")
    if any(
        name.startswith("blk.") and name not in block_names
        for name in head_tensor_names
    ):
        raise ValueError("The head carries trunk blocks; it is not an MTP-only head")
    updates: dict = {
        f"{arch}.block_count": head_blocks,
        f"{arch}.nextn_predict_layers": nextn,
    }
    for key, value in target_meta.items():
        if isinstance(value, list) and len(value) == trunk_blocks:
            head_value = head_meta.get(key)
            if not isinstance(head_value, list) or len(head_value) != head_blocks:
                raise ValueError(
                    f"{key} has {trunk_blocks} per-block entries but the head "
                    "does not provide the extended array"
                )
            if head_value[:trunk_blocks] != value:
                raise ValueError(f"{key} differs between target and head trunk")
            updates[key] = head_value
    return arch, updates, block_names


def _plain_metadata(reader) -> dict:
    return {
        field.name: field.contents()
        for field in reader.fields.values()
        if not field.name.startswith("GGUF.")
    }


def merge(target: Path, head: Path, output: Path, gguf) -> int:
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    parts = split_parts(target)
    readers = [gguf.GGUFReader(part) for part in parts]
    head_reader = gguf.GGUFReader(head)
    target_meta = _plain_metadata(readers[0])
    head_meta = _plain_metadata(head_reader)
    arch, updates, block_names = plan_merge(
        target_meta, head_meta, [t.name for t in head_reader.tensors]
    )
    seen = {t.name for reader in readers for t in reader.tensors}
    clash = sorted(seen & set(block_names))
    if clash:
        raise ValueError("Target already has: " + ", ".join(clash[:4]))

    writer = gguf.GGUFWriter(output, arch=arch, endianess=readers[0].endianess)
    alignment = readers[0].get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    head_fields = {f.name: f for f in head_reader.fields.values()}
    for field in readers[0].fields.values():
        name = field.name
        if (
            name.startswith("GGUF.")
            or name.startswith("split.")
            or name == gguf.Keys.General.ARCHITECTURE
            or name == f"{arch}.nextn_predict_layers"
        ):
            continue
        source = field
        value = field.contents()
        if name in updates:
            value = updates[name]
            if isinstance(value, list):
                source = head_fields[name]
        value_type = source.types[0]
        sub_type = source.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, value, value_type, sub_type=sub_type)
    nextn_key = f"{arch}.nextn_predict_layers"
    writer.add_key_value(nextn_key, updates[nextn_key], gguf.GGUFValueType.UINT32)

    tensors = [t for reader in readers for t in reader.tensors]
    tensors += [t for t in head_reader.tensors if t.name in set(block_names)]
    total = 0
    for tensor in tensors:
        total += tensor.n_bytes
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
    print(
        f"{len(tensors)} tensors ({len(block_names)} from the head), "
        f"{total / 1024**3:.2f} GiB -> {output}"
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    done = 0
    for index, tensor in enumerate(tensors, 1):
        writer.write_tensor_data(tensor.data)
        done += tensor.n_bytes
        if index % 200 == 0 or index == len(tensors):
            print(
                f"  {index}/{len(tensors)} tensors, {done / 1024**3:.1f} GiB",
                flush=True,
            )
    writer.close()
    return len(block_names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gguf-py", default=None)
    args = parser.parse_args()
    gguf_py = _find_gguf_py(args.gguf_py)
    if gguf_py is None:
        print("gguf-py not found: pass --gguf-py or set LLAMA_CPP_DIR", file=sys.stderr)
        return 2
    sys.path.insert(0, str(gguf_py))
    import gguf

    try:
        merge(args.target, args.head, args.output, gguf)
    except ValueError as exc:
        print(f"Cannot merge: {exc}", file=sys.stderr)
        return 1
    print(
        "Done. Rescan the model folder in AutoTuner; the head file is no longer needed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
