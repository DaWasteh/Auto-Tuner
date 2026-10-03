"""merge_mtp_head: pair validation and metadata plan (no gguf-py needed)."""

import pytest

from merge_mtp_head import plan_merge, split_parts


def _target(**extra):
    meta = {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 6,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4, 0, 0],
        "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
        "qwen4exp.ple.layers": [0],
    }
    meta.update(extra)
    return meta


def _head(**extra):
    meta = {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 7,
        "qwen4exp.nextn_predict_layers": 1,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4, 0, 0, 0],
        "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
    }
    meta.update(extra)
    return meta


HEAD_TENSORS = [
    "output.weight",
    "token_embd.weight",
    "blk.6.attn_q.weight",
    "blk.6.nextn.eh_proj.weight",
    "blk.6.nextn.hc_head_norm.weight",
]


def test_plan_extends_block_count_and_per_block_arrays():
    arch, updates, names = plan_merge(_target(), _head(), HEAD_TENSORS)
    assert arch == "qwen4exp"
    assert updates == {
        "qwen4exp.block_count": 7,
        "qwen4exp.nextn_predict_layers": 1,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4, 0, 0, 0],
    }
    # The head's embedding/output copies are dropped; only its block moves.
    assert names == HEAD_TENSORS[2:]


@pytest.mark.parametrize(
    "target,head,tensors,message",
    [
        (_target(), _head(**{"general.architecture": "qwen35"}), HEAD_TENSORS, "share"),
        (
            _target(**{"qwen4exp.nextn_predict_layers": 1}),
            _head(),
            HEAD_TENSORS,
            "already declares",
        ),
        (_target(), _head(**{"qwen4exp.block_count": 8}), HEAD_TENSORS, "line up"),
        (
            _target(),
            _head(**{"qwen4exp.nextn_predict_layers": 0}),
            HEAD_TENSORS,
            "line up",
        ),
        (_target(), _head(), ["token_embd.weight", "blk.6.attn_q.weight"], "nextn"),
        (_target(), _head(), HEAD_TENSORS + ["blk.0.attn_q.weight"], "trunk blocks"),
        (
            _target(),
            _head(**{"qwen4exp.attention.compress_ratios": [0, 0, 0, 4, 0, 0]}),
            HEAD_TENSORS,
            "extended array",
        ),
        (
            _target(),
            _head(**{"qwen4exp.attention.compress_ratios": [4, 0, 0, 4, 0, 0, 0]}),
            HEAD_TENSORS,
            "differs",
        ),
    ],
)
def test_plan_rejects_files_that_are_not_a_trunk_and_head_pair(
    target, head, tensors, message
):
    with pytest.raises(ValueError, match=message):
        plan_merge(target, head, tensors)


def test_split_parts_lists_every_shard_and_reports_missing(tmp_path):
    single = tmp_path / "model.gguf"
    single.write_bytes(b"GGUF")
    assert split_parts(single) == [single]
    parts = [tmp_path / f"m-{n:05d}-of-00003.gguf" for n in (1, 2, 3)]
    for part in parts[:2]:
        part.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="m-00003-of-00003.gguf"):
        split_parts(parts[0])
    parts[2].write_bytes(b"GGUF")
    assert split_parts(parts[0]) == parts
