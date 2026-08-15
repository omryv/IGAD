"""
tests/test_trace_schema.py

The trace contract is the one piece of the post-gate pipeline that can be
tested without a checkpoint, so it is tested properly.

The records below are hand-built fixtures, not captured data, and nothing here
implies that a real MoE 3D generator was run. They exist so that whoever
instruments one finds a schema violation at capture time rather than after the
GPU hours are spent.
"""

import json

import pytest

from experiments.trace_schema import (
    validate_file, validate_manifest, validate_trace_record,
    verify_pre_topk_capture,
)


def make_manifest(**overrides):
    m = {
        "run_id": "run-0001",
        "checkpoint_hash": "a" * 64,
        "checkpoint_id": "some-org/some-moe-3d-model@rev",
        "model_config": {"hidden": 1024},
        "n_experts": 8,
        "top_k": 2,
        "routing_mode": "token-choice",
        "moe_layer_ids": [4, 5, 6, 7],
        "n_layers": 12,
        "shared_experts": 1,
        "seeds": [0, 1, 2],
        "generator_settings": {"steps": 50, "guidance": 7.5},
        "quality_metric_versions": {"trimesh": "4.4.0"},
        "capture_code_commit": "0123456789abcdef",
        "pre_topk_verified": True,
    }
    m.update(overrides)
    return m


def make_record(probs=None, **overrides):
    probs = probs or [0.30, 0.22, 0.14, 0.11, 0.09, 0.07, 0.04, 0.03]
    rec = {
        "object_id": "obj-0001",
        "seed": 0,
        "conditioning_id": "img-0001",
        "step": 3,
        "n_steps": 50,
        "layer": 5,
        "token": 128,
        "router_probs": probs,
        "selected_experts": [0, 1],
        "selected_weights": [0.577, 0.423],
        "output_id": "obj-0001.glb",
        "router_logits": [1.0, 0.7, 0.25, 0.0, -0.2, -0.45, -1.0, -1.3],
    }
    rec.update(overrides)
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_record_passes():
    assert validate_trace_record(make_record(), n_experts=8, top_k=2) == []


def test_post_topk_vector_is_rejected():
    """The whole point of the contract: k - top_k hard zeros means the hook
    was placed after the mask."""
    masked = [0.577, 0.423] + [0.0] * 6
    problems = validate_trace_record(make_record(probs=masked),
                                     n_experts=8, top_k=2)
    assert any("POST-top-k" in p for p in problems), problems


def test_probabilities_must_be_a_simplex():
    problems = validate_trace_record(
        make_record(probs=[0.5] * 8), n_experts=8, top_k=2)
    assert any("sums to" in p for p in problems), problems

    problems = validate_trace_record(
        make_record(probs=[-0.1, 0.3, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]),
        n_experts=8, top_k=2)
    assert any("negative" in p for p in problems), problems


def test_missing_required_field_is_rejected():
    rec = make_record()
    del rec["object_id"]
    problems = validate_trace_record(rec, n_experts=8, top_k=2)
    assert any("object_id" in p for p in problems), problems


def test_selected_experts_are_checked_against_top_k():
    problems = validate_trace_record(
        make_record(selected_experts=[0, 1, 2], selected_weights=[0.4, 0.4, 0.2]),
        n_experts=8, top_k=2)
    assert any("selected experts" in p for p in problems), problems

    problems = validate_trace_record(
        make_record(selected_experts=[0, 0], selected_weights=[0.5, 0.5]),
        n_experts=8, top_k=2)
    assert any("duplicates" in p for p in problems), problems

    problems = validate_trace_record(
        make_record(selected_experts=[0, 99]), n_experts=8, top_k=2)
    assert any("out of range" in p for p in problems), problems


def test_expert_count_mismatch_is_caught():
    problems = validate_trace_record(make_record(), n_experts=64, top_k=2)
    assert any("n_experts" in p for p in problems), problems


def test_step_must_be_inside_the_schedule():
    problems = validate_trace_record(make_record(step=50), n_experts=8, top_k=2)
    assert any("outside" in p for p in problems), problems


def test_unknown_fields_are_reported():
    problems = validate_trace_record(make_record(extra_field="x"),
                                     n_experts=8, top_k=2)
    assert any("unknown field" in p for p in problems), problems


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_manifest_passes():
    assert validate_manifest(make_manifest()) == []


@pytest.mark.parametrize("overrides,fragment", [
    ({"top_k": 16}, "exceeds n_experts"),
    ({"routing_mode": "magic"}, "routing_mode"),
    ({"moe_layer_ids": []}, "empty"),
    ({"moe_layer_ids": [4, 99]}, "outside"),
    ({"checkpoint_hash": "abc"}, "too short"),
    ({"pre_topk_verified": False}, "pre_topk_verified"),
    ({"quality_metric_versions": {}}, "quality"),
])
def test_manifest_rejections(overrides, fragment):
    problems = validate_manifest(make_manifest(**overrides))
    assert any(fragment in p for p in problems), problems


# ─────────────────────────────────────────────────────────────────────────────
# The pre-top-k acceptance test
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_pre_topk_accepts_softmax_output():
    ok, msg = verify_pre_topk_capture([make_record() for _ in range(10)], top_k=2)
    assert ok, msg


def test_verify_pre_topk_rejects_masked_output():
    masked = [make_record(probs=[0.6, 0.4] + [0.0] * 6) for _ in range(10)]
    ok, msg = verify_pre_topk_capture(masked, top_k=2)
    assert not ok and "post-top-k" in msg


def test_verify_pre_topk_rejects_stray_zeros():
    """Not the top-k signature, but softmax output still cannot contain zeros."""
    recs = [make_record(probs=[0.34, 0.22, 0.14, 0.11, 0.09, 0.07, 0.03, 0.0])
            for _ in range(10)]
    ok, msg = verify_pre_topk_capture(recs, top_k=2)
    assert not ok and "exact zero" in msg


def test_verify_pre_topk_declines_when_it_cannot_discriminate():
    ok, msg = verify_pre_topk_capture([make_record()], top_k=8)
    assert not ok and "cannot discriminate" in msg


# ─────────────────────────────────────────────────────────────────────────────
# File level
# ─────────────────────────────────────────────────────────────────────────────

def _write(tmp_path, manifest, records):
    mpath = tmp_path / "manifest.json"
    tpath = tmp_path / "traces.jsonl"
    mpath.write_text(json.dumps(manifest))
    tpath.write_text("\n".join(json.dumps(r) for r in records))
    return str(mpath), str(tpath)


def test_validate_file_accepts_a_well_formed_pair(tmp_path):
    records = [make_record(object_id="obj-%04d" % i, token=i) for i in range(4)]
    mpath, tpath = _write(tmp_path, make_manifest(), records)
    _, n, n_obj, problems = validate_file(mpath, tpath)
    assert (n, n_obj, problems) == (4, 4, [])


def test_validate_file_flags_single_object_datasets(tmp_path):
    """Object-level bootstrap is impossible with one object; say so at capture
    time rather than after the analysis is written (brief section 3)."""
    records = [make_record(token=i) for i in range(4)]
    mpath, tpath = _write(tmp_path, make_manifest(), records)
    _, _, n_obj, problems = validate_file(mpath, tpath)
    assert n_obj == 1
    assert any("independent statistical unit" in p for p in problems), problems


def test_validate_file_reports_malformed_lines(tmp_path):
    mpath = tmp_path / "manifest.json"
    tpath = tmp_path / "traces.jsonl"
    mpath.write_text(json.dumps(make_manifest()))
    tpath.write_text(json.dumps(make_record()) + "\n{not json\n")
    _, _, _, problems = validate_file(str(mpath), str(tpath))
    assert any("not JSON" in p for p in problems), problems
