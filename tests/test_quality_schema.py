"""
tests/test_quality_schema.py

Checks on the 3D-quality contract in `experiments/quality_schema.py`.

The two tests that matter most are the ones enforcing the brief's rules about
how the target variable may be constructed:

  - a failure label derived from router statistics is rejected outright;
  - a manifest whose thresholds were not fixed before the router analysis is
    rejected.

Both are the kind of mistake that produces a perfectly plausible AUC.

No values here are measurements. They are fixtures with the shape of a
measurement, used to check the validator.
"""

import json

import pytest

from experiments.quality_schema import (
    FAILURE_CATEGORIES, validate_file, validate_quality_manifest,
    validate_quality_record,
)


def make_record(**overrides):
    rec = {
        "output_id": "obj-0001.glb",
        "object_id": "obj-0001",
        "checkpoint_hash": "b" * 64,
        "watertight": True,
        "n_connected_components": 1,
        "n_boundary_edges": 0,
        "boundary_edge_length": 0.0,
        "n_self_intersecting_faces": 0,
        "n_degenerate_faces": 2,
        "n_vertices": 15234,
        "n_faces": 30460,
        "surface_area": 6.1834,
        "volume": 0.9271,
        "mean_normal_consistency": 0.973,
        "failure_categories": [],
        "label_provenance": {},
    }
    rec.update(overrides)
    return rec


def make_manifest(**overrides):
    m = {
        "quality_run_id": "quality-0001",
        "checkpoint_hash": "b" * 64,
        "metric_versions": {"trimesh": "4.4.0", "open3d": "0.18.0"},
        "thresholds": {
            "holes": {"field": "boundary_edge_length", "op": ">", "value": 0.01},
            "duplicated_components": {"field": "n_connected_components",
                                      "op": ">", "value": 1},
        },
        "thresholds_registered_before_router_analysis": True,
        "threshold_registration_commit": "0123456789abcdef",
    }
    m.update(overrides)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_record_passes():
    assert validate_quality_record(make_record()) == []


def test_a_record_with_no_measurement_is_rejected():
    """Labels alone cannot serve as the target variable."""
    rec = {"output_id": "o", "object_id": "obj", "checkpoint_hash": "b" * 64,
           "failure_categories": ["holes"],
           "label_provenance": {"holes": "human_annotation"}}
    problems = validate_quality_record(rec)
    assert any("no quantitative measurement" in p for p in problems), problems


def test_router_derived_labels_are_rejected():
    rec = make_record(failure_categories=["geometric_collapse"],
                      label_provenance={"geometric_collapse": "router"})
    problems = validate_quality_record(rec)
    assert any("derived from router statistics" in p for p in problems), problems


@pytest.mark.parametrize("provenance", [
    "router", "router_entropy_threshold", "Router covariance drift"])
def test_router_provenance_is_caught_however_it_is_spelled(provenance):
    rec = make_record(failure_categories=["holes"],
                      label_provenance={"holes": provenance})
    problems = validate_quality_record(rec)
    assert any("derived from router statistics" in p for p in problems), problems


def test_unknown_provenance_is_rejected():
    rec = make_record(failure_categories=["holes"],
                      label_provenance={"holes": "vibes"})
    problems = validate_quality_record(rec)
    assert any("label_provenance" in p for p in problems), problems


def test_label_without_provenance_is_rejected():
    rec = make_record(failure_categories=["holes"], label_provenance={})
    problems = validate_quality_record(rec)
    assert any("no entry in label_provenance" in p for p in problems), problems


def test_unknown_failure_category_is_rejected():
    rec = make_record(failure_categories=["looks_bad"],
                      label_provenance={"looks_bad": "human_annotation"})
    problems = validate_quality_record(rec)
    assert any("unknown failure categor" in p for p in problems), problems


@pytest.mark.parametrize("category", FAILURE_CATEGORIES)
def test_every_documented_category_is_accepted(category):
    rec = make_record(failure_categories=[category],
                      label_provenance={category: "threshold_on_metric"})
    assert validate_quality_record(rec) == []


def test_reference_based_metric_needs_a_reference():
    rec = make_record(chamfer_distance=0.0134)
    problems = validate_quality_record(rec)
    assert any("reference_asset_id" in p for p in problems), problems
    rec["reference_asset_id"] = "gt-0001"
    assert validate_quality_record(rec) == []


def test_f_score_without_its_threshold_is_not_interpretable():
    rec = make_record(f_score=0.83, reference_asset_id="gt-0001")
    problems = validate_quality_record(rec)
    assert any("f_score_threshold" in p for p in problems), problems


@pytest.mark.parametrize("field,value", [
    ("occupancy_iou", 1.5),
    ("occupancy_iou", -0.1),
    ("chamfer_distance", -1.0),
    ("n_connected_components", -2),
    ("mean_normal_consistency", 1.4),
])
def test_out_of_range_values_are_rejected(field, value):
    rec = make_record(**{field: value})
    if field in ("chamfer_distance", "occupancy_iou"):
        rec["reference_asset_id"] = "gt-0001"
    problems = validate_quality_record(rec)
    assert any(field in p for p in problems), problems


def test_non_finite_values_are_rejected():
    rec = make_record(surface_area=float("inf"))
    problems = validate_quality_record(rec)
    assert any("not finite" in p for p in problems), problems


def test_unknown_fields_are_reported():
    rec = make_record(secret_score=0.5)
    problems = validate_quality_record(rec)
    assert any("unknown field" in p for p in problems), problems


def test_missing_identity_is_rejected():
    rec = make_record()
    del rec["object_id"]
    problems = validate_quality_record(rec)
    assert any("object_id" in p for p in problems), problems


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_manifest_passes():
    assert validate_quality_manifest(make_manifest()) == []


def test_unregistered_thresholds_are_rejected():
    problems = validate_quality_manifest(
        make_manifest(thresholds_registered_before_router_analysis=False))
    assert any("fitted to the answer" in p for p in problems), problems


def test_empty_metric_versions_are_rejected():
    problems = validate_quality_manifest(make_manifest(metric_versions={}))
    assert any("metric_versions is empty" in p for p in problems), problems


def test_threshold_naming_an_unknown_field_is_rejected():
    problems = validate_quality_manifest(make_manifest(
        thresholds={"holes": {"field": "router_entropy", "op": ">", "value": 1}}))
    assert any("unknown field" in p for p in problems), problems


def test_threshold_for_an_unknown_category_is_rejected():
    problems = validate_quality_manifest(make_manifest(
        thresholds={"vibes": {"field": "surface_area", "op": ">", "value": 1}}))
    assert any("unknown category" in p for p in problems), problems


# ─────────────────────────────────────────────────────────────────────────────
# File level
# ─────────────────────────────────────────────────────────────────────────────

def _write(tmp_path, manifest, records):
    mpath = tmp_path / "quality_manifest.json"
    qpath = tmp_path / "quality.jsonl"
    mpath.write_text(json.dumps(manifest))
    qpath.write_text("\n".join(json.dumps(r) for r in records))
    return str(mpath), str(qpath)


def test_validate_file_accepts_a_well_formed_pair(tmp_path):
    recs = [make_record(object_id="obj-%04d" % i, output_id="obj-%04d.glb" % i)
            for i in range(4)]
    mpath, qpath = _write(tmp_path, make_manifest(), recs)
    _, n, n_obj, problems = validate_file(mpath, qpath)
    assert (n, n_obj, problems) == (4, 4, [])


def test_validate_file_flags_a_single_object_dataset(tmp_path):
    recs = [make_record() for _ in range(3)]
    mpath, qpath = _write(tmp_path, make_manifest(), recs)
    _, _, n_obj, problems = validate_file(mpath, qpath)
    assert n_obj == 1
    assert any("object-level splits" in p for p in problems), problems


def test_validate_file_notices_traces_without_quality(tmp_path):
    recs = [make_record(object_id="obj-%04d" % i, output_id="obj-%04d.glb" % i)
            for i in range(2)]
    mpath, qpath = _write(tmp_path, make_manifest(), recs)
    trace_outputs = ["obj-0000.glb", "obj-0001.glb", "obj-0002.glb"]
    _, _, _, problems = validate_file(mpath, qpath, trace_output_ids=trace_outputs)
    assert any("no quality record" in p for p in problems), problems
