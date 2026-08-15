"""
experiments/quality_schema.py

The record format for the *target* variable: real 3D quality measured from a
generated output (brief section 4).

**No values are fabricated here.** This is the contract only. Every field is
something a mesh library computes from an actual generated asset, and the
validator's job is to reject a record that was filled in any other way.

Two rules the validator enforces, because the brief singles them out
-------------------------------------------------------------------
1. **"Never define 'bad output' from router statistics themselves."**
   Every failure label carries a `derived_from` provenance tag. A label
   derived from routing is rejected outright: scoring a router statistic
   against a label built from that same statistic measures nothing.

2. **Thresholds are pre-registered.** The categorical failure labels in
   section 13 come from thresholding continuous metrics, and a threshold
   chosen after seeing the router statistics is a free parameter fitted to the
   answer. The manifest carries `thresholds_registered_before_router_analysis`
   and the validator refuses a record set whose thresholds were not fixed
   first.

Reference-free vs reference-based
---------------------------------
Reference-free metrics (watertightness, component count, boundary edges,
self-intersections) need only the generated mesh, so they work on any prompt
set. Reference-based metrics (Chamfer, F-score, normal consistency against a
reference, SDF error, occupancy IoU) need ground truth, so they need a
benchmark set -- and the validator requires `reference_asset_id` whenever one
is present, so a Chamfer distance can never be recorded against nothing.

    python -m experiments.quality_schema            # print the schema
    python -m experiments.quality_schema --quality q.jsonl --manifest m.json
"""

import argparse
import json
import math
import sys

# name -> (types, required, range, description)
UNBOUNDED = (None, None)
NON_NEGATIVE = (0.0, None)
UNIT = (0.0, 1.0)

QUALITY_FIELDS = {
    # identity -- joins to the router traces
    "output_id": ((str,), True, UNBOUNDED,
                  "identifier of the generated 3D output; must match "
                  "`output_id` in the router trace records"),
    "object_id": ((str,), True, UNBOUNDED,
                  "generation run identity -- the independent statistical "
                  "unit (brief section 3)"),
    "checkpoint_hash": ((str,), True, UNBOUNDED,
                        "sha256 of the weights that produced this output"),
    "reference_asset_id": ((str,), False, UNBOUNDED,
                           "ground-truth asset, required by every "
                           "reference-based metric below"),

    # geometry, reference-free
    "watertight": ((bool,), False, UNBOUNDED, "closed manifold surface"),
    "n_connected_components": ((int,), False, NON_NEGATIVE,
                               "> 1 for a single-object prompt is the "
                               "'duplicated components' failure"),
    "n_boundary_edges": ((int,), False, NON_NEGATIVE,
                         "edges adjacent to exactly one face -- holes"),
    "boundary_edge_length": ((int, float), False, NON_NEGATIVE,
                             "total length of those edges, so hole size is "
                             "separable from hole count"),
    "n_self_intersecting_faces": ((int,), False, NON_NEGATIVE, ""),
    "n_degenerate_faces": ((int,), False, NON_NEGATIVE,
                           "zero-area or duplicate-vertex triangles"),
    "n_vertices": ((int,), False, NON_NEGATIVE, ""),
    "n_faces": ((int,), False, NON_NEGATIVE, ""),
    "surface_area": ((int, float), False, NON_NEGATIVE, ""),
    "volume": ((int, float), False, UNBOUNDED,
               "signed; negative indicates inverted normals"),
    "mean_normal_consistency": ((int, float), False, (-1.0, 1.0),
                                "mean cosine between adjacent face normals"),

    # geometry, reference-based
    "chamfer_distance": ((int, float), False, NON_NEGATIVE, ""),
    "f_score": ((int, float), False, UNIT,
                "at a stated distance threshold; see `f_score_threshold`"),
    "f_score_threshold": ((int, float), False, NON_NEGATIVE, ""),
    "normal_consistency_vs_reference": ((int, float), False, (-1.0, 1.0), ""),
    "sdf_rmse": ((int, float), False, NON_NEGATIVE, ""),
    "occupancy_iou": ((int, float), False, UNIT, ""),

    # conditioning alignment
    "render_similarity": ((int, float), False, UNBOUNDED,
                          "similarity between the conditioning image and a "
                          "render from the conditioning viewpoint; state the "
                          "scorer in the manifest"),
    "multiview_consistency": ((int, float), False, UNBOUNDED,
                              "agreement across renders from several views"),

    # labels
    "failure_categories": ((list,), False, UNBOUNDED,
                           "zero or more entries from FAILURE_CATEGORIES"),
    "label_provenance": ((dict,), True, UNBOUNDED,
                         "category -> how it was derived; see PROVENANCE. "
                         "'router' is rejected"),
    "human_quality_label": ((int, float), False, UNBOUNDED,
                            "optional human rating; state the scale in the "
                            "manifest"),
    "generator_failure_category": ((str,), False, UNBOUNDED,
                                   "generator-specific failure the pipeline "
                                   "itself reports, if any"),
    "notes": ((str,), False, UNBOUNDED, ""),
}

# brief section 13
FAILURE_CATEGORIES = (
    "missing_geometry",
    "thin_structure_failure",
    "holes",
    "duplicated_components",
    "geometric_collapse",
    "malformed_topology",
    "poor_image_alignment",
    "over_smoothed_detail",
)

PROVENANCE = (
    "threshold_on_metric",   # a pre-registered rule over a metric in this record
    "human_annotation",
    "generator_report",      # the pipeline itself flagged it
)

REFERENCE_BASED = ("chamfer_distance", "f_score",
                   "normal_consistency_vs_reference", "sdf_rmse",
                   "occupancy_iou")

QUANTITATIVE = tuple(
    n for n, (types, _, _, _) in QUALITY_FIELDS.items()
    if types in ((int,), (int, float)) or types == (bool,)
)

MANIFEST_FIELDS = {
    "quality_run_id": ((str,), True, "identifier for this measurement pass"),
    "checkpoint_hash": ((str,), True, "weights the outputs came from"),
    "metric_versions": ((dict,), True,
                        "metric or library -> version/commit; a quality "
                        "number that cannot be re-derived is not a result"),
    "thresholds": ((dict,), True,
                   "category -> the rule that produces it, e.g. "
                   "{'holes': {'field': 'boundary_edge_length', 'op': '>', "
                   "'value': 0.01}}"),
    "thresholds_registered_before_router_analysis": ((bool,), True,
                                                     "must be true"),
    "threshold_registration_commit": ((str,), True,
                                      "git commit in which the thresholds "
                                      "were fixed, so the claim is checkable"),
    "reference_asset_set": ((str,), False,
                            "identifier of the ground-truth set, if any"),
    "render_scorer": ((str,), False,
                      "what computed render_similarity, if present"),
    "human_label_scale": ((str,), False, "if human labels are present"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_quality_record(rec, label="quality"):
    problems = []
    if not isinstance(rec, dict):
        return ["%s is not an object" % label]

    for name, (types, required, rng, _) in QUALITY_FIELDS.items():
        if name not in rec or rec[name] is None:
            if required:
                problems.append("%s: missing required field '%s'" % (label, name))
            continue
        value = rec[name]
        if isinstance(value, bool) and types != (bool,):
            problems.append("%s: field '%s' is a bool" % (label, name))
            continue
        if not isinstance(value, types):
            problems.append("%s: field '%s' has type %s, expected %s"
                            % (label, name, type(value).__name__,
                               "/".join(t.__name__ for t in types)))
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value != value or value in (float("inf"), float("-inf")):
                problems.append("%s: field '%s' is not finite" % (label, name))
                continue
            lo, hi = rng
            if lo is not None and value < lo:
                problems.append("%s: field '%s' = %r is below %r"
                                % (label, name, value, lo))
            if hi is not None and value > hi:
                problems.append("%s: field '%s' = %r is above %r"
                                % (label, name, value, hi))

    unknown = set(rec) - set(QUALITY_FIELDS)
    if unknown:
        problems.append("%s: unknown field(s) %s" % (label, sorted(unknown)))

    measured = [n for n in QUANTITATIVE if rec.get(n) is not None]
    if not measured:
        problems.append("%s: no quantitative measurement present; a record "
                        "with only labels cannot serve as the target variable "
                        "(brief section 4)" % label)

    if any(rec.get(n) is not None for n in REFERENCE_BASED) \
            and not rec.get("reference_asset_id"):
        problems.append("%s: a reference-based metric is present without "
                        "reference_asset_id" % label)
    if rec.get("f_score") is not None and rec.get("f_score_threshold") is None:
        problems.append("%s: f_score without f_score_threshold is not "
                        "interpretable" % label)

    cats = rec.get("failure_categories") or []
    bad = [c for c in cats if c not in FAILURE_CATEGORIES]
    if bad:
        problems.append("%s: unknown failure categor%s %s; expected from %s"
                        % (label, "y" if len(bad) == 1 else "ies", bad,
                           list(FAILURE_CATEGORIES)))
    prov = rec.get("label_provenance") or {}
    for cat in cats:
        if cat not in prov:
            problems.append("%s: failure category '%s' has no entry in "
                            "label_provenance" % (label, cat))
    for cat, how in prov.items():
        if how == "router" or (isinstance(how, str) and "router" in how.lower()):
            problems.append(
                "%s: label '%s' is marked as derived from router statistics. "
                "The target variable must be measured from the 3D output; a "
                "router statistic scored against a router-derived label "
                "measures nothing (brief section 4)." % (label, cat))
        elif how not in PROVENANCE:
            problems.append("%s: label_provenance['%s'] = %r; expected one of %s"
                            % (label, cat, how, list(PROVENANCE)))
    return problems


def validate_quality_manifest(manifest, label="quality manifest"):
    problems = []
    if not isinstance(manifest, dict):
        return ["%s is not an object" % label]
    for name, (types, required, _) in MANIFEST_FIELDS.items():
        if name not in manifest or manifest[name] is None:
            if required:
                problems.append("%s: missing required field '%s'" % (label, name))
            continue
        if not isinstance(manifest[name], types):
            problems.append("%s: field '%s' has type %s, expected %s"
                            % (label, name, type(manifest[name]).__name__,
                               "/".join(t.__name__ for t in types)))
    unknown = set(manifest) - set(MANIFEST_FIELDS)
    if unknown:
        problems.append("%s: unknown field(s) %s" % (label, sorted(unknown)))
    if problems:
        return problems

    if not manifest["metric_versions"]:
        problems.append("%s: metric_versions is empty" % label)
    if not manifest["thresholds_registered_before_router_analysis"]:
        problems.append(
            "%s: thresholds_registered_before_router_analysis is false. Fix "
            "the failure thresholds and record the commit before computing "
            "any router statistic, or the categories in brief section 13 are "
            "fitted to the answer." % label)
    for cat, rule in manifest["thresholds"].items():
        if cat not in FAILURE_CATEGORIES:
            problems.append("%s: threshold for unknown category '%s'"
                            % (label, cat))
        if not isinstance(rule, dict) or "field" not in rule:
            problems.append("%s: threshold for '%s' does not name a field"
                            % (label, cat))
        elif rule["field"] not in QUALITY_FIELDS:
            problems.append("%s: threshold for '%s' names unknown field '%s'"
                            % (label, cat, rule["field"]))
    return problems


def validate_file(manifest_path, quality_path, trace_output_ids=None):
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    problems = validate_quality_manifest(manifest)
    seen_objects, seen_outputs = set(), set()
    n = 0
    with open(quality_path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                problems.append("quality line %d: not JSON (%s)" % (i + 1, exc))
                continue
            problems.extend(validate_quality_record(
                rec, label="quality line %d" % (i + 1)))
            seen_objects.add(rec.get("object_id"))
            seen_outputs.add(rec.get("output_id"))
            n += 1
    if len(seen_objects) < 2:
        problems.append("only %d distinct object_id across %d records: "
                        "object-level splits and bootstrap need more than one "
                        "object (brief section 3)" % (len(seen_objects), n))
    if trace_output_ids is not None:
        missing = sorted(set(trace_output_ids) - seen_outputs)
        if missing:
            problems.append("%d generated outputs have router traces but no "
                            "quality record (e.g. %s)"
                            % (len(missing), missing[:3]))
    return manifest, n, len(seen_objects), problems


def print_schema():
    print("=" * 78)
    print("QualityManifest")
    print("=" * 78)
    for name, (types, required, desc) in MANIFEST_FIELDS.items():
        print("  %-46s %-8s %s" % (name, "/".join(t.__name__ for t in types),
                                   "required" if required else "optional"))
        if desc:
            print("      %s" % desc)
    print()
    print("=" * 78)
    print("QualityRecord")
    print("=" * 78)
    for name, (types, required, rng, desc) in QUALITY_FIELDS.items():
        bounds = ""
        if rng != UNBOUNDED:
            bounds = "  range %s" % (rng,)
        print("  %-34s %-12s %-9s%s"
              % (name, "/".join(t.__name__ for t in types),
                 "required" if required else "optional", bounds))
        if desc:
            print("      %s" % desc)
    print()
    print("failure categories: %s" % ", ".join(FAILURE_CATEGORIES))
    print("label provenance  : %s   ('router' is rejected)"
          % ", ".join(PROVENANCE))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--quality")
    a = p.parse_args()
    if not a.manifest or not a.quality:
        print_schema()
        print("no --manifest/--quality given; printed the schema instead.")
        print("Gate status: python -m experiments.audit_environment")
        return 0
    _, n, n_obj, problems = validate_file(a.manifest, a.quality)
    print("manifest: %s" % a.manifest)
    print("quality : %s (%d records, %d distinct objects)"
          % (a.quality, n, n_obj))
    if problems:
        print("\n%d problem(s):" % len(problems))
        for pr in problems[:200]:
            print("  - %s" % pr)
        return 1
    print("\nOK -- schema satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
