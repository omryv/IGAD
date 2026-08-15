"""
experiments/trace_schema.py

The record format a real MoE 3D generator has to emit before any of the
experiments in `docs/acquisition_checklist.md` can run, plus a validator.

This is the one piece of the post-gate pipeline that can be built and tested
without a checkpoint: it fixes the contract, so that whoever instruments the
generator finds out at capture time -- not at analysis time -- that a field is
missing, that the recorded vector is post-top-k rather than pre-top-k, or that
the run manifest cannot identify which checkpoint produced it.

Two record types:

`RouterTraceRecord`
    one per (object, generation step, MoE layer, token). Carries the full
    pre-top-k probability simplex, not the selected expert IDs. The brief is
    explicit about this: the covariance and spectral statistics in section 7
    are computed on log-ratio coordinates of the *whole* simplex, and a
    post-top-k vector has k - top_k structural zeros that destroy them.

`RunManifest`
    one per generation run. Carries everything section 18 requires to
    reproduce a number: checkpoint hash, model configuration, expert count,
    top-k, MoE layer IDs, seed, generator settings, quality metric versions.

Usage at capture time:

    from experiments.trace_schema import validate_manifest, validate_trace_record
    problems = validate_manifest(manifest)
    if problems:
        raise SystemExit("manifest rejected: " + "; ".join(problems))

Usage after capture:

    python -m experiments.trace_schema --manifest run.json --traces traces.jsonl
"""

import argparse
import json
import math
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Field specifications
# ─────────────────────────────────────────────────────────────────────────────

# name -> (python types, required, description)
TRACE_FIELDS = {
    "object_id": ((str,), True,
                  "generated object / generation run identity -- the "
                  "independent statistical unit (brief section 3)"),
    "seed": ((int,), True, "random seed for this generation run"),
    "conditioning_id": ((str,), True,
                        "input image or prompt identity; distinct objects may "
                        "share one conditioning across seeds"),
    "step": ((int,), True,
             "generation timestep index, 0 = first step emitted"),
    "n_steps": ((int,), True,
                "total steps in the run, so prefixes can be expressed as "
                "fractions without a second lookup"),
    "layer": ((int,), True, "MoE layer index within the model"),
    "token": ((int,), True, "token / patch / voxel index within the step"),
    "router_probs": ((list,), True,
                     "FULL pre-top-k distribution, softmax(z) over all k "
                     "experts, summing to 1"),
    "selected_experts": ((list,), True,
                         "expert indices retained after top-k (or after "
                         "expert-choice assignment)"),
    "selected_weights": ((list,), True,
                         "combine weights actually applied to those experts"),
    "output_id": ((str,), True,
                  "identifier of the final 3D output this trace belongs to; "
                  "joins to quality_metrics.json"),
    "router_logits": ((list,), False,
                      "pre-softmax z, optional but preferred: softmax of a "
                      "stored float16 probability loses precision that the "
                      "log-ratio transform then amplifies"),
}

MANIFEST_FIELDS = {
    "run_id": ((str,), True, "identifier for this capture run"),
    "checkpoint_hash": ((str,), True,
                        "sha256 of the weight file(s); the brief requires "
                        "every result to preserve it"),
    "checkpoint_id": ((str,), True, "human-readable model/revision name"),
    "model_config": ((dict,), True, "full model configuration as loaded"),
    "n_experts": ((int,), True, "k -- routed experts per MoE layer"),
    "top_k": ((int,), True, "experts selected per token"),
    "routing_mode": ((str,), True,
                     "'token-choice' or 'expert-choice'; the baselines in "
                     "brief section 6 are not comparable across the two -- "
                     "expert-choice balances load by construction, so the "
                     "load baseline is degenerate and must be reported as such"),
    "moe_layer_ids": ((list,), True, "indices of layers carrying an MoE block"),
    "n_layers": ((int,), True, "total layers, MoE or not"),
    "shared_experts": ((int,), False,
                       "always-on experts outside the routed set, if any"),
    "seeds": ((list,), True, "seeds used in this run"),
    "generator_settings": ((dict,), True,
                           "sampler, step count, guidance scale, resolution, "
                           "dtype -- anything that changes the output"),
    "quality_metric_versions": ((dict,), True,
                                "metric name -> library version/commit, so a "
                                "quality number can be re-derived"),
    "capture_code_commit": ((str,), True,
                            "git commit of the capture script itself"),
    "pre_topk_verified": ((bool,), True,
                          "set only after the check in "
                          "`verify_pre_topk_capture` has passed on real data"),
}

SIMPLEX_TOL = 1e-4          # float16 storage of a k=64 simplex drifts ~1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────────────────────────────────────

def _check_fields(rec, spec, label):
    problems = []
    if not isinstance(rec, dict):
        return ["%s is not an object" % label]
    for name, (types, required, _) in spec.items():
        if name not in rec:
            if required:
                problems.append("%s: missing required field '%s'" % (label, name))
            continue
        if rec[name] is None:
            if required:
                problems.append("%s: field '%s' is null" % (label, name))
            continue
        if not isinstance(rec[name], types):
            problems.append("%s: field '%s' has type %s, expected %s"
                            % (label, name, type(rec[name]).__name__,
                               "/".join(t.__name__ for t in types)))
    unknown = set(rec) - set(spec)
    if unknown:
        problems.append("%s: unknown field(s) %s" % (label, sorted(unknown)))
    return problems


def validate_trace_record(rec, n_experts=None, top_k=None, label="trace"):
    """Structural and semantic checks on one router trace record."""
    problems = _check_fields(rec, TRACE_FIELDS, label)
    if problems:
        return problems

    p = rec["router_probs"]
    k = len(p)
    if k < 2:
        problems.append("%s: router_probs has length %d" % (label, k))
        return problems
    if n_experts is not None and k != n_experts:
        problems.append("%s: router_probs has length %d, manifest says "
                        "n_experts=%d" % (label, k, n_experts))
    if any(not isinstance(v, (int, float)) or v != v for v in p):
        problems.append("%s: router_probs contains a non-finite entry" % label)
        return problems
    if any(v < 0.0 for v in p):
        problems.append("%s: router_probs contains a negative entry" % label)
    total = math.fsum(p)
    if abs(total - 1.0) > SIMPLEX_TOL:
        problems.append("%s: router_probs sums to %.6f, not 1" % (label, total))

    # The decisive check: a post-top-k vector has exactly k - top_k hard zeros.
    n_zero = sum(1 for v in p if v == 0.0)
    if top_k is not None and n_zero == k - top_k and top_k < k:
        problems.append(
            "%s: router_probs has exactly k - top_k = %d hard zeros, which is "
            "the signature of a POST-top-k vector. The brief requires the "
            "pre-top-k simplex; capture the softmax before masking."
            % (label, k - top_k))

    sel = rec["selected_experts"]
    if top_k is not None and len(sel) != top_k:
        problems.append("%s: %d selected experts, manifest says top_k=%d"
                        % (label, len(sel), top_k))
    if any(not isinstance(i, int) or i < 0 or i >= k for i in sel):
        problems.append("%s: selected_experts out of range [0, %d)" % (label, k))
    if len(set(sel)) != len(sel):
        problems.append("%s: selected_experts contains duplicates" % label)
    if len(rec["selected_weights"]) != len(sel):
        problems.append("%s: selected_weights and selected_experts differ in "
                        "length" % label)

    if rec.get("router_logits") is not None and len(rec["router_logits"]) != k:
        problems.append("%s: router_logits and router_probs differ in length"
                        % label)
    if not 0 <= rec["step"] < rec["n_steps"]:
        problems.append("%s: step %d outside [0, n_steps=%d)"
                        % (label, rec["step"], rec["n_steps"]))
    return problems


def validate_manifest(manifest, label="manifest"):
    problems = _check_fields(manifest, MANIFEST_FIELDS, label)
    if problems:
        return problems
    if manifest["top_k"] > manifest["n_experts"]:
        problems.append("%s: top_k %d exceeds n_experts %d"
                        % (label, manifest["top_k"], manifest["n_experts"]))
    if manifest["routing_mode"] not in ("token-choice", "expert-choice"):
        problems.append("%s: routing_mode %r is neither 'token-choice' nor "
                        "'expert-choice'" % (label, manifest["routing_mode"]))
    if not manifest["moe_layer_ids"]:
        problems.append("%s: moe_layer_ids is empty" % label)
    if any(not isinstance(i, int) or not 0 <= i < manifest["n_layers"]
           for i in manifest["moe_layer_ids"]):
        problems.append("%s: moe_layer_ids outside [0, n_layers)" % label)
    if len(manifest["checkpoint_hash"]) < 32:
        problems.append("%s: checkpoint_hash %r is too short to be a digest"
                        % (label, manifest["checkpoint_hash"]))
    if not manifest["pre_topk_verified"]:
        problems.append("%s: pre_topk_verified is false -- run "
                        "verify_pre_topk_capture on the real hook output "
                        "before capturing a full dataset" % label)
    if not manifest["quality_metric_versions"]:
        problems.append("%s: quality_metric_versions is empty; a quality "
                        "number that cannot be re-derived is not a result"
                        % label)
    return problems


def verify_pre_topk_capture(records, top_k):
    """Evidence that a hook captured probabilities BEFORE top-k masking.

    Returns (ok, message). A pre-top-k capture has essentially no exact zeros,
    because softmax output is strictly positive. This is the check whose
    result `RunManifest.pre_topk_verified` records, and it is worth running on
    a handful of records before spending GPU-hours on a full capture.
    """
    if not records:
        return False, "no records supplied"
    k = len(records[0]["router_probs"])
    if top_k >= k:
        return False, ("top_k=%d equals the expert count, so no masking "
                       "happens and the check cannot discriminate" % top_k)
    exact_zeros = [sum(1 for v in r["router_probs"] if v == 0.0) for r in records]
    n_masked = sum(1 for z in exact_zeros if z == k - top_k)
    if n_masked > 0.5 * len(records):
        return False, ("%d of %d records carry exactly k - top_k = %d hard "
                       "zeros: this is a post-top-k capture"
                       % (n_masked, len(records), k - top_k))
    n_any = sum(1 for z in exact_zeros if z > 0)
    if n_any > 0.05 * len(records):
        return False, ("%d of %d records contain an exact zero; softmax "
                       "output is strictly positive, so something is masking "
                       "or truncating before storage"
                       % (n_any, len(records)))
    return True, "%d records, no top-k masking signature" % len(records)


def validate_file(manifest_path, traces_path, limit=None):
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    problems = validate_manifest(manifest)
    n = 0
    seen_objects = set()
    with open(traces_path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if limit is not None and n >= limit:
                break
            try:
                rec = json.loads(line)
            except ValueError as exc:
                problems.append("trace line %d: not JSON (%s)" % (i + 1, exc))
                continue
            problems.extend(validate_trace_record(
                rec, manifest.get("n_experts"), manifest.get("top_k"),
                label="trace line %d" % (i + 1)))
            seen_objects.add(rec.get("object_id"))
            n += 1
    if len(seen_objects) < 2:
        problems.append("only %d distinct object_id across %d records: the "
                        "independent statistical unit is the generated object, "
                        "so object-level splits and bootstrap are impossible "
                        "here (brief section 3)" % (len(seen_objects), n))
    return manifest, n, len(seen_objects), problems


def print_schema():
    for title, spec in (("RunManifest", MANIFEST_FIELDS),
                        ("RouterTraceRecord", TRACE_FIELDS)):
        print("=" * 78)
        print(title)
        print("=" * 78)
        for name, (types, required, desc) in spec.items():
            print("  %-24s %-10s %s" % (
                name, "/".join(t.__name__ for t in types),
                "required" if required else "optional"))
            for line in _wrap(desc, 68):
                print("      %s" % line)
        print()


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--traces")
    p.add_argument("--limit", type=int)
    a = p.parse_args()

    if not a.manifest or not a.traces:
        print_schema()
        print("no --manifest/--traces given; printed the schema instead.")
        print("Gate status: python -m experiments.audit_environment")
        return 0

    manifest, n, n_obj, problems = validate_file(a.manifest, a.traces, a.limit)
    print("manifest : %s" % a.manifest)
    print("traces   : %s (%d records, %d distinct objects)" % (a.traces, n, n_obj))
    if problems:
        print("\n%d problem(s):" % len(problems))
        for pr in problems[:200]:
            print("  - %s" % pr)
        return 1
    print("\nOK -- schema satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
