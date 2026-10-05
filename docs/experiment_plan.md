# The early-warning experiment, specified

This is the experiment that runs the moment a suitable checkpoint exists. It is
written so that nothing has to be redesigned at that point: every analysis
function it names is already implemented and tested in this repository, and
every contract it depends on already rejects a malformed capture.

**It has not been run.** No number in this document is a result. See
`docs/acquisition_checklist.md` for what is missing and
`docs/handoff.md` for the current status.

The hypothesis, stated so it can fail:

> Router measurements taken from an **early prefix** of generation predict the
> **final 3D quality** of that object better, or earlier, than expert load and
> routing entropy do.

---

## 0. Preconditions

`python -m experiments.audit_environment` must exit zero. Until it does,
everything below is a specification.

---

## 1. Per object

For each generated object, in this order. The order matters: quality
thresholds are fixed before any router statistic is computed, so the target
variable cannot be fitted to the detector.

1. **Register the failure thresholds** and commit them.
   `quality_schema.validate_quality_manifest` rejects a manifest with
   `thresholds_registered_before_router_analysis: false`, and records the
   commit in which they were fixed.
2. **Generate the object** at a recorded seed and conditioning ID.
3. **Capture router probabilities before top-k**, at every MoE layer and every
   generation timestep. `trace_schema.verify_pre_topk_capture` must pass on a
   sample before the full run; a post-top-k capture carries exactly
   `k - top_k` hard zeros and is rejected.
4. **Measure real 3D quality** from the finished output -
   `quality_schema.QUALITY_FIELDS`. At least one quantitative metric is
   required; a record carrying only labels is rejected.
5. **Derive failure labels** from the pre-registered thresholds, with
   provenance recorded. A label whose provenance mentions routing is rejected.

Only then does any router statistic get computed.

---

## 2. Prefix evaluation - the forecasting constraint

For prefix fraction `f`, a detector may read router information from steps
`0 .. floor(f * n_steps)` and nothing later. Prefixes:

```
f in {0.05, 0.10, 0.20, 0.40, 0.60, 1.00}
```

Reference states are fitted on **training objects only**, and per-timestep
references at prefix `f` use only steps `<= f * n_steps`. Fitting a reference
on all objects, or on the whole schedule, leaks the future into the
"early" score and is the single easiest way to manufacture a positive result.

For each `(object, prefix, layer)`:

```python
summary = router_stats.summarize_window(rows)          # one window
scores  = router_stats.drift_scores(summary, reference)
```

and then, per object and prefix:

```python
object_score = evaluation.aggregate_object_scores(
    [(object_id, s) for ...], how="max")
```

`aggregate_object_scores` is the only sanctioned path from windows to an
observation. After it, one object is one row, and n is the object count.

---

## 3. Detectors

Cheap operational baselines (brief §6) - these are the controls the monitor
must beat, not decoration:

| statistic | note |
| --- | --- |
| mean expert load / imbalance | **degenerate under expert-choice routing** - uniform by construction. Report `routing_mode` beside it or the comparison is meaningless. |
| mean entropy | |
| mean max probability | |
| top-1/top-2 margin | |
| total routing variance `tr(C)` | |

Structure-aware candidates (brief §7), on log-ratio coordinates
`y_i = log(p_i / p_k)`:

| statistic | note |
| --- | --- |
| `lambda_max`, `anisotropy`, `spectral_entropy`, `effective_rank` | spectrum of the log-ratio covariance |
| `covariance_drift` | Frobenius; scale-dependent, included as the obvious thing to try |
| `correlation_drift` | separates "more variable" from "differently coupled" |
| `affine_invariant` | `= sqrt(2) * d_FR`. Entered **once**: Fisher-Rao distance and affine-invariant covariance distance have identical rankings, so they are one detector, not two (brief §8). |
| `mean_drift` | Mahalanobis drift of the log-ratio mean |

**Scalar curvature is not a candidate.** It lost to `affine_invariant` at every
window in the previous pass and is close to a relabelling of concentration
(`rho(R, alpha_0) = 0.841`). It stays in `igad` as a validated numerical
routine. If a future fit makes it non-constant *and* it survives a same-fit
control, that is the moment to reconsider - not before.

---

## 4. What is measured

Binary quality failure, per prefix and per detector:

```python
evaluation.roc_auc(labels, scores)
evaluation.pr_auc(labels, scores)          # positives will be rare; ROC flatters
evaluation.sensitivity_at_fpr(labels, scores, fpr)   # 0.01, 0.05, 0.10
```

Check `evaluation.achievable_fpr_grid(n_negatives)` before quoting a 1% FPR
number: with 60 negatives the achievable grid is 0, 1/60, 2/60, … and "1%" does
not exist. Say so rather than quoting the 0% value.

Continuous quality:

```python
evaluation.spearman(scores, quality)
evaluation.pearson(scores, quality)
evaluation.cross_validated_r2(scores, quality, folds=evaluation.object_kfold(...))
```

Earliest reliable warning (brief §11) - the headline comparison:

```python
evaluation.warning_time_table(detector_aucs, prefix_fractions, threshold=0.80)
```

`earliest_warning` requires the threshold to hold at that prefix **and every
later one**. A detector that touches 0.80 at 10%, collapses at 20% and recovers
at 80% has not warned at 10%.

Layer localisation (brief §12): the same table computed per MoE layer, giving a
layer × prefix matrix. `moe_layer_ids` is not `range(n_layers)` in general -
some architectures keep early blocks dense - so the matrix has structural
holes, not missing data.

---

## 5. Uncertainty - object level, always

```python
evaluation.object_bootstrap_ci(objects, labels, scores)      # resamples objects
evaluation.paired_detector_test(objects, labels, a, b)       # same objects both
evaluation.object_train_test_split(ids, groups=conditioning) # no leakage
```

`object_train_test_split` and `object_kfold` take a `groups` map so that
objects sharing a conditioning ID stay together: two seeds of the same input
image are not independent observations.

**Sizing.** `evaluation.minimum_objects_for_auc(0.85, auc_null=0.70)` gives the
order of magnitude. Distinguishing a structure-aware AUC of 0.85 from an
entropy AUC of 0.70 at 80% power, with a 15% failure rate, is in the high
hundreds of objects. Size the generation run from the effect you need to
detect, not from what fits in a weekend.

---

## 6. Controls (brief §14)

| control | expected behaviour | why it matters |
| --- | --- | --- |
| permute expert identities consistently | permutation-invariant statistics unchanged | catches a detector that has memorised expert indices |
| shuffle token order within a window | order-free statistics unchanged | catches accidental dependence on sequence position |
| permute quality labels | every detector collapses to chance | catches leakage from label to score |
| destroy covariance, preserve marginals | structure-aware detectors respond, cheap ones do not | this is the whole hypothesis, stated as a control |
| preserve covariance, perturb higher-order structure | if a richer model still responds, there is something beyond second moments | needs the raw-trace audit subset, not sufficient statistics |

The last one is why the acquisition checklist reserves full raw traces for a
pre-registered 10% subset instead of storing sufficient statistics for
everything.

---

## 7. Model selection (brief §15)

Start at empirical covariance. Move to logistic-normal only if held-out
diagnostics show the covariance summary is inadequate; move to a mixture only
if the logistic-normal is. Each step has to earn itself on held-out objects.

---

## 8. The decision rule, fixed in advance

| outcome | evidence | conclusion |
| --- | --- | --- |
| **A** | load/entropy match structure-aware statistics on both final AUC and warning time | router structure adds no operational value. **Stop the direction.** |
| **B** | covariance/spectral statistics beat load and entropy on final AUC, on held-out objects, with a paired test | hidden router organisation is a useful health signal. Product-worthy. No curvature involved. |
| **C** | same final AUC, but `T_structure << T_entropy` | valuable as an early-warning monitor, possibly more so than a final-AUC gain |
| **D** | a richer geometric statistic beats controls derived from the *same fitted model* | information geometry adds incremental value - and only then. Requires replication across layers, checkpoints, categories, seeds and quality measures. |

Outcome A is a real result and gets written up as one. The brief's §17 list of
non-results applies throughout: beating random is not success, beating entropy
on one synthetic benchmark is not success, and AUC 1.0 at 100% of generation is
not early warning.

---

## 9. Reproducibility

Raw before figures, always. Layout:

```
experiments/results/
    router_traces/            per-object JSONL, plus the audit subset
    quality_metrics.jsonl     one record per output
    detector_scores.json
    early_warning.json
```

Every result preserves the checkpoint hash, model configuration, expert count,
top-k, routing mode, MoE layer IDs, seed, generator settings and quality-metric
versions - the manifest schemas in `trace_schema.py` and `quality_schema.py`
require all of them, and reject a record set that omits any.
