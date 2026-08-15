# Acquisition and instrumentation checklist

**Part 0 verdict: the hard gate FAILS. The real-data benchmark was not run,
and no synthetic substitute was published in its place.**

Reproduce the verdict:

```bash
python -m experiments.audit_environment      # exits non-zero when the gate fails
```

Raw output: `experiments/results/environment_audit.json`.

---

## 1. What was measured

| # | Condition | Met | Measurement |
| --- | --- | --- | --- |
| 1 | a runnable 3D generator with actual MoE routing | **no** | no `torch`, `jax` or `tensorflow`; no 3D generator package |
| 2 | a real checkpoint | **no** | zero `.safetensors` / `.ckpt` / `.pth` / `.gguf` anywhere on the filesystem; the only `.pt` and `.bin` hits are Go and Vim fixtures |
| 3 | access to router logits or probabilities | **no** | no MoE module exists to hook |
| 4 | a way to generate multiple 3D outputs | **no** | no generator; no `nvidia-smi`, no `/dev/nvidia*`, no CUDA-capable runtime |
| 5 | at least one quantitative 3D-quality measurement | **no** | no `trimesh`, `open3d`, `pytorch3d`, `kaolin`, `pymeshlab`; no mesh files |

Compounding all five: **there is no package or model egress.**
`pypi.org`, `files.pythonhosted.org` and `huggingface.co` all return 403 from
the organisation's egress proxy, and `archive.ubuntu.com` returns 403 for
`apt`. Nothing can be installed and no weights can be downloaded, so none of
the five conditions can be repaired from inside this environment. Even
`numpy` is absent, which is why every experiment in this repository is written
against the standard library.

## 2. Why this is a stop and not a workaround

The previous pass (`docs/validation_report.md`) ran the router experiments on
synthetic logistic-normal routing vectors and reported `AffineInv` AUC = 1.000
against entropy 0.54–0.62. That is a statement about data that was generated
to contain the structure being detected. It is evidence that a structure-aware
detector *can* separate distributions that entropy cannot; it is not evidence
about a real router, and it is not evidence about 3D quality at all.

Repeating it with new statistics would add nothing. The brief's rule —
*real checkpoint first, real quality target second* — is the correct one, so
the remainder of this document is the list of what has to be true before
sections 2–19 of the brief become executable.

---

## 3. Blocking item A — compute and software

| requirement | why | acceptance test |
| --- | --- | --- |
| 1 GPU, ≥ 40 GB VRAM (80 GB if the model is upcycled from a 4B+ dense base) | a 17B-total / 2B-active MoE in bf16 needs ~34 GB of weights resident even though only 2B activate | `torch.cuda.is_available()` and a full forward pass at the target resolution |
| ≥ 2 TB fast local disk | raw router traces dominate; see the volume budget in §6 | `df` after a 10-object dry run, extrapolated |
| egress to `pypi.org` and `huggingface.co`, or an internal mirror | weights and mesh tooling | `audit_environment.py` reports both reachable |
| `torch`, `transformers`, `safetensors`, `accelerate` | model loading and hooks | import + version pinned in the manifest |
| `trimesh` **and** one of `open3d` / `pytorch3d` / `point_cloud_utils` | Chamfer, normal consistency, watertightness, component counts | each metric reproduces a known value on a unit-sphere fixture |
| `numpy`, `scipy`, `scikit-learn` | the detectors in brief §7 and the AUC/PR machinery in §10 | the existing test suite runs |

None of these is a research risk. They are a procurement item.

---

## 4. Blocking item B — the model. This is the hard one.

The brief needs a generator that is **simultaneously** (a) 3D and (b)
token-level MoE with accessible pre-top-k router outputs. A literature and
model-hub search on 2026-08-15 did not find one with released weights.
The facts below came from web search summaries and **must be re-verified at
acquisition time** — this environment cannot reach the model hub to confirm
them directly.

**What exists, and why each falls short:**

| candidate | 3D? | token-level MoE? | weights released? |
| --- | --- | --- | --- |
| [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) (MIT) | yes — image-to-3D, two flow stages (Sparse Structure Flow for geometry, SLAT Flow for appearance) | **no** — dense flow-matching transformer | yes |
| [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Apache 2.0) | **no** — text-to-image | yes — 29 of 32 blocks are sparse MoE, 64 routed experts + 1 shared, ~2B of 17B active | yes; described as the first fully open MoE diffusion model at this quality tier |
| [3D-MoE](https://arxiv.org/abs/2501.16698) | partly — 3D vision + pose diffusion MLLM | yes | no release found |
| [FastDiT-3D](https://arxiv.org/abs/2312.07231) | yes — point clouds | yes — MoE for multi-category generation | no usable release found |

So there are four routes, in increasing order of cost and decreasing order of
risk that the answer will be uninterpretable.

### Path A — find one (cheapest, may simply fail)

Re-run the search at acquisition time against the model hub directly. The
field is moving fast enough that this table may be stale. Accept only a
checkpoint where the router is **token-choice or expert-choice over a routed
expert set**, not a "mixture of expert denoisers" that switches whole networks
per timestep — the latter has no per-token distribution and section 7 has
nothing to compute.

### Path B — upcycle a real 3D generator (recommended)

Take TRELLIS.2-4B (or the current best open image-to-3D model), replace the
FFN in the transformer blocks with an MoE layer, and continue training.
The recipe is established:

- [Sparse Upcycling](https://arxiv.org/abs/2212.05055): copy the dense FFN
  into each of E experts, initialise only the gate randomly, carry everything
  else over from the dense checkpoint.
- [Drop-Upcycling](https://arxiv.org/abs/2502.19261): partially re-initialise
  the copies, because identical experts start out degenerate and the router
  has nothing to separate.
- [Sparse MoE routing in visual diffusion transformers](https://arxiv.org/abs/2605.19378)
  applies exactly this to a ~5B dense visual DiT (routed experts clone the
  original FFN, shared experts start at near-zero noise, only the gates are
  random) and catalogues the failure modes — routing collapse and selective
  deadlock — that this brief's monitor would be trying to detect early.

**This path has a large advantage the others do not**: routing collapse can be
*induced*. Train two or three variants with deliberately different
load-balancing pressure, and you get objects that fail for a known routing
reason alongside objects that fail for reasons unrelated to routing. That is
the difference between a correlational study and one that can answer
"does router structure predict quality" with a control group.

Cost: this is a continued-pretraining job, not an afternoon. Budget it as
such, and treat any smaller version as a pilot.

### Path C — MoE language backbone + mesh tokens

Autoregressive mesh generators tokenise geometry and generate it like text
([MeshAnything](https://buaacyw.github.io/mesh-anything/), LLaMA-Mesh). Swap
the dense backbone for an open MoE LLM and every layer has a genuine
token-level router, with the token axis being mesh faces. Attractive because
the routers are unambiguous and the 3D output is a real mesh that all of
brief §4's geometry metrics apply to directly. Requires a fine-tune, not a
pretrain.

### Path D — rehearse the methodology on Nucleus-Image, and label it as such

Nucleus-Image gives real routers from a real generative checkpoint today. It
produces images, not 3D. Running the full pipeline against 2D quality metrics
validates the *machinery* — hooks, schema, object-level bootstrap, detector
implementations, early-warning curves — and would surface every engineering
bug before GPU time is spent on the real question.

It does **not** answer the brief's question, and any result from it must be
reported under a heading that says so. The brief's §17 list of things that do
not count as success exists precisely to stop a 2D or synthetic result being
promoted into a 3D claim.

---

## 5. Architecture-dependent hazards to settle *before* capture

These change what the experiments mean, and two of them silently invalidate
baselines from brief §6.

1. **Expert-choice routing makes the load baseline degenerate.**
   Nucleus-Image uses expert-choice routing, which "guarantees balanced expert
   utilisation without auxiliary load-balancing losses". Under expert-choice,
   `mean load` is uniform *by construction*. A monitor that beats it has
   beaten a constant. Record `routing_mode` in the manifest and, if it is
   expert-choice, report the load baseline as structurally uninformative
   rather than as a defeated competitor.

2. **Shared experts sit outside the routed simplex.** Nucleus-Image has one
   shared expert per MoE layer. `router_probs` must cover the routed set only,
   and the manifest must say how many shared experts bypass it, or the
   simplex will not sum to 1 and the log-ratio coordinates will be wrong.

3. **Timestep-conditioned routing is expected, not anomalous.**
   Nucleus-Image explicitly decouples routing from modulation to stop
   "adaptive modulation scale ... collapsing expert specialisation into
   timestep-dependent selection". Router statistics will therefore vary
   strongly with the diffusion timestep for entirely healthy runs. This is why
   brief §9 requires per-timestep reference states: a single pooled
   `Sigma_ref` will manufacture anomalies at the ends of the schedule.

4. **The first layers may be dense.** Nucleus-Image keeps 3 of 32 blocks dense
   for training stability. `moe_layer_ids` is not `range(n_layers)`, and the
   layer × timestep heatmap of brief §12 has structural holes.

---

## 6. Blocking item C — instrumentation

### Where to hook

A forward hook on each MoE gate module, capturing `softmax(z)` **before**
top-k masking. `experiments/trace_schema.py::verify_pre_topk_capture` is the
acceptance test: a post-top-k vector carries exactly `k - top_k` hard zeros,
and softmax output never does. Run it on ten records before spending GPU
hours; the manifest field `pre_topk_verified` may not be set otherwise.

### What to record

`python -m experiments.trace_schema` prints the full field list. Per record:
`object_id`, `seed`, `conditioning_id`, `step`, `n_steps`, `layer`, `token`,
`router_probs` (full pre-top-k simplex), `selected_experts`,
`selected_weights`, `output_id`, and `router_logits` where available. Per run:
checkpoint hash, model config, expert count, top-k, routing mode, MoE layer
IDs, seeds, generator settings, quality-metric versions, capture-code commit.

Prefer storing **logits** over probabilities. The detectors in brief §7 work
in log-ratio coordinates `y_i = log(p_i / p_k)`; recovering those from a
float16 probability costs precision exactly where `p_k` is small, which is
where the log-ratio is largest.

### Volume budget — plan for this, it is the practical blocker

A TRELLIS-scale run at 50 steps × ~24 MoE layers × ~4096 tokens × 64 experts
is 4.9 × 10⁶ router vectors per object; at float16 that is **~0.6 GB per
generated object**, so 500 objects is ~300 GB of raw traces.

Recommended split:

- **Sufficient statistics for every object.** Per `(object, layer, step)`
  accumulate the mean log-ratio vector and its second-moment matrix:
  `(k-1) + (k-1)k/2` numbers, ~2.1 × 10³ floats instead of 2.6 × 10⁵ — a 126×
  reduction that loses nothing the covariance, spectral, affine-invariant or
  Fisher-Rao detectors need.
- **Full raw traces for an audit subset** (~10% of objects, chosen by seed
  before generation, not after seeing quality). Brief §14's higher-order
  controls — "preserve covariance while perturbing higher-order structure" —
  cannot be run on second moments alone.

Record which objects are in the audit subset in the manifest, before
generating.

---

## 7. Blocking item D — real 3D quality outcomes

The target variable must be measured from the output, never from the router.
Minimum viable set, in the order they should be added:

**Reference-free (works without ground truth, so it works on any prompt set)**
watertightness; connected-component count; count and total area of boundary
(hole) edges; self-intersecting face count; degenerate-triangle count;
surface-area and volume outliers against the category distribution;
normal-consistency of adjacent faces.

**Reference-based (needs a ground-truth mesh, so needs a benchmark set)**
Chamfer distance; F-score at a fixed threshold; normal consistency against
reference; SDF/occupancy IoU.

**Conditioning alignment (image-to-3D)**
render the generated asset from the conditioning viewpoint and score
image–render similarity; multi-view consistency across renders. Only include
CLIP/DINO variants if the project already depends on them — a new pretrained
scorer is a new dependency and a new failure mode.

**Failure labels.** Brief §13 needs categorical outcomes, and no single scalar
supplies them. Derive them from the reference-free metrics with pre-registered
thresholds (e.g. *holes* = boundary-edge length > τ, *duplicated components* =
component count > 1 for a single-object prompt), and hand-label a subset to
estimate the label noise. Fix the thresholds **before** looking at any router
statistic. Never define "bad output" from router statistics.

Pin every metric's library version in `quality_metric_versions`; the schema
validator rejects a manifest without it.

---

## 8. Blocking item E — statistical protocol

The independent unit is the **generated object**, not the token. One object
contributes ~10⁶ router vectors; treating them as independent inflates every
confidence interval by roughly the square root of that. Concretely:

- train/test splits, bootstrap resampling, confidence intervals and paired
  comparisons all resample **objects**;
- token windows may be used to *construct* a detector score for an object, but
  never to *count* toward n;
- objects sharing a `conditioning_id` across seeds are not independent either
  — group by conditioning when splitting, or report both groupings;
- **minimum n:** distinguishing AUC 0.85 from AUC 0.70 at 80% power needs on
  the order of 100 failures and 100 successes. At a realistic 10–20% failure
  rate that is 600–2000 generated objects. Size the generation run from the
  effect you need to detect, not from what fits in a weekend.

Leakage rule for brief §5 and §11: an early-warning score at step *s* may read
`X_{<=s}` only. Reference states `Sigma_ref` must be estimated on the training
objects only, and per-timestep references must use step ≤ *s* data.

---

## 9. What already exists here and is reusable on day one

| component | file | status |
| --- | --- | --- |
| trace record + manifest schema, pre-top-k detector | `experiments/trace_schema.py` | built, tested |
| gate audit | `experiments/audit_environment.py` | built, tested |
| cheap baselines (§6): mean load, entropy, max-prob, L2 mass, trace-cov | `experiments/_router_common.py` | built |
| log-ratio transform, empirical covariance, matrix log/power, affine-invariant distance | `experiments/_router_common.py` | built |
| AUC, Cohen's d, seed aggregation, JSON persistence | `experiments/_router_common.py` | built |
| Dirichlet MLE with a convergence gate | `experiments/_router_common.py`, `igad/families.py` | built |
| exact O(k) Dirichlet curvature | `igad/curvature.py`, `docs/sherman_morrison.md` | built, benchmarked |
| numerical reliability boundary for curvature | `docs/numerical_reliability.md` | built, measured |

Not built, deliberately: `capture_router_traces.py`, `evaluate_3d_quality.py`,
`analyze_router_structure.py`, `evaluate_early_warning.py`. Writing a pipeline
against a model that cannot be loaded produces code that has never executed a
single line against real data, and the brief asked for a checklist rather than
a hopeful scaffold. The schema is the exception because it is a contract that
can be tested on its own.

---

## 10. Two settled results that carry forward

1. **Fisher-Rao distance and affine-invariant covariance distance are the same
   detector.** On the fixed-mean covariance manifold `d_FR = (1/sqrt 2) d_AI`,
   verified to 4.2e-7 in `experiments/demo_router_logistic_normal_geometry.py`.
   They have identical rankings and therefore identical AUC. Report one, cite
   the geometry as the *reason* it is the natural statistic, and do not enter
   both as competing detectors.

2. **Scalar curvature is excluded from the detector benchmark unless the
   fitted family makes it non-constant.** It lost to `AffineInv` at every
   window in the previous pass and is a near-relabelling of concentration
   (`rho(R, alpha_0) = 0.841`). It remains in the repository as a validated,
   now O(k), numerical routine — not as a candidate monitor.

---

## 11. The one-line acceptance test

The gate is passed when this exits zero:

```bash
python -m experiments.audit_environment && \
python -m experiments.trace_schema --manifest run_manifest.json --traces traces.jsonl
```

and the manifest reports `pre_topk_verified: true` on real hook output, with
at least two distinct `object_id` values in the traces. Until then, sections
2–19 of the brief are specifications, and every entry in the final decision
table that depends on real data reads **Untested** rather than a number.
