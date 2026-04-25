# Mathematical Background: Scalar Curvature of Exponential Families

**Status**: Sections 1-4 are **known results** in Hessian geometry.
The novel contribution is the IGAD score (Section 5).

## 1. Setup

Let M = {p(x;theta)} be a regular exponential family:

    p(x;theta) = exp(<theta, T(x)> - A(theta)) h(x)

Fisher metric:     g_{ij}(theta) = d^2 A / d theta_i d theta_j
Third cumulant:    T_{ijk}(theta) = d^3 A / d theta_i d theta_j d theta_k

## 2. Christoffel Symbols (Known)

Because g_{ij} is the Hessian of A, and T_{ijk} = d_k g_{ij} is fully symmetric:

    Gamma_{ij,k} = 1/2 T_{ijk}

Reference: Amari and Nagaoka (2000), Chapter 2.

## 3. Fourth Cumulant Cancellation (Known)

The Riemann tensor R^l_{ijk} involves fourth cumulant terms from d_i Gamma^l_{jk}.
These are symmetric in (i,j), but R is antisymmetric in (i,j).
Therefore they cancel exactly. R is purely quadratic in T.

Reference: Standard property of Hessian metrics. See Ruppeiner (1995).

## 4. Scalar Curvature Formula (Known)

    R(theta) = 1/4 * ( ||S||^2_g - ||T||^2_g )

where S_m = g^{ab} T_{abm} = d_m log det g.

When det g = const: R = -1/4 ||T||^2_g <= 0.

## 5. IGAD Score (Novel Contribution)

    IGAD(batch) = |R(theta_ref) - R(theta_local)|

where theta_ref is the global MLE and theta_local is the batch MLE.

Interpretation: IGAD measures deviation in local third-cumulant structure,
making it sensitive to distributional shape changes invisible to
location-scale detectors.

Conceptual precursor: Ruppeiner (1979) used scalar curvature to detect
phase transitions in thermodynamic systems.

Failure modes:
1. 1D families (R = 0 identically)
2. Constant-curvature manifolds
3. Pure location-scale anomalies
4. Model misspecification at large sample sizes

## References

- Amari, S. (1985). Differential-Geometrical Methods in Statistics.
- Amari, S. and Nagaoka, H. (2000). Methods of Information Geometry.
- Ruppeiner, G. (1979). Thermodynamics: A Riemannian geometric model. Phys. Rev. A.
- Ruppeiner, G. (1995). Riemannian geometry in thermodynamic fluctuation theory. Rev. Mod. Phys.

---

## 6. Known Failure Mode: Constant-Curvature Families

### Gaussian Manifold (Empirically Confirmed)

The scalar curvature of the multivariate Gaussian manifold is constant
(the manifold is isometric to hyperbolic space). Therefore:

    |R_ref - R_local| ≈ 0  for all parameter choices

Verified experimentally (experiments/demo_gaussian2d.py):

    rho_ref=0.20, rho_anom=0.80  →  |R_diff| = 0.003308
    rho_ref=0.50, rho_anom=0.70  →  |R_diff| = 0.000645
    rho_ref=0.50, rho_anom=0.55  →  |R_diff| = 0.000049

All baselines (IGAD, MLE-correlation, raw correlation) reached AUC=1.0
at n=200 for rho=0.2 vs rho=0.8 — not because of curvature, but because
the correlation difference (0.6) is large enough for any method to detect.
IGAD contributed nothing unique in this setting.

### What This Means

IGAD requires families where R(theta) varies meaningfully with parameters.
This holds when the third cumulant tensor T_{ijk} changes substantially
across the parameter space — which is the case for Gamma but not Gaussian.

### Families Where IGAD Is Applicable

| Family       | dim | R varies? | IGAD applicable? |
|---|---|---|---|
| Poisson      | 1   | No (R=0)  | No               |
| Exponential  | 1   | No (R=0)  | No               |
| Gamma        | 2   | Yes       | Yes (confirmed)  |
| Gaussian     | 3   | No (const)| No               |
| Dirichlet    | k-1 | Yes       | Promising        |
| Neg-Binomial | 2   | Yes       | Untested         |

---

## 7. Closed-Form T_{ijk} for the Dirichlet Family

### Setup

The Dirichlet family in natural parameters theta_i = alpha_i - 1 has log-partition:

    A(theta) = sum_i lgamma(alpha_i) - lgamma(alpha_0)

where alpha_i = theta_i + 1 and alpha_0 = sum_i alpha_i.

### Derivation

The third cumulant tensor follows directly from the exponential-family identity
(Amari & Nagaoka 2000, Ch. 2):

    T_{ijk}(theta) = d^3 A / d theta_i d theta_j d theta_k

**Step 1.** First partial with respect to theta_m (alpha_m' = 1):

    dA/d theta_m = psi(alpha_m) - psi(alpha_0)

where psi = digamma.

**Step 2.** Second partial with respect to theta_n:

    d^2 A / d theta_m d theta_n = psi'(alpha_m) * [m == n]  -  psi'(alpha_0)

where psi' = polygamma(1, ·) = trigamma.  This recovers the Fisher metric
g_{mn} = polygamma(1, alpha_m) * [m == n] - polygamma(1, alpha_0).

**Step 3.** Third partial with respect to theta_p:

Because alpha_0 depends on all theta_i, d/d theta_p of -psi'(alpha_0) = -psi''(alpha_0).
The diagonal term psi'(alpha_m) * [m == n] contributes psi''(alpha_m) * [m == n == p].

Therefore:

    T_{ijk} = -polygamma(2, alpha_0)                        for all (i, j, k)
    T_{i,i,i} += polygamma(2, alpha_i)                      (diagonal correction only)

where polygamma(2, ·) = psi'' is the tetragamma function.

### Vectorized Form

    T = np.full((k, k, k), -polygamma(2, alpha_0))
    T[range(k), range(k), range(k)] += polygamma(2, alpha)

This is O(k^3) memory, O(1) arithmetic operations beyond the fill.

### Properties

- **Full symmetry**: T_{ijk} = T_{perm(i,j,k)} for all 6 permutations — manifest
  from the definition as a mixed third derivative.
- **Off-diagonal uniformity**: all non-pure-diagonal entries equal -polygamma(2, alpha_0).
- **Discretization note**: finite-difference approximation of the diagonal entries
  achieves only ~1e-3 relative accuracy at the optimal step size (5-point stencil,
  truncation O(h^2) with h ~ 5e-3). The error grows with |polygamma(2, alpha_i)|
  (e.g., ~17 at alpha_i=0.5), so absolute accuracy degrades for extreme parameter
  values. The analytical formula is exact.

### Attribution

The derivation is mechanical application of the exponential-family identity
T_{ijk} = d^3 A / d theta_i d theta_j d theta_k from Amari & Nagaoka (2000).
No novelty is claimed for this calculation. The closed form for the Dirichlet
was not available in the codebase prior to this work and is added here for
numerical precision in the IGAD scoring path.

