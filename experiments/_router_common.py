"""
experiments/_router_common.py

Shared machinery for the MoE-router experiments.

Dependency note
---------------
This module and the router experiments that import it are written against the
Python standard library only, with no numpy. That is deliberate: these scripts
exist to make published numbers reproducible, and a reproduction script that
cannot be executed in the environment that published the numbers is not a
reproduction. `tests/test_router_common_mirror.py` cross-checks every Dirichlet
routine here against `igad` itself whenever numpy is installed, so the mirror
cannot silently drift from the package.

Contents
--------
special functions -> digamma / trigamma / tetragamma
linear algebra    -> inverse, symmetric eigendecomposition, matrix log, Cholesky
Dirichlet         -> Fisher metric, structured third cumulant, exact O(k^2) R, MLE
sampling          -> Dirichlet, Dirichlet mixture, multivariate normal
detectors         -> mean load, entropy, max-prob, L2 mass, MMD
evaluation        -> AUC, effect size, seed aggregation, JSON persistence
"""

import json
import math
import os
import random

# ─────────────────────────────────────────────────────────────────────────────
# Special functions
# ─────────────────────────────────────────────────────────────────────────────

# The Euler-Maclaurin tails below are truncated after B_12 and the recurrence
# lifts the argument past SHIFT before they are used. The first omitted term
# then sits below 1e-19 relative for all three functions, i.e. under the
# float64 unit roundoff, so these are correctly rounded to within a few ulp.
#
# The earlier SHIFT of 12 with three or four Bernoulli terms left a relative
# error near 1e-12, which propagated into R(alpha) as an error near 1e-11 --
# larger than anything the linear algebra contributes, and identical across
# every curvature route because all of them share these inputs. See
# `experiments/highprec_reliability.py` (section C2) for the measurement, and
# `tests/test_highprec.py` for the pin against the 120-digit reference.
SHIFT = 30.0


def _digamma_parts(x):
    """(recurrence sum, asymptotic tail) -- psi(x) is their sum."""
    r = 0.0
    while x < SHIFT:
        r -= 1.0 / x
        x += 1.0
    i = 1.0 / x
    i2 = i * i
    # log x - 1/(2x) - sum_{n>=1} B_{2n} / (2n x^{2n})
    tail = math.log(x) - 0.5 * i - i2 * (
        1/12. - i2 * (1/120. - i2 * (1/252. - i2 * (
            1/240. - i2 * (1/132. - i2 * (691/32760.))))))
    return r, tail


def _trigamma_parts(x):
    r = 0.0
    while x < SHIFT:
        r += 1.0 / (x * x)
        x += 1.0
    i = 1.0 / x
    i2 = i * i
    # 1/x + 1/(2x^2) + sum_{n>=1} B_{2n} x^{-(2n+1)}
    tail = i * (1 + i * (0.5 + i * (
        1/6. - i2 * (1/30. - i2 * (1/42. - i2 * (
            1/30. - i2 * (5/66. - i2 * (691/2730.))))))))
    return r, tail


def _tetragamma_parts(x):
    r = 0.0
    while x < SHIFT:
        r -= 2.0 / (x ** 3)
        x += 1.0
    i = 1.0 / x
    i2 = i * i
    # -1/x^2 - 1/x^3 - sum_{n>=1} (2n+1) B_{2n} x^{-(2n+2)}
    tail = -i2 - i2 * i - i2 * i2 * (
        0.5 - i2 * (1/6. - i2 * (1/6. - i2 * (
            0.3 - i2 * (5/6. - i2 * (691/210.))))))
    return r, tail


def digamma(x):
    """psi(x) = d/dx log Gamma(x), for x > 0."""
    r, tail = _digamma_parts(x)
    return r + tail


def trigamma(x):
    """psi'(x) = sum_{n>=0} 1/(x+n)^2, for x > 0."""
    r, tail = _trigamma_parts(x)
    return r + tail


def tetragamma(x):
    """psi''(x) = -2 sum_{n>=0} 1/(x+n)^3, for x > 0.

    The series is the term-by-term derivative of trigamma's.
    """
    r, tail = _tetragamma_parts(x)
    return r + tail


_POLYGAMMA_PARTS = {"digamma": _digamma_parts, "trigamma": _trigamma_parts,
                    "tetragamma": _tetragamma_parts}


def polygamma_cancellation(name, x):
    """(|recurrence| + |tail|) / |result| -- the condition number of the sum.

    This is exactly 1 when the two halves share a sign and grows without bound
    as they cancel. psi' and psi'' accumulate terms of a single sign, so their
    factor is 1 everywhere and their relative accuracy is uniform. psi
    subtracts a recurrence sum from an asymptotic tail, and psi has a root at
    x ~ 1.4616321: the two halves cancel there, the factor reaches ~185 at
    x = 1.5, and no recurrence-based evaluation of psi can avoid it. Relative
    and ulp error are therefore not meaningful measures for psi in that
    neighbourhood; absolute error is, and
    `experiments/special_function_accuracy.py` reports both.

    Note the convention differs from `_highprec.cancellation_ratio`, which
    divides the *largest* of R's five intermediates by the final difference.
    Both measure magnitude-relative-to-result; the constants in front are
    calibrated separately, and R's was fitted against measured error with an
    intercept that recovers log10(eps).

    Computed from the implementation's own intermediates, not from a second
    copy of the series.
    """
    r, tail = _POLYGAMMA_PARTS[name](x)
    total = r + tail
    if total == 0.0:
        return float("inf")
    return (abs(r) + abs(tail)) / abs(total)


def inv_digamma(y):
    """Newton inverse of digamma (Minka 2000), with an early convergence break."""
    x = math.exp(y) + 0.5 if y >= -2.22 else -1.0 / (y + digamma(1.0))
    for _ in range(50):
        step = (digamma(x) - y) / trigamma(x)
        x -= step
        if x <= 0:
            x = 1e-8
        if abs(step) < 1e-13:
            break
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Linear algebra
# ─────────────────────────────────────────────────────────────────────────────

class SingularMatrix(Exception):
    pass


def mat_inv(M):
    n = len(M)
    A = [list(M[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-300:
            raise SingularMatrix("pivot %d vanished" % c)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [v / pv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0.0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [row[n:] for row in A]


def sym_eig(M, sweeps=100, tol=1e-12):
    """Jacobi eigenvalue algorithm for symmetric M. Returns (eigvals desc, eigvecs)."""
    n = len(M)
    A = [list(r) for r in M]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = math.sqrt(sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300:
                    continue
                theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    pairs = sorted(((A[i][i], [V[r][i] for r in range(n)]) for i in range(n)),
                   key=lambda t: -t[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def spd_log(M):
    """Matrix logarithm of a symmetric positive-definite matrix."""
    vals, vecs = sym_eig(M)
    n = len(M)
    if min(vals) <= 0:
        raise SingularMatrix("non-positive eigenvalue %.3e" % min(vals))
    out = [[0.0] * n for _ in range(n)]
    for a in range(n):
        lg = math.log(vals[a])
        v = vecs[a]
        for i in range(n):
            for j in range(n):
                out[i][j] += lg * v[i] * v[j]
    return out


def spd_pow(M, p):
    vals, vecs = sym_eig(M)
    n = len(M)
    if min(vals) <= 0:
        raise SingularMatrix("non-positive eigenvalue %.3e" % min(vals))
    out = [[0.0] * n for _ in range(n)]
    for a in range(n):
        f = vals[a] ** p
        v = vecs[a]
        for i in range(n):
            for j in range(n):
                out[i][j] += f * v[i] * v[j]
    return out


def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def frob(M):
    return math.sqrt(sum(v * v for row in M for v in row))


def cholesky(M):
    n = len(M)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = M[i][i] - s
                if d <= 0:
                    raise SingularMatrix("not positive definite at %d" % i)
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (M[i][j] - s) / L[j][j]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet family — mirrors igad.families.DirichletFamily
# ─────────────────────────────────────────────────────────────────────────────

def dir_fisher(alpha):
    k = len(alpha)
    t0 = trigamma(sum(alpha))
    g = [[-t0] * k for _ in range(k)]
    for i in range(k):
        g[i][i] = trigamma(alpha[i]) - t0
    return g


def dir_cumulant_structure(alpha):
    """T_{ijk} = c + d_i*delta_{ijk}; mirrors third_cumulant_structure."""
    return -tetragamma(sum(alpha)), [tetragamma(a) for a in alpha]


def dir_scalar_curvature(alpha):
    """Exact route via a generic inverse; mirrors scalar_curvature_structured.

    The contraction is O(k^2) but `mat_inv` is O(k^3), so the *complete* path
    is O(k^3). `dir_scalar_curvature_sm` removes that bottleneck.
    """
    return dir_curvature_dense_inverse(*dir_polygamma_inputs(alpha))


def dir_scalar_curvature_sm(alpha):
    """Exact O(k) route via Sherman-Morrison; no matrix is ever formed."""
    return dir_curvature_sm_closed(*dir_polygamma_inputs(alpha))


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet curvature, decomposed so the polygamma inputs can be supplied
# ─────────────────────────────────────────────────────────────────────────────
#
# Every route below takes (tri, tri0, tet, tet0) rather than alpha. That split
# is what lets `experiments/highprec_reliability.py` separate error introduced
# by the special-function implementations from error introduced by the
# arithmetic that follows them: feed the same inputs to every route and any
# remaining disagreement is the arithmetic's.

def dir_polygamma_inputs(alpha):
    """(psi'(alpha_i), psi'(alpha_0), psi''(alpha_i), psi''(alpha_0))."""
    a0 = sum(alpha)
    return ([trigamma(a) for a in alpha], trigamma(a0),
            [tetragamma(a) for a in alpha], tetragamma(a0))


def dir_fisher_from(tri, tri0):
    """g = D - tri0 * 1 1^T with D_ii = tri_i. O(k^2)."""
    k = len(tri)
    g = [[-tri0] * k for _ in range(k)]
    for i in range(k):
        g[i][i] = tri[i] - tri0
    return g


def dir_fisher_inverse_sm(tri, tri0):
    """g^-1 in closed form, O(k^2) -- no elimination.

    With D_ii = tri_i, c = tri0, u = D^-1 1 and U = 1^T D^-1 1,

        g^-1 = D^-1 + [ c / (1 - c U) ] D^-1 1 1^T D^-1
             = D^-1 + beta u u^T,          beta = c / (1 - c U).

    Building the k x k result is itself O(k^2), which is optimal for an
    explicit inverse. `dir_curvature_sm_closed` avoids building it at all.
    """
    k = len(tri)
    u = [1.0 / t for t in tri]
    w = 1.0 - tri0 * sum(u)
    if w == 0.0:
        raise SingularMatrix("Sherman-Morrison denominator vanished")
    beta = tri0 / w
    M = [[beta * u[i] * u[a] for a in range(k)] for i in range(k)]
    for i in range(k):
        M[i][i] += u[i]
    return M


def _curvature_from_inverse(M, c, d):
    """R from an explicit g^-1 and the structured cumulant tensor. O(k^2)."""
    k = len(d)
    r = [sum(M[i][a] for i in range(k)) for a in range(k)]
    s = sum(r)
    S = [c * s + M[m][m] * d[m] for m in range(k)]
    S_sq = sum(S[m] * M[m][n] * S[n] for m in range(k) for n in range(k))
    T_sq = (c * c * s ** 3
            + 2.0 * c * sum(d[a] * r[a] ** 3 for a in range(k))
            + sum(d[i] * (M[i][a] ** 3) * d[a] for i in range(k) for a in range(k)))
    return 0.25 * (T_sq - S_sq)


def dir_curvature_dense_inverse(tri, tri0, tet, tet0):
    """Structured contraction on top of a generic O(k^3) matrix inverse."""
    c, d = -tet0, list(tet)
    return _curvature_from_inverse(mat_inv(dir_fisher_from(tri, tri0)), c, d)


def dir_curvature_sm_matrix(tri, tri0, tet, tet0):
    """Structured contraction on top of the O(k^2) Sherman-Morrison inverse."""
    c, d = -tet0, list(tet)
    return _curvature_from_inverse(dir_fisher_inverse_sm(tri, tri0), c, d)


def dir_curvature_sm_closed(tri, tri0, tet, tet0):
    """R(alpha) in O(k) time and O(k) memory. Nothing k x k is allocated.

    Substituting g^-1 = D^-1 + beta u u^T into the structured contraction and
    using U = sum u_i, w = 1 - tri0 * U, beta = tri0 / w:

        r_a        = u_a / w                    (column sums of g^-1)
        s          = U / w
        (g^-1)_mm  = u_m (1 + beta u_m)
        S_m        = c s + (g^-1)_mm d_m
        ||S||^2_g  = sum_m S_m^2 u_m + beta (sum_m S_m u_m)^2
        ||T||^2_g  = c^2 s^3
                     + 2c sum_a d_a r_a^3
                     + beta^3 [ (sum_i d_i u_i^3)^2 - sum_i (d_i u_i^3)^2 ]
                     + sum_i d_i^2 u_i^3 (1 + beta u_i)^3

    The last two lines are the split of sum_{i,a} d_i d_a (g^-1)_{ia}^3 into
    its off-diagonal part (where (g^-1)_{ia} = beta u_i u_a factorises, so the
    double sum becomes a square of a single sum) and its diagonal part.

    Derivation and complexity proof: docs/sherman_morrison.md.
    """
    k = len(tri)
    c, d = -tet0, tet
    u = [1.0 / t for t in tri]
    U = sum(u)
    w = 1.0 - tri0 * U
    if w == 0.0:
        raise SingularMatrix("Sherman-Morrison denominator vanished")
    beta = tri0 / w

    s = U / w
    S_sq = 0.0
    Su = 0.0
    for m in range(k):
        S_m = c * s + u[m] * (1.0 + beta * u[m]) * d[m]
        S_sq += S_m * S_m * u[m]
        Su += S_m * u[m]
    S_sq += beta * Su * Su

    term_a = c * c * s ** 3
    term_b = 0.0
    du3_sum = 0.0
    du3_sq = 0.0
    diag = 0.0
    for i in range(k):
        u3 = u[i] ** 3
        term_b += d[i] * u3
        du3 = d[i] * u3
        du3_sum += du3
        du3_sq += du3 * du3
        diag += d[i] * du3 * (1.0 + beta * u[i]) ** 3
    term_b = 2.0 * c * term_b / (w ** 3)          # r_a^3 = u_a^3 / w^3
    term_c = beta ** 3 * (du3_sum * du3_sum - du3_sq) + diag
    return 0.25 * ((term_a + term_b + term_c) - S_sq)


EPS = 2.0 ** -53
SIZE_EXPONENT_CLOSED_FORM = 1
RELIABILITY_SAFETY = 8.0


def dir_curvature_reliability(tri, tri0, tet, tet0):
    """Diagnostic: how many digits of R are trustworthy here.

    Mirrors `igad.curvature.curvature_reliability`. **Read-only** -- it does
    not change R, and R is not computed differently because of it.

    Returns a dict with R, the cancellation ratio rho, the estimated number of
    correct significant decimal digits (16 - log10(rho * k)), the implied
    relative-error bound, and the Sherman-Morrison denominator w.
    """
    R, rho = dir_curvature_sm_closed_diagnostic(tri, tri0, tet, tet0)
    k = len(tri)
    u_sum = sum(1.0 / t for t in tri)
    w = 1.0 - tri0 * u_sum
    bound = RELIABILITY_SAFETY * EPS * rho * k ** SIZE_EXPONENT_CLOSED_FORM
    digits = 0.0 if bound >= 1.0 else -math.log10(max(bound, EPS))
    return {"R": R, "cancellation_ratio": rho,
            "trusted_digits": max(digits, 0.0),
            "relative_error_bound": bound, "sm_denominator": w}


def dir_curvature_sm_closed_diagnostic(tri, tri0, tet, tet0):
    """(R, rho_hat) -- the curvature and a float64 estimate of its own accuracy.

    `rho_hat` is the cancellation ratio: the largest intermediate magnitude
    divided by |S^2 - T^2|. The expected relative error in R is eps * rho_hat,
    so `16 - log10(rho_hat)` is roughly how many decimal digits of R survive.
    Both come out of the same O(k) pass.

    `experiments/highprec_reliability.py` measures how well this float64
    estimate tracks the exact cancellation ratio, and how well eps * rho_hat
    bounds the error. See docs/numerical_reliability.md.
    """
    k = len(tri)
    c, d = -tet0, tet
    u = [1.0 / t for t in tri]
    U = sum(u)
    w = 1.0 - tri0 * U
    if w == 0.0:
        raise SingularMatrix("Sherman-Morrison denominator vanished")
    beta = tri0 / w

    s = U / w
    S_sq = 0.0
    Su = 0.0
    for m in range(k):
        S_m = c * s + u[m] * (1.0 + beta * u[m]) * d[m]
        S_sq += S_m * S_m * u[m]
        Su += S_m * u[m]
    S_sq += beta * Su * Su

    term_a = c * c * s ** 3
    tb = 0.0
    du3_sum = 0.0
    du3_sq = 0.0
    diag = 0.0
    for i in range(k):
        u3 = u[i] ** 3
        du3 = d[i] * u3
        tb += du3
        du3_sum += du3
        du3_sq += du3 * du3
        diag += d[i] * du3 * (1.0 + beta * u[i]) ** 3
    term_b = 2.0 * c * tb / (w ** 3)
    term_c = beta ** 3 * (du3_sum * du3_sum - du3_sq) + diag
    T_sq = term_a + term_b + term_c

    gap = T_sq - S_sq
    biggest = max(abs(S_sq), abs(T_sq), abs(term_a), abs(term_b), abs(term_c))
    rho_hat = float("inf") if gap == 0.0 else biggest / abs(gap)
    return 0.25 * gap, rho_hat


class ConvergenceError(Exception):
    pass


def dir_suff_stat(batch, k):
    n = len(batch)
    acc = [0.0] * k
    for row in batch:
        for i in range(k):
            acc[i] += math.log(row[i] if row[i] > 1e-15 else 1e-15)
    return [a / n for a in acc]


def dir_mle(mean_log_x, k, max_iter=1000, tol=1e-8):
    """Minka fixed point; mirrors DirichletFamily.mle including its gate."""
    alpha = [1.0] * k
    for _ in range(max_iter):
        prev = alpha[:]
        psi0 = digamma(sum(alpha))
        for i in range(k):
            alpha[i] = max(inv_digamma(psi0 + mean_log_x[i]), 1e-8)
        if max(abs(a - b) for a, b in zip(alpha, prev)) < tol:
            break
    a0 = sum(alpha)
    resid = max(abs((digamma(a) - digamma(a0)) - m)
                for a, m in zip(alpha, mean_log_x))
    if resid > 1e-4:
        raise ConvergenceError("sufficient-statistic residual %.2e" % resid)
    return alpha


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_dirichlet(alpha, n, rng):
    k = len(alpha)
    out = []
    for _ in range(n):
        g = [rng.gammavariate(alpha[i], 1.0) for i in range(k)]
        s = sum(g) or 1.0
        out.append([v / s for v in g])
    return out


def sample_dirichlet_mixture(modes, n, rng, weights=None):
    out = []
    for _ in range(n):
        m = rng.choice(modes) if weights is None else _weighted(modes, weights, rng)
        out.extend(sample_dirichlet(m, 1, rng))
    return out


def _weighted(items, weights, rng):
    u = rng.random() * sum(weights)
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if u <= acc:
            return it
    return items[-1]


def sample_mvn(mu, cov, n, rng):
    d = len(mu)
    L = cholesky(cov)
    out = []
    for _ in range(n):
        z = [rng.gauss(0.0, 1.0) for _ in range(d)]
        out.append([mu[i] + sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(d)])
    return out


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def to_logratio(x):
    """y_i = log(x_i / x_k), i = 1..k-1. Identifiable coordinates for the simplex."""
    k = len(x)
    xk = x[k - 1] if x[k - 1] > 1e-15 else 1e-15
    return [math.log((x[i] if x[i] > 1e-15 else 1e-15) / xk) for i in range(k - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Cheap detector statistics
# ─────────────────────────────────────────────────────────────────────────────

def mean_load(batch):
    k = len(batch[0])
    n = len(batch)
    return [sum(r[i] for r in batch) / n for i in range(k)]


def mean_entropy(batch):
    tot = 0.0
    for row in batch:
        h = 0.0
        for v in row:
            if v > 1e-12:
                h -= v * math.log(v)
        tot += h
    return tot / len(batch)


def entropy_variance(batch):
    hs = []
    for row in batch:
        h = 0.0
        for v in row:
            if v > 1e-12:
                h -= v * math.log(v)
        hs.append(h)
    m = sum(hs) / len(hs)
    return sum((h - m) ** 2 for h in hs) / len(hs)


def mean_max_prob(batch):
    return sum(max(r) for r in batch) / len(batch)


def mean_l2_mass(batch):
    """E[||x||_2^2] -- a cheap decisiveness / participation statistic."""
    return sum(sum(v * v for v in r) for r in batch) / len(batch)


def empirical_cov(rows):
    d = len(rows[0])
    n = len(rows)
    mu = [sum(r[i] for r in rows) / n for i in range(d)]
    C = [[0.0] * d for _ in range(d)]
    for r in rows:
        dv = [r[i] - mu[i] for i in range(d)]
        for i in range(d):
            for j in range(i, d):
                C[i][j] += dv[i] * dv[j]
    denom = max(n - 1, 1)
    for i in range(d):
        for j in range(i, d):
            C[i][j] /= denom
            C[j][i] = C[i][j]
    return mu, C


def trace(M):
    return sum(M[i][i] for i in range(len(M)))


def mmd_rbf(X, Y):
    """MMD^2, RBF kernel, median-heuristic bandwidth on the pooled sample.

    One pairwise distance matrix is built for the pooled set and the three
    kernel blocks are read off as submatrices -- algebraically identical to
    computing them separately, as demo_dirichlet.py does.
    """
    Z = list(X) + list(Y)
    n, nx = len(Z), len(X)
    d2 = [[0.0] * n for _ in range(n)]
    flat = []
    for i in range(n):
        zi = Z[i]
        for j in range(i + 1, n):
            s = 0.0
            for a, b in zip(zi, Z[j]):
                t = a - b
                s += t * t
            d2[i][j] = d2[j][i] = s
            flat.append(s)
    flat.sort()
    m = len(flat)
    sigma2 = (flat[m // 2] if m % 2 else 0.5 * (flat[m // 2 - 1] + flat[m // 2])) + 1e-8

    def block(ii, jj, drop_diag):
        tot = 0.0
        for i in ii:
            for j in jj:
                if drop_diag and i == j:
                    continue
                tot += math.exp(-d2[i][j] / (2 * sigma2))
        return tot

    ny = n - nx
    return (block(range(nx), range(nx), True) / (nx * (nx - 1))
            + block(range(nx, n), range(nx, n), True) / (ny * (ny - 1))
            - 2.0 * block(range(nx), range(nx, n), False) / (nx * ny))


def l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

FAILED_FIT = 1e12


def auc(labels, scores):
    """Mann-Whitney AUC with average ranks for ties."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    npos = sum(labels)
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    rsum = sum(r for r, l in zip(ranks, labels) if l)
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


def cohens_d(labels, scores):
    """Standardised mean difference between anomalous and normal scores."""
    a = [s for s, l in zip(scores, labels) if l and s < FAILED_FIT]
    b = [s for s, l in zip(scores, labels) if not l and s < FAILED_FIT]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((v - ma) ** 2 for v in a) / (len(a) - 1)
    vb = sum((v - mb) ** 2 for v in b) / (len(b) - 1)
    sp = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return (ma - mb) / sp if sp > 0 else float("nan")


def mean_sd(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def save_results(name, payload):
    """Persist raw numbers before any figure is drawn."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, name + ".json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def load_results(name):
    with open(os.path.join(RESULT_DIR, name + ".json")) as fh:
        return json.load(fh)


def print_table(title, header, rows, widths=None):
    print("=" * 78)
    print(title)
    print("=" * 78)
    widths = widths or [max(len(str(header[i])),
                            max((len(str(r[i])) for r in rows), default=0)) + 2
                        for i in range(len(header))]
    print("".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("-" * sum(widths))
    for r in rows:
        print("".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print()
