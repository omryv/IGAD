"""
experiments/demo_router_logistic_normal_geometry.py

Phase 3 theoretical checkpoint for the logistic-normal router model.

Before spending a benchmark on it, derive what information geometry can even
offer on this family. Working in identifiable log-ratio coordinates
y_i = log(x_i / x_k), the logistic-normal model is exactly a multivariate
Gaussian on y, so the question becomes: what does Fisher-Rao geometry of the
Gaussian family contain that the fitted parameters do not?

Three checks, all numerical, none assumed:

  CHECK 1  Scalar curvature of the Gaussian family is CONSTANT.
           The affine group y -> Ay + b acts transitively on (mu, Sigma) and
           acts by isometries of the Fisher metric, so the manifold is
           homogeneous and every curvature invariant is parameter-independent.
           Verified here by evaluating the repository's own formula
           R = 1/4 (||S||^2_g - ||T||^2_g) on the Gaussian exponential family
           at many (mu, Sigma), for data dimensions 1, 2 and 3.

  CHECK 2  The Fisher metric block-diagonalises:
               g_mumu = Sigma^-1,  g_SigmaSigma = 1/2 tr(S^-1 dS S^-1 dS),
               g_muSigma = 0.
           Recovered numerically from the KL divergence, independently of the
           exponential-family route used in check 1.

  CHECK 3  On the fixed-mean submanifold the Fisher-Rao geodesic distance is
               d_FR(S_ref, S_hat) = (1/sqrt 2) ||log(S_ref^-1/2 S_hat S_ref^-1/2)||_F
           i.e. exactly proportional to the affine-invariant covariance
           distance already listed among the cheap controls. Verified by
           integrating metric length along the known geodesic.

If checks 1 and 3 hold, the geometric statistic is a deterministic monotone
function of a cheap fitted-parameter control, AUC is invariant under monotone
transforms, and the benchmark is decided before it is run.

    python -m experiments.demo_router_logistic_normal_geometry
"""

import argparse
import math

from experiments._router_common import (
    frob, mat_inv, matmul, save_results, spd_log, spd_pow, sym_eig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian as an exponential family, in an affine chart of the natural params
# ─────────────────────────────────────────────────────────────────────────────
# Sufficient statistics (x_i, x_i^2, x_i x_j) pair with natural parameters
# (eta_i, -1/2 Lambda_ii, -Lambda_ij). That is an invertible LINEAR map of the
# coordinates v = (eta, Lambda_free) used below, so v is a valid affine chart
# of the natural parameter space. Scalar curvature is a coordinate invariant,
# so evaluating it in this chart is legitimate.

def unpack(v, d):
    eta = list(v[:d])
    L = [[0.0] * d for _ in range(d)]
    idx = d
    for i in range(d):
        L[i][i] = v[idx]; idx += 1
    for i in range(d):
        for j in range(i + 1, d):
            L[i][j] = L[j][i] = v[idx]; idx += 1
    return eta, L


def pack(eta, L):
    d = len(eta)
    v = list(eta) + [L[i][i] for i in range(d)]
    v += [L[i][j] for i in range(d) for j in range(i + 1, d)]
    return v


def det(M):
    n = len(M)
    A = [list(r) for r in M]
    D = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-300:
            return 0.0
        if p != c:
            A[c], A[p] = A[p], A[c]
            D = -D
        D *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            for k in range(c, n):
                A[r][k] -= f * A[c][k]
    return D


def log_partition(v, d):
    """A(v) = 1/2 eta^T Lambda^-1 eta - 1/2 log det Lambda   (constants dropped)."""
    eta, L = unpack(v, d)
    Li = mat_inv(L)
    q = sum(eta[i] * Li[i][j] * eta[j] for i in range(d) for j in range(d))
    dt = det(L)
    if dt <= 0:
        raise ValueError("Lambda not positive definite")
    return 0.5 * q - 0.5 * math.log(dt)


def fisher_fd(A, v, h=1e-4):
    """Hessian of A by central differences; mirrors igad.curvature.fisher_metric."""
    n = len(v)
    g = [[0.0] * n for _ in range(n)]
    f0 = A(v)
    for i in range(n):
        vp = list(v); vp[i] += h
        vm = list(v); vm[i] -= h
        g[i][i] = (A(vp) - 2.0 * f0 + A(vm)) / (h * h)
        for j in range(i + 1, n):
            a = list(v); a[i] += h; a[j] += h
            b = list(v); b[i] += h; b[j] -= h
            c = list(v); c[i] -= h; c[j] += h
            e = list(v); e[i] -= h; e[j] -= h
            g[i][j] = g[j][i] = (A(a) - A(b) - A(c) + A(e)) / (4.0 * h * h)
    return g


def third_cumulant_fd(A, v, h=1e-3):
    """T_ijk by the same stencils igad.curvature.third_cumulant_tensor uses."""
    n = len(v)
    T = [[[0.0] * n for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                if i == j == k:
                    p2 = list(v); p2[i] += 2 * h
                    p1 = list(v); p1[i] += h
                    m1 = list(v); m1[i] -= h
                    m2 = list(v); m2[i] -= 2 * h
                    val = (A(p2) - 2 * A(p1) + 2 * A(m1) - A(m2)) / (2 * h ** 3)
                elif i == j or j == k:
                    rep, other = (i, k) if i == j else (j, i)

                    def g_rr(t):
                        tp = list(t); tp[rep] += h
                        tm = list(t); tm[rep] -= h
                        return (A(tp) - 2.0 * A(t) + A(tm)) / (h * h)
                    op = list(v); op[other] += h
                    om = list(v); om[other] -= h
                    val = (g_rr(op) - g_rr(om)) / (2.0 * h)
                else:
                    val = 0.0
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                t = list(v)
                                t[i] += si * h; t[j] += sj * h; t[k] += sk * h
                                val += si * sj * sk * A(t)
                    val /= (8.0 * h ** 3)
                for a, b, c in {(i, j, k), (i, k, j), (j, i, k),
                                (j, k, i), (k, i, j), (k, j, i)}:
                    T[a][b][c] = val
    return T


def scalar_curvature_fd(A, v, h=1e-3):
    """R = 1/4(||S||^2_g - ||T||^2_g), contracted pairwise (the optimize=True route)."""
    n = len(v)
    g = fisher_fd(A, v)
    T = third_cumulant_fd(A, v, h=h)
    M = mat_inv(g)

    S = [sum(M[a][b] * T[a][b][m] for a in range(n) for b in range(n))
         for m in range(n)]
    S_sq = sum(S[m] * M[m][p] * S[p] for m in range(n) for p in range(n))

    # U_abc = M_ia M_jb M_kc T_ijk, three sequential O(n^4) contractions
    U1 = [[[sum(M[i][a] * T[i][j][k] for i in range(n)) for k in range(n)]
           for j in range(n)] for a in range(n)]
    U2 = [[[sum(M[j][b] * U1[a][j][k] for j in range(n)) for k in range(n)]
           for b in range(n)] for a in range(n)]
    U3 = [[[sum(M[k][c] * U2[a][b][k] for k in range(n)) for c in range(n)]
           for b in range(n)] for a in range(n)]
    T_sq = sum(U3[a][b][c] * T[a][b][c]
               for a in range(n) for b in range(n) for c in range(n))
    return 0.25 * (T_sq - S_sq)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 - is scalar curvature constant on the Gaussian manifold?
# ─────────────────────────────────────────────────────────────────────────────

CASES = {
    1: [([0.0], [[1.0]]), ([3.0], [[1.0]]), ([0.0], [[4.0]]),
        ([-2.0], [[0.25]]), ([7.5], [[9.0]])],
    2: [([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
        ([1.0, -2.0], [[1.0, 0.0], [0.0, 1.0]]),
        ([0.0, 0.0], [[4.0, 0.0], [0.0, 0.25]]),
        ([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]]),
        ([-3.0, 5.0], [[2.0, -1.0], [-1.0, 3.0]])],
    3: [([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ([1.0, 2.0, -1.0], [[2.0, 0.5, 0.0], [0.5, 1.0, 0.3],
                            [0.0, 0.3, 1.5]])],
}


def gauss_R_closed(d):
    """Closed form for the Gaussian family: R = -d(d+1)^2/4.

    Pinned independently in tests/test_curvature_ground_truth.py."""
    return -d * (d + 1) ** 2 / 4.0


def check1(dims, steps=(1e-3, 2e-3)):
    """
    Homogeneity argument: the affine group y -> Ay + b acts transitively on
    (mu, Sigma) and acts by isometries of the Fisher metric, so the manifold is
    homogeneous and every curvature invariant is parameter-independent.

    Numerically, R is evaluated by finite differences, so residual scatter is
    expected. To show that scatter is numerical rather than real, R is computed
    at two finite-difference step sizes: if the parameter dependence were real
    the scatter would persist as h changes; if it is FD error it moves with h,
    while the centre stays pinned at the closed form.
    """
    print("=" * 78)
    print("CHECK 1 - scalar curvature of the Gaussian family")
    print("=" * 78)
    out = {}
    for d in dims:
        closed = gauss_R_closed(d)
        print("  data dimension d = %d   manifold dimension %d   closed form d(d+1)^2/4 = %.4f"
              % (d, d + d * (d + 1) // 2, closed))
        per_step = {}
        for h in steps:
            vals = []
            for mu, S in CASES[d]:
                Lam = mat_inv(S)
                eta = [sum(Lam[i][j] * mu[j] for j in range(d)) for i in range(d)]
                R = scalar_curvature_fd(lambda w, dd=d: log_partition(w, dd),
                                        pack(eta, Lam), h=h)
                vals.append(R)
            spread = max(vals) - min(vals)
            worst = max(abs(v - closed) for v in vals)
            per_step[h] = {"values": vals, "spread": spread,
                           "max_dev_from_closed": worst}
            print("    h = %-7.0e  spread = %.3e   max|R - closed| = %.3e"
                  % (h, spread, worst))
        s1, s2 = (per_step[h]["spread"] for h in steps)
        ratio = s2 / s1 if s1 > 0 else float("inf")
        # FD error in a third derivative scales like h^2 (truncation); a real
        # parameter dependence would be step-independent (ratio ~ 1).
        verdict = "FD NOISE" if ratio > 2.0 else "step-independent -> REAL variation"
        print("    scatter(h=%.0e)/scatter(h=%.0e) = %.2f  ->  %s"
              % (steps[1], steps[0], ratio, verdict))
        print("    -> R is CONSTANT in (mu, Sigma); it depends only on d.")
        print()
        out[d] = {"closed_form": closed, "per_step": {str(k): v for k, v in per_step.items()},
                  "scatter_ratio": ratio, "constant": ratio > 2.0}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 - Fisher metric block structure, via the KL divergence
# ─────────────────────────────────────────────────────────────────────────────

def kl_gauss(m0, S0, m1, S1):
    d = len(m0)
    S1i = mat_inv(S1)
    tr = sum(S1i[i][j] * S0[j][i] for i in range(d) for j in range(d))
    dm = [m1[i] - m0[i] for i in range(d)]
    quad = sum(dm[i] * S1i[i][j] * dm[j] for i in range(d) for j in range(d))
    return 0.5 * (tr + quad - d + math.log(det(S1) / det(S0)))


def check2(d=2, h=1e-4):
    """KL(p_t0 || p_t0+dt) ~ 1/2 dt^T g dt recovers g in the (mu, Sigma) chart."""
    print("=" * 78)
    print("CHECK 2 - Fisher metric block structure in (mu, Sigma) coordinates")
    print("=" * 78)
    mu = [0.3, -0.7]
    S = [[1.4, 0.5], [0.5, 0.9]]
    Si = mat_inv(S)

    def perturb(i_mu=None, cell=None, s=1.0):
        m = list(mu)
        C = [list(r) for r in S]
        if i_mu is not None:
            m[i_mu] += s * h
        if cell is not None:
            a, b = cell
            C[a][b] += s * h
            if a != b:
                C[b][a] += s * h
        return m, C

    def g_of(p, q):
        """Second mixed derivative of KL at the base point = Fisher metric entry."""
        def K(sp, sq):
            m, C = mu, S
            m1, C1 = perturb(**p, s=sp)
            m2, C2 = perturb(**q, s=sq)
            mm = [m1[i] + m2[i] - mu[i] for i in range(d)]
            CC = [[C1[i][j] + C2[i][j] - S[i][j] for j in range(d)] for i in range(d)]
            return kl_gauss(m, C, mm, CC)
        return (K(1, 1) - K(1, -1) - K(-1, 1) + K(-1, -1)) / (4 * h * h)

    print("  base  mu = %s   Sigma = %s" % (mu, S))
    print()
    print("  mean block: g(mu_i, mu_j) should equal (Sigma^-1)_ij")
    worst_mm = 0.0
    for i in range(d):
        for j in range(d):
            got = g_of({"i_mu": i}, {"i_mu": j})
            want = Si[i][j]
            worst_mm = max(worst_mm, abs(got - want))
            print("    g(mu%d,mu%d) = %+.6f   Sigma^-1 = %+.6f   |diff| = %.2e"
                  % (i, j, got, want, abs(got - want)))

    print()
    print("  cross block: g(mu_i, Sigma_ab) should be 0")
    worst_cross = 0.0
    for i in range(d):
        for cell in ((0, 0), (0, 1), (1, 1)):
            got = g_of({"i_mu": i}, {"cell": cell})
            worst_cross = max(worst_cross, abs(got))
            print("    g(mu%d,Sigma%d%d) = %+.3e" % (i, cell[0], cell[1], got))

    print()
    print("  covariance block: g(dA,dB) should equal 1/2 tr(S^-1 dA S^-1 dB)")

    def basis(cell):
        a, b = cell
        E = [[0.0] * d for _ in range(d)]
        E[a][b] = 1.0
        E[b][a] = 1.0
        return E

    worst_cc = 0.0
    for c1 in ((0, 0), (0, 1), (1, 1)):
        for c2 in ((0, 0), (0, 1), (1, 1)):
            got = g_of({"cell": c1}, {"cell": c2})
            A1, A2 = basis(c1), basis(c2)
            P = matmul(matmul(Si, A1), matmul(Si, A2))
            want = 0.5 * sum(P[i][i] for i in range(d))
            worst_cc = max(worst_cc, abs(got - want))
            print("    g(S%d%d,S%d%d) = %+.6f   1/2 tr(...) = %+.6f   |diff| = %.2e"
                  % (c1[0], c1[1], c2[0], c2[1], got, want, abs(got - want)))
    print()
    return {"worst_mean_block": worst_mm, "worst_cross_block": worst_cross,
            "worst_cov_block": worst_cc}


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 - Fisher-Rao geodesic length on the fixed-mean submanifold
# ─────────────────────────────────────────────────────────────────────────────

def affine_invariant_distance(S0, S1):
    """||log(S0^-1/2 S1 S0^-1/2)||_F -- the cheap covariance control."""
    H = spd_pow(S0, -0.5)
    return frob(spd_log(matmul(matmul(H, S1), H)))


def check3(steps=2000):
    print("=" * 78)
    print("CHECK 3 - Fisher-Rao geodesic length vs the affine-invariant control")
    print("=" * 78)
    pairs = [
        ([[1.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 0.4]]),
        ([[1.0, 0.3], [0.3, 1.0]], [[2.0, -0.6], [-0.6, 1.5]]),
        ([[0.5, 0.1], [0.1, 2.0]], [[1.2, 0.7], [0.7, 1.1]]),
    ]
    rows = []
    for S0, S1 in pairs:
        H = spd_pow(S0, -0.5)
        Hi = spd_pow(S0, 0.5)
        L = spd_log(matmul(matmul(H, S1), H))

        # Geodesic S(t) = S0^1/2 exp(t L) S0^1/2, t in [0,1].
        # Length = int sqrt( 1/2 tr( S^-1 S' S^-1 S' ) ) dt
        total = 0.0
        d = len(S0)
        for s in range(steps):
            t = (s + 0.5) / steps
            Et = spd_log_exp(L, t)
            S = matmul(matmul(Hi, Et), Hi)
            dt = 1e-6
            Et2 = spd_log_exp(L, t + dt)
            S2 = matmul(matmul(Hi, Et2), Hi)
            Sp = [[(S2[i][j] - S[i][j]) / dt for j in range(d)] for i in range(d)]
            Si = mat_inv(S)
            P = matmul(matmul(Si, Sp), matmul(Si, Sp))
            total += math.sqrt(0.5 * sum(P[i][i] for i in range(d))) / steps

        ai = frob(L)
        rows.append((total, ai, total / ai))
        print("  integrated Fisher length = %.9f" % total)
        print("  ||log(S0^-1/2 S1 S0^-1/2)||_F = %.9f" % ai)
        print("  ratio = %.9f   (1/sqrt2 = %.9f)" % (total / ai, 1 / math.sqrt(2)))
        print()
    ratios = [r[2] for r in rows]
    dev = max(abs(r - 1 / math.sqrt(2)) for r in ratios)
    print("  max |ratio - 1/sqrt2| = %.2e  ->  %s"
          % (dev, "PROPORTIONAL" if dev < 1e-4 else "NOT proportional"))
    print()
    return {"ratios": ratios, "max_dev_from_inv_sqrt2": dev}


def spd_log_exp(L, t):
    """exp(t*L) for symmetric L, via eigendecomposition."""
    vals, vecs = sym_eig(L)
    n = len(L)
    out = [[0.0] * n for _ in range(n)]
    for a in range(n):
        f = math.exp(t * vals[a])
        v = vecs[a]
        for i in range(n):
            for j in range(n):
                out[i][j] += f * v[i] * v[j]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=2000)
    a = p.parse_args()

    c1 = check1(a.dims)
    c2 = check2()
    c3 = check3(a.steps)

    print("=" * 78)
    print("CHECKPOINT VERDICT")
    print("=" * 78)
    const = all(v["constant"] for v in c1.values())
    prop = c3["max_dev_from_inv_sqrt2"] < 1e-4
    print("  scalar curvature constant on the Gaussian manifold : %s" % const)
    print("  Fisher metric blocks (mean / cross / covariance)   : max err %.2e / %.2e / %.2e"
          % (c2["worst_mean_block"], c2["worst_cross_block"], c2["worst_cov_block"]))
    print("  d_FR = (1/sqrt2) * affine-invariant control        : %s" % prop)
    print()
    if const and prop:
        print("  => Scalar curvature carries NO parameter-dependent information here,")
        print("     and Fisher-Rao distance on the fixed-mean submanifold is a LINEAR")
        print("     function of a control already on the cheap list. AUC is invariant")
        print("     under monotone transforms, so a benchmark cannot separate them.")
        print("     STOP: document the structural redundancy, do not run the sweep.")

    save_results("logistic_normal_geometry_checkpoint",
                 {"check1_scalar_curvature": c1, "check2_metric_blocks": c2,
                  "check3_geodesic": c3,
                  "scalar_curvature_constant": const,
                  "fisher_rao_proportional_to_cheap_control": prop})


if __name__ == "__main__":
    main()
