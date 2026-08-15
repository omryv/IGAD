"""
Scalar curvature of Fisher-Rao manifolds for exponential families.

Background identity (known in Hessian geometry; restated for completeness):

    R(theta) = 1/4 * ( ||grad log det g||^2_g  -  ||T||^2_g )

where:
    g_{ij}(theta)   = d^2 A / d theta_i d theta_j          (Fisher metric)
    T_{ijk}(theta)  = d^3 A / d theta_i d theta_j d theta_k (third cumulant tensor)
    S_m             = g^{ab} T_{abm}                        (trace vector)

References:
    - Amari & Nagaoka, Methods of Information Geometry (2000)
    - Ruppeiner, Riemannian geometry in thermodynamic fluctuation theory (1995)
"""

import numpy as np
from typing import Callable, Optional


def fisher_metric(
    log_partition: Callable[[np.ndarray], float],
    theta: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Compute Fisher information matrix g_{ij} = d^2 A / d theta_i d theta_j
    via central finite differences on the log-partition function A(theta).
    """
    theta = np.asarray(theta, dtype=np.float64)
    d = theta.shape[0]
    g = np.zeros((d, d))
    A = log_partition
    h = eps
    f0 = A(theta)

    for i in range(d):
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        g[i, i] = (A(tp) - 2.0 * f0 + A(tm)) / (h ** 2)

        for j in range(i + 1, d):
            tpp = theta.copy(); tpp[i] += h; tpp[j] += h
            tpm = theta.copy(); tpm[i] += h; tpm[j] -= h
            tmp = theta.copy(); tmp[i] -= h; tmp[j] += h
            tmm = theta.copy(); tmm[i] -= h; tmm[j] -= h
            g[i, j] = (A(tpp) - A(tpm) - A(tmp) + A(tmm)) / (4.0 * h ** 2)
            g[j, i] = g[i, j]

    return g


def third_cumulant_tensor(
    log_partition: Callable[[np.ndarray], float],
    theta: np.ndarray,
    eps: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """
    Compute T_{ijk} = d^3 A / d theta_i d theta_j d theta_k directly
    via dedicated finite-difference stencils for each index pattern.
    """
    theta = np.asarray(theta, dtype=np.float64)
    d = theta.shape[0]
    T = np.zeros((d, d, d))
    A = log_partition
    h = eps

    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):

                if i == j == k:
                    # d^3 A / d theta_i^3
                    tp2 = theta.copy(); tp2[i] += 2 * h
                    tp1 = theta.copy(); tp1[i] += h
                    tm1 = theta.copy(); tm1[i] -= h
                    tm2 = theta.copy(); tm2[i] -= 2 * h
                    val = (A(tp2) - 2*A(tp1) + 2*A(tm1) - A(tm2)) / (2 * h**3)

                elif i == j:
                    # d^3 A / d theta_i^2 d theta_k  (i==j != k)
                    def g_ii(t):
                        tp = t.copy(); tp[i] += h
                        tm = t.copy(); tm[i] -= h
                        return (A(tp) - 2.0*A(t) + A(tm)) / (h**2)
                    tkp = theta.copy(); tkp[k] += h
                    tkm = theta.copy(); tkm[k] -= h
                    val = (g_ii(tkp) - g_ii(tkm)) / (2.0 * h)

                elif j == k:
                    # d^3 A / d theta_i d theta_j^2  (i != j==k)
                    def g_jj(t):
                        tp = t.copy(); tp[j] += h
                        tm = t.copy(); tm[j] -= h
                        return (A(tp) - 2.0*A(t) + A(tm)) / (h**2)
                    tip = theta.copy(); tip[i] += h
                    tim = theta.copy(); tim[i] -= h
                    val = (g_jj(tip) - g_jj(tim)) / (2.0 * h)

                else:
                    # d^3 A / d theta_i d theta_j d theta_k (all different)
                    val = 0.0
                    for si in (+1, -1):
                        for sj in (+1, -1):
                            for sk in (+1, -1):
                                t = theta.copy()
                                t[i] += si * h
                                t[j] += sj * h
                                t[k] += sk * h
                                val += si * sj * sk * A(t)
                    val /= (8.0 * h**3)

                # Assign to all permutations
                for a, b, c in {(i,j,k),(i,k,j),(j,i,k),
                                (j,k,i),(k,i,j),(k,j,i)}:
                    T[a, b, c] = val

    return T


def scalar_curvature(
    log_partition: Callable[[np.ndarray], float],
    theta: np.ndarray,
    g: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
) -> float:
    """
    Compute scalar curvature R(theta) of the Fisher-Rao manifold.

        R = 1/4 * ( ||S||^2_g - ||T||^2_g )

    where S_m = g^{ab} T_{abm}.
    """
    theta = np.asarray(theta, dtype=np.float64)

    if g is None:
        g = fisher_metric(log_partition, theta)
    if T is None:
        T = third_cumulant_tensor(log_partition, theta)

    g_inv = np.linalg.inv(g)

    # Trace vector: S_m = g^{ab} T_{abm}
    S = np.einsum("ab,abm->m", g_inv, T)

    # ||S||^2_g = g^{mn} S_m S_n
    S_norm_sq = np.einsum("mn,m,n->", g_inv, S, S)

    # ||T||^2_g = g^{ia} g^{jb} g^{kc} T_{ijk} T_{abc}
    # optimize=True contracts pairwise (O(d^4)) instead of the naive
    # six-index loop (O(d^6)); the value is unchanged.
    T_norm_sq = np.einsum(
        "ia,jb,kc,ijk,abc->", g_inv, g_inv, g_inv, T, T, optimize=True
    )

    return 0.25 * (S_norm_sq - T_norm_sq)


def scalar_curvature_structured(
    g: np.ndarray,
    c: float,
    d: np.ndarray,
) -> float:
    """
    Scalar curvature for families whose third cumulant tensor has the form

        T_{ijk} = c + d_i * delta_{ijk}

    i.e. a constant offset plus a correction carried only on the triple
    diagonal. The Dirichlet family has exactly this structure; see
    ``families.DirichletFamily.third_cumulant_structure``.

    Writing M = g^{-1}, r_a = sum_i M_{ia} and s = sum_a r_a, the two
    contractions in R = 1/4 * (||S||^2_g - ||T||^2_g) collapse to

        S_m       = c * s + M_{mm} * d_m
        ||S||^2_g = S^T M S
        ||T||^2_g = c^2 * s^3
                    + 2c * sum_a d_a r_a^3
                    + sum_{i,a} d_i d_a M_{ia}^3

    This is O(k^2) rather than the O(k^6) contraction in
    :func:`scalar_curvature`, and it is exact -- not an approximation.
    Materialising T at all is avoided, which matters once k reaches the
    dozens (a k=64 evaluation is ~6.9e10 einsum operations otherwise).

    Parameters
    ----------
    g : ndarray of shape (k, k)
        Fisher metric at the parameter point.
    c : float
        Constant offset of the third cumulant tensor.
    d : ndarray of shape (k,)
        Triple-diagonal correction, T_{iii} = c + d_i.

    Returns
    -------
    float
        Scalar curvature R(theta).
    """
    g = np.asarray(g, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    M = np.linalg.inv(g)
    r = M.sum(axis=0)            # r_a = sum_i M_{ia}
    s = float(r.sum())           # s   = sum_{ia} M_{ia}

    # S_m = g^{ab} T_{abm} = c*s + M_{mm} d_m
    S = c * s + np.diag(M) * d
    S_norm_sq = float(S @ M @ S)

    # ||T||^2_g, expanded over T = c*E + D with E_{ijk}=1, D_{ijk}=d_i delta_{ijk}
    T_norm_sq = (
        c * c * s ** 3
        + 2.0 * c * float(d @ r ** 3)
        + float(d @ (M ** 3) @ d)
    )

    return 0.25 * (S_norm_sq - T_norm_sq)


def dirichlet_fisher_inverse(tri: np.ndarray, tri0: float) -> np.ndarray:
    """Inverse of the Dirichlet Fisher metric in closed form. O(k^2).

    The metric is a diagonal matrix plus a rank-one term,

        g = D - c * 1 1^T,    D_ii = psi'(alpha_i),    c = psi'(alpha_0),

    so the Sherman-Morrison identity gives g^{-1} without any elimination:

        g^{-1} = D^{-1}
                 + [ c / (1 - c * 1^T D^{-1} 1) ] D^{-1} 1 1^T D^{-1}
               = D^{-1} + beta u u^T,
        u      = D^{-1} 1,   U = 1^T u,   w = 1 - c U,   beta = c / w.

    O(k^2) is optimal here only because the *result* has k^2 entries; the
    factored form (u, beta) is O(k), and `scalar_curvature_dirichlet` uses it
    directly rather than materialising this matrix.

    Parameters
    ----------
    tri : ndarray of shape (k,)
        psi'(alpha_i).
    tri0 : float
        psi'(alpha_0), alpha_0 = sum_i alpha_i.

    Returns
    -------
    ndarray of shape (k, k)

    Raises
    ------
    ZeroDivisionError
        If 1 - c * 1^T D^{-1} 1 vanishes. For a genuine Dirichlet parameter
        this quantity is strictly positive -- it equals det(g) / prod_i D_ii,
        and g is positive definite -- so a zero here means the inputs are not
        a valid (psi'(alpha), psi'(alpha_0)) pair.
    """
    tri = np.asarray(tri, dtype=np.float64)
    u = 1.0 / tri
    w = 1.0 - tri0 * float(u.sum())
    if w == 0.0:
        raise ZeroDivisionError("Sherman-Morrison denominator 1 - c*1'D^-1*1 vanished")
    beta = tri0 / w
    return np.diag(u) + beta * np.outer(u, u)


def scalar_curvature_dirichlet(
    tri: np.ndarray,
    tri0: float,
    tet: np.ndarray,
    tet0: float,
) -> float:
    """Scalar curvature of the Dirichlet manifold in O(k) time and O(k) memory.

    Neither g nor g^{-1} is formed. Substituting the Sherman-Morrison factored
    inverse g^{-1} = D^{-1} + beta u u^T into the structured contraction of
    :func:`scalar_curvature_structured` collapses every remaining sum to a
    single pass over the k experts:

        r_a       = u_a / w
        s         = U / w
        g^{-1}_mm = u_m (1 + beta u_m)
        S_m       = c s + g^{-1}_mm d_m
        ||S||^2_g = sum_m S_m^2 u_m + beta (sum_m S_m u_m)^2
        ||T||^2_g = c^2 s^3
                    + (2c / w^3) sum_a d_a u_a^3
                    + beta^3 [ (sum_i d_i u_i^3)^2 - sum_i (d_i u_i^3)^2 ]
                    + sum_i d_i^2 u_i^3 (1 + beta u_i)^3

    with c = -psi''(alpha_0) and d_i = psi''(alpha_i). The third line is the
    off-diagonal part of sum_{i,a} d_i d_a (g^{-1})_{ia}^3: off the diagonal
    (g^{-1})_{ia} = beta u_i u_a factorises, so the double sum is a square of
    a single sum. Exact, not an approximation.

    Derivation and measured scaling: docs/sherman_morrison.md.

    Parameters
    ----------
    tri, tet : ndarray of shape (k,)
        psi'(alpha_i) and psi''(alpha_i).
    tri0, tet0 : float
        psi'(alpha_0) and psi''(alpha_0).

    Returns
    -------
    float
    """
    tri = np.asarray(tri, dtype=np.float64)
    tet = np.asarray(tet, dtype=np.float64)
    c = -float(tet0)
    d = tet

    u = 1.0 / tri
    U = float(u.sum())
    w = 1.0 - tri0 * U
    if w == 0.0:
        raise ZeroDivisionError("Sherman-Morrison denominator 1 - c*1'D^-1*1 vanished")
    beta = tri0 / w
    s = U / w

    S = c * s + u * (1.0 + beta * u) * d
    S_norm_sq = float((S * S * u).sum()) + beta * float((S * u).sum()) ** 2

    u3 = u ** 3
    du3 = d * u3
    T_norm_sq = (
        c * c * s ** 3
        + 2.0 * c * float((d * u3).sum()) / w ** 3
        + beta ** 3 * (float(du3.sum()) ** 2 - float((du3 * du3).sum()))
        + float((d * du3 * (1.0 + beta * u) ** 3).sum())
    )

    return 0.25 * (S_norm_sq - T_norm_sq)
