"""
Scalar curvature of Fisher-Rao manifolds for exponential families.

Background identity (known in Hessian geometry; restated for completeness):

    R_igad(theta) = 1/4 * ( ||S||^2_g  -  ||T||^2_g )

where:
    g_{ij}(theta)   = d^2 A / d theta_i d theta_j          (Fisher metric)
    T_{ijk}(theta)  = d^3 A / d theta_i d theta_j d theta_k (third cumulant tensor)
    S_m             = g^{ab} T_{abm}                        (trace vector)

Sign convention note:
    The quantity computed here equals -R under the standard Levi-Civita /
    differential-geometry convention (positive ~2 for a 2D Gaussian manifold,
    rather than the correct Riemannian value of -1).  This follows the
    Ruppeiner / thermodynamic sign convention (Ruppeiner 1995, eq. 4.3).
    The sign is consistent throughout the codebase; what matters for the
    IGAD score is |R_igad(theta_ref) - R_igad(theta_local)|, which is
    sign-invariant.

References:
    - Amari & Nagaoka, Methods of Information Geometry (2000)
    - Ruppeiner, Riemannian geometry in thermodynamic fluctuation theory
      (Rev. Mod. Phys. 67, 605, 1995) — sign convention follows eq. 4.3
    - Shima, The Geometry of Hessian Structures (2007), Theorem 3.1
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
                    def g_ii(t, _i=i, _h=h):
                        tp = t.copy(); tp[_i] += _h
                        tm = t.copy(); tm[_i] -= _h
                        return (A(tp) - 2.0*A(t) + A(tm)) / (_h**2)
                    tkp = theta.copy(); tkp[k] += h
                    tkm = theta.copy(); tkm[k] -= h
                    val = (g_ii(tkp) - g_ii(tkm)) / (2.0 * h)

                elif j == k:
                    # d^3 A / d theta_i d theta_j^2  (i != j==k)
                    def g_jj(t, _j=j, _h=h):
                        tp = t.copy(); tp[_j] += _h
                        tm = t.copy(); tm[_j] -= _h
                        return (A(tp) - 2.0*A(t) + A(tm)) / (_h**2)
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
    Compute scalar curvature R_igad(theta) of the Fisher-Rao manifold.

        R_igad = 1/4 * ( ||S||^2_g - ||T||^2_g )

    where S_m = g^{ab} T_{abm}.

    Note: This quantity equals -R under the standard Levi-Civita convention.
    See module docstring for the sign convention discussion.
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
    T_norm_sq = np.einsum("ia,jb,kc,ijk,abc->", g_inv, g_inv, g_inv, T, T)

    return 0.25 * (S_norm_sq - T_norm_sq)
