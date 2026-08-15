"""
experiments/_highprec.py

Arbitrary-precision reference arithmetic for the Dirichlet scalar-curvature
path. Standard library only (`decimal`, `fractions`, `math`).

Why this module exists
----------------------
`docs/validation_report.md` recorded four Dirichlet parameter points where the
dense and the structured float64 routes to R(alpha) disagree by more than the
1e-11 target, and attributed the residual to "the shared, ill-conditioned
g^-1 and the polygamma inputs". That attribution was reached by comparing two
float64 implementations *against each other*, which cannot establish which one
is wrong, whether either is, or what the error actually tracks.

This module supplies the missing third party: the same quantity carried at
>= 120 correct decimal digits, against which every float64 route is measured.

Contents
--------
precision control -> `precision()` context manager (working precision = digits + guard)
constants         -> pi (Machin), sin, cos, exp/ln not needed
special functions -> psi'(x) (trigamma), psi''(x) (tetragamma), Bernoulli numbers
linear algebra    -> Gauss-Jordan inverse over Decimal
curvature         -> four independent routes to the same R(alpha):
                       hp_curvature_dense_naive     literal 6-index contraction
                       hp_curvature_dense_pairwise  generic tensor, O(k^4)
                       hp_curvature_structured      c + d_i delta_ijk collapse
                       hp_curvature_sherman_morrison  never forms a matrix
                     Each returns the intermediates as well as R, so the
                     cancellation structure can be measured rather than guessed.

Accuracy argument
-----------------
psi'(x) is evaluated by upward recurrence

    psi'(x) = 1/x^2 + psi'(x+1)

until the argument exceeds `SHIFT_TARGET`, then by the Euler-Maclaurin
asymptotic series truncated after `BERNOULLI_TERMS` Bernoulli terms. The first
omitted term bounds the truncation error of an asymptotic series of this type;
with the defaults below it is below 1e-180, i.e. far under the working
precision. `validate_special_functions()` checks the result against the
recurrence, the Legendre duplication formula and the reflection formula, none
of which the implementation is allowed to assume.
"""

import contextlib
import decimal
import functools
import math
from decimal import Decimal, localcontext
from fractions import Fraction

# ─────────────────────────────────────────────────────────────────────────────
# Precision control
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DIGITS = 120
GUARD_DIGITS = 40
BERNOULLI_TERMS = 50      # B_2 .. B_100
SHIFT_TARGET = 350        # recurrence lifts the argument above this


@contextlib.contextmanager
def precision(digits=DEFAULT_DIGITS, guard=GUARD_DIGITS):
    """Run a block at `digits + guard` significant decimal digits."""
    with localcontext() as ctx:
        ctx.prec = digits + guard
        ctx.Emax = 10 ** 9
        ctx.Emin = -10 ** 9
        yield ctx


def to_decimal(x):
    """Lift a value to Decimal without changing what it means.

    float  -> the exact binary value of that double (so a float64 input is
              represented exactly, not re-parsed from its decimal spelling)
    str    -> the exact decimal literal
    int    -> exact
    """
    if isinstance(x, Decimal):
        return x
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(x)


def rel_error(approx, exact):
    """|approx - exact| / |exact| as a float, both arguments Decimal."""
    exact = to_decimal(exact)
    approx = to_decimal(approx)
    if exact == 0:
        return float(abs(approx))
    return float(abs((approx - exact) / exact))


# ─────────────────────────────────────────────────────────────────────────────
# Bernoulli numbers, exact
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def bernoulli_exact(n_max):
    """B_0 .. B_{n_max} as exact Fractions, via sum_{j<=m} C(m+1,j) B_j = 0."""
    B = [Fraction(0)] * (n_max + 1)
    B[0] = Fraction(1)
    for n in range(1, n_max + 1):
        acc = Fraction(0)
        for j in range(n):
            if B[j]:
                acc += math.comb(n + 1, j) * B[j]
        B[n] = -acc / (n + 1)
    return tuple(B)


def _bernoulli_even_decimal(terms):
    """[B_2, B_4, ..., B_{2*terms}] as Decimals at the current precision."""
    exact = bernoulli_exact(2 * terms)
    return [Decimal(exact[2 * i].numerator) / Decimal(exact[2 * i].denominator)
            for i in range(1, terms + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Constants and elementary functions (only what the validation needs)
# ─────────────────────────────────────────────────────────────────────────────

def hp_pi():
    """pi via Machin's formula, to the current working precision."""
    def arctan_inv(n):
        n = Decimal(n)
        n2 = n * n
        term = Decimal(1) / n
        total = term
        k = 1
        while True:
            term = -term / n2
            new = total + term / (2 * k + 1)
            if new == total:
                return new
            total = new
            k += 1
    return 16 * arctan_inv(5) - 4 * arctan_inv(239)


def hp_sin(x):
    """sin(x) by Taylor series; adequate for |x| <= ~4, which is all we use."""
    x = to_decimal(x)
    term = x
    total = x
    n = 1
    x2 = x * x
    while True:
        term = -term * x2 / ((2 * n) * (2 * n + 1))
        new = total + term
        if new == total:
            return new
        total = new
        n += 1


def hp_cos(x):
    x = to_decimal(x)
    term = Decimal(1)
    total = term
    n = 1
    x2 = x * x
    while True:
        term = -term * x2 / ((2 * n - 1) * (2 * n))
        new = total + term
        if new == total:
            return new
        total = new
        n += 1


# ─────────────────────────────────────────────────────────────────────────────
# Polygamma
# ─────────────────────────────────────────────────────────────────────────────

def hp_digamma(x, terms=BERNOULLI_TERMS, shift_target=SHIFT_TARGET):
    """psi(x) = d/dx log Gamma(x), for x > 0.

    Upward recurrence psi(x) = psi(x+1) - 1/x until x > shift_target, then

        psi(x) ~ log x - 1/(2x) - sum_{n>=1} B_{2n} / (2n x^{2n}).
    """
    x = to_decimal(x)
    if x <= 0:
        raise ValueError("hp_digamma requires x > 0, got %s" % x)
    acc = Decimal(0)
    target = Decimal(shift_target)
    while x < target:
        acc -= 1 / x
        x += 1
    inv = 1 / x
    inv2 = inv * inv
    total = x.ln() - inv / 2
    power = inv2                              # x^{-2}
    for n, b in enumerate(_bernoulli_even_decimal(terms), start=1):
        total -= b * power / (2 * n)
        power *= inv2
    return acc + total


def hp_trigamma(x, terms=BERNOULLI_TERMS, shift_target=SHIFT_TARGET):
    """psi'(x) = sum_{n>=0} 1/(x+n)^2, for x > 0.

    Upward recurrence psi'(x) = 1/x^2 + psi'(x+1) until x > shift_target, then

        psi'(x) ~ 1/x + 1/(2x^2) + sum_{n>=1} B_{2n} x^{-(2n+1)}.
    """
    x = to_decimal(x)
    if x <= 0:
        raise ValueError("hp_trigamma requires x > 0, got %s" % x)
    acc = Decimal(0)
    target = Decimal(shift_target)
    while x < target:
        acc += 1 / (x * x)
        x += 1
    inv = 1 / x
    inv2 = inv * inv
    total = inv + inv2 / 2
    power = inv2 * inv                        # x^{-3}
    for b in _bernoulli_even_decimal(terms):  # B_2, B_4, ...
        total += b * power
        power *= inv2
    return acc + total


def hp_tetragamma(x, terms=BERNOULLI_TERMS, shift_target=SHIFT_TARGET):
    """psi''(x) = -2 * sum_{n>=0} 1/(x+n)^3, for x > 0.

    Term-by-term derivative of the trigamma expansion:

        psi''(x) ~ -1/x^2 - 1/x^3 - sum_{n>=1} (2n+1) B_{2n} x^{-(2n+2)}.
    """
    x = to_decimal(x)
    if x <= 0:
        raise ValueError("hp_tetragamma requires x > 0, got %s" % x)
    acc = Decimal(0)
    target = Decimal(shift_target)
    while x < target:
        acc -= 2 / (x * x * x)
        x += 1
    inv = 1 / x
    inv2 = inv * inv
    total = -inv2 - inv2 * inv
    power = inv2 * inv2                       # x^{-4}
    for n, b in enumerate(_bernoulli_even_decimal(terms), start=1):
        total -= (2 * n + 1) * b * power
        power *= inv2
    return acc + total


def validate_special_functions(digits=DEFAULT_DIGITS):
    """Check psi' and psi'' against identities the implementation does not use.

    Returns a dict of relative residuals; every entry should be ~1e-`digits`.
    """
    out = {}
    with precision(digits):
        tol_scale = Decimal(1)

        # 1. psi'(1) = pi^2 / 6, psi'(1/2) = pi^2 / 2
        pi = hp_pi()
        out["trigamma_at_1_vs_pi2_over_6"] = rel_error(
            hp_trigamma(1), pi * pi / 6)
        out["trigamma_at_half_vs_pi2_over_2"] = rel_error(
            hp_trigamma(Decimal("0.5")), pi * pi / 2)

        # 2. recurrence psi'(x) - psi'(x+1) = 1/x^2   (x below the shift target,
        #    so this exercises the recurrence/asymptotic junction)
        for x in ("0.001", "0.37", "2.5", "17.25", "349.5"):
            xd = Decimal(x)
            lhs = hp_trigamma(xd) - hp_trigamma(xd + 1)
            out["trigamma_recurrence_x=%s" % x] = rel_error(lhs, 1 / (xd * xd))

        # 3. Legendre duplication: psi'(2x) = (psi'(x) + psi'(x+1/2)) / 4
        for x in ("0.3", "1.7", "13.0", "260.0"):
            xd = Decimal(x)
            rhs = (hp_trigamma(xd) + hp_trigamma(xd + Decimal("0.5"))) / 4
            out["trigamma_duplication_x=%s" % x] = rel_error(hp_trigamma(2 * xd), rhs)

        # 4. reflection: psi'(x) + psi'(1-x) = pi^2 / sin^2(pi x), 0 < x < 1
        for x in ("0.25", "0.4", "0.7"):
            xd = Decimal(x)
            s = hp_sin(pi * xd)
            out["trigamma_reflection_x=%s" % x] = rel_error(
                hp_trigamma(xd) + hp_trigamma(1 - xd), pi * pi / (s * s))

        # 5. psi'' recurrence, duplication, reflection
        for x in ("0.001", "0.37", "2.5", "349.5"):
            xd = Decimal(x)
            lhs = hp_tetragamma(xd) - hp_tetragamma(xd + 1)
            out["tetragamma_recurrence_x=%s" % x] = rel_error(
                lhs, -2 / (xd * xd * xd))
        for x in ("0.3", "1.7", "13.0", "260.0"):
            xd = Decimal(x)
            rhs = (hp_tetragamma(xd) + hp_tetragamma(xd + Decimal("0.5"))) / 8
            out["tetragamma_duplication_x=%s" % x] = rel_error(
                hp_tetragamma(2 * xd), rhs)
        for x in ("0.25", "0.4", "0.7"):
            xd = Decimal(x)
            s = hp_sin(pi * xd)
            c = hp_cos(pi * xd)
            rhs = -2 * pi * pi * pi * c / (s * s * s)
            out["tetragamma_reflection_x=%s" % x] = rel_error(
                hp_tetragamma(xd) - hp_tetragamma(1 - xd), rhs)

        # 6. psi recurrence and duplication
        for x in ("0.001", "0.37", "2.5", "349.5"):
            xd = Decimal(x)
            out["digamma_recurrence_x=%s" % x] = rel_error(
                hp_digamma(xd + 1) - hp_digamma(xd), 1 / xd)
        ln2 = Decimal(2).ln()
        for x in ("0.3", "1.7", "13.0", "260.0"):
            xd = Decimal(x)
            rhs = (hp_digamma(xd) + hp_digamma(xd + Decimal("0.5"))) / 2 + ln2
            out["digamma_duplication_x=%s" % x] = rel_error(hp_digamma(2 * xd), rhs)

        # 7. precision self-consistency: recompute with more Bernoulli terms and
        #    a higher shift target; the answer must not move.
        for x in ("0.05", "2.0", "200.0", "5000.0"):
            xd = Decimal(x)
            for label, fn in (("digamma", hp_digamma), ("trigamma", hp_trigamma),
                              ("tetragamma", hp_tetragamma)):
                out["%s_selfconsistency_x=%s" % (label, x)] = rel_error(
                    fn(xd, terms=BERNOULLI_TERMS + 10, shift_target=600), fn(xd))
        del tol_scale
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Decimal linear algebra
# ─────────────────────────────────────────────────────────────────────────────

class HPSingularMatrix(Exception):
    pass


def hp_mat_inv(M):
    """Gauss-Jordan inverse with partial pivoting, over Decimal."""
    n = len(M)
    A = [list(M[i]) + [Decimal(1) if i == j else Decimal(0) for j in range(n)]
         for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if A[p][c] == 0:
            raise HPSingularMatrix("pivot %d vanished" % c)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [v / pv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [row[n:] for row in A]


def hp_sym_eigenvalues(M, sweeps=200):
    """Jacobi eigenvalues of a symmetric Decimal matrix, descending.

    A rotation is skipped once the off-diagonal entry is negligible against the
    two diagonal entries it couples. Without that test the sweep keeps rotating
    on entries that are already at the precision floor, and the rotation angle
    overflows.
    """
    n = len(M)
    A = [[to_decimal(v) for v in row] for row in M]
    zero = Decimal(0)
    eps = Decimal(10) ** -(decimal.getcontext().prec - 4)
    for _ in range(sweeps):
        off = sum(abs(A[i][j]) for i in range(n) for j in range(n) if i != j)
        if off == zero:
            break
        moved = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) <= eps * (abs(A[p][p]) + abs(A[q][q])):
                    A[p][q] = A[q][p] = zero
                    continue
                moved = True
                theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
                sign = Decimal(1) if theta >= 0 else Decimal(-1)
                t = sign / (abs(theta) + (theta * theta + 1).sqrt())
                c = 1 / (t * t + 1).sqrt()
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
        if not moved:
            break
    return sorted((A[i][i] for i in range(n)), reverse=True)


def hp_condition_number(M):
    vals = [abs(v) for v in hp_sym_eigenvalues(M)]
    lo = min(vals)
    if lo == 0:
        return float("inf")
    return float(max(vals) / lo)


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet inputs
# ─────────────────────────────────────────────────────────────────────────────

def hp_dirichlet_inputs(alpha):
    """(tri, tri0, tet, tet0) at high precision, from Decimal-lifted alpha."""
    a = [to_decimal(v) for v in alpha]
    a0 = sum(a)
    return ([hp_trigamma(v) for v in a], hp_trigamma(a0),
            [hp_tetragamma(v) for v in a], hp_tetragamma(a0))


def hp_fisher(tri, tri0):
    """g = D - tri0 * 1 1^T, D_ii = tri_i."""
    k = len(tri)
    g = [[-tri0] * k for _ in range(k)]
    for i in range(k):
        g[i][i] = tri[i] - tri0
    return g


def hp_cumulant_structure(tet, tet0):
    """T_ijk = c + d_i delta_ijk with c = -psi''(alpha_0), d_i = psi''(alpha_i)."""
    return -tet0, list(tet)


def hp_dense_tensor(c, d):
    k = len(d)
    T = [[[c] * k for _ in range(k)] for _ in range(k)]
    for i in range(k):
        T[i][i][i] = c + d[i]
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Four routes to R(alpha), all at high precision
# ─────────────────────────────────────────────────────────────────────────────
#
# R = 1/4 * ( ||S||^2_g - ||T||^2_g ),   S_m = g^{ab} T_{abm}
#
# Route 1 and route 2 know nothing about the c + d_i delta structure: they
# materialise T and contract it generically. Route 3 uses the structural
# collapse that `igad.curvature.scalar_curvature_structured` implements.
# Route 4 additionally replaces the matrix inverse by Sherman-Morrison and
# never forms a matrix at all.

def hp_curvature_dense_naive(tri, tri0, tet, tet0):
    """Literal six-index contraction, O(k^6). Ground truth by construction."""
    k = len(tri)
    c, d = hp_cumulant_structure(tet, tet0)
    T = hp_dense_tensor(c, d)
    M = hp_mat_inv(hp_fisher(tri, tri0))
    S = [sum(M[a][b] * T[a][b][m] for a in range(k) for b in range(k))
         for m in range(k)]
    S_sq = sum(S[m] * M[m][n] * S[n] for m in range(k) for n in range(k))
    T_sq = Decimal(0)
    for i in range(k):
        for j in range(k):
            for kk in range(k):
                t = T[i][j][kk]
                if t == 0:
                    continue
                for a in range(k):
                    mia = M[i][a]
                    for b in range(k):
                        mjb = M[j][b]
                        for cc in range(k):
                            T_sq += mia * mjb * M[kk][cc] * t * T[a][b][cc]
    return {"R": (S_sq - T_sq) / 4, "S_sq": S_sq, "T_sq": T_sq}


def hp_curvature_dense_pairwise(tri, tri0, tet, tet0):
    """Generic tensor, contracted one index at a time. O(k^4).

    Also reports the absolute mass of the final contraction, i.e.
    sum |term| / |sum term|, which is the cancellation actually incurred by
    this route rather than by the structured one.
    """
    k = len(tri)
    c, d = hp_cumulant_structure(tet, tet0)
    T = hp_dense_tensor(c, d)
    M = hp_mat_inv(hp_fisher(tri, tri0))
    S = [sum(M[a][b] * T[a][b][m] for a in range(k) for b in range(k))
         for m in range(k)]
    S_sq = sum(S[m] * M[m][n] * S[n] for m in range(k) for n in range(k))
    U1 = [[[sum(M[i][a] * T[i][j][kk] for i in range(k)) for kk in range(k)]
           for j in range(k)] for a in range(k)]
    U2 = [[[sum(M[j][b] * U1[a][j][kk] for j in range(k)) for kk in range(k)]
           for b in range(k)] for a in range(k)]
    U3 = [[[sum(M[kk][cc] * U2[a][b][kk] for kk in range(k)) for cc in range(k)]
           for b in range(k)] for a in range(k)]
    terms = [U3[a][b][cc] * T[a][b][cc]
             for a in range(k) for b in range(k) for cc in range(k)]
    T_sq = sum(terms)
    abs_mass = sum(abs(t) for t in terms)
    return {"R": (S_sq - T_sq) / 4, "S_sq": S_sq, "T_sq": T_sq,
            "dense_abs_mass": abs_mass}


def hp_curvature_structured(tri, tri0, tet, tet0):
    """The c + d_i delta_ijk collapse, with an explicit g^-1. O(k^2) after the
    inverse. Mirrors `igad.curvature.scalar_curvature_structured`."""
    k = len(tri)
    c, d = hp_cumulant_structure(tet, tet0)
    M = hp_mat_inv(hp_fisher(tri, tri0))
    r = [sum(M[i][a] for i in range(k)) for a in range(k)]
    s = sum(r)
    S = [c * s + M[m][m] * d[m] for m in range(k)]
    S_sq = sum(S[m] * M[m][n] * S[n] for m in range(k) for n in range(k))
    term_a = c * c * s ** 3
    term_b = 2 * c * sum(d[a] * r[a] ** 3 for a in range(k))
    term_c = sum(d[i] * (M[i][a] ** 3) * d[a]
                 for i in range(k) for a in range(k))
    T_sq = term_a + term_b + term_c
    return {"R": (S_sq - T_sq) / 4, "S_sq": S_sq, "T_sq": T_sq,
            "term_a": term_a, "term_b": term_b, "term_c": term_c,
            "s": s, "c": c}


def hp_curvature_sherman_morrison(tri, tri0, tet, tet0):
    """The O(k) route: g^-1 is never formed, only (u, beta) are carried.

        g   = D - c_g 1 1^T,  D_ii = tri_i,  c_g = tri0
        u   = D^-1 1,  U = 1^T u,  w = 1 - c_g U,  beta = c_g / w
        g^-1 = D^-1 + beta u u^T

    Every contraction in R then collapses to O(k) sums; see
    docs/sherman_morrison.md for the derivation.
    """
    k = len(tri)
    c, d = hp_cumulant_structure(tet, tet0)
    c_g = tri0
    u = [1 / t for t in tri]
    U = sum(u)
    w = 1 - c_g * U
    if w == 0:
        raise HPSingularMatrix("Sherman-Morrison denominator vanished")
    beta = c_g / w

    s = U / w                                   # sum_{i,a} (g^-1)_{ia}
    r = [ui / w for ui in u]                    # r_a = sum_i (g^-1)_{ia}
    Mdiag = [u[i] + beta * u[i] * u[i] for i in range(k)]
    S = [c * s + Mdiag[m] * d[m] for m in range(k)]

    # ||S||^2_g = S^T (D^-1 + beta u u^T) S
    S_sq = sum(S[m] * S[m] * u[m] for m in range(k)) \
        + beta * (sum(S[m] * u[m] for m in range(k))) ** 2

    term_a = c * c * s ** 3
    term_b = 2 * c * sum(d[a] * r[a] ** 3 for a in range(k))
    # sum_{i,a} d_i d_a (g^-1)_{ia}^3, split into off-diagonal and diagonal
    du3 = [d[i] * u[i] ** 3 for i in range(k)]
    off = beta ** 3 * ((sum(du3)) ** 2 - sum(v * v for v in du3))
    diag = sum(d[i] * d[i] * u[i] ** 3 * (1 + beta * u[i]) ** 3
               for i in range(k))
    term_c = off + diag
    T_sq = term_a + term_b + term_c
    return {"R": (S_sq - T_sq) / 4, "S_sq": S_sq, "T_sq": T_sq,
            "term_a": term_a, "term_b": term_b, "term_c": term_c,
            "s": s, "c": c, "w": w, "beta": beta, "U": U}


# ─────────────────────────────────────────────────────────────────────────────
# Cancellation measurement
# ─────────────────────────────────────────────────────────────────────────────

def cancellation_ratio(parts):
    """rho = (largest intermediate magnitude) / |S_sq - T_sq|.

    R = (S_sq - T_sq)/4, so a float64 evaluation of this expression loses
    about log10(rho) decimal digits regardless of how the pieces are grouped:
    eps * rho is the expected relative error in R.

    `parts` is the dict returned by any of the curvature routes.
    """
    mags = [abs(parts["S_sq"]), abs(parts["T_sq"])]
    for key in ("term_a", "term_b", "term_c"):
        if key in parts:
            mags.append(abs(parts[key]))
    if "dense_abs_mass" in parts:
        mags.append(abs(parts["dense_abs_mass"]))
    denom = abs(parts["S_sq"] - parts["T_sq"])
    if denom == 0:
        return float("inf")
    return float(max(mags) / denom)
