"""
tests/test_sherman_morrison.py

Pins the O(k) Dirichlet curvature route.

Three float64 implementations must produce the same number as each other and
as the high-precision reference:

    dense-inverse   generic O(k^3) inverse + structured contraction
    sm-matrix       Sherman-Morrison inverse materialised, O(k^2)
    sm-closed       Sherman-Morrison inverse factored, O(k)

and, when numpy is installed, so must the package versions in `igad`.

Accuracy is asserted against a bound derived from the cancellation ratio of
the expression itself (see docs/numerical_reliability.md), not against a flat
tolerance: R(alpha) genuinely cannot be evaluated to 1e-11 relative in float64
at every alpha, and a flat tolerance would either be vacuous or would fail on
points where every possible implementation fails.
"""

import math

import pytest

from experiments._highprec import (
    cancellation_ratio, hp_curvature_sherman_morrison, hp_dirichlet_inputs,
    hp_fisher, hp_mat_inv, precision, rel_error, to_decimal,
)
from experiments._router_common import (
    SingularMatrix, dir_curvature_dense_inverse, dir_curvature_reliability,
    dir_curvature_sm_closed, dir_curvature_sm_closed_diagnostic,
    dir_curvature_sm_matrix, dir_fisher_from, dir_fisher_inverse_sm,
    dir_polygamma_inputs, dir_scalar_curvature, dir_scalar_curvature_sm,
    mat_inv, trigamma,
)

EPS = 2.0 ** -53

ALPHAS = [
    [4.0, 4.0, 4.0],
    [1.5, 4.0, 6.5],
    [0.05, 0.05, 0.05],
    [14.5, 0.5, 0.5, 0.5],
    [0.02, 0.03, 0.02, 0.05],
    [200.0, 200.0, 200.0],
    [500.0, 480.0, 510.0, 495.0],
    [1e-3, 1.0, 5e3],
    [1e-4, 1e-2, 1.0, 1e3],
    [1e-5] * 5,
    [0.05, 3.0, 12.0, 0.4, 6.0],
    [0.6, 0.9, 1.4, 0.3, 2.1, 0.8, 1.1, 0.5],
    [0.4 + 0.1 * i for i in range(16)],
    [2.0] * 32,
]

ROUTES = [dir_curvature_dense_inverse, dir_curvature_sm_matrix,
          dir_curvature_sm_closed]

# Section C5 of experiments/highprec_reliability.py measures how the error
# grows with k once the cancellation ratio is divided out: k^1.11 for the O(k)
# route, k^2.0-2.2 for the two routes that build a k x k matrix. The bound uses
# the next integer up, with a 64x constant.
SIZE_EXPONENT = {"dir_curvature_sm_closed": 1,
                 "dir_curvature_sm_matrix": 2,
                 "dir_curvature_dense_inverse": 2}


def _tolerance(rho, k, exponent):
    return max(64.0 * EPS * rho * k ** exponent, 64.0 * EPS)


def _reference(alpha):
    """(R, cancellation ratio) at 120 digits."""
    with precision(120):
        parts = hp_curvature_sherman_morrison(*hp_dirichlet_inputs(alpha))
        return parts["R"], cancellation_ratio(parts)


# ─────────────────────────────────────────────────────────────────────────────
# The inverse itself
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", ALPHAS)
def test_sherman_morrison_inverse_matches_elimination(alpha):
    tri, tri0, _, _ = dir_polygamma_inputs(alpha)
    sm = dir_fisher_inverse_sm(tri, tri0)
    gj = mat_inv(dir_fisher_from(tri, tri0))
    scale = max(abs(v) for row in gj for v in row)
    worst = max(abs(a - b) for ra, rb in zip(sm, gj) for a, b in zip(ra, rb))
    assert worst <= 1e-8 * scale, "worst entry difference %.3e (scale %.3e)" % (
        worst, scale)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_sherman_morrison_inverse_is_a_left_and_right_inverse(alpha):
    tri, tri0, _, _ = dir_polygamma_inputs(alpha)
    g = dir_fisher_from(tri, tri0)
    m = dir_fisher_inverse_sm(tri, tri0)
    k = len(alpha)
    with precision(120):
        # residual is measured at high precision so the check does not merely
        # re-test float64 matrix multiplication
        gd = [[to_decimal(v) for v in row] for row in g]
        md = [[to_decimal(v) for v in row] for row in m]
        worst = 0.0
        for i in range(k):
            for j in range(k):
                acc = sum(gd[i][a] * md[a][j] for a in range(k))
                worst = max(worst, abs(float(acc - (1 if i == j else 0))))
    cond_proxy = max(abs(v) for row in m for v in row) * max(
        abs(v) for row in g for v in row)
    assert worst <= 1e-9 * max(cond_proxy, 1.0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_sherman_morrison_denominator_is_positive(alpha):
    """w = 1 - psi'(alpha_0) * sum_i 1/psi'(alpha_i) equals det(g)/prod(D_ii),
    which is positive because the Fisher metric is positive definite."""
    tri, tri0, _, _ = dir_polygamma_inputs(alpha)
    w = 1.0 - tri0 * sum(1.0 / t for t in tri)
    assert w > 0.0, "w = %.3e for alpha=%s" % (w, alpha)


def test_degenerate_denominator_raises():
    """A (tri, tri0) pair that is not a real Dirichlet parameter must raise
    rather than return a number."""
    tri = [1.0, 1.0]
    tri0 = 0.5                    # forces w = 1 - 0.5*2 = 0
    with pytest.raises(SingularMatrix):
        dir_fisher_inverse_sm(tri, tri0)
    with pytest.raises(SingularMatrix):
        dir_curvature_sm_closed(tri, tri0, [-1.0, -1.0], -0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Curvature
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", ALPHAS)
def test_all_float64_routes_match_the_reference(alpha):
    ref, rho = _reference(alpha)
    inputs = dir_polygamma_inputs(alpha)
    with precision(120):
        for fn in ROUTES:
            tol = _tolerance(rho, len(alpha), SIZE_EXPONENT[fn.__name__])
            err = rel_error(to_decimal(fn(*inputs)), ref)
            assert err < tol, "%s: rel err %.3e, bound %.3e (rho %.2e, k %d)" % (
                fn.__name__, err, tol, rho, len(alpha))


@pytest.mark.parametrize("alpha", ALPHAS)
def test_alpha_level_entrypoints_agree(alpha):
    a = dir_scalar_curvature(alpha)
    b = dir_scalar_curvature_sm(alpha)
    _, rho = _reference(alpha)
    # the dense route is the looser of the two, so its exponent governs
    tol = _tolerance(rho, len(alpha), 2) * abs(a)
    assert abs(a - b) <= max(tol, 1e-14)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_diagnostic_rho_matches_the_exact_cancellation_ratio(alpha):
    """The float64 guard rail has to be right to within an order of magnitude,
    or the reported digit count is not usable."""
    r, rho_hat = dir_curvature_sm_closed_diagnostic(*dir_polygamma_inputs(alpha))
    ref, rho = _reference(alpha)
    assert 0.1 <= rho_hat / rho <= 10.0, "rho_hat %.3e vs rho %.3e" % (rho_hat, rho)
    assert r == dir_curvature_sm_closed(*dir_polygamma_inputs(alpha))


@pytest.mark.parametrize("alpha", ALPHAS)
def test_reliability_diagnostic_is_read_only_and_conservative(alpha):
    """Two properties the diagnostic must have to be worth shipping:

    1. it does not change R -- the value it returns is bit-identical to the
       one the plain route returns;
    2. it does not over-promise -- the digits it claims are actually there.
    """
    inputs = dir_polygamma_inputs(alpha)
    info = dir_curvature_reliability(*inputs)
    assert info["R"] == dir_curvature_sm_closed(*inputs)

    ref, _ = _reference(alpha)
    with precision(120):
        err = rel_error(to_decimal(info["R"]), ref)
    actual_digits = -math.log10(max(err, 1e-17))
    assert actual_digits >= info["trusted_digits"] - 1e-9, (
        "promised %.2f digits, delivered %.2f"
        % (info["trusted_digits"], actual_digits))
    assert err <= info["relative_error_bound"]
    assert info["sm_denominator"] > 0.0
    assert info["cancellation_ratio"] >= 1.0


def test_sm_closed_allocates_nothing_quadratic():
    """O(k) memory: peak allocation must not grow like k^2.

    Doubling k should roughly double the peak, not quadruple it. Compared
    against sm-matrix, which does allocate k x k.
    """
    import tracemalloc

    def peak(fn, k):
        inputs = dir_polygamma_inputs([2.0] * k)
        fn(*inputs)                          # warm any lazy allocation
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        fn(*inputs)
        pk = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return max(pk - base, 1)

    closed_ratio = peak(dir_curvature_sm_closed, 512) / peak(
        dir_curvature_sm_closed, 256)
    matrix_ratio = peak(dir_curvature_sm_matrix, 512) / peak(
        dir_curvature_sm_matrix, 256)
    assert closed_ratio < 3.0, "sm-closed peak grew %.1fx on doubling k" % closed_ratio
    assert matrix_ratio > 3.0, "sm-matrix peak grew only %.1fx on doubling k" % matrix_ratio


def test_large_k_is_reachable():
    """The point of the O(k) route: expert counts that the O(k^3) route
    cannot reach in reasonable time."""
    k = 8192
    alpha = [2.0] * k
    r = dir_scalar_curvature_sm(alpha)
    assert math.isfinite(r)
    tri = trigamma(2.0)
    assert tri > 0
    # R grows with k on the symmetric subfamily; a sanity floor, not a claim
    assert r > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Package (numpy) mirror
# ─────────────────────────────────────────────────────────────────────────────
#
# `importorskip` is called inside each test rather than at module scope: at
# module scope it skips the whole file, taking the standard-library tests above
# with it, which is exactly the failure mode that hid them on the first run.

def _numpy():
    return pytest.importorskip("numpy", reason="package routes need numpy")


@pytest.mark.parametrize("alpha", ALPHAS)
def test_package_matches_stdlib_mirror(alpha):
    np = _numpy()
    from igad.curvature import dirichlet_fisher_inverse, scalar_curvature_dirichlet
    from igad.families import DirichletFamily

    theta = DirichletFamily.to_natural(np.array(alpha))
    tri, tri0, tet, tet0 = DirichletFamily.polygamma_inputs(theta)

    fast = scalar_curvature_dirichlet(tri, tri0, tet, tet0)
    analytical = DirichletFamily.scalar_curvature_analytical(theta)
    _, rho = _reference(alpha)
    tol = max(_tolerance(rho, len(alpha), 2) * abs(analytical), 1e-13)
    assert abs(fast - analytical) <= tol
    assert abs(DirichletFamily.scalar_curvature_fast(theta) - fast) <= tol

    inv = dirichlet_fisher_inverse(tri, tri0)
    g = DirichletFamily.fisher_metric_analytical(theta)
    resid = np.abs(g @ inv - np.eye(len(alpha))).max()
    assert resid <= 1e-9 * max(np.abs(inv).max() * np.abs(g).max(), 1.0)
    assert np.abs(inv - DirichletFamily.fisher_metric_inverse_analytical(theta)).max() == 0.0


@pytest.mark.parametrize("alpha", [[4.0, 4.0, 4.0], [1.5, 4.0, 6.5], [1.0] * 5])
def test_package_matches_the_generic_dense_contraction(alpha):
    """The O(k) route against the fully generic contraction in `igad`."""
    np = _numpy()
    from igad.curvature import scalar_curvature
    from igad.families import DirichletFamily

    theta = DirichletFamily.to_natural(np.array(alpha))
    g = DirichletFamily.fisher_metric_analytical(theta)
    T = DirichletFamily.third_cumulant_analytical(theta)
    generic = scalar_curvature(DirichletFamily.log_partition, theta, g=g, T=T)
    fast = DirichletFamily.scalar_curvature_fast(theta)
    assert abs(fast - generic) <= 1e-10 * max(abs(generic), 1.0)


@pytest.mark.parametrize("alpha", [[4.0, 4.0, 4.0], [0.5] * 8, [2.0] * 16])
def test_the_preferred_dirichlet_route_is_the_O_k_one(alpha):
    """`scalar_curvature_analytical` is the name every existing caller uses,
    including IGADDetector. It must now dispatch to the O(k) route, and the
    O(k^3) route must still be reachable for cross-checking."""
    np = _numpy()
    from igad.curvature import scalar_curvature_dirichlet
    from igad.families import DirichletFamily

    theta = DirichletFamily.to_natural(np.array(alpha))
    preferred = DirichletFamily.scalar_curvature_analytical(theta)
    closed = scalar_curvature_dirichlet(*DirichletFamily.polygamma_inputs(theta))
    assert preferred == closed          # bit-identical: same code path
    assert DirichletFamily.scalar_curvature_fast(theta) == closed

    via_inverse = DirichletFamily.scalar_curvature_via_inverse(theta)
    _, rho = _reference(alpha)
    assert abs(preferred - via_inverse) <= max(
        _tolerance(rho, len(alpha), 2) * abs(via_inverse), 1e-13)


def test_detector_uses_the_O_k_route_without_modification():
    """The detector prefers `family.scalar_curvature_analytical`; since that
    name now points at the O(k) implementation, the detector picks it up with
    no change to its own code."""
    np = _numpy()
    from igad.detector import IGADDetector
    from igad.families import DirichletFamily

    d = IGADDetector(family=DirichletFamily, k_neighbors=5)
    theta = DirichletFamily.to_natural(np.array([2.0] * 12))
    assert d._scalar_curvature(theta) == DirichletFamily.scalar_curvature_analytical(theta)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_package_reliability_matches_the_stdlib_mirror(alpha):
    np = _numpy()
    from igad.curvature import curvature_reliability
    from igad.families import DirichletFamily

    theta = DirichletFamily.to_natural(np.array(alpha))
    pkg = curvature_reliability(*DirichletFamily.polygamma_inputs(theta))
    mirror = dir_curvature_reliability(*dir_polygamma_inputs(alpha))
    for key in ("cancellation_ratio", "trusted_digits", "relative_error_bound",
                "sm_denominator"):
        assert pkg[key] == pytest.approx(mirror[key], rel=1e-9), key
    assert pkg["R"] == pytest.approx(mirror["R"], rel=1e-9)
