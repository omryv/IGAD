"""
tests/test_highprec.py

Pins the high-precision reference in `experiments/_highprec.py`, and pins the
float64 special functions against it.

The reference is what Part 1.2 uses to arbitrate between float64 curvature
routes, so it has to be checked against identities it does not itself use --
recurrence, Legendre duplication, reflection -- rather than against another
implementation of the same series.

Standard library only; these run whether or not numpy is installed.
"""

import math

import pytest

from experiments._highprec import (
    cancellation_ratio, hp_curvature_dense_naive, hp_curvature_dense_pairwise,
    hp_curvature_sherman_morrison, hp_curvature_structured, hp_digamma,
    hp_dirichlet_inputs, hp_tetragamma, hp_trigamma, precision, rel_error,
    to_decimal, validate_special_functions,
)
from experiments._router_common import digamma, tetragamma, trigamma

ULP = 2.0 ** -52          # one ulp of a float64 significand

CURVATURE_CASES = [
    [4.0, 4.0, 4.0],
    [1.5, 4.0, 6.5],
    [0.05, 0.05, 0.05],
    [200.0, 200.0, 200.0],
    [1e-3, 1.0, 5e3],
    [0.05, 3.0, 12.0, 0.4, 6.0],
    [1e-5] * 5,
    [1.0] * 6,
]


def test_special_function_identities_hold_to_100_digits():
    """psi, psi', psi'' against recurrence, duplication and reflection."""
    residuals = validate_special_functions(120)
    assert len(residuals) >= 40
    worst = max(residuals.values())
    assert worst < 1e-100, "worst identity residual %.3e" % worst


@pytest.mark.parametrize("x", [1e-6, 1e-3, 0.05, 0.5, 1.0, 2.0, 11.9, 29.9,
                               30.1, 200.0, 5000.0, 1e6])
def test_float64_polygamma_is_within_a_few_ulp(x):
    """The stdlib mirror's psi/psi'/psi'' must be correctly rounded to ~4 ulp.

    Before the shift target was raised from 12 to 30 these carried a relative
    error near 1e-12, which propagated into R(alpha) as ~1e-11 and -- because
    every curvature route shares these inputs -- was invisible to any
    route-versus-route comparison.
    """
    with precision(120):
        assert rel_error(to_decimal(digamma(x)), hp_digamma(x)) < 4 * ULP
        assert rel_error(to_decimal(trigamma(x)), hp_trigamma(x)) < 4 * ULP
        assert rel_error(to_decimal(tetragamma(x)), hp_tetragamma(x)) < 4 * ULP


@pytest.mark.parametrize("alpha", CURVATURE_CASES)
def test_hp_curvature_routes_agree(alpha):
    """Dense and structured are the same number, to far more digits than
    float64 could ever show."""
    with precision(120):
        inputs = hp_dirichlet_inputs(alpha)
        ref = hp_curvature_sherman_morrison(*inputs)
        rho = cancellation_ratio(ref)
        # 160 working digits minus whatever the cancellation costs, with margin
        tol = 10.0 ** -(120 - math.log10(max(rho, 1.0)))
        for fn in (hp_curvature_dense_pairwise, hp_curvature_structured):
            assert rel_error(fn(*inputs)["R"], ref["R"]) < tol
        if len(alpha) <= 6:
            assert rel_error(hp_curvature_dense_naive(*inputs)["R"],
                             ref["R"]) < tol


@pytest.mark.parametrize("alpha", CURVATURE_CASES)
def test_reference_is_converged(alpha):
    """Raising the precision must not move the reference."""
    with precision(120):
        lo = hp_curvature_sherman_morrison(*hp_dirichlet_inputs(alpha))["R"]
        rho = cancellation_ratio(
            hp_curvature_sherman_morrison(*hp_dirichlet_inputs(alpha)))
    with precision(200):
        hi = hp_curvature_sherman_morrison(*hp_dirichlet_inputs(alpha))["R"]
    tol = 10.0 ** -(100 - math.log10(max(rho, 1.0)))
    assert rel_error(to_decimal(lo), hi) < tol


def test_trigamma_rejects_nonpositive():
    with precision(50):
        with pytest.raises(ValueError):
            hp_trigamma(0)
        with pytest.raises(ValueError):
            hp_tetragamma(-1)
        with pytest.raises(ValueError):
            hp_digamma(0)


def test_cancellation_ratio_is_at_least_one():
    with precision(120):
        for alpha in CURVATURE_CASES:
            parts = hp_curvature_sherman_morrison(*hp_dirichlet_inputs(alpha))
            assert cancellation_ratio(parts) >= 1.0
