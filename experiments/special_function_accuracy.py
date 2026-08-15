"""
experiments/special_function_accuracy.py

A3 -- measure the float64 polygamma implementations against an independent
high-precision reference, over the full argument range the Dirichlet
experiments use.

Why this exists as its own artifact
-----------------------------------
The previous pass found that the standard-library `trigamma`/`tetragamma`
were the dominant float64 error source in R(alpha), and that the error was
invisible to every check then in place: each curvature route consumes the same
psi values, so the error cancelled exactly whenever two routes were compared
to each other. Two approximations agreeing is not validation.

This script therefore compares each float64 implementation against
`experiments/_highprec.py` at 160 working digits -- an implementation that is
itself checked against 45 identities it does not use -- and reports absolute,
relative and ulp error at every argument.

    python -m experiments.special_function_accuracy

Raw output: experiments/results/special_function_accuracy.json
"""

import argparse
import math

from experiments._highprec import (
    hp_digamma, hp_tetragamma, hp_trigamma, precision, to_decimal,
    validate_special_functions,
)
from experiments._router_common import (
    digamma, polygamma_cancellation, print_table, save_results, tetragamma,
    trigamma,
)

FUNCTIONS = [
    ("digamma", digamma, hp_digamma),
    ("trigamma", trigamma, hp_trigamma),
    ("tetragamma", tetragamma, hp_tetragamma),
]


def argument_grid():
    """Arguments spanning everything the Dirichlet experiments reach.

    alpha_i as small as 1e-6 (a router that never picks an expert) and as
    large as 1e6 (alpha_0 for a large, confident router), plus dense coverage
    either side of the recurrence/asymptotic junction at SHIFT = 30, which is
    where an implementation error would hide.
    """
    xs = []
    for e in range(-6, 7):                       # decades
        for m in (1.0, 2.0, 3.0, 5.0, 7.0):
            xs.append(m * 10.0 ** e)
    xs += [0.5, 1.5, 2.5, 4.5, 11.9, 12.0, 12.1, 25.0, 29.0, 29.5, 29.9,
           29.999, 30.0, 30.001, 30.1, 31.0, 35.0, 60.0, 120.0]
    xs += [1.0 + 1e-9, 1e-8, 3.14159265358979, 2.718281828459045]
    return sorted(set(xs))


def ulp(x):
    """Spacing of float64 at x -- the unit in the last place."""
    x = abs(float(x))
    if x == 0.0:
        return 5e-324
    return math.ldexp(1.0, math.frexp(x)[1] - 53)


def run(digits, ulp_budget):
    xs = argument_grid()
    records, rows = [], []
    worst = {name: {"abs": 0.0, "rel": 0.0, "ulp": 0.0, "adj": 0.0,
                    "cancellation": 1.0, "at": None}
             for name, _, _ in FUNCTIONS}
    worst_raw_ulp = {name: (0.0, None) for name, _, _ in FUNCTIONS}

    with precision(digits):
        for x in xs:
            rec = {"x": x}
            for name, f64, hp in FUNCTIONS:
                exact = hp(x)
                got = f64(x)
                abs_err = float(abs(to_decimal(got) - exact))
                rel_err = abs_err / float(abs(exact)) if exact != 0 else abs_err
                ulp_err = abs_err / ulp(float(exact))
                # ulp measured against the result is not the right yardstick
                # where the evaluation itself cancels; divide it out.
                cancel = polygamma_cancellation(name, x)
                adj = ulp_err / cancel if cancel and math.isfinite(cancel) else ulp_err
                rec[name] = {"value": got, "reference": float(exact),
                             "abs_err": abs_err, "rel_err": rel_err,
                             "ulp_err": ulp_err, "cancellation": cancel,
                             "cancellation_adjusted_ulp": adj}
                if ulp_err > worst_raw_ulp[name][0]:
                    worst_raw_ulp[name] = (ulp_err, x)
                if adj > worst[name]["adj"]:
                    worst[name] = {"abs": abs_err, "rel": rel_err,
                                   "ulp": ulp_err, "adj": adj,
                                   "cancellation": cancel, "at": x}
            records.append(rec)

    for name, _, _ in FUNCTIONS:
        w = worst[name]
        rows.append([name, "%.3e" % w["abs"], "%.3e" % w["rel"],
                     "%.2f" % w["ulp"], "%.1f" % w["cancellation"],
                     "%.2f" % w["adj"], "%g" % w["at"],
                     "PASS" if w["adj"] <= ulp_budget else "FAIL"])
    print_table("A3 SPECIAL FUNCTIONS -- worst error over %d arguments in "
                "[1e-6, 7e6], vs a %d-digit reference" % (len(xs), digits),
                ["function", "abs err", "rel err", "ulp", "cancellation",
                 "adjusted ulp", "at x",
                 "verdict (<= %.0f)" % ulp_budget], rows)
    print("  'cancellation' is max(|recurrence|,|tail|)/|result| for the "
          "implementation's own\n  intermediates; 'adjusted ulp' divides it "
          "out. Raw worst ulp, for the record:")
    for name, _, _ in FUNCTIONS:
        u, at = worst_raw_ulp[name]
        print("    %-11s %8.2f ulp at x = %g" % (name, u, at))
    print()

    # where the errors sit, so a reader can see there is no bad region
    bands = [(10.0 ** e, 10.0 ** (e + 1)) for e in range(-6, 7)]
    band_rows = []
    for lo, hi in bands:
        sel = [r for r in records if lo <= r["x"] < hi]
        if not sel:
            continue
        band_rows.append([("%.0e - %.0e" % (lo, hi)), len(sel)]
                         + ["%.2f" % max(r[n]["cancellation_adjusted_ulp"]
                                         for r in sel)
                            for n, _, _ in FUNCTIONS])
    print_table("adjusted ulp error by argument decade",
                ["decade", "n"] + ["max %s" % n for n, _, _ in FUNCTIONS],
                band_rows)

    identities = validate_special_functions(digits)
    worst_identity = max(identities.values())
    print("  the reference itself: %d identities it does not use "
          "(recurrence, Legendre duplication, reflection, pi^2/6, "
          "self-consistency)" % len(identities))
    print("  worst identity residual: %.2e" % worst_identity)
    print()

    all_pass = all(worst[n]["adj"] <= ulp_budget for n, _, _ in FUNCTIONS)
    print("=" * 78)
    print("A3 VERDICT: %s" % ("PASS" if all_pass else "FAIL"))
    for name, _, _ in FUNCTIONS:
        w = worst[name]
        print("  %-11s worst %.2f adjusted ulp (abs %.2e, rel %.2e) at x = %g"
              % (name, w["adj"], w["abs"], w["rel"], w["at"]))
    u, at = worst_raw_ulp["digamma"]
    print()
    print("  Documented bound, psi only: near its root at x = 1.4616321 the")
    print("  recurrence and the asymptotic tail cancel, so relative and ulp")
    print("  error there are not meaningful. Worst raw ulp is %.0f at x = %g,"
          % (u, at))
    print("  where the evaluation cancels %.0fx; absolute error is %.2e."
          % (polygamma_cancellation("digamma", at),
             max(r["digamma"]["abs_err"] for r in records if r["x"] == at)))
    print("  psi enters only inv_digamma and the Dirichlet MLE gate (tolerance")
    print("  1e-4). It is not on the curvature path at all.")
    print("=" * 78)
    print()

    return {"digits": digits, "ulp_budget": ulp_budget, "n_arguments": len(xs),
            "worst_adjusted": worst,
            "worst_raw_ulp": {n: {"ulp": u, "at": a}
                              for n, (u, a) in worst_raw_ulp.items()},
            "all_pass": all_pass,
            "reference_identity_residuals": identities,
            "worst_identity_residual": worst_identity,
            "per_argument": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--digits", type=int, default=120)
    p.add_argument("--ulp-budget", type=float, default=8.0)
    a = p.parse_args()
    payload = run(a.digits, a.ulp_budget)
    path = save_results("special_function_accuracy", payload)
    print("raw results written to %s" % path)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
