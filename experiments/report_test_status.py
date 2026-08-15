"""
experiments/report_test_status.py

A8 -- say exactly which tests ran, which could not, and why.

The failure mode this exists to prevent
---------------------------------------
"141 passed" is not a status report if 18 tests silently skipped and 3 modules
were never collected. Worse, a module-level `pytest.importorskip` skips an
entire file, so a numpy guard placed at the top of a file takes the file's
standard-library tests down with it -- which happened in this repository and
hid a whole test module for one commit.

So this script classifies every test into exactly one of three buckets and
refuses to blur them:

  verified-locally      the test executed in this environment and passed
  skipped-no-dependency the test could not run because a package is absent
  not-collected         the module could not even be imported

and reports CI separately, because a workflow run that never receives a runner
is not a code result in either direction.

    python -m experiments.report_test_status

Raw output: experiments/results/test_status.json
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

from experiments._router_common import print_table, save_results

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Packages a test module may require. Absence is a fact about the environment,
# not about the code.
OPTIONAL_DEPENDENCIES = ("numpy", "scipy", "sklearn", "matplotlib", "torch",
                         "trimesh", "open3d")


def available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def find_pytest():
    """How to invoke pytest here, as an argv prefix.

    `python -m pytest` is the obvious answer and is wrong in any environment
    where pytest is installed as a standalone tool rather than into the
    interpreter running this script -- which is the case here, and which
    silently produced a report of zero tests until it was caught.
    """
    try:
        subprocess.run([sys.executable, "-c", "import pytest"], check=True,
                       capture_output=True)
        return [sys.executable, "-m", "pytest"]
    except (subprocess.CalledProcessError, OSError):
        pass
    exe = shutil.which("pytest")
    if exe:
        return [exe]
    return None


def run_pytest(extra_args):
    """Run pytest and parse its `-rA` short summary.

    pytest-json-report is a dev dependency and may not be installed, so the
    parsing depends only on `-rA` output, which every pytest version produces.
    """
    prefix = find_pytest()
    if prefix is None:
        return None, None
    # --continue-on-collection-errors is essential, not cosmetic: without it a
    # single unimportable module aborts the entire session, and the report
    # would say "0 tests verified" while conflating "did not run" with
    # "could not be collected".
    cmd = prefix + ["tests/", "-q", "--no-header", "-rA",
                    "--continue-on-collection-errors",
                    "-p", "no:cacheprovider"] + list(extra_args)
    return " ".join(cmd), subprocess.run(cmd, cwd=REPO, capture_output=True,
                                         text=True)


def parse_report(text):
    """Extract PASSED / SKIPPED / ERROR lines from `-rA` short summary."""
    outcomes = {"passed": [], "skipped": [], "failed": [], "error": []}
    for line in text.splitlines():
        line = line.strip()
        for tag, key in (("PASSED ", "passed"), ("SKIPPED ", "skipped"),
                         ("FAILED ", "failed"), ("ERROR ", "error")):
            if line.startswith(tag):
                outcomes[key].append(line[len(tag):].strip())
                break
    return outcomes


def classify_skip_reasons(text):
    """Map a skipped test to the dependency that caused it, where stated."""
    reasons = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("SKIPPED "):
            continue
        body = line[len("SKIPPED "):]
        dep = next((d for d in OPTIONAL_DEPENDENCIES if d in body), None)
        reasons[body] = dep or "unspecified"
    return reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pytest-args", nargs="*", default=[])
    a = p.parse_args()

    env = {name: available(name) for name in OPTIONAL_DEPENDENCIES}
    missing = sorted(n for n, ok in env.items() if not ok)

    print_table("ENVIRONMENT", ["package", "available"],
                [[n, "yes" if ok else "no"] for n, ok in sorted(env.items())])

    cmdline, proc = run_pytest(a.pytest_args)
    if proc is None:
        print("pytest is not installed for %s and is not on PATH; cannot "
              "report test status." % sys.executable)
        save_results("test_status", {"environment": env,
                                     "missing_packages": missing,
                                     "error": "pytest not found"})
        return 2
    text = proc.stdout + proc.stderr
    outcomes = parse_report(text)
    skip_reasons = classify_skip_reasons(text)

    # modules that could not be imported at all
    not_collected = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ERROR ") and "::" not in line:
            not_collected.append(line[len("ERROR "):].split(" ")[0])

    summary_line = ""
    for line in reversed(text.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            summary_line = line.strip()
            break

    print("  pytest: %s" % cmdline)
    print()
    print_table("TEST STATUS -- this environment",
                ["bucket", "count", "meaning"], [
        ["verified locally", len(outcomes["passed"]),
         "executed here and passed"],
        ["failed", len(outcomes["failed"]),
         "executed here and failed -- a code result"],
        ["skipped (no dependency)", len(outcomes["skipped"]),
         "could not run; a fact about the environment"],
        ["not collected", len(set(not_collected)),
         "module could not be imported at all"],
    ])

    if outcomes["skipped"]:
        by_dep = {}
        for body, dep in skip_reasons.items():
            by_dep.setdefault(dep, []).append(body)
        print_table("SKIPPED, BY MISSING DEPENDENCY",
                    ["dependency", "tests skipped"],
                    [[dep, len(v)] for dep, v in sorted(by_dep.items())])

    if not_collected:
        print_table("NOT COLLECTED", ["module"],
                    [[m] for m in sorted(set(not_collected))])

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  %d tests verified locally." % len(outcomes["passed"]))
    if outcomes["failed"]:
        print("  %d FAILED -- these are code results and must be fixed."
              % len(outcomes["failed"]))
    if missing:
        print("  %d test(s) skipped and %d module(s) uncollected because "
              "these packages are absent:" % (len(outcomes["skipped"]),
                                              len(set(not_collected))))
        print("      %s" % ", ".join(missing))
        print("  Those code paths are NOT verified. Do not report them as "
              "passing until they run.")
    print()
    print("  CI status is reported separately: a GitHub Actions job that ends")
    print("  with runner_id 0 and no executed steps never received a runner,")
    print("  which is infrastructure unavailability, not a test failure.")
    print("=" * 78)
    print()

    payload = {
        "environment": env,
        "missing_packages": missing,
        "pytest_command": cmdline,
        "pytest_returncode": proc.returncode,
        "pytest_summary_line": summary_line,
        "verified_locally": sorted(outcomes["passed"]),
        "failed": sorted(outcomes["failed"]),
        "skipped_no_dependency": sorted(outcomes["skipped"]),
        "skip_reasons": skip_reasons,
        "not_collected": sorted(set(not_collected)),
        "counts": {
            "verified_locally": len(outcomes["passed"]),
            "failed": len(outcomes["failed"]),
            "skipped_no_dependency": len(outcomes["skipped"]),
            "not_collected": len(set(not_collected)),
        },
        "note": "Buckets are exclusive. 'verified locally' means the test "
                "executed in this environment and passed; it is the only "
                "bucket that may be described as verified.",
    }
    path = save_results("test_status", payload)
    print("raw status written to %s" % path)
    return 1 if outcomes["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
