# Security Policy

## Project Status

IGAD is a static research library. There is no hosted service, no backend,
no authentication, and no user data collection. The attack surface is limited
to the Python package itself and its dependencies.

Only the latest commit on `main` is supported. There are no versioned releases.

## Scope

**In scope:**
- Arbitrary code execution via malicious input to `IGADDetector`, `curvature.py`,
  or `families.py`
- Dependency vulnerabilities in `numpy`, `scipy`, or any package in `pyproject.toml`
- Supply chain issues (compromised dependency, typosquatting)

**Out of scope:**
- Numerical instability that does not cross into exploitability
- Performance or accuracy issues
- Findings from automated scanners with no proof of exploitability

## Reporting a Vulnerability

**Do not open a public GitHub issue.**

Report via GitHub private advisory:
1. Go to the **Security** tab of this repository
2. Click **Report a vulnerability**
3. Include: description, steps to reproduce, and potential impact

You will receive an acknowledgement within 72 hours. If accepted, a fix will
be committed to `main` and you will be credited unless you prefer otherwise.
If declined, you will receive a clear explanation why.

## Disclosure Policy

Coordinated disclosure. Please allow 90 days from your report before publishing.
You will be notified when the fix is live.
