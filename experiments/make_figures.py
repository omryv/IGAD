"""
experiments/make_figures.py

Figures for the Part 1 results. Reads only from `experiments/results/*.json`
and writes SVG to `docs/figures/`. No figure is drawn from a number that is
not already committed as raw output, and nothing here recomputes a result.

SVG rather than matplotlib because matplotlib is not installable in this
environment, and because a text format diffs.

    python -m experiments.make_figures

Figures produced (from `sherman_morrison_benchmark.json` and
`highprec_reliability.json`):

    scaling_wallclock.svg     wall clock vs k, three routes, fitted slopes
    scaling_memory.svg        peak allocation vs k, three routes
    error_vs_cancellation.svg float64 error vs the cancellation ratio
    error_vs_condition.svg    the same error vs cond(g) -- the null result
    error_decomposition.svg   input rounding / special functions / arithmetic

The figures the brief asks for in section 19 -- quality-vs-router-score
scatter, AUC vs generation progress, earliest-warning comparison, layer x
timestep heatmap, eigenvalue spectra for good vs failed outputs -- are NOT
produced, because the hard gate in section 0 failed and no 3D quality data
exists. See docs/acquisition_checklist.md.
"""

import argparse
import json
import math
import os

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "docs", "figures")

# Validated 3-slot categorical palette (light mode, all-pairs safe):
# worst CVD dE 9.2, worst normal-vision dE 24.0. Aqua sits below 3:1 against
# the surface, so every series carries a legend chip AND a direct label, and
# the underlying table is committed as JSON -- the relief rule.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Single-quoted in the emitted attribute, so the quoted family name is legal XML.
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _text(x, y, s, size=11.5, fill=INK_2, anchor="start", weight="400",
          tabular=False, raw=None):
    extra = ' font-variant-numeric="tabular-nums"' if tabular else ""
    return ("<text x='%.1f' y='%.1f' font-family='%s' font-size='%s' fill='%s' "
            "text-anchor='%s' font-weight='%s'%s>%s</text>"
            % (x, y, FONT, size, fill, anchor, weight, extra,
               raw if raw is not None else _esc(s)))


class Chart:
    """Log-log canvas. The right margin is reserved for the legend; direct
    labels are placed inside the plot, above each series' final point."""

    def __init__(self, width=800, height=440, pad=(66, 196, 58, 76),
                 title="", subtitle="", xlabel="", ylabel=""):
        self.w, self.h = width, height
        self.pt, self.pr, self.pb, self.pl = pad
        self.parts = []
        self.title, self.subtitle = title, subtitle
        self.xlabel, self.ylabel = xlabel, ylabel
        self.x0 = self.x1 = self.y0 = self.y1 = 0.0

    # -- domain --------------------------------------------------------------
    def set_log_domain(self, xs, ys, pad_decades=0.15):
        lx = [math.log10(v) for v in xs if v > 0]
        ly = [math.log10(v) for v in ys if v > 0]
        self.x0, self.x1 = min(lx) - pad_decades, max(lx) + pad_decades
        self.y0, self.y1 = min(ly) - pad_decades, max(ly) + pad_decades
        if self.x1 - self.x0 < 0.5:
            self.x0, self.x1 = self.x0 - 0.25, self.x1 + 0.25
        if self.y1 - self.y0 < 0.5:
            self.y0, self.y1 = self.y0 - 0.25, self.y1 + 0.25

    def px(self, v):
        t = (math.log10(v) - self.x0) / (self.x1 - self.x0)
        return self.pl + t * (self.w - self.pl - self.pr)

    def py(self, v):
        t = (math.log10(v) - self.y0) / (self.y1 - self.y0)
        return self.h - self.pb - t * (self.h - self.pt - self.pb)

    # -- primitives ----------------------------------------------------------
    def text(self, *a, **kw):
        self.parts.append(_text(*a, **kw))

    def line(self, x1, y1, x2, y2, stroke, width=2.0, dash=None, opacity=1.0):
        d = " stroke-dasharray='%s'" % dash if dash else ""
        self.parts.append(
            "<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='%s' "
            "stroke-width='%.1f' stroke-linecap='round' opacity='%.2f'%s/>"
            % (x1, y1, x2, y2, stroke, width, opacity, d))

    def polyline(self, pts, stroke, width=2.0):
        self.parts.append(
            "<polyline points='%s' fill='none' stroke='%s' stroke-width='%.1f' "
            "stroke-linejoin='round' stroke-linecap='round'/>"
            % (" ".join("%.1f,%.1f" % p for p in pts), stroke, width))

    def dot(self, x, y, fill, r=4.5):
        # 2px surface ring keeps overlapping marks separable
        self.parts.append(
            "<circle cx='%.1f' cy='%.1f' r='%.1f' fill='%s' stroke='%s' "
            "stroke-width='2'/>" % (x, y, r, fill, SURFACE))

    # -- axes ----------------------------------------------------------------
    def log_axes(self, xticks, xfmt, ylabelfmt=None):
        left, right = self.pl, self.w - self.pr
        top, bottom = self.pt, self.h - self.pb
        ylabelfmt = ylabelfmt or _pow10

        for e in range(math.floor(self.y0), math.ceil(self.y1) + 1):
            if not self.y0 <= e <= self.y1:
                continue
            y = self.py(10.0 ** e)
            self.line(left, y, right, y, GRID, 1.0)
            self.parts.append(_text(left - 10, y + 4, None, 11, MUTED, "end",
                                    tabular=True, raw=ylabelfmt(e)))

        # every tick gets a gridline; only the labelled subset gets text, so
        # 8 .. 131072 does not collide at 11px
        for v, label in xticks:
            if not self.x0 <= math.log10(v) <= self.x1:
                continue
            x = self.px(v)
            self.line(x, top, x, bottom, GRID, 1.0)
            if label:
                self.text(x, bottom + 20, xfmt(v), 11, MUTED, "middle",
                          tabular=True)

        self.line(left, bottom, right, bottom, AXIS, 1.5)
        self.line(left, top, left, bottom, AXIS, 1.5)

    def direct_label(self, x, y, s, color):
        """Above the final point, extending left over empty plot area."""
        yy = max(y - 13, self.pt + 11)
        self.parts.append("<circle cx='%.1f' cy='%.1f' r='3.5' fill='%s'/>"
                          % (x - 5, yy - 4, color))
        self.text(x - 12, yy, s, 11, INK_2, "end", "500")

    def legend(self, entries):
        """entries: (label, [sub-lines], color). Sub-lines are short enough to
        stay inside the reserved right margin."""
        x, y = self.w - self.pr + 14, self.pt + 8
        for label, subs, color in entries:
            self.parts.append("<rect x='%.1f' y='%.1f' width='11' height='11' "
                              "rx='3' fill='%s'/>" % (x, y - 9, color))
            self.text(x + 17, y, label, 11.5, INK_2, "start", "500")
            y += 15
            for sub in subs:
                self.text(x + 17, y, sub, 10.5, MUTED)
                y += 13
            y += 9

    def note(self, lines, y_offset=0):
        x = self.w - self.pr + 14
        y = self.h - self.pb - len(lines) * 14 - y_offset
        for i, s in enumerate(lines):
            self.text(x, y + i * 14, s, 10.5, MUTED)

    def render(self):
        head = ("<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' "
                "viewBox='0 0 %d %d' role='img'>" % (self.w, self.h, self.w, self.h))
        bg = "<rect width='%d' height='%d' fill='%s'/>" % (self.w, self.h, SURFACE)
        chrome = [_text(24, 26, self.title, 15, INK, "start", "600")]
        if self.subtitle:
            chrome.append(_text(24, 45, self.subtitle, 11.5, MUTED))
        if self.xlabel:
            chrome.append(_text((self.pl + self.w - self.pr) / 2, self.h - 12,
                                self.xlabel, 11.5, INK_2, "middle", "500"))
        if self.ylabel:
            chrome.append(
                "<text transform='translate(17,%.1f) rotate(-90)' "
                "font-family='%s' font-size='11.5' fill='%s' "
                "text-anchor='middle' font-weight='500'>%s</text>"
                % ((self.pt + self.h - self.pb) / 2, FONT, INK_2,
                   _esc(self.ylabel)))
        return head + bg + "".join(chrome) + "".join(self.parts) + "</svg>"

    def save(self, name):
        os.makedirs(FIGURE_DIR, exist_ok=True)
        path = os.path.join(FIGURE_DIR, name)
        with open(path, "w") as fh:
            fh.write(self.render())
        return path


def _pow10(e):
    """Superscript exponent via tspan; returns raw SVG."""
    if e == 0:
        return "1"
    return "10<tspan dy='-4' font-size='8.5'>%s</tspan>" % (
        str(e).replace("-", "−"))


def _si(v):
    v = int(round(v))
    return "%dk" % (v // 1024) if v >= 1024 and v % 1024 == 0 else str(v)


def _pow2_ticks(xs, label_every=2):
    """(value, label?) pairs: gridline everywhere, text on a subset."""
    vals = sorted(set(xs))
    return [(v, (i % label_every == 0 or i == len(vals) - 1))
            for i, v in enumerate(vals)]


def _decade_ticks(x0, x1, max_labels=10):
    """Gridline on every decade; text on a stride that keeps labels apart."""
    exps = list(range(math.floor(x0), math.ceil(x1) + 1))
    stride = max(1, -(-len(exps) // max_labels))
    return [(10.0 ** e, i % stride == 0) for i, e in enumerate(exps)]


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _load(name):
    with open(os.path.join(RESULT_DIR, name + ".json")) as fh:
        return json.load(fh)


ROUTE_LABELS = [("dense-inverse", "O(k^3)"), ("sm-matrix", "O(k^2)"),
                ("sm-closed", "O(k)")]


def fig_scaling(bench, key, name, title, subtitle, ylabel, scale=1.0):
    series = []
    for (route, theo), color in zip(ROUTE_LABELS, SERIES):
        pts = [(r["k"], r[route + key] * scale) for r in bench
               if r.get(route + key)]
        if pts:
            series.append((route, theo, color, pts))
    if not series:
        return None

    xs = [p[0] for _, _, _, pts in series for p in pts]
    ys = [p[1] for _, _, _, pts in series for p in pts]
    ch = Chart(title=title, subtitle=subtitle,
               xlabel="k   (routed experts)", ylabel=ylabel)
    ch.set_log_domain(xs, ys)
    ch.log_axes(_pow2_ticks(xs), _si)

    legend = []
    for route, theo, color, pts in series:
        ch.polyline([(ch.px(a), ch.py(b)) for a, b in pts], color, 2.0)
        for a, b in pts:
            ch.dot(ch.px(a), ch.py(b), color)
        ax, ay = pts[-1]
        ch.direct_label(ch.px(ax), ch.py(ay), route, color)
        slope = _slope(pts)
        subs = ["predicted %s" % theo]
        if slope is not None:
            subs.append("measured slope %.2f" % slope)
        legend.append((route, subs, color))
    ch.legend(legend)
    return ch.save(name)


def _slope(pts, fit_from=64):
    tail = [(a, b) for a, b in pts if a >= fit_from and b > 0]
    if len(tail) < 2:
        tail = [(a, b) for a, b in pts if b > 0]
    if len(tail) < 2:
        return None
    lx = [math.log2(a) for a, _ in tail]
    ly = [math.log2(b) for _, b in tail]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    den = sum((v - mx) ** 2 for v in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den else None


def fig_error_vs(records, predictor, name, title, subtitle, xlabel,
                 route="sm-closed", floor=1e-17, show_bound=False):
    xs = [max(r[predictor], 1.0) for r in records]
    ys = [max(r["total_error"][route], floor) for r in records]
    ch = Chart(title=title, subtitle=subtitle, xlabel=xlabel,
               ylabel="relative error of R vs the 120-digit reference")
    ch.set_log_domain(xs, ys + [floor])
    ch.log_axes(_decade_ticks(ch.x0, ch.x1), lambda v: _si_pow(v))

    if show_bound:
        eps = 2.0 ** -53
        lo, hi = 10.0 ** ch.x0, 10.0 ** ch.x1
        y_lo, y_hi = max(eps * lo, 10.0 ** ch.y0), min(eps * hi, 10.0 ** ch.y1)
        ch.line(ch.px(lo), ch.py(y_lo), ch.px(hi), ch.py(y_hi),
                INK_2, 1.5, dash="5 4", opacity=0.6)
        ch.text(ch.px(hi) - 6, ch.py(y_hi) - 10,
                "predicted floor   eps x rho", 11, INK_2, "end", "500")

    for x, y in zip(xs, ys):
        ch.dot(ch.px(x), ch.py(y), SERIES[0], 4.5)

    rho_r, n = _spearman(xs, ys)
    ch.text(ch.w - ch.pr + 14, ch.pt + 8, "Spearman", 11, MUTED)
    ch.text(ch.w - ch.pr + 14, ch.pt + 30, "%+.3f" % rho_r, 19, INK, "start", "600")
    ch.note(["%d Dirichlet" % n, "parameter points", "route: %s" % route,
             "errors below 1e-17", "are plotted at the floor"])
    return ch.save(name)


def _si_pow(v):
    e = int(round(math.log10(v)))
    return "1" if e == 0 else "1e%d" % e


def _spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return (num / (dx * dy) if dx and dy else float("nan")), n


def fig_decomposition(records, name, route="sm-closed", floor=1e-17):
    recs = sorted(records, key=lambda r: r["cancellation_rho"])
    xs = list(range(1, len(recs) + 1))
    comps = [
        ("input rounding floor",
         [max(r["input_rounding_floor"], floor) for r in recs],
         "exact result from doubles"),
        ("special functions",
         [max(r["special_function_error"], floor) for r in recs],
         "psi' and psi'' at ~2 ulp"),
        ("arithmetic (%s)" % route,
         [max(r["arithmetic_error"][route], floor) for r in recs],
         "the only term tracking rho"),
    ]
    ch = Chart(title="Where the float64 error in R(alpha) comes from",
               subtitle="%d Dirichlet parameter points, ordered by cancellation "
                        "ratio rho" % len(recs),
               xlabel="parameter point, ordered by rho   (left = least cancellation)",
               ylabel="relative contribution to the error in R")
    ch.set_log_domain(xs, [v for _, vals, _ in comps for v in vals])
    ch.log_axes([(v, True) for v in (1, 3, 10, 30, len(recs))],
                lambda v: str(int(round(v))))

    for (label, vals, _), color in zip(comps, SERIES):
        ch.polyline([(ch.px(a), ch.py(b)) for a, b in zip(xs, vals)], color, 2.0)
        ch.direct_label(ch.px(xs[-1]), ch.py(vals[-1]), label.split(" (")[0], color)
    ch.legend([(label, [sub], c) for (label, _, sub), c in zip(comps, SERIES)])
    return ch.save(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", default="sm-closed")
    a = p.parse_args()
    written = []

    try:
        bench = _load("sherman_morrison_benchmark")["B2_B3_measurements"]
    except (OSError, KeyError):
        bench = None
        print("skipped scaling figures: run "
              "`python -m experiments.benchmark_sherman_morrison` first")
    if bench:
        written.append(fig_scaling(
            bench, "_seconds", "scaling_wallclock.svg",
            "Complete Dirichlet curvature path: measured wall clock",
            "standard-library Python, minimum over repeats; slope fitted on k >= 64",
            "seconds per evaluation"))
        written.append(fig_scaling(
            bench, "_peak_bytes", "scaling_memory.svg",
            "Peak allocation per evaluation",
            "tracemalloc; sm-closed never allocates anything k x k",
            "peak KiB allocated", scale=1.0 / 1024.0))

    try:
        rel = _load("highprec_reliability")["C2_error_decomposition"]
    except (OSError, KeyError):
        rel = None
        print("skipped reliability figures: run "
              "`python -m experiments.highprec_reliability` first")
    if rel:
        written.append(fig_error_vs(
            rel, "cancellation_rho", "error_vs_cancellation.svg",
            "float64 error in R(alpha) against cancellation",
            "rho = largest intermediate magnitude / |S^2 - T^2|",
            "cancellation ratio rho", a.route, show_bound=True))
        written.append(fig_error_vs(
            rel, "cond_g", "error_vs_condition.svg",
            "The same error against cond(g) -- the null result",
            "conditioning of the Fisher metric does not order the error",
            "cond(g)", a.route))
        written.append(fig_decomposition(rel, "error_decomposition.svg", a.route))

    for path in written:
        if path:
            print("wrote %s" % path)
    if not written:
        return 1
    print("\nNo section-19 figure was produced: the Part 0 gate failed, so no "
          "3D quality data exists. See docs/acquisition_checklist.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
