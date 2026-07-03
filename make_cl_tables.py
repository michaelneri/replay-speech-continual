#!/usr/bin/env python
"""
make_cl_tables.py — generate LaTeX tables (.tex) summarising the continual
learning benchmark (baseline / TSB / EWC / GPM) from the cl_results_*.json
files produced by train_cl.py and training_cl_algorithms.py.

Outputs:
  cl_paper_tables.tex
    Table 1  — global CL metrics (AA, AIA, FM, BWT, IM, FWT) per algorithm,
               mean ± 95% CI half-width.  Algorithms whose results file does
               not exist yet are shown as a "pending" row so the table
               structure is ready before EWC / GPM finish.
    Table 2  — paired comparison vs. the naive-fine-tuning baseline
               (paired t-test AND Wilcoxon signed-rank test on matched
               ordering×run observations), with significance stars.

Usage:
  python make_cl_tables.py [out_dir]

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT

# ---------------------------------------------------------------------------
# Configuration — algorithm display order / labels / result files
# ---------------------------------------------------------------------------
ALGOS = [
    ("baseline", "cl_results.json",      "Naive fine-tuning (baseline)"),
    ("tsb",      "cl_results_tsb.json",  "TSB"),
    ("ewc",      "cl_results_ewc.json",  "EWC"),
    ("gpm",      "cl_results_gpm.json",  "GPM"),
]
BASELINE_KEY = "baseline"

# Direction of improvement for each metric's raw Delta sign:
#   -1  -> negative Delta (vs. baseline) is an improvement (lower-is-better metric)
#   +1  -> positive Delta is an improvement (higher-is-better metric)
IMPROVE_SIGN = {"AA": -1, "AIA": -1, "FM": -1, "BWT": +1, "IM": -1, "FWT": +1}

# Colorblind-safe pair (Okabe-Ito): blue = improvement, orange = degradation.
COLOR_IMPROVE = "CLImprove"
COLOR_WORSEN  = "CLWorsen"

METRICS = ["AA", "AIA", "FM", "BWT", "IM", "FWT"]
METRIC_LATEX = {
    "AA":  r"AA $\downarrow$",
    "AIA": r"AIA $\downarrow$",
    "FM":  r"FM $\downarrow$",
    "BWT": r"BWT $\uparrow$",
    "IM":  r"IM $\downarrow$",
    "FWT": r"FWT $\uparrow$",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def t_halfwidth(mean: float, std: float, n: int, alpha: float = 0.05) -> float:
    if n < 2 or not np.isfinite(std):
        return float("nan")
    return float(st.t.ppf(1 - alpha / 2, df=n - 1) * std / np.sqrt(n))


def fmt_mci(mean: float, hw: float, dec: int = 4) -> str:
    if not np.isfinite(hw):
        return f"{mean:.{dec}f}"
    return f"\\meanci{{{mean:.{dec}f}}}{{{hw:.{dec}f}}}"


def sig_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return r"$^{***}$"
    if p < 0.01:
        return r"$^{**}$"
    if p < 0.05:
        return r"$^{*}$"
    return ""


def load_results() -> dict:
    """Returns {algo_key: json_dict_or_None}."""
    out = {}
    for key, fname, _ in ALGOS:
        path = ROOT / fname
        out[key] = json.loads(path.read_text()) if path.exists() else None
    return out


def paired_values(results: dict, algo_key: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Matched (baseline_values, algo_values) across all (ordering, run) pairs
    for a given metric, in the same order for both arrays.
    """
    base = results[BASELINE_KEY]
    algo = results[algo_key]
    orders = base["per_order_summary"].keys()
    b_vals, a_vals = [], []
    for o in orders:
        bv = base["per_order_summary"][o][metric]["values"] if base["per_order_summary"][o][metric] else None
        av = algo["per_order_summary"].get(o, {}).get(metric, {}).get("values") if algo["per_order_summary"].get(o) else None
        if bv is None or av is None or len(bv) != len(av):
            continue
        b_vals.extend(bv)
        a_vals.extend(av)
    return np.array(b_vals), np.array(a_vals)


# ---------------------------------------------------------------------------
# Table 1 — global metrics per algorithm
# ---------------------------------------------------------------------------
def make_global_table(results: dict) -> str:
    L = []
    L.append(r"% =========================================================================")
    L.append(r"% Table 1: Global CL metrics per algorithm (mean +/- 95% CI half-width)")
    L.append(r"% =========================================================================")
    L.append(r"\begin{table*}[t]")
    L.append(r"\caption{Continual learning benchmark on ReMASC (Device 2, 16\,kHz), "
              r"pooled over 24 environment orderings $\times$ 5 runs ($n=120$ unless "
              r"noted). All metrics are computed on EER (\%); $\downarrow$ = lower is "
              r"better, $\uparrow$ = higher is better. Values are mean $\pm$ 95\% CI "
              r"half-width ($t$-distribution).}")
    L.append(r"\label{tab:cl_global}")
    L.append(r"\centering")
    L.append(r"\begin{adjustbox}{max width=\textwidth}")
    L.append(r"\begin{tabular}{l" + "c" * len(METRICS) + "}")
    L.append(r"\toprule")
    L.append(r"Algorithm & " + " & ".join(METRIC_LATEX[m] for m in METRICS) + r" \\")
    L.append(r"\midrule")

    for key, _, label in ALGOS:
        data = results[key]
        if data is None:
            cells = [label] + ["pending"] * len(METRICS)
            L.append(" & ".join(cells) + r" \\[2pt]")
            continue
        gs = data["global_summary"]
        cells = [label]
        for m in METRICS:
            mean, std, n = gs[m]["mean"], gs[m]["std"], gs[m]["n"]
            hw = t_halfwidth(mean, std, n)
            cells.append(fmt_mci(mean, hw, dec=4))
        L.append(" & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{adjustbox}")
    L.append(r"\end{table*}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Table 2 — paired comparison vs. baseline
# ---------------------------------------------------------------------------
def make_paired_table(results: dict) -> str:
    L = []
    L.append(r"% =========================================================================")
    L.append(r"% Table 2: Paired comparison vs. naive fine-tuning baseline")
    L.append(r"% =========================================================================")
    L.append(r"\begin{table*}[t]")
    L.append(r"\caption{Paired comparison against the naive fine-tuning baseline, "
              r"matched by (ordering, run) so shared variance across the 24 orderings "
              r"$\times$ 5 runs is removed. $\Delta$ is mean(algorithm) $-$ mean(baseline) "
              r"in score space ($n=120$ paired observations unless noted); $p_t$ is the "
              r"paired $t$-test $p$-value, $p_W$ the Wilcoxon signed-rank $p$-value. "
              r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
              rf"\colorbox{{{COLOR_IMPROVE}!20}}{{\strut}}\,/\,"
              rf"\colorbox{{{COLOR_WORSEN}!20}}{{\strut}} shading marks a "
              r"significant ($p_W<0.05$) improvement / degradation vs.\ baseline.}")
    L.append(r"\label{tab:cl_paired}")
    L.append(r"\centering")
    L.append(r"\begin{adjustbox}{max width=\textwidth}")
    L.append(r"\begin{tabular}{llccc}")
    L.append(r"\toprule")
    L.append(r"Algorithm & Metric & $\Delta$ (vs.\ baseline) & $p_t$ & $p_W$ \\")
    L.append(r"\midrule")

    have_baseline = results[BASELINE_KEY] is not None
    for key, _, label in ALGOS:
        if key == BASELINE_KEY:
            continue
        if not have_baseline or results[key] is None:
            L.append(f"{label} & \\multicolumn{{4}}{{c}}{{pending — rerun once baseline and "
                      f"{label} results are both available}} \\\\")
            L.append(r"\addlinespace")
            continue

        for i, m in enumerate(METRICS):
            b_vals, a_vals = paired_values(results, key, m)
            n = len(b_vals)
            if n < 2:
                row = [label if i == 0 else "", METRIC_LATEX[m], "---", "---", "---"]
                L.append(" & ".join(row) + r" \\")
                continue
            diffs = a_vals - b_vals
            mean_diff = float(diffs.mean())
            hw = t_halfwidth(mean_diff, float(diffs.std(ddof=1)), n)
            t_p = float(st.ttest_rel(a_vals, b_vals).pvalue)
            try:
                w_p = float(st.wilcoxon(a_vals, b_vals).pvalue)
            except ValueError:
                w_p = float("nan")

            delta_cell = fmt_mci(mean_diff, hw, dec=4)
            # Shade by significance on the more conservative test (Wilcoxon);
            # color encodes direction (blue=better, orange=worse) so the
            # signal survives for colorblind readers without relying on stars alone.
            if np.isfinite(w_p) and w_p < 0.05:
                is_improvement = (mean_diff * IMPROVE_SIGN[m]) > 0
                color = COLOR_IMPROVE if is_improvement else COLOR_WORSEN
                delta_cell = f"\\cellcolor{{{color}!20}}{delta_cell}"

            row = [
                label if i == 0 else "",
                METRIC_LATEX[m],
                delta_cell,
                f"{t_p:.4f}{sig_marker(t_p)}",
                f"{w_p:.4f}{sig_marker(w_p)}" if np.isfinite(w_p) else "---",
            ]
            L.append(" & ".join(row) + r" \\")
        L.append(r"\addlinespace")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{adjustbox}")
    L.append(r"\end{table*}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    results = load_results()

    found = [label for key, _, label in ALGOS if results[key] is not None]
    pending = [label for key, _, label in ALGOS if results[key] is None]
    print(f"Found results for: {found}")
    if pending:
        print(f"Pending (will show as placeholder rows): {pending}")

    blocks = [
        r"% Auto-generated by make_cl_tables.py — DO NOT EDIT BY HAND.",
        r"% Re-run the script once EWC/GPM finish to refresh.",
        r"%",
        r"% Add this macro to your preamble (or paste it here once):",
        r"% \newcommand{\meanci}[2]{#1\,{\scriptsize$\pm$#2}}",
        r"%",
        r"% Colorblind-safe pair (Okabe-Ito) for significance shading — add once:",
        rf"% \definecolor{{{COLOR_IMPROVE}}}{{HTML}}{{0072B2}}  % blue  = improvement",
        rf"% \definecolor{{{COLOR_WORSEN}}}{{HTML}}{{E69F00}}  % orange = degradation",
        r"%",
        r"% Required packages: booktabs, adjustbox, amsmath, xcolor (load with [table] option)",
        r"% \usepackage[table]{xcolor}",
        r"%",
        make_global_table(results),
        "",
        make_paired_table(results),
        "",
    ]

    out_path = OUT_DIR / "cl_paper_tables.tex"
    out_path.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
