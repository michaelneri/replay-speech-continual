#!/usr/bin/env python
"""
plot_ordering_analysis.py — effect of last environment on AA across algorithms.

Since the first environment has negligible influence (verified separately),
we focus on the last environment in the CL sequence, which dominates
performance. Shows mean AA (± 95% CI) grouped by last environment,
with one bar per algorithm, so ordering sensitivity can be compared
directly across methods.

Output: ordering_analysis.pdf  (and .png for quick preview)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

ROOT = Path(__file__).parent

ALGOS = [
    ("baseline", "cl_results.json",     "Baseline"),
    ("tsb",      "cl_results_tsb.json", "TSB"),
    ("ewc",      "cl_results_ewc.json", "EWC"),
    ("gpm",      "cl_results_gpm.json", "GPM"),
]
ENVS = [1, 2, 3, 4]
ENV_LABELS = [
    "Outdoor\n(traffic & wind)",
    "Indoor quiet\n(18 spk-mic configs)",
    "Indoor lounge\n(music & TV)",
    "Vehicle\n(moving car)",
]

# Okabe-Ito palette — one color per algorithm
ALGO_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def collect_last_aa(data: dict) -> dict[int, list[float]]:
    """Returns {env: [aa_values]} for all orderings where env is last."""
    last_aa: dict[int, list] = {e: [] for e in ENVS}
    for order_str, summary in data["per_order_summary"].items():
        last_env = int(order_str.split("_")[-1])
        last_aa[last_env].extend(summary["AA"]["values"])
    return last_aa


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    n = len(arr)
    mean = float(arr.mean())
    hw = float(st.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n))
    return mean, hw


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot() -> None:
    available = [(key, label, color)
                 for (key, fname, label), color in zip(ALGOS, ALGO_COLORS)
                 if (ROOT / fname).exists()]
    missing = [label for (key, fname, label), _ in zip(ALGOS, ALGO_COLORS)
               if not (ROOT / fname).exists()]
    if missing:
        print(f"Missing (omitted): {missing}")

    n_algos = len(available)
    total_width = 0.7
    width = total_width / n_algos
    x = np.arange(len(ENVS))

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for a_idx, (key, label, color) in enumerate(available):
        data = json.loads((ROOT / (
            next(fname for k, fname, _ in ALGOS if k == key)
        )).read_text())
        last_aa = collect_last_aa(data)

        offsets = (np.arange(n_algos) - (n_algos - 1) / 2) * width
        means, hws = [], []
        for env in ENVS:
            m, hw = mean_ci(last_aa[env])
            means.append(m)
            hws.append(hw)

        ax.bar(x + offsets[a_idx], means, width * 0.9,
               color=color, alpha=0.85,
               yerr=hws, capsize=3, error_kw=dict(linewidth=1.0),
               label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(ENV_LABELS, fontsize=8)
    ax.set_ylabel("Mean AA — EER (lower is better)", fontsize=9)
    ax.set_title("Effect of last environment on CL accuracy\n(all algorithms, 95% CI)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout(pad=1.2)
    for ext in ("pdf", "png"):
        out = ROOT / f"ordering_analysis.{ext}"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
