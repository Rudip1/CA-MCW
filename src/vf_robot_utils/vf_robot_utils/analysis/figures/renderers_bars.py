"""renderers_bars.py — per-tier bar-panel renderer.

Renderer library for the cross-controller comparison figures
(``comparison_figures.py``). Not a CLI entry point.

``render_tier_panel`` draws one compact multi-panel bar figure for a
single evaluation tier: every metric of the tier in a 3-column grid
(so the figure is at most two rows tall — report-friendly, never a
full page). Bar + Wilson 95 % CI for rate metrics, bar + bootstrap
95 % CI for continuous metrics, with the Classical / Fixed-wt /
Trained / Imitation category bands and red dashed separators.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vf_robot_utils.analysis.style import (
    apply_style, controller_color, order_by_category, category_separators,
    category_spans, SEPARATOR_COLOR,
)
from vf_robot_utils.analysis.statistical_tests import wilson_ci, bootstrap_ci


def _short(c: str) -> str:
    if c.startswith("oraclewt_"): return "O-" + c.removeprefix("oraclewt_")
    if c.startswith("rawwt_"):    return "R-" + c.removeprefix("rawwt_")
    if c.startswith("imitationwt_"): return "I-" + c.removeprefix("imitationwt_")
    return c


def _cell(df: pd.DataFrame, ctrl: str, col: str, is_prop: bool,
          drop_inf: bool) -> tuple[float, float, float, int]:
    """Return (mean, lo_err, hi_err, n).

    Wilson 95% CI for proportions; bootstrap 95% CI (B=10000, seed=0)
    for continuous metrics — the per-controller n is small (one episode
    per goal), so a Gaussian SEM would understate the asymmetry of
    heavy-tailed metrics like jerk and clearance.
    """
    arr = pd.to_numeric(
        df.loc[df["controller"] == ctrl, col], errors="coerce"
    ).to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr) if drop_inf else ~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), 0.0, 0.0, 0
    mean = float(arr.mean())
    if is_prop:
        k = int(round(float(arr.sum())))
        lo, hi = wilson_ci(k, n, alpha=0.05)
        return mean, max(0.0, mean - float(lo)), max(0.0, float(hi) - mean), n
    lo, hi = bootstrap_ci(arr, B=10_000, seed=0)
    return mean, max(0.0, mean - lo), max(0.0, hi - mean), n


def render_tier_panel(
    df: pd.DataFrame, controllers: list[str],
    metrics: list[tuple[str, str, bool, bool, int]],
    out_path: Path, tier_header: str,
    label_map: dict[str, str] | None = None,
    legend_label_map: dict[str, str] | None = None,
    highlight: str | None = None,
    use_legend: bool = False,
    scope: str = "full",
) -> list[dict]:
    """One compact multi-panel bar figure for a single evaluation tier.

    All metrics of the tier are shown at once in a 3-column grid (so the
    figure is at most two rows tall — report-friendly, never a full page).
    Bar + Wilson 95 % CI for rate metrics, bar + bootstrap 95 % CI for
    continuous metrics. ``metrics`` items are
    ``(col, label, is_prop, drop_inf, direction)``.

    ``tier_header`` is drawn large + bold at the top (e.g. "Tier 3:
    Safety"). The thesis' own controller (``highlight``) gets a thick
    bar edge throughout, and a bold entry in the legend / x-axis.

    ``use_legend`` picks the controller-naming style:
      - ``True``  — no x-tick labels; a single shared legend at the
        bottom carries the (possibly long) controller names. Best for
        the small headline set, keeps panels short.
      - ``False`` — short x-tick labels under every panel + the
        Classical / Fixed-wt / Trained / Imitation group labels. Best
        for the full controller set.

    ``label_map`` sets the (short) x-axis tick labels; ``legend_label_map``
    sets the legend names (typically the longer descriptive form). If
    ``legend_label_map`` is omitted the legend falls back to ``label_map``.
    Returns per-(controller, metric) summary rows for the CSV.
    """
    apply_style()
    if "timestamp" in df.columns:
        df = (df.sort_values("timestamp")
                .drop_duplicates(subset=["controller", "goal_folder"],
                                 keep="last")
                .reset_index(drop=True))
    controllers = order_by_category(controllers)
    label_map = label_map or {}
    legend_label_map = legend_label_map or label_map
    n = len(metrics)
    many = len(controllers) > 12

    # 3-column grid -> at most 2 rows -> compact, report-friendly height.
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    if many:
        panel_w, panel_h = 4.4, 3.0
        tick_fs, title_fs, label_fs, xlab_fs = 7.0, 11.0, 9.5, 6.5
        val_fs = 6.0
    else:
        # Headline 6-controller layout. LaTeX now includes these at the
        # full \textwidth (was 0.88×), so the matplotlib panels are
        # widened to fill the extra margin and fonts are bumped a step.
        panel_w, panel_h = 5.0, 4.3
        tick_fs, title_fs, label_fs, xlab_fs = 13.0, 15.0, 12.5, 14.0
        val_fs = 12.0

    header_in = 0.55
    legend_in = 1.35 if use_legend else 0.0
    fig_w = panel_w * ncols
    fig_h = panel_h * nrows + header_in + legend_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False)
    flat = list(axes.flat)
    xs = np.arange(len(controllers), dtype=float)
    colors = [controller_color(c, scope=scope, highlight=highlight)
              for c in controllers]
    edge_w = [1.8 if c == highlight else 0.4 for c in controllers]
    rows: list[dict] = []

    for idx, (col, mlabel, is_prop, drop_inf, direction) in enumerate(metrics):
        ax = flat[idx]
        means = np.full(len(controllers), np.nan)
        lo_err = np.zeros(len(controllers))
        hi_err = np.zeros(len(controllers))
        for i, ctrl in enumerate(controllers):
            m, lo, hi, ncell = _cell(df, ctrl, col, is_prop, drop_inf)
            means[i] = m; lo_err[i] = lo; hi_err[i] = hi
            rows.append(dict(Metric=col, Label=mlabel, Controller=ctrl,
                             n=ncell, mean=m,
                             ci_lo=(m - lo), ci_hi=(m + hi)))
        ax.bar(xs, means, color=colors, edgecolor="black",
               linewidth=edge_w, width=0.82)
        ax.errorbar(xs, means, yerr=[lo_err, hi_err], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=2)
        # Value labels on top of each bar (above the upper error whisker)
        # to make per-bar magnitudes readable without consulting the table.
        # Format adapts to magnitude: rates as 0.00, larger continuous as
        # short numbers; NaN bars get no label.
        for xi, mi, hi in zip(xs, means, hi_err):
            if not np.isfinite(mi):
                continue
            if is_prop or abs(mi) < 10.0:
                txt = f"{mi:.2f}"
            elif abs(mi) < 100.0:
                txt = f"{mi:.1f}"
            else:
                txt = f"{mi:.0f}"
            y_top = mi + (hi if np.isfinite(hi) else 0.0)
            ax.text(xi, y_top, txt, ha="center", va="bottom",
                    fontsize=val_fs, color="#222222")
        # Slanted small x-tick labels — shown on every panel.
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [label_map.get(c, _short(c)) for c in controllers],
            rotation=70, ha="right", fontsize=xlab_fs)
        for tl, c in zip(ax.get_xticklabels(), controllers):
            if c == highlight:
                tl.set_fontweight("bold")
                tl.set_fontsize(xlab_fs + 1.5)
                tl.set_color("#000000")
        arrow = ("  ↑ better" if direction > 0
                 else "  ↓ better" if direction < 0 else "")
        ax.set_title(mlabel + arrow, fontsize=title_fs)
        if is_prop:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("rate (Wilson 95% CI)", fontsize=label_fs)
        else:
            ax.set_ylabel("mean ± 95% CI", fontsize=label_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.grid(axis="y", alpha=0.25)
        # Thin black box around the metric panel (all four spines).
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)
            sp.set_color("black")
        # Headroom for the value labels on top of each bar plus the
        # single-level category-label band. A bit more than before
        # because the value labels sit above the upper error whisker.
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.32 * (y1 - y0))
        # Red dashed separators between the controller-category groups
        # (classical | fixedwt | trained | imitation). The category
        # labels above each band were removed (unreadable at print
        # scale); the legend below the figure names every controller.
        for sx in category_separators(controllers):
            ax.axvline(sx, color=SEPARATOR_COLOR, linestyle=(0, (4, 3)),
                       linewidth=1.0, zorder=1)

    for j in range(n, nrows * ncols):
        flat[j].set_visible(False)

    fig.suptitle(tier_header, fontsize=15, fontweight="bold")

    if use_legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=controller_color(c, scope=scope, highlight=highlight),
                         edgecolor="black",
                         linewidth=(1.8 if c == highlight else 0.4),
                         label=legend_label_map.get(c, _short(c)))
                   for c in controllers]
        leg = fig.legend(handles=handles, loc="lower center",
                         ncol=min(3, len(controllers)), frameon=False,
                         fontsize=14.0, bbox_to_anchor=(0.5, 0.0),
                         handlelength=2.2, handleheight=1.5,
                         columnspacing=2.4, labelspacing=0.9)
        for txt, c in zip(leg.get_texts(), controllers):
            if c == highlight:
                txt.set_fontweight("bold")

    rect = (0.0, legend_in / fig_h, 1.0, 1.0 - header_in / fig_h)
    # h_pad / w_pad: extra gap between panel rows and columns (figsize
    # stays fixed, so the panels shrink slightly — overall size unchanged).
    fig.tight_layout(rect=rect, h_pad=2.6, w_pad=2.6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return rows
