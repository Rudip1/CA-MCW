"""renderers_xte.py — cross-track-error (XTE) renderer.

Renderer library for the cross-controller comparison figures
(``comparison_figures.py``). Not a CLI entry point.

Reference = each controller's own time-active ``/plan``. For every
robot pose at ``sim_time[k]``, the metric uses the most-recent
``/plan`` that was published at-or-before that timestamp (sim_time
bisect over ``global_path_plans/plan_times`` in the HDF5). This
measures execution fidelity of the controller against the plan it was
actually being given at each instant — the right metric when Nav2's
BT may issue replans / recovery behaviours mid-episode. Falls back to
straight-line start->goal for legacy HDF5s that predate the
``global_path_plans/`` schema (2026-05-16).

Renderers:
  render_violin        per-step |XTE| violin, pooled over goals.
  render_profile       |XTE| vs normalised progress, one line per
                       controller, per goal.
  render_envelope      mean +/-1sigma / 2sigma envelope across goals
                       for a single controller.
  render_xte_overview  three-panel envelope | profile | violin.
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vf_robot_utils.analysis.style import (
    apply_style, controller_color, order_by_category, category_separators,
    category_spans, SEPARATOR_COLOR,
)


def _short(c: str) -> str:
    if c.startswith("oraclewt_"):    return "O-" + c.removeprefix("oraclewt_")
    if c.startswith("rawwt_"):       return "R-" + c.removeprefix("rawwt_")
    if c.startswith("imitationwt_"): return "I-" + c.removeprefix("imitationwt_")
    return c


def _xte_arrays(h5_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Legacy fallback: straight-line start→goal XTE for one episode.

    Used only when ``_xte_arrays_vs_active_plan`` returns empty (HDF5
    predates the ``global_path_plans/`` group). Matches the
    straight-line fallback branch in ``metrics_from_h5._tier6``.
    """
    with h5py.File(h5_path, "r") as f:
        if "robot_pose" not in f or f["robot_pose"].shape[0] < 2:
            return np.array([]), np.array([])
        rp = f["robot_pose"][:].astype(np.float64)
        goal = f["goal"][-1] if "goal" in f else None
    if goal is None: return np.array([]), np.array([])
    sx, sy = float(rp[0, 0]), float(rp[0, 1])
    gx, gy = float(goal[0]), float(goal[1])
    dx, dy = gx - sx, gy - sy
    L = math.hypot(dx, dy)
    if L < 1e-6: return np.array([]), np.array([])
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    px = rp[:, 0] - sx; py = rp[:, 1] - sy
    signed = px * nx + py * ny
    # x-axis: robot's own executed arc-length, normalised — same rationale
    # as _xte_arrays_vs_active_plan (projecting onto a line collapses it).
    step = np.diff(rp[:, :2], axis=0)
    cum = np.concatenate(([0.0], np.cumsum(np.sqrt((step ** 2).sum(axis=1)))))
    along = cum / cum[-1] if cum[-1] > 1e-6 else cum
    return signed, along


def _xte_arrays_vs_active_plan(
    h5_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XTE against each step's time-active /plan reference.

    Reads ``global_path_plans/{plan_times, plan_NNNNN}`` and, for each
    robot pose at ``sim_time[k]``, finds the most-recent /plan that was
    published at-or-before that timestamp (sim_time bisect). XTE is the
    nearest-point distance to that plan's polyline.

    ``along_frac`` is the robot's OWN executed arc-length, normalised to
    [0, 1]. Projecting onto a single /plan is unreliable: Nav2 replans
    many times per episode and the last plan in the history is usually a
    short near-goal stub, so projecting every pose onto it collapses
    almost the whole trajectory onto x~0. The executed-path arc-length
    is monotonic and covers the whole episode by construction.

    Returns ``(signed_xte, abs_xte, along_frac)``.  Empty arrays if the
    HDF5 lacks plan history or robot_pose. Sign convention: positive =
    robot is left of the segment's direction of travel.
    """
    with h5py.File(h5_path, "r") as f:
        if "robot_pose" not in f or f["robot_pose"].shape[0] < 2:
            return np.array([]), np.array([]), np.array([])
        rp = f["robot_pose"][:, :2].astype(np.float64)
        if "sim_time" not in f:
            return np.array([]), np.array([]), np.array([])
        rt = f["sim_time"][:].astype(np.float64)
        if "global_path_plans" not in f:
            return np.array([]), np.array([]), np.array([])
        g = f["global_path_plans"]
        if "plan_times" not in g:
            return np.array([]), np.array([]), np.array([])
        plan_times = g["plan_times"][:].astype(np.float64)
        plan_polys: list[np.ndarray] = []
        for i in range(plan_times.size):
            key = f"plan_{i:05d}"
            if key in g:
                plan_polys.append(g[key][:, :2].astype(np.float64))
            else:
                plan_polys.append(np.empty((0, 2)))

    if rp.shape[0] != rt.shape[0] or not plan_polys:
        return np.array([]), np.array([]), np.array([])

    # Which /plan was active at each robot step? Largest i s.t. plan_times[i] <= rt[k].
    idx = np.searchsorted(plan_times, rt, side="right") - 1
    idx = np.clip(idx, 0, len(plan_polys) - 1)

    signed = np.full(rp.shape[0], np.nan, dtype=np.float64)
    abs_d  = np.full(rp.shape[0], np.nan, dtype=np.float64)
    for i, poly in enumerate(plan_polys):
        if poly.shape[0] < 2:
            continue
        mask = (idx == i)
        if not mask.any():
            continue
        a = poly[:-1]; b = poly[1:]
        seg = b - a
        seg_len_sq = (seg ** 2).sum(axis=1)
        seg_len_sq = np.where(seg_len_sq < 1e-12, 1e-12, seg_len_sq)
        pts = rp[mask]
        rel = pts[:, None, :] - a[None, :, :]
        t = np.clip((rel * seg[None, :, :]).sum(axis=2) / seg_len_sq[None, :], 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * seg[None, :, :]
        diffs = pts[:, None, :] - closest
        dist_sq = (diffs ** 2).sum(axis=2)
        abs_d[mask] = np.sqrt(dist_sq.min(axis=1))
        nearest = np.argmin(dist_sq, axis=1)
        rows = np.arange(pts.shape[0])
        seg_at = seg[nearest]
        seg_len = np.sqrt((seg_at ** 2).sum(axis=1))
        seg_len = np.where(seg_len < 1e-12, 1.0, seg_len)
        nx = -seg_at[:, 1] / seg_len
        ny =  seg_at[:, 0] / seg_len
        diff_at = diffs[rows, nearest]
        signed[mask] = diff_at[:, 0] * nx + diff_at[:, 1] * ny

    # Common x-axis: the robot's OWN executed arc-length, normalised to
    # [0, 1]. Monotonic and full-coverage regardless of how often Nav2
    # replans (projecting onto the final /plan collapses the axis — that
    # plan is usually a short near-goal stub).
    step = np.diff(rp, axis=0)
    cum = np.concatenate(([0.0], np.cumsum(np.sqrt((step ** 2).sum(axis=1)))))
    total = float(cum[-1])
    if total < 1e-6:
        return np.array([]), np.array([]), np.array([])
    along_frac = cum / total

    valid = np.isfinite(abs_d)
    if not valid.any():
        return np.array([]), np.array([]), np.array([])
    return signed[valid], abs_d[valid], along_frac[valid]


# ---------------------------------------------------------------------------
# Violin distribution
# ---------------------------------------------------------------------------

def _violin_into_ax(ax, df: pd.DataFrame, controllers: list[str],
                    label_map: dict[str, str] | None = None,
                    highlight: str | None = None,
                    show_category_labels: bool = True,
                    scope: str = "full") -> None:
    """Draw the per-controller |XTE| violin into ``ax``.

    XTE is computed against each controller's own time-active /plan;
    HDF5s lacking ``global_path_plans/`` fall back to the straight-line
    start→goal proxy.
    """
    controllers = order_by_category(controllers)
    label_map = label_map or {}
    data: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    plotted: list[str] = []
    for ctrl in controllers:
        ctrl_rows = df[df["controller"] == ctrl]
        if ctrl_rows.empty: continue
        pooled: list[np.ndarray] = []
        for _, row in ctrl_rows.iterrows():
            _, abs_xte, _ = _xte_arrays_vs_active_plan(row["hdf5_path"])
            if abs_xte.size == 0:
                signed, _ = _xte_arrays(row["hdf5_path"])
                abs_xte = np.abs(signed) if signed.size else signed
            if abs_xte.size:
                pooled.append(abs_xte)
        if not pooled: continue
        arr = np.concatenate(pooled)
        # Sub-sample if huge.
        if arr.size > 3000:
            idx = np.linspace(0, arr.size - 1, 3000).astype(int)
            arr = arr[idx]
        data.append(arr)
        labels.append(label_map.get(ctrl, _short(ctrl)))
        colors.append(controller_color(ctrl, scope=scope, highlight=highlight))
        plotted.append(ctrl)
    if not data:
        return

    parts = ax.violinplot(data, positions=np.arange(len(data)),
                          showmeans=False, showmedians=False, showextrema=False,
                          widths=0.8)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_edgecolor("black"); body.set_alpha(0.55)
    # IQR box + median tick inside each violin.
    for i, arr in enumerate(data):
        q1, med, q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
        ax.vlines(i, q1, q3, color="black", linewidth=5.0, alpha=0.85, zorder=3)
        ax.plot([i - 0.13, i + 0.13], [med, med], color="white",
                linewidth=1.6, zorder=4)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=14)
    # Emphasise the thesis' own controller — bold, larger, black.
    for tl, c in zip(ax.get_xticklabels(), plotted):
        if c == highlight:
            tl.set_fontweight("bold")
            tl.set_fontsize(16)
            tl.set_color("#000000")
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("|XTE| (m)  —  vs own active /plan  (symlog)", fontsize=12.5)
    ax.grid(axis="y", alpha=0.3, which="both")
    # Symmetric-log y-axis with linthresh = 0.05 m so the linear region
    # covers the bulk of the completing controllers (|XTE| < 0.05 m)
    # while the imitation tail is log-compressed and does not collapse
    # the rest. Linear-only would render the five completing controllers
    # as visually identical thin bars next to the imitation outlier.
    ax.set_yscale("symlog", linthresh=0.05)
    # Headroom band at the top so the category labels sit clear above the
    # tallest violin.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.22 * (y1 - y0))
    for sx in category_separators(plotted):
        ax.axvline(sx, color=SEPARATOR_COLOR, linestyle=(0, (4, 3)),
                   linewidth=1.0, zorder=1)
    # Category labels (Classical / Fixed-wt / Trained / Imitation) used
    # to sit above the violins but were unreadable at print scale; the
    # legend below the figure names every controller.


def render_violin(df: pd.DataFrame, controllers: list[str], out_path: Path,
                  label_map: dict[str, str] | None = None,
                  highlight: str | None = None,
                  show_category_labels: bool = True,
                  scope: str = "full") -> None:
    """Standalone per-controller |XTE| violin figure (pooled over goals)."""
    apply_style()
    n_goals = int(df["goal_folder"].nunique())
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(controllers)), 4.5),
                           constrained_layout=True)
    _violin_into_ax(ax, df, controllers, label_map, highlight,
                    show_category_labels, scope=scope)
    ax.set_title("Per-step |XTE| distribution per controller "
                 f"(pooled over {n_goals} goals, ref: own active /plan)",
                 fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Profile — XTE vs normalized trajectory progress, per goal
# ---------------------------------------------------------------------------

def render_profile(df: pd.DataFrame, goal: str, controllers: list[str],
                   out_path: Path,
                   color_map: dict[str, str] | None = None,
                   label_map: dict[str, str] | None = None,
                   highlight: str | None = None) -> None:
    """|XTE| vs normalized trajectory progress, one line per controller.

    XTE is computed per-step against the time-active /plan; the x-axis is
    each episode's own executed arc-length normalised to [0, 1]. One thin
    solid line per controller in its own colour; the thesis' own
    controller is drawn a little thicker.
    """
    apply_style()
    label_map = label_map or {}
    controllers = order_by_category(controllers)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    plotted_any = False
    for ctrl in controllers:
        rows = df[(df["controller"] == ctrl) & (df["goal_folder"] == goal)]
        if rows.empty: continue
        _, abs_xte, along = _xte_arrays_vs_active_plan(rows.iloc[-1]["hdf5_path"])
        if abs_xte.size == 0:
            signed, along = _xte_arrays(rows.iloc[-1]["hdf5_path"])
            abs_xte = np.abs(signed) if signed.size else signed
        if abs_xte.size == 0: continue
        bins = np.linspace(0, 1, 26)        # 25 bins — smoother than 40
        binned = np.full(len(bins) - 1, np.nan)
        for j in range(len(bins) - 1):
            mask = (along >= bins[j]) & (along < bins[j + 1])
            if mask.any():
                binned[j] = float(abs_xte[mask].mean())
        x_mid = (bins[:-1] + bins[1:]) / 2
        color = (color_map.get(ctrl) if color_map and ctrl in color_map
                 else controller_color(ctrl))
        is_hl = (ctrl == highlight)
        ax.plot(x_mid, binned, color=color,
                linewidth=(1.2 if is_hl else 0.8), alpha=0.95,
                label=label_map.get(ctrl, _short(ctrl)),
                zorder=(5 if is_hl else 3))
        plotted_any = True
    if not plotted_any:
        plt.close(fig); return
    ax.set_xlabel("normalized progress along executed trajectory "
                  "(0 = start, 1 = goal)")
    ax.set_ylabel("|XTE| (m)  —  vs own active /plan")
    ax.set_title(f"|XTE| profile along executed trajectory — {goal}  "
                 "(ref: own active /plan)")
    ax.axhline(0.0, color="#333333", linestyle=(0, (2, 2)), linewidth=1.0,
               alpha=0.7, zorder=1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, frameon=False, ncol=1)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Envelope — single controller across all goals
# ---------------------------------------------------------------------------

def render_envelope(df: pd.DataFrame, ctrl: str, out_path: Path,
                    color: str | None = None,
                    ) -> None:
    """|XTE| envelope across all goals for a single controller.

    Per-step XTE is computed against the time-active /plan; the x-axis is
    each episode's own executed arc-length normalised to [0, 1]. Mean and
    ±1σ / ±2σ are computed across the goals after binning each episode
    onto the same normalised progress grid.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    bins = np.linspace(0, 1, 50)
    all_binned: list[np.ndarray] = []
    if color is None:
        color = controller_color(ctrl)
    for goal in sorted(df.loc[df["controller"] == ctrl, "goal_folder"].unique()):
        rows = df[(df["controller"] == ctrl) & (df["goal_folder"] == goal)]
        if rows.empty: continue
        _, abs_xte, along = _xte_arrays_vs_active_plan(rows.iloc[-1]["hdf5_path"])
        if abs_xte.size == 0:
            signed, along = _xte_arrays(rows.iloc[-1]["hdf5_path"])
            abs_xte = np.abs(signed) if signed.size else signed
        if abs_xte.size == 0: continue
        binned = np.full(len(bins) - 1, np.nan)
        for i in range(len(bins) - 1):
            mask = (along >= bins[i]) & (along < bins[i + 1])
            if mask.any():
                binned[i] = float(abs_xte[mask].mean())
        all_binned.append(binned)
        x_mid = (bins[:-1] + bins[1:]) / 2
        ax.plot(x_mid, binned, color=color, linewidth=0.6, alpha=0.5)
    if not all_binned:
        plt.close(fig); return
    A = np.vstack(all_binned)
    mean = np.nanmean(A, axis=0)
    sd   = np.nanstd(A, axis=0, ddof=1) if A.shape[0] >= 2 else np.zeros_like(mean)
    x_mid = (bins[:-1] + bins[1:]) / 2
    ax.fill_between(x_mid, mean - 2 * sd, mean + 2 * sd,
                    color=color, alpha=0.15, label="±2σ")
    ax.fill_between(x_mid, mean - sd, mean + sd,
                    color=color, alpha=0.30, label="±1σ")
    ax.plot(x_mid, mean, color="black", linewidth=1.5, label="mean")
    ax.axhline(0.0, color="#333333", linestyle=(0, (2, 2)), linewidth=1.0,
               alpha=0.7, zorder=1)
    ax.set_xlabel("normalized progress along executed trajectory")
    ax.set_ylabel("|XTE| (m)  —  vs own active /plan")
    ax.set_title(f"|XTE| envelope — {ctrl} "
                 f"(across {len(all_binned)} goals, ref: own active /plan)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined XTE — one figure, every controller's goal-averaged curve
# ---------------------------------------------------------------------------

def _controller_mean_xte(df: pd.DataFrame, ctrl: str,
                         bins: np.ndarray) -> np.ndarray | None:
    """Per-controller |XTE|-vs-progress curves, one row per goal.

    Returns a (n_goals x n_bins) array of binned |XTE| (NaN where a bin
    had no samples), or ``None`` if the controller has no usable episode.
    """
    per_goal: list[np.ndarray] = []
    for goal in sorted(df.loc[df["controller"] == ctrl, "goal_folder"].unique()):
        rows = df[(df["controller"] == ctrl) & (df["goal_folder"] == goal)]
        if rows.empty:
            continue
        _, abs_xte, along = _xte_arrays_vs_active_plan(rows.iloc[-1]["hdf5_path"])
        if abs_xte.size == 0:
            signed, along = _xte_arrays(rows.iloc[-1]["hdf5_path"])
            abs_xte = np.abs(signed) if signed.size else signed
        if abs_xte.size == 0:
            continue
        binned = np.full(len(bins) - 1, np.nan)
        for i in range(len(bins) - 1):
            mask = (along >= bins[i]) & (along < bins[i + 1])
            if mask.any():
                binned[i] = float(abs_xte[mask].mean())
        per_goal.append(binned)
    return np.vstack(per_goal) if per_goal else None


def _combined_curves_into_ax(ax, df: pd.DataFrame, controllers: list[str],
                             label_map: dict[str, str] | None = None,
                             highlight: str | None = None,
                             with_band: bool = False,
                             scope: str = "full") -> None:
    """Draw goal-averaged |XTE| curves (one per controller) into ``ax``.

    For each controller the per-goal binned |XTE|-vs-progress curves are
    averaged into a single mean curve, drawn as a thin solid line in that
    controller's colour. ``with_band=True`` shades a faint ±1σ-across-
    goals band (envelope view). The thesis' own controller is drawn a
    little thicker.
    """
    controllers = order_by_category(controllers)
    label_map = label_map or {}
    bins = np.linspace(0, 1, 26)        # 25 bins — smoother than 50
    x_mid = (bins[:-1] + bins[1:]) / 2
    for ctrl in controllers:
        A = _controller_mean_xte(df, ctrl, bins)
        if A is None:
            continue
        mean = np.nanmean(A, axis=0)
        color = controller_color(ctrl, scope=scope, highlight=highlight)
        is_hl = (ctrl == highlight)
        ax.plot(x_mid, mean, color=color,
                linewidth=(1.2 if is_hl else 0.8), alpha=0.95,
                label=label_map.get(ctrl, _short(ctrl)),
                zorder=(5 if is_hl else 3))
        if with_band and A.shape[0] >= 2:
            sd = np.nanstd(A, axis=0, ddof=1)
            ax.fill_between(x_mid, mean - sd, mean + sd,
                            color=color, alpha=0.06, linewidth=0.0, zorder=1)
    ax.set_xlabel("normalized progress (0 = start, 1 = goal)", fontsize=12.5)
    ax.set_ylabel("|XTE| (m)  —  vs own active /plan", fontsize=12.5)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.3)


def render_xte_overview(df: pd.DataFrame, controllers: list[str],
                        out_path: Path,
                        label_map: dict[str, str] | None = None,
                        legend_label_map: dict[str, str] | None = None,
                        highlight: str | None = None,
                        scope: str = "full",
                        orientation: str = "vertical") -> None:
    """Three-panel XTE overview in one figure.

    Columns: goal-averaged |XTE| envelope (±1sigma band) | goal-averaged
    |XTE| profile (mean lines) | per-step |XTE| violin. A single shared
    legend below names every controller (full names). The detailed
    per-goal profile and per-controller envelope figures are produced
    separately.

    ``orientation`` selects the panel arrangement: ``"vertical"`` (the
    default, 3 rows x 1 col) yields the portrait figure used in the
    thesis; ``"horizontal"`` (1 row x 3 cols) yields a wide, short strip
    for the poster, where the three panels need to share one row's height.
    """
    apply_style()
    controllers = order_by_category(controllers)
    legend_label_map = legend_label_map or label_map or {}

    # Reserve more bottom room when the legend needs several rows.
    # Cap at 3 columns so 6 controllers wrap to a 2-row legend (matches
    # the Tier 2/3/4 bar figures and keeps the descriptive labels on
    # one line each). The wide horizontal strip is broad enough to lay
    # every entry on a single row.
    ncol = len(controllers) if orientation == "horizontal" \
        else min(3, len(controllers))
    legend_rows = math.ceil(len(controllers) / ncol)
    legend_in = 0.45 + 0.40 * legend_rows
    if orientation == "horizontal":
        # One row x three cols: a wide, short strip that fills the poster
        # column width with all three panels readable at the same height.
        panel_w, panel_h = 6.0, 4.6
        fig_w = panel_w * 3
        fig_h = panel_h + legend_in + 0.4
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    else:
        # Stack the three panels vertically (3 rows x 1 col) so the figure
        # is portrait-oriented; \includegraphics[width=\textwidth] then
        # produces a tall, readable figure rather than a thin, wide strip.
        fig_w, panel_h = 9.5, 3.6
        fig_h = panel_h * 3 + legend_in + 0.4
        fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h))
    _combined_curves_into_ax(axes[0], df, controllers, label_map,
                             highlight, with_band=True, scope=scope)
    axes[0].set_title("Mean |XTE| envelope (±1σ across goals)", fontsize=15)
    _combined_curves_into_ax(axes[1], df, controllers, label_map,
                             highlight, with_band=False, scope=scope)
    axes[1].set_title("Mean |XTE| profile (goal-averaged)", fontsize=15)
    _violin_into_ax(axes[2], df, controllers, label_map, highlight,
                    show_category_labels=True, scope=scope)
    axes[2].set_title("Per-step |XTE| distribution", fontsize=15)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=controller_color(c, scope=scope, highlight=highlight),
                     edgecolor="black",
                     linewidth=(1.8 if c == highlight else 0.4),
                     label=legend_label_map.get(c, _short(c)))
               for c in controllers]
    leg = fig.legend(handles=handles, loc="lower center", ncol=ncol,
                     frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.0),
                     handlelength=2.2, handleheight=1.5,
                     columnspacing=(1.4 if orientation == "horizontal" else 2.4),
                     labelspacing=0.9)
    for txt, c in zip(leg.get_texts(), controllers):
        if c == highlight:
            txt.set_fontweight("bold")

    fig.suptitle("XTE overview — controller comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, legend_in / fig_h, 1, 1 - 0.5 / fig_h))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
