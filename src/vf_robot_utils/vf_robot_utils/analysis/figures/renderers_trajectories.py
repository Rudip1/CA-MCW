"""renderers_trajectories.py — per-goal trajectory-overlay renderer.

Renderer library for the cross-controller comparison figures
(``comparison_figures.py``). Not a CLI entry point.

``render_per_goal`` overlays every controller's actual robot path on
the map for one goal. ``_load_map`` / ``_autodetect_map_yaml`` resolve
the map background image + origin from the map YAML.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vf_robot_utils.analysis.style import apply_style, controller_color


def _load_map(yaml_path: Path) -> tuple[np.ndarray, float, float, float]:
    import yaml
    from PIL import Image
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    pgm = cfg["image"]
    if not Path(pgm).is_absolute():
        pgm = yaml_path.parent / pgm
    res = float(cfg["resolution"])
    ox = float(cfg["origin"][0]); oy = float(cfg["origin"][1])
    img = np.array(Image.open(pgm).convert("L"))
    return img, res, ox, oy


def _autodetect_map_yaml(root_name: str) -> Path | None:
    try:
        from vf_robot_utils.constants import MAPS_ROOT
        cand = Path(MAPS_ROOT) / root_name / f"{root_name}.yaml"
        if cand.exists():
            return cand
    except Exception:
        pass
    return None


def _short(c: str) -> str:
    if c.startswith("oraclewt_"):    return "O-" + c.removeprefix("oraclewt_")
    if c.startswith("rawwt_"):       return "R-" + c.removeprefix("rawwt_")
    if c.startswith("imitationwt_"): return "I-" + c.removeprefix("imitationwt_")
    return c.upper()


def _draw_map(ax, img: np.ndarray, res: float, ox: float, oy: float) -> None:
    h, w = img.shape
    extent = (ox, ox + w * res, oy, oy + h * res)
    ax.imshow(img, cmap="gray", origin="upper", extent=extent, alpha=0.6,
              zorder=0)


def _row_failed(row) -> bool:
    """True if the episode did not succeed.

    Reads the CSV ``t2_success`` column (the authoritative post-processed
    outcome). Handles bool, "True"/"False" strings, and NaN — a bare
    ``bool("False")`` is truthy, so string values must be parsed.
    """
    v = row.get("t2_success", False)
    if isinstance(v, str):
        return v.strip().lower() not in ("true", "1", "1.0", "yes")
    if v != v:            # NaN
        return True
    return not bool(v)


def _plot_path(ax, h5_path: str, color: str, label: str,
               failure: bool, alpha: float = 0.85) -> None:
    with h5py.File(h5_path, "r") as f:
        rp = f["robot_pose"][:] if "robot_pose" in f else None
    if rp is None or rp.shape[0] == 0:
        return
    ax.plot(rp[:, 0], rp[:, 1], color=color, linewidth=1.2,
            alpha=alpha, label=label, zorder=2)
    ax.scatter([rp[0, 0]], [rp[0, 1]], marker="o", s=22,
               facecolor=color, edgecolor="black", linewidth=0.4, zorder=4)
    if failure:
        ax.scatter([rp[-1, 0]], [rp[-1, 1]], marker="x", s=42,
                   color="red", linewidth=1.4, zorder=5)


def _goal_xy_from_results(df: pd.DataFrame, goal: str) -> tuple[float, float] | None:
    sub = df[df["goal_folder"] == goal]
    if sub.empty: return None
    # Pull the goal coords from the HDF5 of any successful row.
    for _, row in sub.iterrows():
        try:
            with h5py.File(row["hdf5_path"], "r") as f:
                if "goal" in f and f["goal"].shape[0] > 0:
                    g = f["goal"][-1]
                    return float(g[0]), float(g[1])
        except Exception:
            continue
    return None


def render_per_goal(
    df: pd.DataFrame, goal: str, controllers: list[str],
    img_tuple: tuple, out_path: Path, title_suffix: str,
    color_map: dict[str, str] | None = None,
    global_path_xy: np.ndarray | None = None,
    scope: str = "full",
    highlight: str | None = None,
) -> None:
    """color_map: optional {controller: hex} override (per-rank coloring etc.).
    global_path_xy: optional (N, 2) planner-path overlay (dotted dark grey)."""
    apply_style()
    img, res, ox, oy = img_tuple
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    _draw_map(ax, img, res, ox, oy)

    # Global planner path under controller paths, above the map.
    if global_path_xy is not None and global_path_xy.size:
        ax.plot(global_path_xy[:, 0], global_path_xy[:, 1],
                linestyle=(0, (2, 2)), linewidth=1.4,
                color="#333333", alpha=0.85, zorder=1,
                label="global planner path")

    goal_xy = _goal_xy_from_results(df, goal)
    if goal_xy:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker="v", s=90,
                   facecolor="gold", edgecolor="black", linewidth=0.8,
                   zorder=6, label="goal")

    # Plot paths in stable controller order.
    g_df = df[df["goal_folder"] == goal]
    # Dedupe duplicate runs per controller.
    if "timestamp" in g_df.columns:
        g_df = (g_df.sort_values("timestamp")
                    .drop_duplicates(subset="controller", keep="last"))
    for ctrl in controllers:
        rows = g_df[g_df["controller"] == ctrl]
        if rows.empty: continue
        row = rows.iloc[-1]
        color = (color_map.get(ctrl) if color_map and ctrl in color_map
                 else controller_color(ctrl, scope=scope, highlight=highlight))
        _plot_path(ax, row["hdf5_path"],
                   color, _short(ctrl),
                   failure=_row_failed(row))

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectories @ {goal}  ({title_suffix})")
    # Legend outside the axes for readability.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, frameon=False, ncol=1)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Combined tour overview ────────────────────────────────────────────────
# The evaluation tour is sequential: Start -> G1 -> G2 -> ... -> G8, where
# each leg's nominal start is the previous waypoint. A controller that
# aborts a leg never physically reaches the proper start for the next, so
# the raw odometry of later legs starts wherever that controller stalled.
# For a single readable overview, each controller's leg-k path is rigidly
# translated so its first pose coincides with the *nominal* leg-k start
# waypoint. This is a presentation-only re-anchoring; the quantitative
# outcomes (success, abort, clearance, ...) are reported unchanged in the
# per-category figures and Table 5.1.

# Canonical tour waypoints (x, y), Start then G1..G8 — matches the
# evaluation-tour table of Chapter 4 / Appendix B.
_TOUR_WAYPOINTS: list[tuple[float, float]] = [
    ( 4.661,  1.443),   # Start
    ( 0.522, -1.851),   # G1
    (-6.209, -7.575),   # G2
    (10.535, -8.511),   # G3
    (11.213,  5.437),   # G4
    ( 7.302,  7.169),   # G5
    (-0.662,  6.171),   # G6
    (-2.652, -1.000),   # G7
    ( 3.015, -4.800),   # G8
]


def _goal_folder_xy(folder: str) -> tuple[float, float] | None:
    """Parse 'goal_x0.52_yn1.85_t3.13' -> (0.52, -1.85). 'n' = minus."""
    try:
        body = folder.split("goal_", 1)[1]
        parts = body.split("_")
        def num(tok: str, pfx: str) -> float:
            s = tok[len(pfx):]
            neg = s.startswith("n")
            if neg:
                s = s[1:]
            return (-1.0 if neg else 1.0) * float(s)
        return num(parts[0], "x"), num(parts[1], "y")
    except Exception:
        return None


def render_tour_overview(
    df: pd.DataFrame, goals: list[str], controllers: list[str],
    img_tuple: tuple, out_path: Path,
    label_map: dict[str, str] | None = None,
    highlight: str | None = None,
    scope: str = "headline",
) -> None:
    """One map: all eight goals and every controller's full tour.

    Each controller's per-leg path is rigidly re-anchored to the nominal
    leg-start waypoint so all controllers share a common Start and the
    goal markers G1..G8 sit at their nominal locations. Disclosed as a
    schematic in the LaTeX caption.
    """
    apply_style()
    label_map = label_map or {}
    img, res, ox, oy = img_tuple

    # Per-controller mean success rate over all evaluation goals. The
    # overview only draws controllers that completed every leg (success
    # rate 1.0); a partially-completing controller has no path on the
    # legs it aborted, which would clutter the single-map view. The
    # excluded controllers and their success rate are listed on the
    # figure and remain fully reported in Table 5.1.
    def _succ(v) -> float:
        if isinstance(v, str):
            return 1.0 if v.strip().lower() in ("true", "1", "1.0", "yes") else 0.0
        if v != v:
            return 0.0
        return float(bool(v))
    succ_rate: dict[str, float] = {}
    for ctrl in controllers:
        sub = df[df["controller"] == ctrl]
        if sub.empty:
            continue
        vals = [_succ(x) for x in sub.get("t2_success", [])]
        if vals:
            succ_rate[ctrl] = sum(vals) / len(vals)
    keep = [c for c in controllers if succ_rate.get(c, 0.0) >= 0.999]
    excluded = [(c, succ_rate.get(c, 0.0))
                for c in controllers if c not in keep]

    fig, ax = plt.subplots(figsize=(9.6, 8.6), constrained_layout=True)
    _draw_map(ax, img, res, ox, oy)

    # Order the goal_folders into tour sequence by nearest waypoint.
    wp = _TOUR_WAYPOINTS
    folder_for_leg: dict[int, str] = {}
    for g in goals:
        gx = _goal_folder_xy(g)
        if gx is None:
            continue
        # Match to G1..G8 (indices 1..8 in wp).
        best_k, best_d = None, 1e9
        for k in range(1, len(wp)):
            d = (gx[0] - wp[k][0]) ** 2 + (gx[1] - wp[k][1]) ** 2
            if d < best_d:
                best_d, best_k = d, k
        if best_k is not None and best_d < 1.0:
            folder_for_leg[best_k] = g

    if "timestamp" in df.columns:
        df = (df.sort_values("timestamp")
                .drop_duplicates(subset=["controller", "goal_folder"],
                                 keep="last"))

    drawn_ctrl: set[str] = set()
    for k in sorted(folder_for_leg):
        goal = folder_for_leg[k]
        nom_start = np.array(wp[k - 1], dtype=float)
        g_df = df[df["goal_folder"] == goal]
        for ctrl in keep:
            rows = g_df[g_df["controller"] == ctrl]
            if rows.empty:
                continue
            row = rows.iloc[-1]
            try:
                with h5py.File(row["hdf5_path"], "r") as f:
                    rp = f["robot_pose"][:] if "robot_pose" in f else None
            except Exception:
                rp = None
            if rp is None or rp.shape[0] == 0:
                continue
            xy = rp[:, :2].astype(float)
            # Rigid translation: first executed pose -> nominal leg start.
            xy = xy - xy[0] + nom_start
            color = controller_color(ctrl, scope=scope, highlight=highlight)
            lw = 2.4 if ctrl == highlight else 1.1
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw,
                    alpha=0.85,
                    label=(label_map.get(ctrl, _short(ctrl))
                           if ctrl not in drawn_ctrl else None),
                    zorder=(4 if ctrl == highlight else 2))
            if _row_failed(row):
                ax.scatter([xy[-1, 0]], [xy[-1, 1]], marker="x", s=34,
                           color="red", linewidth=1.2, zorder=5)
            drawn_ctrl.add(ctrl)

    # Waypoint markers: Start (square) then G1..G8 (numbered).
    sx, sy = wp[0]
    ax.scatter([sx], [sy], marker="s", s=120, facecolor="white",
               edgecolor="black", linewidth=1.3, zorder=7)
    ax.annotate("Start", (sx, sy), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=13, fontweight="bold")
    for k in range(1, len(wp)):
        gx, gy = wp[k]
        ax.scatter([gx], [gy], marker="v", s=120, facecolor="gold",
                   edgecolor="black", linewidth=0.9, zorder=7)
        ax.annotate(f"G{k}", (gx, gy), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=13,
                    fontweight="bold")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Evaluation tour — every goal, the fully-successful "
                 "controllers (per-leg re-anchored to nominal start)",
                 fontsize=14)
    if excluded:
        note = "Omitted (success rate $<1$): " + ", ".join(
            f"{label_map.get(c, _short(c))} ({r:.0%})" for c, r in excluded)
        ax.text(0.5, -0.13, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=11.5, style="italic", color="#444444")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncol=min(4, max(1, len(keep))), fontsize=13, frameon=False,
              handlelength=2.4, columnspacing=3.0, labelspacing=0.9,
              handletextpad=0.6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
