"""Centralised matplotlib style for vf_robot_utils figures.

Imports of this module are cheap (no figures created at import time).
Call ``apply_style()`` once at the top of any figure script — it sets
``rcParams`` consistent with T-RO/IJRR submission expectations.

Two palettes are exposed:

- ``OKABE_ITO``           — colour-blind-safe categorical palette (8 hues).
- ``controller_color()``  — stable controller -> hex mapping; same
                            controller always gets the same colour
                            across figures.

For sequential heatmaps and per-scenario gradients use
``matplotlib.cm.viridis`` (or ``"viridis"``) directly — no helper
needed.

Reference: Okabe & Ito (2008), "Color Universal Design".
"""
from __future__ import annotations

from typing import Iterable

import matplotlib as mpl


# Okabe-Ito palette — colour-blind safe, prints well in greyscale.
OKABE_ITO: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)


# Stable controller -> colour map. Two scopes:
#
#   HEADLINE scope (used by run_comparison.sh, headline 6 controllers):
#   each controller is a distinct, bold, colour-blind-safe hue. The
#   deployed adaptive controller (resolved at runtime by TOPSIS, family
#   ``rawwt``) is rendered in bright red — the eye-catching colour of
#   the thesis.
#
#   FULL scope (used by run_comparison_all.sh, all 23 controllers):
#   the 18 trained-grid variants are split into a WARM family (raw,
#   nine shades of red/orange) and a COOL family (oracle, nine shades
#   of blue/teal) so the reader can read ``(family, channel, regime)``
#   straight off the colour. Naming convention for the grid:
#     - hue family:    rawwt -> warm reds; oraclewt -> cool blues
#     - lightness:     normal (saturated) -> tuned (deep) -> hardreg (darkest)
#     - chroma:        v1 (most saturated) -> v2 (mid) -> v3 (lightest)
#
#   The six non-grid controllers (mppi, dwb, rpp, FW-MPPI, IL-V,
#   graceful) keep their HEADLINE hue in both scopes.

# ── Headline-scope colours (six bold, distinct hues) ──────────────────
_HEADLINE_COLORS: dict[str, str] = {
    "mppi":         "#444444",    # dark grey (classical)
    "dwb":          "#009e73",    # bluish green (classical)
    "rpp":          "#e69f00",    # vermillion-orange (classical)
    "graceful":     "#f0e442",    # yellow (excluded classical)
    "fixedwt":      "#56b4e9",    # sky blue (fixed-weight baseline)
    "imitationwt":  "#cc79a7",    # reddish purple (imitation baseline)
    # Headline override: the deployed adaptive controller, whichever
    # rawwt variant TOPSIS picks at runtime, is recoloured to bright
    # red by the caller using ``controller_color(name, scope="headline",
    # highlight=name)``.
    "rawwt":        "#d62728",    # bright red (deployed CA-MCW)
}

# ── Full-scope shades for the 18-variant trained grid ─────────────────
# Raw-critic (warm) family — nine shades of red.
_RAW_GRID: dict[str, str] = {
    "rawwt_normal_v1":   "#d62728",  # base red — same shade as CA-MCW headline
    "rawwt_normal_v2":   "#e7654a",  # red-orange
    "rawwt_normal_v3":   "#ee8b71",  # light coral
    "rawwt_tuned_v1":    "#a51d20",  # deep red
    "rawwt_tuned_v2":    "#b8462e",  # rust
    "rawwt_tuned_v3":    "#c66050",  # rose
    "rawwt_hardreg_v1":  "#7c1417",  # dark crimson
    "rawwt_hardreg_v2":  "#8b2f1d",  # brown-red
    "rawwt_hardreg_v3":  "#9b4538",  # terracotta
}

# Oracle (cool) family — nine shades of blue.
_ORACLE_GRID: dict[str, str] = {
    "oraclewt_normal_v1":   "#1f77b4",  # blue
    "oraclewt_normal_v2":   "#3a93cc",  # mid blue
    "oraclewt_normal_v3":   "#56b4e9",  # sky blue
    "oraclewt_tuned_v1":    "#155988",  # deep blue
    "oraclewt_tuned_v2":    "#266d9f",  # navy
    "oraclewt_tuned_v3":    "#3782b6",  # steel
    "oraclewt_hardreg_v1":  "#0a3d5c",  # very dark blue
    "oraclewt_hardreg_v2":  "#1a4e76",  # midnight
    "oraclewt_hardreg_v3":  "#2a5f90",  # dark steel
}

# ── Aliased imitation variants ────────────────────────────────────────
# The imitation family has a single shipped variant; alias both.
_IMIT_VARIANTS: dict[str, str] = {
    "imitationwt_normal_v1": _HEADLINE_COLORS["imitationwt"],
}

# Master full-scope map (built once at import time).
_FULL_COLORS: dict[str, str] = {
    **_HEADLINE_COLORS,
    **_RAW_GRID,
    **_ORACLE_GRID,
    **_IMIT_VARIANTS,
}

_DEFAULT_FALLBACK = "#777777"


def controller_color(name: str, scope: str = "full",
                     highlight: str | None = None) -> str:
    """Return the canonical colour for ``name``.

    Parameters
    ----------
    name : str
        Controller identifier (slug, e.g. ``rawwt_tuned_v3`` or
        ``fixedwt`` or ``mppi``).
    scope : {"headline", "full"}, default "full"
        ``"headline"`` returns the headline-6 palette and recolours the
        deployed adaptive controller (passed via ``highlight``) to
        bright red, regardless of which rawwt variant it is.
        ``"full"`` returns the per-variant shade from the warm/cool grid.
    highlight : str, optional
        Used only in headline scope: the controller identifier of the
        deployed adaptive controller (the TOPSIS-selected rawwt
        variant). When ``name == highlight`` and scope is headline, the
        colour is forced to the bright red headline override.
    """
    # Headline scope: deployed adaptive controller gets the bright red.
    if scope == "headline" and highlight is not None and name == highlight:
        return _HEADLINE_COLORS["rawwt"]

    # Headline scope: any other rawwt variant (very unusual in headline
    # figures) falls back to the family bright red as well, for legend
    # consistency. fixedwt / imitationwt etc. use _HEADLINE_COLORS.
    if scope == "headline":
        if name in _HEADLINE_COLORS:
            return _HEADLINE_COLORS[name]
        for family in ("rawwt", "oraclewt", "imitationwt"):
            if name.startswith(family + "_") or name == family:
                return _HEADLINE_COLORS[family]
        return _DEFAULT_FALLBACK

    # Full scope: per-variant shades from the grid.
    if name in _FULL_COLORS:
        return _FULL_COLORS[name]
    for family in ("rawwt", "oraclewt", "imitationwt"):
        if name.startswith(family + "_") or name == family:
            return _FULL_COLORS.get(family, _HEADLINE_COLORS[family])
    return _DEFAULT_FALLBACK


def cycler_for(controllers: Iterable[str], scope: str = "full",
               highlight: str | None = None) -> list[str]:
    """List of stable colours in the requested controller order."""
    return [controller_color(c, scope=scope, highlight=highlight)
            for c in controllers]


# ── Controller categories — stable ordering for comparison figures ──────────
# Comparison bar / violin plots lay controllers out in this category order,
# with a thin dashed separator between groups.
_CLASSICAL: tuple[str, ...] = ("mppi", "dwb", "rpp", "graceful")
CATEGORY_ORDER: tuple[str, ...] = ("classical", "fixedwt", "trained", "imitation")

# Human-readable group label drawn under the x-axis of comparison figures.
# Kept to a single short word — in the headline figure the fixedwt /
# trained / imitation blocks are each a single bar wide and adjacent, so
# a multi-word label would overlap the neighbouring group.
CATEGORY_LABEL: dict[str, str] = {
    "classical":  "Classical",
    "fixedwt":    "Fixed-wt",
    "trained":    "Trained",
    "imitation":  "Imitation",
}

# Colour of the dashed group separator in comparison figures.
SEPARATOR_COLOR: str = "#d62728"   # red


def controller_category(name: str) -> str:
    """Coarse category used to group controllers in comparison figures."""
    if name in _CLASSICAL:
        return "classical"
    if name == "fixedwt":
        return "fixedwt"
    if name.startswith("imitationwt"):
        return "imitation"
    return "trained"          # oraclewt_* / rawwt_*


def order_by_category(controllers: Iterable[str]) -> list[str]:
    """Reorder controllers classical -> fixedwt -> trained -> imitation,
    preserving the original order within each category."""
    items = list(controllers)
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    return sorted(
        items,
        key=lambda c: (rank.get(controller_category(c), len(rank)),
                       items.index(c)),
    )


def category_separators(controllers: list[str]) -> list[float]:
    """Between-bar x-positions where the controller category changes.

    Draw a thin dashed vertical line at each (``ax.axvline(x, ...)``) to
    separate the classical / fixedwt / trained / imitation groups.
    ``controllers`` must already be in plotted order.
    """
    cats = [controller_category(c) for c in controllers]
    return [i - 0.5 for i in range(1, len(cats)) if cats[i] != cats[i - 1]]


def category_spans(controllers: list[str]) -> list[tuple[float, str]]:
    """``(centre_x, label)`` for each contiguous controller-category block.

    ``centre_x`` is the mid bar-index of the block — place the group
    label there under the x-axis. ``controllers`` must be in plotted
    order (see ``order_by_category``).
    """
    spans: list[tuple[float, str]] = []
    if not controllers:
        return spans
    cats = [controller_category(c) for c in controllers]
    start = 0
    for i in range(1, len(cats) + 1):
        if i == len(cats) or cats[i] != cats[start]:
            centre = (start + i - 1) / 2.0
            spans.append((centre, CATEGORY_LABEL.get(cats[start], cats[start])))
            start = i
    return spans


def apply_style() -> None:
    """Set matplotlib rcParams to a consistent journal-figure style.

    Idempotent — calling twice is a no-op. Safe to call from inside any
    figure-building function. Uses ``mpl.rcParams`` directly (not
    ``rc_context``) so the style sticks for the whole script.
    """
    rc = mpl.rcParams
    rc["font.family"] = "DejaVu Sans"
    rc["font.size"] = 10.0
    rc["axes.titlesize"] = 11.0
    rc["axes.labelsize"] = 10.0
    rc["xtick.labelsize"] = 9.0
    rc["ytick.labelsize"] = 9.0
    rc["legend.fontsize"] = 9.0
    rc["figure.titlesize"] = 12.0

    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.grid"] = True
    rc["grid.alpha"] = 0.30
    rc["grid.linestyle"] = ":"

    rc["axes.prop_cycle"] = mpl.cycler(color=list(OKABE_ITO))

    rc["figure.dpi"] = 110
    rc["savefig.dpi"] = 200
    rc["savefig.bbox"] = "tight"
