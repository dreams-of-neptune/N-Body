"""
plotting.py -- Three-Body Visualisation
========================================
All matplotlib rendering logic.  No physics; depends only on
SimulationResult from simulation.py.

Public functions
----------------
plot_3d(result, ...)         -- single-scenario detailed plot
plot_3d_panel(results, ...)  -- multi-scenario 2xN grid of 3-D axes

Typesetting conventions
-----------------------
- All source text (comments, docstrings, string literals) uses plain ASCII
  only.  No Unicode dash variants, arrow glyphs, subscript digits, or
  special letters appear anywhere in the source.
- Mathematical symbols rendered on plots are expressed as LaTeX strings
  passed through matplotlib's mathtext engine, e.g.:
      r"$\\Delta E \\;/\\; E_0$"   for Delta E / E_0
      r"$\\rightarrow$"             for a right-pointing arrow
      r"$\\blacklozenge$"           for the filled diamond start marker
- The helper _safe_title() uses a regex to strip any non-ASCII bytes that
  might be injected via user-supplied title strings.
"""

from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401 (registers 3d)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np
from typing import Dict, Optional, Sequence, Tuple

from simulation import SimulationResult

# ---------------------------------------------------------------------------
# Palette / style constants  (plain ASCII identifiers and hex strings only)
# ---------------------------------------------------------------------------

_BG       = "#0d0d14"    # figure / axes background
_BG_PANEL = "#111120"    # inset panel background
_EDGE     = "#333344"    # pane / spine edge colour
_TICK     = "#888899"    # tick label colour
_LABEL    = "#aaaacc"    # axis label colour
_TITLE    = "#ccccdd"    # subplot title colour

BODY_CMAPS  = ("Blues", "Oranges", "Greens")
BODY_COLORS = ("#4fc3f7", "#ffb74d", "#81c784")
BODY_LABELS = ("Star 1", "Star 2", "Interloper")

# ---------------------------------------------------------------------------
# LaTeX label constants
# All plot-facing strings that require special characters live here so they
# are easy to audit in one place.  Raw strings (r"...") are used throughout
# so that backslashes reach matplotlib's mathtext engine unescaped.
# ---------------------------------------------------------------------------

# Axis labels
_LAB_X_AU    = r"$x$  [AU]"
_LAB_Y_AU    = r"$y$  [AU]"
_LAB_Z_AU    = r"$z$  [AU]"
_LAB_X_SHORT = r"$x$"
_LAB_Y_SHORT = r"$y$"
_LAB_Z_SHORT = r"$z$"
_LAB_TIME    = r"time  [yr]"
_LAB_DELTA_E = r"$\Delta E \;/\; E_0$"
_LAB_STEP_H  = r"step size  $H$"

# Panel titles (plain ASCII -- no special characters needed)
_TITLE_TRAJ   = "Trajectories"
_TITLE_ENERGY = "Energy Conservation"
_TITLE_STEP   = "Adaptive Step Size"

# Legend annotation rendered at the bottom of the detailed figure.
# r"$\diamondsuit$"  renders the open diamond (start marker; matches "D" scatter).
# r"$\spadesuit$"    renders the spade glyph  (end marker; matches "*" scatter).
# r"$\rightarrow$"   renders a right-pointing arrow.
# \blacklozenge and \bigstar are NOT available in matplotlib's built-in mathtext
# font; \diamondsuit and \spadesuit are confirmed present.
_LEGEND_NOTE = (
    r"$\diamondsuit$ = start    "
    r"$\spadesuit$ = end    "
    r"colour fades: dim $\rightarrow$ bright = early $\rightarrow$ late"
)

# ---------------------------------------------------------------------------
# Regex for sanitising user-supplied title strings
# Matches any byte outside the printable ASCII range U+0020..U+007E.
# ---------------------------------------------------------------------------

_NON_ASCII_RE = re.compile(r"[^\x20-\x7E]")


def _safe_title(text: str) -> str:
    """
    Return *text* with all non-ASCII characters replaced by '?' and any
    run of whitespace collapsed to a single space.

    This prevents Unicode glyphs in scenario names from reaching matplotlib
    title calls, where font-fallback behaviour is unpredictable.
    """
    text = _NON_ASCII_RE.sub("?", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Internal style helpers
# ---------------------------------------------------------------------------

def _style_3d_ax(ax) -> None:
    """Apply the dark-theme style to a 3-D Axes object."""
    ax.set_facecolor(_BG)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor(_EDGE)
    ax.grid(True, color="#222230", linewidth=0.4)
    ax.tick_params(colors=_TICK, labelsize=6)
    for spine in ax.spines.values():
        spine.set_color(_EDGE)


def _style_2d_ax(ax) -> None:
    """Apply the dark-theme style to a 2-D Axes object."""
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_TICK, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(_EDGE)
    ax.yaxis.get_offset_text().set_color(_TICK)


def _gradient_line3d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cmap: str,
    lw: float = 1.5,
    alpha: float = 0.85,
) -> Line3DCollection:
    """
    Draw a 3-D polyline coloured from dim (early timesteps) to bright (late).

    Builds a Line3DCollection of single-segment pieces.  Each segment is
    assigned a colour sampled linearly from `cmap` over the interval
    [0.25, 1.0], so the trail visually fades from dark at the start to the
    full colourmap hue at the end.

    Returns the collection so the caller can add it to a legend if needed.
    """
    pts  = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = plt.get_cmap(cmap)(np.linspace(0.25, 1.0, len(segs)))
    lc   = Line3DCollection(segs, colors=cols, linewidth=lw, alpha=alpha)
    ax.add_collection(lc)
    return lc


def _clip_positions(
    result: SimulationResult,
    clip: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (positions, times) with rows removed where any body exceeds `clip`.

    Falls back to the full unclipped arrays if fewer than 3 rows would remain
    after masking, so callers always receive a plottable dataset.
    """
    pos, t = result.positions.copy(), result.times.copy()
    if clip is not None:
        mask = np.all(np.linalg.norm(pos, axis=2) < clip, axis=1)
        if mask.sum() > 2:
            pos, t = pos[mask], t[mask]
    return pos, t


def _draw_trajectories(
    ax,
    pos: np.ndarray,
    cmaps:  Sequence[str] = BODY_CMAPS,
    colors: Sequence[str] = BODY_COLORS,
    labels: Sequence[str] = BODY_LABELS,
    lw: float = 1.8,
) -> None:
    """
    Plot gradient-coloured 3-D trajectories for every body in *pos*.

    Start positions are marked with a filled diamond scatter point (marker
    code "D").  End positions are marked with a star ("*") and registered
    in the axes legend via the `label` keyword so the caller can call
    ax.legend() without further arguments.
    """
    for i, (lbl, cm, mc) in enumerate(zip(labels, cmaps, colors)):
        x, y, z = pos[:, i, 0], pos[:, i, 1], pos[:, i, 2]
        _gradient_line3d(ax, x, y, z, cm, lw=lw)
        # Start marker -- filled diamond
        ax.scatter(*[a[0]  for a in (x, y, z)],
                   s=60,  marker="D", color=mc, zorder=10,
                   edgecolors="white", linewidths=0.5)
        # End marker -- star, added to the legend
        ax.scatter(*[a[-1] for a in (x, y, z)],
                   s=120, marker="*", color=mc, zorder=10,
                   edgecolors="white", linewidths=0.5, label=lbl)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_3d(
    result:    SimulationResult,
    labels:    Sequence[str]   = BODY_LABELS,
    cmaps:     Sequence[str]   = BODY_CMAPS,
    title:     str             = "Three-Body Problem -- Bulirsch-Stoer",
    clip:      Optional[float] = None,
    elev:      float           = 25.0,
    azim:      float           = -55.0,
    save_path: str             = "three_body_3d.png",
) -> str:
    """
    Produce a detailed single-scenario figure and save it to disk.

    Layout
    ------
    Left (large panel) : 3-D trajectory plot with gradient-coloured trails.
    Top-right          : Energy conservation (Delta E / E_0) vs time.
    Bottom-right       : Adaptive step size H vs time on a log scale.

    Parameters
    ----------
    result    : SimulationResult returned by simulate().
    labels    : Body name strings; length must equal the number of bodies.
    cmaps     : Matplotlib colourmap name for each body.
    title     : Figure suptitle.  Non-ASCII characters are replaced with '?'
                by _safe_title() before the string reaches matplotlib.
    clip      : Spatial clip radius in AU.  Timestep rows where any body
                exceeds this distance are dropped before plotting, preventing
                an ejected body from collapsing the visible axis range.
    elev      : Elevation angle of the initial 3-D viewpoint in degrees.
    azim      : Azimuth  angle of the initial 3-D viewpoint in degrees.
    save_path : Destination file path (.png recommended).

    Returns
    -------
    save_path : The path that was written, unchanged.
    """
    pos, _ = _clip_positions(result, clip)

    fig = plt.figure(figsize=(14, 10), facecolor=_BG)
    fig.suptitle(_safe_title(title), color="white", fontsize=14, y=0.97,
                 fontfamily="monospace")

    # -- 3-D trajectory panel -------------------------------------------------
    ax3 = fig.add_axes([0.03, 0.18, 0.60, 0.78], projection="3d")
    _style_3d_ax(ax3)
    _draw_trajectories(ax3, pos, cmaps=cmaps, labels=labels)

    ax3.set_xlabel(_LAB_X_AU, color=_LABEL, labelpad=6, fontsize=8)
    ax3.set_ylabel(_LAB_Y_AU, color=_LABEL, labelpad=6, fontsize=8)
    ax3.set_zlabel(_LAB_Z_AU, color=_LABEL, labelpad=6, fontsize=8)
    ax3.set_title(_TITLE_TRAJ, color=_TITLE, fontsize=10, pad=4)
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.25,
               labelcolor="white", facecolor=_BG_PANEL)
    ax3.view_init(elev=elev, azim=azim)

    # -- Energy conservation panel --------------------------------------------
    ax_e = fig.add_axes([0.68, 0.55, 0.29, 0.36])
    _style_2d_ax(ax_e)
    E0 = result.energies[0]
    dE = (result.energies - E0) / abs(E0)
    ax_e.plot(result.times, dE, color="#cf6679", linewidth=1.2)
    ax_e.axhline(0, color="#555566", linewidth=0.8, linestyle="--")
    ax_e.set_xlabel(_LAB_TIME,    color=_LABEL, fontsize=8)
    ax_e.set_ylabel(_LAB_DELTA_E, color=_LABEL, fontsize=8)
    ax_e.set_title(_TITLE_ENERGY, color=_TITLE,  fontsize=9)

    # -- Adaptive step-size panel ---------------------------------------------
    ax_h = fig.add_axes([0.68, 0.10, 0.29, 0.36])
    _style_2d_ax(ax_h)
    ax_h.semilogy(result.times[1:], result.step_sizes[1:],
                  color="#7ec8e3", linewidth=0.9, alpha=0.8)
    ax_h.set_xlabel(_LAB_TIME,   color=_LABEL, fontsize=8)
    ax_h.set_ylabel(_LAB_STEP_H, color=_LABEL, fontsize=8)
    ax_h.set_title(_TITLE_STEP,  color=_TITLE,  fontsize=9)

    # -- Legend annotation at figure bottom -----------------------------------
    fig.text(0.03, 0.08, _LEGEND_NOTE,
             color="#666688", fontsize=8, fontfamily="monospace")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


def plot_3d_panel(
    results:   Dict[str, SimulationResult],
    suptitle:  str             = "Three-Body Problem -- Bulirsch-Stoer",
    cmaps:     Sequence[str]   = BODY_CMAPS,
    labels:    Sequence[str]   = BODY_LABELS,
    clip:      Optional[float] = 50.0,
    ncols:     int             = 2,
    elev:      float           = 22.0,
    azim:      float           = -50.0,
    save_path: str             = "three_body_panel.png",
) -> str:
    """
    Produce a multi-scenario grid of 3-D trajectory subplots.

    Parameters
    ----------
    results   : Mapping of scenario name (str) to SimulationResult.
                Panel order follows the dict insertion order.
    suptitle  : Figure-level title.  Non-ASCII characters are sanitised.
    cmaps     : Matplotlib colourmap name per body, shared across all panels.
    labels    : Body label strings, shared across all panels.
    clip      : Spatial clip radius in AU applied to each panel independently.
    ncols     : Number of columns in the subplot grid.
    elev      : Shared elevation angle for all 3-D viewpoints (degrees).
    azim      : Shared azimuth  angle for all 3-D viewpoints (degrees).
    save_path : Destination file path (.png recommended).

    Returns
    -------
    save_path : The path that was written, unchanged.
    """
    names = list(results.keys())
    nrows = (len(names) + ncols - 1) // ncols

    fig = plt.figure(figsize=(9 * ncols, 7 * nrows), facecolor=_BG)
    fig.suptitle(_safe_title(suptitle), color="white", fontsize=13,
                 fontfamily="monospace", y=0.99)

    for idx, name in enumerate(names):
        result = results[name]
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        _style_3d_ax(ax)

        pos, _ = _clip_positions(result, clip)
        _draw_trajectories(ax, pos, cmaps=cmaps, labels=labels, lw=1.4)

        ax.set_xlabel(_LAB_X_SHORT, color=_TICK, fontsize=7, labelpad=3)
        ax.set_ylabel(_LAB_Y_SHORT, color=_TICK, fontsize=7, labelpad=3)
        ax.set_zlabel(_LAB_Z_SHORT, color=_TICK, fontsize=7, labelpad=3)
        ax.set_title(_safe_title(name), color="#ddddee", fontsize=10, pad=4)
        ax.legend(fontsize=7, framealpha=0.2, labelcolor="white",
                  facecolor=_BG_PANEL, loc="upper left")
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path