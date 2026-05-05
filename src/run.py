"""
run.py - Three-Body Simulation Runner
======================================
Orchestrates the four canonical eccentricity scenarios, delegates all
physics to simulation.py and all rendering to plotting.py.

Usage
-----
    python run.py

Output
------
  three_body_circular.png
  three_body_elliptic.png
  three_body_parabolic.png
  three_body_hyperbolic.png
  three_body_all_eccentricities.png
"""


from __future__ import annotations
import os
from typing import Dict
from simulation import SimulationResult, initial_conditions, simulate
from plotting   import plot_3d, plot_3d_panel

# Output directory

OUT = os.getcwd() + "/outputs"

# Scenario definitions
#
# Each entry:  (display_name, file_stem, e3, t_end)
#
# The shared physical parameters below are the same for all scenarios so that
# the plots are directly comparable - only the third-body eccentricity differs.

_SHARED = dict(
    m1          = 1.0,
    m2          = 0.8,
    m3          = 0.3,
    a_bin       = 1.0,
    e_bin       = 0.0,      # circular binary
    inc_bin_deg = 30.0,     # binary plane tilted 30° off the x-y plane
    r_peri3     = 2.0,      # periapsis of the interloper orbit [AU]
    r3_init     = 10.0,     # starting distance of the interloper [AU]
    inc3_deg    = 15.0,     # inclination of the interloper approach
    omega3_deg  = 175.0,    # argument of periapsis of the interloper
)

SCENARIOS = [
    ("Circular   e=0.0", "circular",   0.0,  12.0),
    ("Elliptic   e=0.5", "elliptic",   0.5,  15.0),
    ("Parabolic  e=1.0", "parabolic",  1.0,  12.0),
    ("Hyperbolic e=1.6", "hyperbolic", 1.6,  10.0),
]

# Runner helpers 

def run_scenario(name: str, e3: float, t_end: float) -> SimulationResult:
    """Build ICs, integrate, and return the result for one scenario."""
    print(f"\n{'='*60}")
    print(f"  {name}  (e3 = {e3})")
    print(f"{'='*60}")

    bodies = initial_conditions(e3=e3, **_SHARED)
    for i, b in enumerate(bodies):
        print(f"  body {i}: {b}")

    return simulate(
        bodies,
        t_end       = t_end,
        H_init      = 0.01,
        tol         = 1e-9,
        clip_radius = 200.0,
        verbose     = True,
    )


def run_all() -> Dict[str, SimulationResult]:
    """Run every scenario and return a name - result mapping."""
    results: Dict[str, SimulationResult] = {}
    for name, stem, e3, t_end in SCENARIOS:
        results[name] = run_scenario(name, e3, t_end)
    return results


#  Entry point 

if __name__ == "__main__":

    results = run_all()

    # Individual detailed plots 
    for name, stem, *_ in SCENARIOS:
        path = f"{OUT}/three_body_{stem}.png"
        plot_3d(
            results[name],
            title     = f"Three-Body Bulirsch-Stoer  |  {name}",
            clip      = 50.0,
            save_path = path,
        )
        print(f"  Saved: {path}")

    #  Combined 4-panel overview 
    panel_path = f"{OUT}/three_body_all_eccentricities.png"
    plot_3d_panel(
        results,
        suptitle  = "Three-Body Problem — Bulirsch-Stoer  |  All Eccentricity Regimes",
        clip      = 50.0,
        ncols     = 2,
        save_path = panel_path,
    )
    print(f"  Saved: {panel_path}")
