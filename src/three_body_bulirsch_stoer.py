"""
Three-Body Problem — Bulirsch-Stoer Integrator (generalised eccentricity)
=========================================================================
Supports ALL eccentricity regimes for the incoming body:
  e = 0          →  circular approach orbit
  0 < e < 1      →  elliptic (bound)  approach
  e = 1          →  parabolic (escape speed exactly)
  e > 1          →  hyperbolic flyby

The binary itself can also be given an arbitrary eccentricity (default 0,
circular).  Initial positions are placed at the correct point on each conic
section via true-anomaly inversion from the starting distance.

Physics:
  G = 1,  units: AU / yr / M_sun  (normalised)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from dataclasses import dataclass
from typing import List, Optional, Tuple

G = 1.0

# ── Keplerian helpers ────────────────────────────────────────────────────────

def keplerian_state(M_central: float, a: float, e: float,
                    inc: float, omega: float, f: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return position & velocity vectors in 3-D for a body on a Keplerian conic.

    Parameters
    ----------
    M_central : mass of the central body (for vis-viva)
    a         : semi-major axis  (< 0 for hyperbola, use |a| = r_peri/(e-1))
    e         : eccentricity  (≥ 0; e=1 → a must be ∞, pass np.inf)
    inc       : inclination  (radians, tilt of orbital plane)
    omega     : argument of periapsis (radians, in-plane rotation)
    f         : true anomaly at the desired position (radians)

    Returns
    -------
    r_vec, v_vec  in the frame where the reference plane is x-y
    """
    if e == 1.0:                            # parabolic  p = 2 * r_peri
        p = -a * 2.0                        # here 'a' encodes r_peri as -a
        r = p / (1.0 + np.cos(f))
    else:
        p = a * (1.0 - e**2)               # semi-latus rectum (works e<1 & e>1)
        r = p / (1.0 + e * np.cos(f))

    # Position in orbital plane
    x_orb = r * np.cos(f)
    y_orb = r * np.sin(f)

    # Velocity in orbital plane (vis-viva based)
    mu = G * M_central
    if e == 1.0:
        vr   =  np.sqrt(mu / p) * np.sin(f)
        vt   =  np.sqrt(mu / p) * (1.0 + np.cos(f))
    else:
        vr   =  np.sqrt(mu / p) * e * np.sin(f)
        vt   =  np.sqrt(mu / p) * (1.0 + e * np.cos(f))

    vx_orb = vr * np.cos(f) - vt * np.sin(f)
    vy_orb = vr * np.sin(f) + vt * np.cos(f)

    # Rotate: argument of periapsis (omega) then inclination (inc) around x-axis
    cos_o, sin_o = np.cos(omega), np.sin(omega)
    cos_i, sin_i = np.cos(inc),   np.sin(inc)

    def rot(x, y):
        # rotate by omega in orbital plane then tilt by inc
        xr =  cos_o * x - sin_o * y
        yr =  sin_o * x + cos_o * y
        return np.array([xr,
                         yr * cos_i,
                         yr * sin_i])

    pos = rot(x_orb, y_orb)
    vel = rot(vx_orb, vy_orb)
    return pos, vel


def true_anomaly_from_distance(r: float, a: float, e: float,
                                approaching: bool = True) -> float:
    """
    Solve r = a(1-e²)/(1+e·cos f) for f.

    For e=1 (parabolic), a is interpreted as -r_peri (see keplerian_state).
    Returns the true anomaly on the approaching leg if `approaching=True`,
    i.e. f < 0 (body moving toward periapsis).
    """
    if e == 1.0:
        r_peri = -a
        p = 2.0 * r_peri
        cos_f = p / r - 1.0
        cos_f = np.clip(cos_f, -1.0, 1.0)
        f = np.arccos(cos_f)
        return -f if approaching else f

    p = a * (1.0 - e**2)
    cos_f = (p / r - 1.0) / e
    cos_f = np.clip(cos_f, -1.0, 1.0)
    f = np.arccos(cos_f)
    return -f if approaching else f  # negative → approaching periapsis


# ── Body / Result containers ─────────────────────────────────────────────────

@dataclass
class Body:
    mass: float
    pos:  np.ndarray   # (3,)
    vel:  np.ndarray   # (3,)

    def __repr__(self):
        p, v = self.pos, self.vel
        return (f"Body(m={self.mass:.3f}  "
                f"pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]  "
                f"vel=[{v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f}])")


@dataclass
class SimulationResult:
    times:      np.ndarray    # (N,)
    positions:  np.ndarray    # (N, 3, 3)
    velocities: np.ndarray    # (N, 3, 3)
    energies:   np.ndarray    # (N,)
    step_sizes: np.ndarray    # (N,)


# ── ODE machinery ────────────────────────────────────────────────────────────

def flatten(bodies: List[Body]) -> np.ndarray:
    return np.concatenate([np.r_[b.pos, b.vel] for b in bodies])


def derivatives(y: np.ndarray, masses: np.ndarray,
                softening: float = 1e-5) -> np.ndarray:
    n    = len(masses)
    dydt = np.zeros_like(y)
    for i in range(n):
        dydt[6*i:6*i+3] = y[6*i+3:6*i+6]
        ri  = y[6*i:6*i+3]
        acc = np.zeros(3)
        for j in range(n):
            if j == i: continue
            dr   = y[6*j:6*j+3] - ri
            dist = np.sqrt(dr @ dr + softening**2)
            acc += G * masses[j] / dist**3 * dr
        dydt[6*i+3:6*i+6] = acc
    return dydt


def total_energy(y: np.ndarray, masses: np.ndarray,
                 softening: float = 1e-5) -> float:
    n  = len(masses)
    E  = 0.0
    for i in range(n):
        E += 0.5 * masses[i] * (y[6*i+3:6*i+6] @ y[6*i+3:6*i+6])
        for j in range(i+1, n):
            dr = y[6*j:6*j+3] - y[6*i:6*i+3]
            r  = np.sqrt(dr @ dr + softening**2)
            E -= G * masses[i] * masses[j] / r
    return E


# ── Modified midpoint (Gragg) ────────────────────────────────────────────────

def modified_midpoint(y, dydt, masses, H, n_sub):
    h  = H / n_sub
    z0 = y.copy()
    z1 = y + h * dydt
    for _ in range(n_sub - 1):
        d  = derivatives(z1, masses)
        z0, z1 = z1, z0 + 2.0*h*d
    d_end = derivatives(z1, masses)
    return 0.5 * (z1 + z0 + h * d_end)


# ── Bulirsch-Stoer step ──────────────────────────────────────────────────────

BS_SEQ = [2, 4, 6, 8, 10, 12, 14, 16, 18]

def bs_step(y, masses, H, tol):
    dydt = derivatives(y, masses)
    T    = [None] * len(BS_SEQ)
    err  = np.inf
    for k, n_sub in enumerate(BS_SEQ):
        T[k] = modified_midpoint(y, dydt, masses, H, n_sub)
        for j in range(k-1, -1, -1):
            ratio = (BS_SEQ[k] / BS_SEQ[j])**2
            T[j]  = T[j+1] + (T[j+1] - T[j]) / (ratio - 1.0)
        if k >= 1:
            delta = T[0] - T[1]
            scale = np.maximum(np.abs(T[0]), np.abs(y)) + 1e-30
            err   = np.sqrt(np.mean((delta / (scale * tol))**2))
            if err <= 1.0:
                exp    = 1.0 / (2*k + 1)
                H_next = H * 0.9 * (1.0 / err)**exp
                H_next = np.clip(H_next, 0.1*H, 5.0*H)
                return T[0], H_next, err
    return T[0], H * 0.25, err


# ── Initial conditions (fully generalised eccentricity) ─────────────────────

def initial_conditions(
    m1: float = 1.0,
    m2: float = 0.8,
    m3: float = 0.3,
    # Binary orbit
    a_bin: float = 1.0,
    e_bin: float = 0.0,          # binary eccentricity  0 → circular
    inc_bin_deg: float = 30.0,   # tilt of binary plane
    omega_bin_deg: float = 0.0,  # argument of periapsis of binary
    f_bin_deg: float = 0.0,      # true anomaly of binary at t=0
    # Incoming body
    e3: float = 0.85,            # ANY eccentricity ≥ 0
    r_peri3: float = 2.0,        # periapsis distance of m3 orbit
    r3_init: float = 10.0,       # starting distance of m3
    inc3_deg: float = 0.0,       # inclination of m3 approach plane
    omega3_deg: float = 180.0,   # argument of periapsis of m3 orbit
) -> List[Body]:
    """
    Build 3-body initial conditions supporting arbitrary eccentricities.

    Binary eccentricity  e_bin:
        0            → circular
        (0,1)        → elliptic
        1            → parabolic  (r_peri → a_bin used as periapsis)
        >1           → hyperbolic (e.g. unbound binary — unusual but valid)

    Third-body eccentricity  e3:
        0            → circular approach (r3_init = radius = constant)
        (0,1)        → elliptic, bound to the binary system
        1            → parabolic (escape-speed approach)
        >1           → hyperbolic flyby  (most realistic for interlopers)
    """
    inc_bin = np.radians(inc_bin_deg)
    omega_b = np.radians(omega_bin_deg)
    f_b     = np.radians(f_bin_deg)
    inc3    = np.radians(inc3_deg)
    omega3  = np.radians(omega3_deg)

    M_bin   = m1 + m2
    M_tot   = m1 + m2 + m3

    # ── Binary ──────────────────────────────────────────────────────────────
    # Reduced-mass two-body: place m1 and m2 symmetrically around their CoM.
    # The "central mass" for the reduced-mass orbit is M_bin.
    if e_bin == 1.0:
        # Parabolic binary — a encodes -r_peri
        a_b_eff = -a_bin
    elif e_bin > 1.0:
        # Hyperbolic: a_bin is periapsis distance, so |a| = a_bin/(e_bin-1)
        a_b_eff = -a_bin / (e_bin - 1.0)
    else:
        a_b_eff = a_bin   # elliptic / circular

    # Relative orbit (m2 around m1 in CoM frame)
    pos_rel, vel_rel = keplerian_state(M_bin, a_b_eff, e_bin,
                                       inc_bin, omega_b, f_b)

    # Split into individual positions/velocities
    mu1 = m1 / M_bin
    mu2 = m2 / M_bin

    pos1 =  mu2 * pos_rel
    pos2 = -mu1 * pos_rel
    vel1 =  mu2 * vel_rel
    vel2 = -mu1 * vel_rel

    # ── Third body ──────────────────────────────────────────────────────────
    if e3 == 0.0:
        # Circular: r3_init is the orbital radius; place at (−r3_init, 0, 0)
        v3_mag = np.sqrt(G * M_tot / r3_init)
        pos3 = np.array([-r3_init, 0.0, 0.0])
        # velocity perpendicular in the approach plane
        vel3 = v3_mag * np.array([0.0,
                                   np.cos(inc3),
                                   np.sin(inc3)])
    else:
        if e3 == 1.0:
            # Parabolic: a encodes -r_peri  (see keplerian_state)
            a3_eff = -r_peri3
        elif e3 > 1.0:
            # Hyperbolic: a = -r_peri / (e3 - 1)  (negative semi-major axis)
            a3_eff = -r_peri3 / (e3 - 1.0)
        else:
            # Elliptic: a = r_peri / (1 - e3)
            a3_eff = r_peri3 / (1.0 - e3)

        # True anomaly at starting distance r3_init (approaching leg → f < 0)
        f3 = true_anomaly_from_distance(r3_init, a3_eff, e3, approaching=True)
        pos3, vel3 = keplerian_state(M_tot, a3_eff, e3, inc3, omega3, f3)

    bodies = [
        Body(mass=m1, pos=pos1, vel=vel1),
        Body(mass=m2, pos=pos2, vel=vel2),
        Body(mass=m3, pos=pos3, vel=vel3),
    ]

    # ── Centre-of-mass correction ────────────────────────────────────────────
    M = sum(b.mass for b in bodies)
    com_pos = sum(b.mass * b.pos for b in bodies) / M
    com_vel = sum(b.mass * b.vel for b in bodies) / M
    for b in bodies:
        b.pos -= com_pos
        b.vel -= com_vel

    return bodies


# ── Main integrator ──────────────────────────────────────────────────────────

def simulate(bodies: List[Body],
             t_end: float = 15.0,
             H_init: float = 0.02,
             tol: float = 1e-9,
             max_steps: int = 300_000,
             clip_radius: Optional[float] = None,
             verbose: bool = True) -> SimulationResult:
    """
    Integrate N-body equations using the Bulirsch-Stoer algorithm.

    Parameters
    ----------
    bodies       : initial conditions
    t_end        : end time
    H_init       : initial macro-step
    tol          : relative error tolerance
    max_steps    : safety cap
    clip_radius  : stop if any body exceeds this distance (ejection guard)
    verbose      : print progress
    """
    masses = np.array([b.mass for b in bodies])
    y      = flatten(bodies)
    t, H   = 0.0, H_init

    ts, ys, Es, hs = [t], [y.copy()], [total_energy(y, masses)], [0.0]

    for step in range(max_steps):
        H = min(H, t_end - t)
        if H <= 0:
            break

        y_new, H_next, err = bs_step(y, masses, H, tol)

        t += H
        y  = y_new
        H  = H_next

        ts.append(t)
        ys.append(y.copy())
        Es.append(total_energy(y, masses))
        hs.append(H)

        if verbose and (step + 1) % 1000 == 0:
            dE = (Es[-1] - Es[0]) / abs(Es[0])
            print(f"  t={t:.3f}  step={step+1}  H={H:.2e}  ΔE/E={dE:.2e}")

        # Ejection guard
        if clip_radius is not None:
            n_b = len(masses)
            for i in range(n_b):
                if np.linalg.norm(y[6*i:6*i+3]) > clip_radius:
                    if verbose:
                        print(f"  Body {i} ejected at t={t:.4f}. Stopping.")
                    break
            else:
                continue
            break

    N   = len(ts)
    nb  = len(bodies)
    pos = np.zeros((N, nb, 3))
    vel = np.zeros((N, nb, 3))
    for i, state in enumerate(ys):
        for b in range(nb):
            pos[i, b] = state[6*b:6*b+3]
            vel[i, b] = state[6*b+3:6*b+6]

    if verbose:
        E0, Ef = Es[0], Es[-1]
        print(f"\n  Done — {N} snapshots, t_final={ts[-1]:.4f}")
        print(f"  Energy conservation ΔE/E = {(Ef-E0)/abs(E0):.2e}\n")

    return SimulationResult(
        times      = np.array(ts),
        positions  = pos,
        velocities = vel,
        energies   = np.array(Es),
        step_sizes = np.array(hs),
    )


# ── 3-D plotting ─────────────────────────────────────────────────────────────

def _gradient_line3d(ax, x, y, z, cmap, lw=1.5, alpha=0.85):
    """Draw a 3-D line with colour fading from dark (start) to bright (end)."""
    pts   = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
    N     = len(segs)
    cols  = plt.get_cmap(cmap)(np.linspace(0.25, 1.0, N))
    lc    = Line3DCollection(segs, colors=cols, linewidth=lw, alpha=alpha)
    ax.add_collection(lc)
    return lc


def plot_3d(result: SimulationResult,
            labels=("Star 1", "Star 2", "Interloper"),
            cmaps=("Blues", "Oranges", "Greens"),
            title: str = "Three-Body Problem — Bulirsch-Stoer",
            clip: Optional[float] = None,
            save_path: str = "three_body_3d.png") -> str:
    """
    Plot 3-D trajectories with gradient colouring (faint→bright = past→present),
    start markers (◆) and end markers (★), plus energy conservation panel.

    Parameters
    ----------
    result    : SimulationResult
    labels    : body name strings
    cmaps     : matplotlib colourmap names per body
    clip      : if set, trim positions beyond this radius for display
    save_path : output PNG path

    Returns
    -------
    save_path
    """
    pos = result.positions.copy()   # (N, 3, 3)
    t   = result.times

    # Optional spatial clip (avoids ejected-body ruining the axes)
    if clip is not None:
        mask = np.all(np.linalg.norm(pos, axis=2) < clip, axis=1)
        if mask.sum() > 2:
            pos = pos[mask]
            t   = t[mask]

    fig = plt.figure(figsize=(14, 10), facecolor="#0d0d14")
    fig.suptitle(title, color="white", fontsize=14, y=0.97,
                 fontfamily="monospace")

    # ── 3-D trajectory panel ────────────────────────────────────────────────
    ax3 = fig.add_axes([0.03, 0.18, 0.60, 0.78], projection="3d")
    ax3.set_facecolor("#0d0d14")
    ax3.xaxis.pane.fill = ax3.yaxis.pane.fill = ax3.zaxis.pane.fill = False
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_edgecolor("#333344")
    ax3.grid(True, color="#222230", linewidth=0.5)
    ax3.tick_params(colors="#888899", labelsize=7)
    for spine in ax3.spines.values():
        spine.set_color("#333344")

    marker_colors = ["#4fc3f7", "#ffb74d", "#81c784"]
    for i, (lbl, cm, mc) in enumerate(zip(labels, cmaps, marker_colors)):
        x, y, z = pos[:, i, 0], pos[:, i, 1], pos[:, i, 2]
        _gradient_line3d(ax3, x, y, z, cm, lw=1.8)
        # Start marker
        ax3.scatter(*[arr[0] for arr in (x, y, z)],
                    s=60, marker="D", color=mc, zorder=10,
                    edgecolors="white", linewidths=0.5)
        # End marker
        ax3.scatter(*[arr[-1] for arr in (x, y, z)],
                    s=120, marker="*", color=mc, zorder=10,
                    edgecolors="white", linewidths=0.5, label=lbl)

    ax3.set_xlabel("x  [AU]", color="#aaaacc", labelpad=6, fontsize=8)
    ax3.set_ylabel("y  [AU]", color="#aaaacc", labelpad=6, fontsize=8)
    ax3.set_zlabel("z  [AU]", color="#aaaacc", labelpad=6, fontsize=8)
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.25,
               labelcolor="white", facecolor="#111120")
    ax3.set_title("Trajectories", color="#ccccdd", fontsize=10, pad=4)
    ax3.view_init(elev=25, azim=-55)

    # ── Energy conservation panel ────────────────────────────────────────────
    ax_e = fig.add_axes([0.68, 0.55, 0.29, 0.36])
    ax_e.set_facecolor("#111120")
    E0   = result.energies[0]
    dE   = (result.energies - E0) / abs(E0)
    ax_e.plot(result.times, dE, color="#cf6679", linewidth=1.2)
    ax_e.axhline(0, color="#555566", linewidth=0.8, linestyle="--")
    ax_e.set_xlabel("time  [yr]", color="#aaaacc", fontsize=8)
    ax_e.set_ylabel("ΔE / E₀", color="#aaaacc", fontsize=8)
    ax_e.set_title("Energy Conservation", color="#ccccdd", fontsize=9)
    ax_e.tick_params(colors="#888899", labelsize=7)
    for sp in ax_e.spines.values():
        sp.set_edgecolor("#333344")
    ax_e.yaxis.get_offset_text().set_color("#888899")

    # ── Step-size panel ──────────────────────────────────────────────────────
    ax_h = fig.add_axes([0.68, 0.10, 0.29, 0.36])
    ax_h.set_facecolor("#111120")
    ax_h.semilogy(result.times[1:], result.step_sizes[1:],
                  color="#7ec8e3", linewidth=0.9, alpha=0.8)
    ax_h.set_xlabel("time  [yr]", color="#aaaacc", fontsize=8)
    ax_h.set_ylabel("step size  H", color="#aaaacc", fontsize=8)
    ax_h.set_title("Adaptive Step Size", color="#ccccdd", fontsize=9)
    ax_h.tick_params(colors="#888899", labelsize=7)
    for sp in ax_h.spines.values():
        sp.set_edgecolor("#333344")

    # ── Eccentricity / info text ─────────────────────────────────────────────
    fig.text(0.03, 0.08,
             "◆ = start    ★ = end    colour fades: dim → bright = early → late",
             color="#666688", fontsize=8, fontfamily="monospace")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


# ── Demo scenarios (one per eccentricity regime) ─────────────────────────────

def run_scenario(name: str, e3: float, t_end: float,
                 extra_kwargs: dict = {}) -> SimulationResult:
    print(f"\n{'='*60}")
    print(f"  Scenario: {name}  (e3 = {e3})")
    print(f"{'='*60}")

    bodies = initial_conditions(
        m1=1.0, m2=0.8, m3=0.3,
        a_bin=1.0, e_bin=0.0,
        inc_bin_deg=30.0,
        e3=e3,
        r_peri3=2.0,
        r3_init=10.0,
        inc3_deg=15.0,
        omega3_deg=175.0,
        **extra_kwargs
    )
    for i, b in enumerate(bodies):
        print(f"  body {i}: {b}")

    return simulate(bodies, t_end=t_end, H_init=0.01, tol=1e-9,
                    clip_radius=200.0, verbose=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Run the four canonical eccentricity cases ────────────────────────────
    scenarios = [
        ("Circular   e=0.0",  0.0,  12.0),
        ("Elliptic   e=0.5",  0.5,  15.0),
        ("Parabolic  e=1.0",  1.0,  12.0),
        ("Hyperbolic e=1.6",  1.6,  10.0),
    ]

    results = {}
    for name, e3, t_end in scenarios:
        results[name] = run_scenario(name, e3, t_end)

    # ── Individual plots ─────────────────────────────────────────────────────
    out_paths = []
    for (name, e3, _), result in zip(scenarios, results.values()):
        label  = name.split()[0].lower()
        e_str  = f"e={e3}"
        path   = f"/mnt/user-data/outputs/three_body_{label}.png"
        plot_3d(result,
                title=f"Three-Body Bulirsch-Stoer  |  {name}",
                clip=50.0,
                save_path=path)
        print(f"  Saved: {path}")
        out_paths.append(path)

    # ── Combined 4-panel plot ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor="#0d0d14")
    fig.suptitle(
        "Three-Body Problem — Bulirsch-Stoer  |  All Eccentricity Regimes",
        color="white", fontsize=13, fontfamily="monospace", y=0.99)

    panel_positions = [(0,0), (0,1), (1,0), (1,1)]
    cmaps  = ("Blues", "Oranges", "Greens")
    mcolors = ["#4fc3f7", "#ffb74d", "#81c784"]
    labels = ("Star 1", "Star 2", "Interloper")

    for idx, ((name, e3, _), result) in enumerate(zip(scenarios, results.values())):
        row, col = panel_positions[idx]
        ax = fig.add_subplot(2, 2, idx+1, projection="3d")
        ax.set_facecolor("#0d0d14")
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("#2a2a3a")
        ax.grid(True, color="#1e1e2e", linewidth=0.4)
        ax.tick_params(colors="#666688", labelsize=6)

        pos = result.positions.copy()
        # Clip display to 50 AU
        clip = 50.0
        mask = np.all(np.linalg.norm(pos, axis=2) < clip, axis=1)
        if mask.sum() > 2:
            pos = pos[mask]

        for i, (lbl, cm, mc) in enumerate(zip(labels, cmaps, mcolors)):
            x, y, z = pos[:,i,0], pos[:,i,1], pos[:,i,2]
            _gradient_line3d(ax, x, y, z, cm, lw=1.4)
            ax.scatter(*[arr[0] for arr in (x,y,z)], s=30, marker="D",
                       color=mc, zorder=8, edgecolors="white", linewidths=0.4)
            ax.scatter(*[arr[-1] for arr in (x,y,z)], s=70, marker="*",
                       color=mc, zorder=8, edgecolors="white", linewidths=0.4,
                       label=lbl)

        ax.set_xlabel("x", color="#888899", fontsize=7, labelpad=3)
        ax.set_ylabel("y", color="#888899", fontsize=7, labelpad=3)
        ax.set_zlabel("z", color="#888899", fontsize=7, labelpad=3)
        ax.set_title(name, color="#ddddee", fontsize=10, pad=4)
        ax.legend(fontsize=7, framealpha=0.2, labelcolor="white",
                  facecolor="#111120", loc="upper left")
        ax.view_init(elev=22, azim=-50)

    combined_path = "/mnt/user-data/outputs/three_body_all_eccentricities.png"
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(combined_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Combined panel saved: {combined_path}")

    # ── Also save the updated Python module ─────────────────────────────────
    import shutil
    shutil.copy(__file__, "/mnt/user-data/outputs/three_body_bulirsch_stoer.py")
    print("  Updated module saved.")
