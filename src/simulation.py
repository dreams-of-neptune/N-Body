"""
simulation.py -- Three-Body Physics Engine
===========================================
Provides:
  Body                 -- point-mass data class
  SimulationResult     -- output container (times, positions, velocities, ...)
  initial_conditions() -- Keplerian ICs for all eccentricity regimes
  simulate()           -- dispatcher: choose integrator via method=

Three integrators are available, each exposing the same SimulationResult
interface so they can be swapped without changing any downstream code.

  method="bs"        Bulirsch-Stoer with adaptive step control.
                     Highest accuracy per function evaluation.
                     Best for long integrations and close encounters.
                     Order: effectively > 10 (extrapolated).

  method="rk4"       Classical 4th-order Runge-Kutta, fixed step.
                     Familiar, robust, simple to reason about.
                     Dissipates energy slowly over long runs (not symplectic).
                     Order: 4.

  method="leapfrog"  Kick-Drift-Kick (Verlet) leapfrog, fixed step.
                     Symplectic: conserves a shadow Hamiltonian exactly,
                     so energy error stays bounded (does not drift) for any
                     integration length at fixed step size.
                     Best for long-duration statistical studies.
                     Order: 2 (but superior long-term energy behaviour vs RK4).

Units:  G = 1,  AU / yr / M_sun  (normalised)

Eccentricity conventions (incoming body e3):
  e = 0        : circular orbit at radius r3_init
  0 < e < 1    : elliptic  (bound to the binary)
  e = 1        : parabolic (exactly escape speed)
  e > 1        : hyperbolic flyby
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

G = 1.0  # gravitational constant in normalised units

# Type alias for the method selector
Method = Literal["bs", "rk4", "leapfrog"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Body:
    """A gravitating point mass."""
    mass: float
    pos:  np.ndarray   # shape (3,)
    vel:  np.ndarray   # shape (3,)

    def __repr__(self) -> str:
        p, v = self.pos, self.vel
        return (f"Body(m={self.mass:.3f}  "
                f"pos=[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]  "
                f"vel=[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}])")


@dataclass
class SimulationResult:
    """Container for integrator output."""
    times:      np.ndarray   # (N,)      -- time at each snapshot
    positions:  np.ndarray   # (N, B, 3) -- positions  of B bodies
    velocities: np.ndarray   # (N, B, 3) -- velocities of B bodies
    energies:   np.ndarray   # (N,)      -- total mechanical energy
    step_sizes: np.ndarray   # (N,)      -- step sizes used (constant for fixed-step methods)
    method:     str          # integrator name for labelling / comparison plots


# ---------------------------------------------------------------------------
# Keplerian helpers
# ---------------------------------------------------------------------------

def keplerian_state(
    M_central: float,
    a: float,
    e: float,
    inc: float,
    omega: float,
    f: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Position and velocity on a Keplerian conic section.

    Parameters
    ----------
    M_central : mass of the attracting body (sets the velocity scale via mu = G*M)
    a         : semi-major axis.
                  e < 1  -- positive (ellipse / circle)
                  e > 1  -- negative, equals  -r_peri / (e - 1)  (hyperbola)
                  e = 1  -- pass -r_peri directly  (parabolic encoding)
    e         : eccentricity (>= 0)
    inc       : inclination in radians -- tilt of orbital plane around the x-axis
    omega     : argument of periapsis in radians -- in-plane rotation
    f         : true anomaly in radians at the desired point on the orbit

    Returns
    -------
    pos, vel  : 3-vectors in the reference frame whose reference plane is x-y
    """
    mu = G * M_central

    if e == 1.0:
        # Parabolic: a encodes -r_peri, so semi-latus rectum p = 2 * r_peri = -2a
        p  = -a * 2.0
        r  = p / (1.0 + np.cos(f))
        vr = np.sqrt(mu / p) * np.sin(f)
        vt = np.sqrt(mu / p) * (1.0 + np.cos(f))
    else:
        # Elliptic (e < 1) and hyperbolic (e > 1): semi-latus rectum p = a(1 - e^2)
        p  = a * (1.0 - e**2)
        r  = p / (1.0 + e * np.cos(f))
        vr = np.sqrt(mu / p) * e * np.sin(f)
        vt = np.sqrt(mu / p) * (1.0 + e * np.cos(f))

    # Coordinates in the orbital plane
    x_o  =  r  * np.cos(f)
    y_o  =  r  * np.sin(f)
    vx_o =  vr * np.cos(f) - vt * np.sin(f)
    vy_o =  vr * np.sin(f) + vt * np.cos(f)

    # Rotate: first by omega (argument of periapsis), then tilt by inc around x-axis
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(inc),   np.sin(inc)

    def _rotate(x: float, y: float) -> np.ndarray:
        xr = co * x - so * y
        yr = so * x + co * y
        return np.array([xr, yr * ci, yr * si])

    return _rotate(x_o, y_o), _rotate(vx_o, vy_o)


def true_anomaly_from_distance(
    r: float,
    a: float,
    e: float,
    approaching: bool = True,
) -> float:
    """
    Invert  r = p / (1 + e * cos(f))  to obtain the true anomaly f.

    For e = 1, `a` must equal -r_peri (parabolic encoding, matching
    keplerian_state).  Set approaching=True to place the body on the inbound
    leg, i.e. f < 0 (body moving toward periapsis).
    """
    if e == 1.0:
        p     = -a * 2.0
        cos_f = np.clip(p / r - 1.0, -1.0, 1.0)
    else:
        p     = a * (1.0 - e**2)
        cos_f = np.clip((p / r - 1.0) / e, -1.0, 1.0)

    f = np.arccos(cos_f)
    return -f if approaching else f


# ---------------------------------------------------------------------------
# State vector helpers (shared by all integrators)
# ---------------------------------------------------------------------------

def _flatten(bodies: List[Body]) -> np.ndarray:
    """
    Pack all body states into a single 1-D array.

    Layout: [x0,y0,z0, vx0,vy0,vz0,  x1,y1,z1, vx1,vy1,vz1, ...]
    """
    return np.concatenate([np.r_[b.pos, b.vel] for b in bodies])


def _unpack_positions(y: np.ndarray, n: int) -> np.ndarray:
    """Return an (n, 3) view of the position sub-vectors in state y."""
    return y.reshape(n, 6)[:, :3]


def _unpack_velocities(y: np.ndarray, n: int) -> np.ndarray:
    """Return an (n, 3) view of the velocity sub-vectors in state y."""
    return y.reshape(n, 6)[:, 3:]


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def _derivatives(y: np.ndarray, masses: np.ndarray,
                 softening: float = 1e-5) -> np.ndarray:
    """
    Compute dy/dt for the gravitational N-body ODE.

    The state vector y contains interleaved positions and velocities:
      y[6i:6i+3]  = position  of body i
      y[6i+3:6i+6] = velocity of body i

    dy/dt layout is identical:
      dydt[6i:6i+3]   = velocity  of body i    (r_dot = v)
      dydt[6i+3:6i+6] = acceleration of body i  (v_dot = a)

    Gravitational acceleration on body i from body j:
      a_i += G * m_j * (r_j - r_i) / |r_j - r_i|^3

    A Plummer softening length epsilon = 1e-5 AU prevents the denominator
    from reaching zero during close passages.
    """
    n    = len(masses)
    dydt = np.zeros_like(y)
    for i in range(n):
        dydt[6*i:6*i+3] = y[6*i+3:6*i+6]   # r_dot = v
        ri  = y[6*i:6*i+3]
        acc = np.zeros(3)
        for j in range(n):
            if j == i:
                continue
            dr   = y[6*j:6*j+3] - ri
            dist = np.sqrt(dr @ dr + softening**2)
            acc += G * masses[j] / dist**3 * dr
        dydt[6*i+3:6*i+6] = acc             # v_dot = a
    return dydt


def _accelerations(y: np.ndarray, masses: np.ndarray,
                   softening: float = 1e-5) -> np.ndarray:
    """
    Return only the acceleration vectors as an (n, 3) array.

    Used by the leapfrog integrator, which updates positions and velocities
    separately and only needs accelerations, not the full derivative vector.
    """
    n   = len(masses)
    acc = np.zeros((n, 3))
    pos = y.reshape(n, 6)[:, :3]
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            dr      = pos[j] - pos[i]
            dist    = np.sqrt(dr @ dr + softening**2)
            acc[i] += G * masses[j] / dist**3 * dr
    return acc


def _total_energy(y: np.ndarray, masses: np.ndarray,
                  softening: float = 1e-5) -> float:
    """
    Total mechanical energy: kinetic + gravitational potential.

    KE = sum_i  0.5 * m_i * |v_i|^2
    PE = sum_{i<j}  -G * m_i * m_j / |r_i - r_j|
    """
    n = len(masses)
    E = 0.0
    for i in range(n):
        E += 0.5 * masses[i] * (y[6*i+3:6*i+6] @ y[6*i+3:6*i+6])
        for j in range(i + 1, n):
            dr = y[6*j:6*j+3] - y[6*i:6*i+3]
            r  = np.sqrt(dr @ dr + softening**2)
            E -= G * masses[i] * masses[j] / r
    return E


# ---------------------------------------------------------------------------
# Integrator 1: Bulirsch-Stoer (adaptive step)
# ---------------------------------------------------------------------------
#
# Algorithm outline:
#   1. For each macro-step H, apply Gragg's modified midpoint rule with
#      successively finer sub-step counts from _BS_SEQUENCE = [2,4,6,8,...].
#   2. Collect the sequence of estimates T_1, T_2, ... and apply Neville's
#      polynomial extrapolation in h^2 to produce a high-order estimate T[0].
#   3. Estimate the local error from the difference between successive
#      extrapolated columns.  Accept when the normalised RMS error < 1.
#   4. Rescale H for the next step using the standard controller formula
#      H_next = H * safety * (1/err)^(1/(2k+1)).
#
# Reference: Press et al., "Numerical Recipes", chapter 17.

_BS_SEQUENCE = [2, 4, 6, 8, 10, 12, 14, 16, 18]


def _modified_midpoint(y: np.ndarray, dydt: np.ndarray,
                        masses: np.ndarray, H: float, n_sub: int) -> np.ndarray:
    """
    Gragg's modified midpoint method: advance y by H using n_sub sub-steps.

    The endpoint smoothing step  y_end = 0.5*(z1 + z0 + h*f(z1))  cancels
    all odd-order terms in the local error, leaving only even powers of h.
    This makes Richardson / polynomial extrapolation in h^2 well-conditioned.
    """
    h  = H / n_sub
    z0 = y.copy()
    z1 = y + h * dydt              # first step: simple Euler
    for _ in range(n_sub - 1):
        d      = _derivatives(z1, masses)
        z0, z1 = z1, z0 + 2.0 * h * d   # interior steps: leapfrog-like
    d_end = _derivatives(z1, masses)
    return 0.5 * (z1 + z0 + h * d_end)  # smoothed endpoint


def _bs_step(
    y: np.ndarray,
    masses: np.ndarray,
    H: float,
    tol: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Attempt one adaptive Bulirsch-Stoer macro-step of size H.

    Returns
    -------
    y_new  : extrapolated state at t + H
    H_next : step size suggestion for the next call
    err    : normalised RMS error estimate (< 1.0 on success)
    """
    dydt = _derivatives(y, masses)
    T    = [None] * len(_BS_SEQUENCE)

    for k, n_sub in enumerate(_BS_SEQUENCE):
        T[k] = _modified_midpoint(y, dydt, masses, H, n_sub)

        # Neville polynomial extrapolation in h^2
        for j in range(k - 1, -1, -1):
            ratio = (_BS_SEQUENCE[k] / _BS_SEQUENCE[j]) ** 2
            T[j]  = T[j+1] + (T[j+1] - T[j]) / (ratio - 1.0)

        if k >= 1:
            delta = T[0] - T[1]
            scale = np.maximum(np.abs(T[0]), np.abs(y)) + 1e-30
            err   = float(np.sqrt(np.mean((delta / (scale * tol)) ** 2)))
            if err <= 1.0:
                exp    = 1.0 / (2 * k + 1)
                H_next = H * 0.9 * (1.0 / err) ** exp
                H_next = float(np.clip(H_next, 0.1 * H, 5.0 * H))
                return T[0], H_next, err

    # Could not converge within the tableau -- shrink and retry next call
    return T[0], H * 0.25, float(np.inf)


# ---------------------------------------------------------------------------
# Integrator 2: classical 4th-order Runge-Kutta (fixed step)
# ---------------------------------------------------------------------------
#
# RK4 Butcher tableau  (standard "3/8-rule" variant not used here):
#
#   k1 = h * f(y)
#   k2 = h * f(y + k1/2)
#   k3 = h * f(y + k2/2)
#   k4 = h * f(y + k3)
#
#   y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
#
# Local truncation error O(h^5), global error O(h^4).
# Not symplectic: energy error grows roughly linearly with time at fixed h.

def _rk4_step(y: np.ndarray, masses: np.ndarray, h: float) -> np.ndarray:
    """
    Advance state y by one fixed step h using the classical RK4 scheme.

    Requires 4 derivative evaluations per step.
    """
    k1 = h * _derivatives(y,            masses)
    k2 = h * _derivatives(y + 0.5 * k1, masses)
    k3 = h * _derivatives(y + 0.5 * k2, masses)
    k4 = h * _derivatives(y + k3,        masses)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


# ---------------------------------------------------------------------------
# Integrator 3: Kick-Drift-Kick leapfrog (fixed step, symplectic)
# ---------------------------------------------------------------------------
#
# The KDK (Kick-Drift-Kick) leapfrog scheme is a second-order symplectic
# integrator.  "Symplectic" means it exactly conserves the phase-space volume
# and preserves a slightly modified (shadow) Hamiltonian for all time, so
# the total energy oscillates around the true value without secular drift.
#
# One KDK step of size h:
#   1. Half-kick:  v_{n+1/2} = v_n + (h/2) * a(r_n)
#   2. Drift:      r_{n+1}   = r_n +  h     * v_{n+1/2}
#   3. Recompute:  a_{n+1}   = a(r_{n+1})
#   4. Half-kick:  v_{n+1}   = v_{n+1/2} + (h/2) * a_{n+1}
#
# This requires 1 force evaluation per step (re-using a_{n+1} as the first
# half-kick of the next step with the DKD variant -- here we compute it fresh
# for clarity).  For an N-body problem the computational bottleneck is the
# O(N^2) force calculation, so saving one evaluation per step is meaningful
# for large N.
#
# Reference: Hockney & Eastwood, "Computer Simulation Using Particles" (1988);
#            Springel, "The cosmological simulation code GADGET-2" (2005).

def _leapfrog_step(y: np.ndarray, masses: np.ndarray, h: float) -> np.ndarray:
    """
    Advance state y by one fixed step h using the KDK leapfrog scheme.

    The state vector is split internally into positions and velocities,
    updated separately, and reassembled.  Requires 1 force evaluation
    at the start (for the first half-kick) plus 1 at the end (after drift),
    totalling 2 per step -- the second will be recomputed as the "start"
    of the next step, so architecturally the cost is 1 per step if pipelined.
    """
    n   = len(masses)
    pos = y.reshape(n, 6)[:, :3].copy()
    vel = y.reshape(n, 6)[:, 3:].copy()

    # Pack a temporary state to evaluate accelerations at current positions
    y_curr    = np.concatenate([np.r_[pos[i], vel[i]] for i in range(n)])
    acc_start = _accelerations(y_curr, masses)   # a(r_n)

    # Step 1 -- half-kick: v_{n+1/2} = v_n + (h/2) * a(r_n)
    vel_half = vel + 0.5 * h * acc_start

    # Step 2 -- drift: r_{n+1} = r_n + h * v_{n+1/2}
    pos_new = pos + h * vel_half

    # Step 3 -- recompute accelerations at the new positions
    y_new_pos = np.concatenate([np.r_[pos_new[i], vel_half[i]] for i in range(n)])
    acc_end   = _accelerations(y_new_pos, masses)   # a(r_{n+1})

    # Step 4 -- half-kick: v_{n+1} = v_{n+1/2} + (h/2) * a(r_{n+1})
    vel_new = vel_half + 0.5 * h * acc_end

    # Reassemble into flat state vector
    return np.concatenate([np.r_[pos_new[i], vel_new[i]] for i in range(n)])


# ---------------------------------------------------------------------------
# Shared output assembly
# ---------------------------------------------------------------------------

def _assemble_result(
    ts: List[float],
    ys: List[np.ndarray],
    Es: List[float],
    hs: List[float],
    n_bodies: int,
    method: str,
    verbose: bool,
) -> SimulationResult:
    """Reshape raw integration lists into a SimulationResult."""
    N  = len(ts)
    nb = n_bodies
    pos = np.zeros((N, nb, 3))
    vel = np.zeros((N, nb, 3))
    for i, state in enumerate(ys):
        for b in range(nb):
            pos[i, b] = state[6*b:6*b+3]
            vel[i, b] = state[6*b+3:6*b+6]

    if verbose:
        dE = (Es[-1] - Es[0]) / abs(Es[0])
        print(f"  Done [{method}] -- {N} snapshots, t_final={ts[-1]:.4f}")
        print(f"  Energy conservation dE/E = {dE:.2e}\n")

    return SimulationResult(
        times      = np.array(ts),
        positions  = pos,
        velocities = vel,
        energies   = np.array(Es),
        step_sizes = np.array(hs),
        method     = method,
    )


# ---------------------------------------------------------------------------
# Individual integrator loops
# ---------------------------------------------------------------------------

def _integrate_bs(
    y: np.ndarray,
    masses: np.ndarray,
    t_end: float,
    H_init: float,
    tol: float,
    max_steps: int,
    clip_radius: Optional[float],
    verbose: bool,
) -> Tuple[List, List, List, List]:
    """Bulirsch-Stoer adaptive integration loop."""
    t, H = 0.0, H_init
    ts, ys, Es, hs = [t], [y.copy()], [_total_energy(y, masses)], [0.0]

    for step in range(max_steps):
        H = min(H, t_end - t)
        if H <= 0:
            break

        y_new, H_next, err = _bs_step(y, masses, H, tol)
        t += H
        y  = y_new
        H  = H_next

        ts.append(t);  ys.append(y.copy())
        Es.append(_total_energy(y, masses));  hs.append(H)

        if verbose and (step + 1) % 1000 == 0:
            dE = (Es[-1] - Es[0]) / abs(Es[0])
            print(f"  [bs]  t={t:.3f}  step={step+1}  H={H:.2e}  dE/E={dE:.2e}")

        if clip_radius is not None:
            if any(np.linalg.norm(y[6*i:6*i+3]) > clip_radius
                   for i in range(len(masses))):
                if verbose:
                    print(f"  [bs]  Ejection detected at t={t:.4f}. Stopping.")
                break

    return ts, ys, Es, hs


def _integrate_rk4(
    y: np.ndarray,
    masses: np.ndarray,
    t_end: float,
    h: float,
    max_steps: int,
    clip_radius: Optional[float],
    verbose: bool,
) -> Tuple[List, List, List, List]:
    """
    RK4 fixed-step integration loop.

    The step size h is held constant throughout (no adaptive control).
    The final step is shortened to land exactly on t_end.
    """
    t = 0.0
    ts, ys, Es, hs = [t], [y.copy()], [_total_energy(y, masses)], [0.0]

    for step in range(max_steps):
        h_actual = min(h, t_end - t)
        if h_actual <= 0:
            break

        y = _rk4_step(y, masses, h_actual)
        t += h_actual

        ts.append(t);  ys.append(y.copy())
        Es.append(_total_energy(y, masses));  hs.append(h_actual)

        if verbose and (step + 1) % 2000 == 0:
            dE = (Es[-1] - Es[0]) / abs(Es[0])
            print(f"  [rk4]  t={t:.3f}  step={step+1}  dE/E={dE:.2e}")

        if clip_radius is not None:
            if any(np.linalg.norm(y[6*i:6*i+3]) > clip_radius
                   for i in range(len(masses))):
                if verbose:
                    print(f"  [rk4]  Ejection detected at t={t:.4f}. Stopping.")
                break

    return ts, ys, Es, hs


def _integrate_leapfrog(
    y: np.ndarray,
    masses: np.ndarray,
    t_end: float,
    h: float,
    max_steps: int,
    clip_radius: Optional[float],
    verbose: bool,
) -> Tuple[List, List, List, List]:
    """
    KDK leapfrog fixed-step integration loop.

    Like RK4, the step size is constant.  The final step is shortened to
    land exactly on t_end.  Because leapfrog is symplectic, the recorded
    total energy will oscillate by O(h^2) around the true value but will
    not show a secular trend even over millions of steps.
    """
    t = 0.0
    ts, ys, Es, hs = [t], [y.copy()], [_total_energy(y, masses)], [0.0]

    for step in range(max_steps):
        h_actual = min(h, t_end - t)
        if h_actual <= 0:
            break

        y = _leapfrog_step(y, masses, h_actual)
        t += h_actual

        ts.append(t);  ys.append(y.copy())
        Es.append(_total_energy(y, masses));  hs.append(h_actual)

        if verbose and (step + 1) % 2000 == 0:
            dE = (Es[-1] - Es[0]) / abs(Es[0])
            print(f"  [leapfrog]  t={t:.3f}  step={step+1}  dE/E={dE:.2e}")

        if clip_radius is not None:
            if any(np.linalg.norm(y[6*i:6*i+3]) > clip_radius
                   for i in range(len(masses))):
                if verbose:
                    print(f"  [leapfrog]  Ejection detected at t={t:.4f}. Stopping.")
                break

    return ts, ys, Es, hs


# ---------------------------------------------------------------------------
# Public API: initial_conditions
# ---------------------------------------------------------------------------

def initial_conditions(
    m1: float = 1.0,
    m2: float = 0.8,
    m3: float = 0.3,
    # Binary parameters
    a_bin: float         = 1.0,
    e_bin: float         = 0.0,
    inc_bin_deg: float   = 30.0,
    omega_bin_deg: float = 0.0,
    f_bin_deg: float     = 0.0,
    # Incoming body parameters
    e3: float            = 0.85,
    r_peri3: float       = 2.0,
    r3_init: float       = 10.0,
    inc3_deg: float      = 0.0,
    omega3_deg: float    = 180.0,
) -> List[Body]:
    """
    Build three-body initial conditions supporting all eccentricity regimes.

    Binary (m1, m2)
    ---------------
    Placed on a Keplerian conic with semi-major axis a_bin and eccentricity
    e_bin, with the orbital plane tilted by inc_bin_deg relative to x-y.

    Incoming body (m3)
    ------------------
    Placed at distance r3_init from the binary centre-of-mass on a Keplerian
    conic with eccentricity e3 and periapsis r_peri3.  The body is on the
    approaching leg of its orbit (true anomaly f < 0).

    Eccentricity support
    --------------------
    e = 0        : circular
    0 < e < 1    : elliptic (bound)
    e = 1        : parabolic
    e > 1        : hyperbolic (unbound flyby)

    Returns
    -------
    List of three Body objects in the centre-of-mass frame.
    """
    inc_bin = np.radians(inc_bin_deg)
    omega_b = np.radians(omega_bin_deg)
    f_b     = np.radians(f_bin_deg)
    inc3    = np.radians(inc3_deg)
    omega3  = np.radians(omega3_deg)

    M_bin = m1 + m2
    M_tot = m1 + m2 + m3

    # -- Binary ---------------------------------------------------------------
    if e_bin == 1.0:
        a_b_eff = -a_bin                    # parabolic: encode r_peri as -a
    elif e_bin > 1.0:
        a_b_eff = -a_bin / (e_bin - 1.0)   # hyperbolic: a_bin is the periapsis
    else:
        a_b_eff = a_bin                     # elliptic / circular

    pos_rel, vel_rel = keplerian_state(M_bin, a_b_eff, e_bin,
                                       inc_bin, omega_b, f_b)
    mu1, mu2 = m1 / M_bin, m2 / M_bin
    pos1, vel1 =  mu2 * pos_rel,  mu2 * vel_rel
    pos2, vel2 = -mu1 * pos_rel, -mu1 * vel_rel

    # -- Incoming body --------------------------------------------------------
    if e3 == 0.0:
        # Circular: place on a circular orbit of radius r3_init
        v3_mag = np.sqrt(G * M_tot / r3_init)
        pos3   = np.array([-r3_init, 0.0, 0.0])
        vel3   = v3_mag * np.array([0.0, np.cos(inc3), np.sin(inc3)])
    else:
        if e3 == 1.0:
            a3_eff = -r_peri3               # parabolic
        elif e3 > 1.0:
            a3_eff = -r_peri3 / (e3 - 1.0) # hyperbolic
        else:
            a3_eff = r_peri3 / (1.0 - e3)  # elliptic

        f3         = true_anomaly_from_distance(r3_init, a3_eff, e3, approaching=True)
        pos3, vel3 = keplerian_state(M_tot, a3_eff, e3, inc3, omega3, f3)

    bodies: List[Body] = [
        Body(mass=m1, pos=pos1, vel=vel1),
        Body(mass=m2, pos=pos2, vel=vel2),
        Body(mass=m3, pos=pos3, vel=vel3),
    ]

    # -- Shift to true centre-of-mass frame -----------------------------------
    M       = sum(b.mass for b in bodies)
    com_pos = sum(b.mass * b.pos for b in bodies) / M
    com_vel = sum(b.mass * b.vel for b in bodies) / M
    for b in bodies:
        b.pos = b.pos - com_pos
        b.vel = b.vel - com_vel

    return bodies


# ---------------------------------------------------------------------------
# Public API: simulate
# ---------------------------------------------------------------------------

def simulate(
    bodies: List[Body],
    method: Method       = "bs",
    t_end: float         = 15.0,
    H_init: float        = 0.02,
    tol: float           = 1e-9,
    max_steps: int       = 500_000,
    clip_radius: Optional[float] = None,
    verbose: bool        = True,
) -> SimulationResult:
    """
    Integrate the N-body equations of motion and return a SimulationResult.

    Parameters
    ----------
    bodies      : List of Body objects defining the initial conditions.
    method      : Integrator to use.  One of:
                    "bs"        -- Bulirsch-Stoer (adaptive, high accuracy)
                    "rk4"       -- Runge-Kutta 4  (fixed step, 4th order)
                    "leapfrog"  -- KDK leapfrog   (fixed step, symplectic)
    t_end       : Integration end time [yr].
    H_init      : Initial (or fixed, for RK4/leapfrog) step size.
                  For "bs" this is the starting macro-step; the integrator
                  will adapt it automatically.  For "rk4" and "leapfrog"
                  this is the constant step used throughout.
    tol         : Relative local error tolerance (Bulirsch-Stoer only).
                  Has no effect for RK4 or leapfrog.
    max_steps   : Hard cap on the total number of integration steps.
    clip_radius : If any body exceeds this distance from the origin (AU),
                  integration stops early.  Useful for detecting ejections.
    verbose     : Print per-step progress (every 1000/2000 steps) and a
                  final energy-conservation summary.

    Returns
    -------
    SimulationResult
        .times      : (N,)      array of snapshot times
        .positions  : (N, B, 3) array of body positions
        .velocities : (N, B, 3) array of body velocities
        .energies   : (N,)      total mechanical energy at each snapshot
        .step_sizes : (N,)      step sizes used (varies for BS, constant for others)
        .method     : name of the integrator used

    Notes
    -----
    Step size guidance:
      "bs"       -- H_init=0.02 is a reasonable starting value; the integrator
                    will shrink or grow it as needed.
      "rk4"      -- For the default scenario, H_init=0.002 gives O(h^4) ~ 1e-11
                    per step; H_init=0.01 is faster but less accurate.
      "leapfrog" -- Use H_init=0.002 or smaller.  Energy oscillates by O(h^2)
                    (~1e-6 at h=0.002) but never drifts secularly.
    """
    if method not in ("bs", "rk4", "leapfrog"):
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'bs', 'rk4', 'leapfrog'."
        )

    masses = np.array([b.mass for b in bodies])
    y      = _flatten(bodies)

    if verbose:
        print(f"  Integrating with method='{method}'  t_end={t_end}  H={H_init}")
        if method == "bs":
            print(f"  Tolerance tol={tol}")

    if method == "bs":
        ts, ys, Es, hs = _integrate_bs(
            y, masses, t_end, H_init, tol, max_steps, clip_radius, verbose)
    elif method == "rk4":
        ts, ys, Es, hs = _integrate_rk4(
            y, masses, t_end, H_init, max_steps, clip_radius, verbose)
    else:  # leapfrog
        ts, ys, Es, hs = _integrate_leapfrog(
            y, masses, t_end, H_init, max_steps, clip_radius, verbose)

    return _assemble_result(ts, ys, Es, hs, len(bodies), method, verbose)