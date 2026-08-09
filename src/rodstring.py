"""
PRAHARI - Rod-string dynamics via the damped wave equation (Gibbs, 1963).

Governing equation, x measured DOWNWARD from the polished rod, u positive
downward:

        d2u/dt2 = (E/rho) * d2u/dx2  +  g_eff  -  c * du/dt

    E/rho = v^2   (v = acoustic velocity in steel ~5135 m/s)
    g_eff = g * (1 - rho_fluid/rho_steel)   (buoyed rod weight)
    c     = viscous damping coefficient [1/s]

Axial tension  T(x,t) = E * A_rod * du/dx.

Boundary conditions (mixed):
    x = 0 : u(0,t) = prescribed polished-rod displacement   (Dirichlet)
    x = L : T(L,t) = F_pump(t) from the pump model          (Neumann/force)

Solved with an explicit second-order finite-difference scheme; the pump-end
force BC is applied with a ghost node. The CFL condition v*dt/dx <= 1 is
enforced and asserted.

WHY THIS MATTERS: fluid pound produces a *sudden* release of the fluid load
at the pump. Because we integrate the real wave equation, the resulting
high-frequency ringing in the surface card emerges from the physics rather
than being drawn in by hand. That is the difference between a simulator and
a shape template.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

E_STEEL = 2.07e11  # Pa
RHO_STEEL = 7850.0  # kg/m3
G = 9.80665  # m/s2
V_SOUND = np.sqrt(E_STEEL / RHO_STEEL)  # ~5135 m/s
CFL_TARGET = 0.85  # explicit-scheme Courant target (damping erodes the limit)

FAULTS = (
    "normal",
    "fluid_pound",
    "gas_interference",
    "tv_leak",
    "sv_leak",
    "worn_pump",
    "rod_part",
    "tagging",
)


@dataclass
class RodString:
    """Uniform (single-taper) rod string."""

    L: float = 1200.0  # m, pump setting depth
    d_rod: float = 0.0222  # m, 7/8 in sucker rod
    damping: float = 2.50  # 1/s, viscous damping coefficient
    rho_fluid: float = 900.0  # kg/m3

    @property
    def A_rod(self) -> float:
        return np.pi * self.d_rod**2 / 4.0

    @property
    def g_eff(self) -> float:
        return G * (1.0 - self.rho_fluid / RHO_STEEL)

    @property
    def weight_buoyed(self) -> float:
        """Total buoyed rod weight [N] - the static polished rod load."""
        return RHO_STEEL * self.A_rod * self.L * self.g_eff


@dataclass
class PumpConfig:
    """Downhole pump geometry, loading and fault parameterisation."""

    d_plunger: float = 0.0381  # m, 1.5 in plunger
    fluid_load: float = 16000.0  # N, Fo = dP * A_plunger at full load
    fault: str = "normal"

    # --- fault parameters (physical, not cosmetic) ---
    fillage: float = 1.0  # pump fillage fraction [0.2, 1.0]
    gas_compress: float = 0.02  # smoothness of load transfer (gas cushion)
    tv_leak_rate: float = 0.0  # 1/s, traveling-valve leak-off rate
    sv_leak_rate: float = 0.0  # 1/s, standing-valve leak-off rate
    rod_part_frac: float = 1.0  # fraction of rod string remaining (1 = intact)
    tag_force: float = 0.0  # N, bottom-tagging impact force
    tag_frac: float = 0.02  # fraction of stroke over which tagging acts

    @property
    def A_plunger(self) -> float:
        return np.pi * self.d_plunger**2 / 4.0


def make_pump(fault: str, rng: np.random.Generator, base: dict | None = None) -> PumpConfig:
    """
    Build a PumpConfig for a given fault class with physically sensible,
    randomised severity. Centralises fault parameterisation so that a fault
    label always implies a genuinely different physical boundary condition.
    """
    base = base or {}
    cfg = PumpConfig(fault=fault, **base)
    if fault == "normal":
        cfg.fillage = float(rng.uniform(0.95, 1.0))
    elif fault == "fluid_pound":
        cfg.fillage = float(rng.uniform(0.30, 0.70))
    elif fault == "gas_interference":
        cfg.fillage = float(rng.uniform(0.45, 0.80))
        cfg.gas_compress = float(rng.uniform(0.08, 0.25))
    elif fault == "tv_leak":
        cfg.tv_leak_rate = float(rng.uniform(0.35, 1.20))
    elif fault == "sv_leak":
        cfg.sv_leak_rate = float(rng.uniform(0.35, 1.20))
    elif fault == "worn_pump":
        cfg.tv_leak_rate = float(rng.uniform(0.45, 1.00))
        cfg.sv_leak_rate = float(rng.uniform(0.45, 1.00))
        cfg.fillage = float(rng.uniform(0.70, 0.95))
    elif fault == "rod_part":
        cfg.rod_part_frac = float(rng.uniform(0.35, 0.80))
    elif fault == "tagging":
        cfg.tag_force = float(rng.uniform(0.20, 0.50)) * cfg.fluid_load
        cfg.tag_frac = float(rng.uniform(0.02, 0.06))
    else:
        raise ValueError(f"unknown fault {fault!r}")
    return cfg


def pump_force(
    x_plunger: np.ndarray | float,
    v_plunger: float,
    stroke: float,
    state: dict,
    cfg: PumpConfig,
    dt: float,
) -> float:
    """
    Pump boundary force F_pump [N] = tension in the rod just above the pump.

    Implemented as a physical state machine over the pump cycle rather than a
    lookup of card shapes. `state` carries the fluid-load memory between steps.

    x_plunger : plunger displacement DOWNWARD from top of stroke [m]
    v_plunger : plunger velocity, positive DOWNWARD [m/s]
    """
    fault = cfg.fault
    Fo = cfg.fluid_load
    F = state.get("F", 0.0)

    if fault == "rod_part":
        # Parted rod: nothing below the break carries fluid load at all.
        state["F"] = 0.0
        return 0.0

    # Effective fillage
    phi = cfg.fillage if fault in ("fluid_pound", "gas_interference", "worn_pump") else 1.0

    up = v_plunger < 0.0  # plunger moving UP (x decreasing)

    if up:
        # ---- UPSTROKE: travelling valve closed, fluid load carried by rods ----
        target = Fo
        state["t_up"] = state.get("t_up", 0.0) + dt

        # TRAVELLING-VALVE LEAK: fluid escapes downward past the plunger while
        # the rods are lifting it -> the load BLEEDS OFF through the upstroke.
        if fault in ("tv_leak", "worn_pump"):
            rate = cfg.tv_leak_rate
            target = Fo * np.exp(-rate * state["t_up"])

        # STANDING-VALVE LEAK: fluid drains back through the standing valve, so
        # the barrel pressurises slowly and the load PICKS UP LATE.
        # (Distinct signature from a TV leak: slanted left flank, not a drooping
        #  top flank. Modelled as a long pick-up time constant.)
        tau = 0.06
        if fault == "gas_interference":
            tau = 0.25
        if fault in ("sv_leak", "worn_pump"):
            tau = 0.06 + 1.8 * cfg.sv_leak_rate

        F += (target - F) * (1.0 - np.exp(-dt / tau))
        state["t_down"] = 0.0
    else:
        # ---- DOWNSTROKE ----
        # The rods keep carrying Fo until the plunger contacts liquid.
        # Contact occurs after travelling (1 - phi) of the stroke.
        travel_frac = np.clip(x_plunger / max(stroke, 1e-9), 0.0, 1.0)
        contacted = travel_frac >= (1.0 - phi)

        if contacted:
            if fault == "gas_interference":
                # Gas cushion -> gradual, compressible load transfer
                excess = travel_frac - (1.0 - phi)
                w = 1.0 - np.exp(-excess / max(cfg.gas_compress, 1e-4))
                F = Fo * (1.0 - w)
            else:
                # Liquid contact -> abrupt unloading == FLUID POUND
                F += (0.0 - F) * (1.0 - np.exp(-dt / 0.010))
        else:
            # Still descending through the void: load stays on the rods
            if fault in ("sv_leak", "worn_pump"):
                F *= np.exp(-cfg.sv_leak_rate * dt)
        state["t_up"] = 0.0

    # Bottom tagging: impact spike near the bottom of the stroke
    if fault == "tagging":
        frac_from_bottom = 1.0 - np.clip(x_plunger / max(stroke, 1e-9), 0.0, 1.0)
        if frac_from_bottom < cfg.tag_frac and not up:
            F += cfg.tag_force * (1.0 - frac_from_bottom / cfg.tag_frac)

    F = float(max(F, 0.0))
    state["F"] = F
    return F


def simulate_rodstring(
    s_of_t: np.ndarray,
    dt: float,
    rod: RodString,
    pump: PumpConfig,
    n_nodes: int = 48,
    n_cycles_warmup: int = 3,
) -> dict:
    """
    Integrate the damped wave equation for a prescribed polished-rod motion.

    s_of_t : polished rod position UPWARD from bottom of stroke, one cycle [m]
    dt     : timestep [s]

    Returns dict with surface load (PRL), downhole pump load/position, for the
    final (converged) cycle.
    """
    L_eff = rod.L * pump.rod_part_frac
    dx = L_eff / n_nodes
    v = V_SOUND

    # --- CFL stability condition, WITH SAFETY MARGIN ---
    # The bare Courant limit for the undamped wave equation is 1.0, but the
    # viscous damping term erodes the stability region. Runs landing at
    # CFL ~= 0.997 (deep string + thick rod + high SPM) were observed to
    # diverge, so we sub-step to a target of CFL_TARGET instead of 1.0.
    cfl = v * dt / dx
    substeps = max(1, int(np.ceil(cfl / CFL_TARGET)))
    dt_sub = dt / substeps
    assert v * dt_sub / dx <= CFL_TARGET + 1e-9, "CFL condition violated"

    stroke = float(s_of_t.max() - s_of_t.min())
    n_t = len(s_of_t)

    # Convert to DOWNWARD displacement from top of stroke
    u_surf = (s_of_t.max() - s_of_t)

    EA = E_STEEL * rod.A_rod
    r2 = (v * dt_sub / dx) ** 2

    # Initialise with the ANALYTIC STATIC STRETCH profile.
    # Solves  v^2 u'' + g_eff = 0,  u(0)=u_surf(0),  EA u'(L) = F0
    # so the simulation starts in equilibrium instead of releasing a large
    # start-up transient that would contaminate the first cycles.
    x = np.linspace(0.0, L_eff, n_nodes + 1)
    F0 = 0.0 if pump.fault == "rod_part" else pump.fluid_load
    C1 = F0 / EA + RHO_STEEL * rod.g_eff * L_eff / E_STEEL
    u_static = -RHO_STEEL * rod.g_eff * x**2 / (2.0 * E_STEEL) + C1 * x
    u_prev = u_surf[0] + u_static
    u_curr = u_prev.copy()

    pstate: dict = {"F": F0, "t_up": 0.0, "t_down": 0.0}

    rec_prl, rec_fpump, rec_xplunger = [], [], []

    total_cycles = n_cycles_warmup + 1
    for cyc in range(total_cycles):
        record = cyc == total_cycles - 1
        for i in range(n_t):
            # Surface displacement must be advanced SMOOTHLY across sub-steps.
            # Holding it constant within an output step would inject a velocity
            # discontinuity every sub-cycle and spuriously excite the string.
            u_a = u_surf[i]
            u_b = u_surf[(i + 1) % n_t]
            for k in range(substeps):
                u_top = u_a + (u_b - u_a) * (k + 1) / substeps
                # plunger kinematics (node n_nodes)
                x_pl = u_curr[-1]
                v_pl = (u_curr[-1] - u_prev[-1]) / dt_sub
                Fp = pump_force(x_pl, v_pl, stroke, pstate, pump, dt_sub)

                u_next = np.empty_like(u_curr)
                # interior nodes
                u_next[1:-1] = (
                    2.0 * u_curr[1:-1]
                    - u_prev[1:-1]
                    + r2 * (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2])
                    + rod.g_eff * dt_sub**2
                    - rod.damping * dt_sub * (u_curr[1:-1] - u_prev[1:-1])
                )
                # surface: Dirichlet
                u_next[0] = u_top
                # pump end: force BC via ghost node u[N+1] = u[N-1] + 2*dx*Fp/(EA)
                ghost = u_curr[-2] + 2.0 * dx * Fp / EA
                u_next[-1] = (
                    2.0 * u_curr[-1]
                    - u_prev[-1]
                    + r2 * (ghost - 2.0 * u_curr[-1] + u_curr[-2])
                    + rod.g_eff * dt_sub**2
                    - rod.damping * dt_sub * (u_curr[-1] - u_prev[-1])
                )
                u_prev, u_curr = u_curr, u_next

            if record:
                # Surface tension = EA * du/dx  (2nd-order one-sided)
                dudx0 = (-3.0 * u_curr[0] + 4.0 * u_curr[1] - u_curr[2]) / (2.0 * dx)
                rec_prl.append(EA * dudx0)
                rec_fpump.append(pstate["F"])
                rec_xplunger.append(u_curr[-1])

    prl = np.asarray(rec_prl)
    # Net plunger stroke: peak-to-peak plunger travel
    xpl = np.asarray(rec_xplunger)
    net_stroke = float(xpl.max() - xpl.min())

    return {
        "PRL": prl,
        "F_pump": np.asarray(rec_fpump),
        "x_plunger": xpl,
        "s_surface": s_of_t,
        "net_stroke": net_stroke,
        "stroke": stroke,
        "cfl": v * dt_sub / dx,
    }
