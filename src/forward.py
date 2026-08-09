"""
PRAHARI - Full forward observation model.

Chain:
    crank angle theta(t)
      -> unit kinematics  (kinematics.py)
      -> rod-string wave equation + pump faults  (rodstring.py)
      -> polished rod load PRL(theta)
      -> net crankshaft torque  T_net = TF*(PRL - SU) - M*sin(theta + tau)
      -> induction motor + gearbox + inertia
      -> ELECTRICAL POWER P_elec(t)          <-- observable #1 (clamp-on CT)
    and
      -> walking-beam angle psi(t)           <-- observable #2 (MEMS IMU)

Only P_elec(t) and psi(t) are visible to PRAHARI. Everything else
(PRL, downhole card, fillage, counterbalance) is hidden ground truth used
ONLY to score the inversion.

The crank speed is NOT constant: an induction motor slips under load and the
crank/flywheel inertia smooths torque. We integrate that explicitly, because
it is precisely the speed ripple that carries load information and makes the
IMU an independent measurement rather than a redundant one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kinematics import UnitGeometry, beam_angle, polished_rod_position, torque_factor
from rodstring import PumpConfig, RodString, simulate_rodstring


@dataclass
class DriveConfig:
    """Prime mover, gearbox and counterbalance."""

    M_cb: float = 22000.0  # N.m, maximum counterbalance moment   <-- UNKNOWN in field
    tau_cb: float = 0.0  # rad, counterbalance phase offset      <-- UNKNOWN in field
    SU: float = 1800.0  # N, structural unbalance               <-- UNKNOWN in field
    spm: float = 6.0  # strokes per minute (nominal)
    J_crank: float = 3500.0  # kg.m2, crank + counterweights + flywheel
    gearbox_eff: float = 0.95
    motor_rated_kw: float = 30.0
    motor_sync_rpm: float = 1500.0  # 4-pole, 50 Hz
    motor_slip_rated: float = 0.080  # NEMA D high-slip, standard for beam units
    gear_ratio: float = 0.0  # motor -> crankshaft; 0 => derive from spm
    motor_eff: float = 0.90
    no_load_kw: float = 1.6  # fixed losses (windage, magnetising, gearbox churn)

    def __post_init__(self) -> None:
        # A beam unit uses a double-reduction gearbox plus a belt drive; the
        # total reduction is set by the required SPM, not chosen freely.
        # 1500 rpm motor -> 6 SPM crank is a ratio of ~250, not ~30.
        if self.gear_ratio <= 0.0:
            self.gear_ratio = (
                self.motor_sync_rpm * (1.0 - self.motor_slip_rated) / self.spm
            )


@dataclass
class WellState:
    geom: UnitGeometry = field(default_factory=UnitGeometry)
    rod: RodString = field(default_factory=RodString)
    pump: PumpConfig = field(default_factory=PumpConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)


def crank_speed_profile(
    theta: np.ndarray, T_net: np.ndarray, drive: DriveConfig
) -> np.ndarray:
    """
    Solve for the non-uniform crank speed omega(theta) from the energy balance
    of the flywheel against net torque, with an induction-motor torque-speed
    characteristic.

    Uses the work-energy form  d(0.5*J*omega^2)/dtheta = T_motor - T_net,
    integrated once around the cycle and relaxed to a periodic solution.
    """
    omega_nom = 2.0 * np.pi * drive.spm / 60.0
    omega_sync_crank = 2.0 * np.pi * drive.motor_sync_rpm / 60.0 / drive.gear_ratio
    T_rated_crank = (
        drive.motor_rated_kw * 1000.0 / (omega_sync_crank * (1 - drive.motor_slip_rated))
    )
    # Linear torque-slip characteristic referred to the crankshaft
    k_motor = T_rated_crank / (drive.motor_slip_rated * omega_sync_crank)

    dtheta = float(theta[1] - theta[0])

    # Integrate  J * omega * d(omega)/d(theta) = T_motor(omega) - T_net(theta)
    # forward around the cycle, repeating until the orbit is periodic.
    # Equivalent to marching the flywheel equation in time, but in theta so the
    # output lands directly on the crank-angle grid.
    # The induction-motor torque-speed characteristic is STIFF: a 1% speed
    # change swings torque by tens of kN.m. Explicit Euler diverges here, so we
    # use a backward-Euler update, which is unconditionally stable and recovers
    # the physically correct behaviour (a few % of slip ripple, not runaway).
    #
    #   J*w*dw/dtheta = k*(w_s - w) - T_net
    #   =>  w_{i+1} = [ w_i + (k*w_s - T_net_i)*a ] / (1 + k*a),  a = dtheta/(J*w_i)
    omega = np.full_like(theta, omega_nom)
    w = omega_nom
    for _ in range(200):
        w_start = w
        for i in range(len(theta)):
            omega[i] = w
            a = dtheta / (drive.J_crank * max(w, 1e-3))
            w = (w + (k_motor * omega_sync_crank - T_net[i]) * a) / (1.0 + k_motor * a)
            w = max(w, 1e-3)
        if abs(w - w_start) < 1e-12:
            break
    return omega


def balanced_counterbalance(TF: np.ndarray, PRL: np.ndarray, theta: np.ndarray,
                            SU: float) -> tuple[float, float]:
    """
    Size the counterbalance the way a field crew would: choose M and tau so the
    counterbalance torque cancels the FIRST HARMONIC of the rod torque, which
    is what minimises peak gearbox torque.

    Returns (M_cb, tau_cb). Used to generate realistically-balanced wells, and
    also as the physical prior in the blind calibration.
    """
    T_rod = TF * (PRL - SU)
    a = 2.0 * np.mean(T_rod * np.sin(theta))
    b = 2.0 * np.mean(T_rod * np.cos(theta))
    # T_rod ~ a*sin(theta) + b*cos(theta) = M*sin(theta + tau)
    M = float(np.hypot(a, b))
    tau = float(np.arctan2(b, a))
    return M, tau


def simulate_well(
    well: WellState,
    n_theta: int = 360,
    noise_power_pct: float = 0.0,
    noise_imu_deg: float = 0.0,
    seed: int | None = None,
    n_nodes: int = 48,
    auto_balance: bool = True,
    cb_detune: float = 1.0,
    cb_phase_err: float = 0.0,
) -> dict:
    """
    Run the complete forward model and return both the hidden ground truth
    and the two noisy field-observable signals.
    """
    rng = np.random.default_rng(seed)
    g, rod, pump, drive = well.geom, well.rod, well.pump, well.drive

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    s = polished_rod_position(theta, g)
    psi = beam_angle(theta, g)
    TF = torque_factor(theta, g)

    period = 60.0 / drive.spm
    dt = period / n_theta

    # --- rod string + pump ---
    sim = simulate_rodstring(s, dt, rod, pump, n_nodes=n_nodes)
    PRL = sim["PRL"]

    # --- counterbalance ---
    # Real units are balanced once at install and then DRIFT (weights slip,
    # loads change, records are lost). We model that: size the counterbalance
    # to the current rod torque, then de-tune it by `cb_detune`.
    if auto_balance:
        M0, tau0 = balanced_counterbalance(TF, PRL, theta, drive.SU)
        drive.M_cb = M0 * cb_detune
        drive.tau_cb = tau0 + cb_phase_err

    # --- crankshaft torque ---
    T_rod = TF * (PRL - drive.SU)
    T_cb = drive.M_cb * np.sin(theta + drive.tau_cb)
    T_net = T_rod - T_cb

    # --- speed ripple ---
    omega = crank_speed_profile(theta, T_net, drive)

    # --- electrical power ---
    P_shaft = T_net * omega  # W
    P_gear = np.where(
        P_shaft >= 0.0, P_shaft / drive.gearbox_eff, P_shaft * drive.gearbox_eff
    )
    P_elec = np.where(
        P_gear >= 0.0, P_gear / drive.motor_eff, P_gear * drive.motor_eff
    )
    P_elec = P_elec + drive.no_load_kw * 1000.0

    # --- measurement noise ---
    P_meas = P_elec.copy()
    if noise_power_pct > 0.0:
        scale = noise_power_pct / 100.0 * np.abs(P_elec).max()
        P_meas = P_meas + rng.normal(0.0, scale, size=P_elec.shape)
    psi_meas = psi.copy()
    if noise_imu_deg > 0.0:
        psi_meas = psi_meas + rng.normal(
            0.0, np.deg2rad(noise_imu_deg), size=psi.shape
        )

    return {
        # ---------- OBSERVABLE (what a ~Rs 5,000 kit actually sees) ----------
        "P_elec_meas": P_meas,
        "psi_meas": psi_meas,
        "omega": omega,
        "theta": theta,
        # ---------- HIDDEN GROUND TRUTH (scoring only) ----------
        "P_elec_true": P_elec,
        "PRL": PRL,
        "F_pump": sim["F_pump"],
        "x_plunger": sim["x_plunger"],
        "s": s,
        "psi": psi,
        "TF": TF,
        "T_net": T_net,
        "net_stroke": sim["net_stroke"],
        "stroke": sim["stroke"],
        "fillage_true": pump.fillage,
        "Fo_true": pump.fluid_load,
        "M_cb_true": drive.M_cb,
        "tau_cb_true": drive.tau_cb,
        "fault": pump.fault,
        "cfl": sim["cfl"],
    }


def card_features(PRL: np.ndarray, s: np.ndarray) -> dict:
    """Physically interpretable descriptors of a surface dynamometer card."""
    lo, hi = float(PRL.min()), float(PRL.max())
    rng_ = hi - lo
    # Enclosed area (net work per stroke) by the shoelace formula
    area = 0.5 * float(np.abs(np.dot(s, np.roll(PRL, -1)) - np.dot(PRL, np.roll(s, -1))))
    return {
        "PRL_min": lo,
        "PRL_max": hi,
        "PRL_range": rng_,
        "card_area_J": area,
        "PRL_mean": float(PRL.mean()),
        "PRL_std": float(PRL.std()),
    }
