"""
PRAHARI - Pumping unit kinematics (conventional crank-balanced beam unit).

Implements the four-bar linkage of an API Spec 11E conventional beam pumping
unit and derives, for a given crank angle theta:

    - beam (walking-beam) rotation angle  psi(theta)      [rad]   <-- what the IMU sees
    - polished rod position               s(theta)        [m]     (0 = bottom of stroke)
    - torque factor                       TF(theta)       [m/rad]

The torque factor is obtained EXACTLY from the principle of virtual work:
the torque required at the crankshaft to hold a polished-rod load F is

        T = F * ds/dtheta        =>      TF(theta) == ds/dtheta

This is exact (no small-angle approximation) and is why we compute TF by
differentiating the kinematic solution rather than using tabulated API
torque factors.

Geometry symbols follow API Spec 11E convention:
    A = distance saddle bearing -> polished rod (front / horsehead arm)  [m]
    C = distance saddle bearing -> equalizer bearing (rear arm)          [m]
    P = pitman length                                                    [m]
    R = crank radius                                                     [m]
    K = distance saddle bearing -> crankshaft centre                     [m]
    phi_K = angle of the line (saddle bearing -> crankshaft centre)
            measured from horizontal                                     [rad]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UnitGeometry:
    """Conventional crank-balanced beam pumping unit geometry."""

    A: float = 3.6576  # front arm, m   (144 in)
    C: float = 2.7432  # rear arm,  m   (108 in)
    P: float = 3.2004  # pitman,    m   (126 in)
    R: float = 0.9144  # crank radius, m (36 in)
    K: float = 4.2672  # saddle bearing -> crankshaft, m (168 in)
    phi_K: float = 0.4014  # rad, ~23 deg below horizontal

    def __post_init__(self) -> None:
        # Four-bar closure feasibility: the crank must be able to rotate fully.
        # Requires |C - P| < J < C + P for all J in [K-R, K+R].
        j_min, j_max = self.K - self.R, self.K + self.R
        if not (abs(self.C - self.P) < j_min and j_max < self.C + self.P):
            raise ValueError(
                f"Infeasible four-bar geometry: J in [{j_min:.3f}, {j_max:.3f}] "
                f"must lie inside (|C-P|, C+P) = ({abs(self.C - self.P):.3f}, "
                f"{self.C + self.P:.3f})"
            )


def beam_angle(theta: np.ndarray, g: UnitGeometry) -> np.ndarray:
    """
    Walking-beam rear-arm angle psi(theta), measured from the line
    (saddle bearing -> crankshaft centre).

    Solves the four-bar linkage by two applications of the law of cosines.

    theta : crank angle [rad], 0 = crank pin pointing away from saddle bearing
    """
    R, K, C, P = g.R, g.K, g.C, g.P

    # J = distance from saddle bearing to crank pin
    J2 = K**2 + R**2 - 2.0 * K * R * np.cos(theta)
    J = np.sqrt(J2)

    # chi = angle at saddle bearing between (SB -> crankshaft) and (SB -> crank pin)
    cos_chi = np.clip((K**2 + J2 - R**2) / (2.0 * K * J), -1.0, 1.0)
    chi = np.arccos(cos_chi)
    # sign of chi follows the sign of sin(theta) so psi sweeps monotonically
    chi = np.sign(np.sin(theta)) * chi

    # beta = angle at saddle bearing in triangle (SB, crank pin) with sides C, P
    cos_beta = np.clip((C**2 + J2 - P**2) / (2.0 * C * J), -1.0, 1.0)
    beta = np.arccos(cos_beta)

    return chi + beta


def polished_rod_position(theta: np.ndarray, g: UnitGeometry) -> np.ndarray:
    """
    Polished rod position s(theta) [m], measured UPWARD from the bottom
    of the stroke. The horsehead arc is approximated as vertical travel
    s = A * psi, which is the standard assumption for a conventional unit
    (the horsehead + wireline keeps the rod vertical).
    """
    psi = beam_angle(theta, g)
    # Reference to the minimum over a full revolution so s >= 0
    theta_full = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    psi_min = beam_angle(theta_full, g).min()
    return g.A * (psi - psi_min)


def stroke_length(g: UnitGeometry) -> float:
    """Total polished-rod stroke length [m]."""
    theta_full = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    s = polished_rod_position(theta_full, g)
    return float(s.max() - s.min())


def torque_factor(theta: np.ndarray, g: UnitGeometry) -> np.ndarray:
    """
    Torque factor TF(theta) = ds/dtheta  [m/rad], computed by central
    differences on the exact kinematic solution.

    Exact by virtual work: crankshaft torque from a polished-rod load F
    is T = F * TF(theta).
    """
    h = 1e-5
    return (
        polished_rod_position(theta + h, g) - polished_rod_position(theta - h, g)
    ) / (2.0 * h)


def kinematics_table(n: int, g: UnitGeometry) -> dict:
    """Pre-computed kinematics over one crank revolution on a uniform grid."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return {
        "theta": theta,
        "psi": beam_angle(theta, g),
        "s": polished_rod_position(theta, g),
        "TF": torque_factor(theta, g),
        "stroke": stroke_length(g),
    }
