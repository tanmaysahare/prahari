"""
PRAHARI - BATCHED rod-string solver.

The inverse problem needs hundreds of forward evaluations per well. On a small
rod string the numpy call overhead dominates the arithmetic, so evaluating one
candidate at a time wastes ~95% of the runtime.

This module solves the damped wave equation for N candidate parameter sets
SIMULTANEOUSLY, with state arrays of shape (n_cand, n_nodes+1). Identical
physics to rodstring.py, but ~30x faster per candidate, which is what makes the
variable-projection inversion practical.

Fault codes (integer, vectorised):
    0 normal   1 fluid_pound   2 gas_interference   3 tv_leak
    4 sv_leak  5 worn_pump     6 rod_part           7 tagging
"""

from __future__ import annotations

import numpy as np

from rodstring import E_STEEL, G, RHO_STEEL, V_SOUND, FAULTS
import rodstring as _rs

FAULT_CODE = {f: i for i, f in enumerate(FAULTS)}
CODE_FAULT = {i: f for f, i in FAULT_CODE.items()}


def pump_force_batch(
    x_pl: np.ndarray,
    v_pl: np.ndarray,
    stroke: float,
    F: np.ndarray,
    t_up: np.ndarray,
    P: dict,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised pump boundary force. P holds per-candidate parameter arrays."""
    code = P["code"]
    Fo = P["Fo"]
    phi = P["phi"]

    up = v_pl < 0.0
    t_up_new = np.where(up, t_up + dt, 0.0)

    # ---------------- UPSTROKE ----------------
    target_up = np.where(
        (code == 3) | (code == 5), Fo * np.exp(-P["tv_rate"] * t_up_new), Fo
    )
    tau_up = np.where(code == 2, 0.25, 0.06)
    tau_up = np.where((code == 4) | (code == 5), 0.06 + 1.8 * P["sv_rate"], tau_up)
    F_up = F + (target_up - F) * (1.0 - np.exp(-dt / tau_up))

    # ---------------- DOWNSTROKE ----------------
    travel = np.clip(x_pl / max(stroke, 1e-9), 0.0, 1.0)
    contacted = travel >= (1.0 - phi)

    excess = np.maximum(travel - (1.0 - phi), 0.0)
    w_gas = 1.0 - np.exp(-excess / np.maximum(P["gas_c"], 1e-4))
    F_gas = Fo * (1.0 - w_gas)
    F_pound = F + (0.0 - F) * (1.0 - np.exp(-dt / 0.010))
    F_contact = np.where(code == 2, F_gas, F_pound)
    F_void = np.where((code == 4) | (code == 5), F * np.exp(-P["sv_rate"] * dt), F)
    F_dn = np.where(contacted, F_contact, F_void)

    F_new = np.where(up, F_up, F_dn)

    # ---------------- tagging ----------------
    frac_bot = 1.0 - travel
    tag = (
        (code == 7)
        & (~up)
        & (frac_bot < P["tag_frac"])
    )
    F_new = F_new + np.where(
        tag, P["tag_force"] * (1.0 - frac_bot / np.maximum(P["tag_frac"], 1e-6)), 0.0
    )

    # ---------------- parted rod carries nothing ----------------
    F_new = np.where(code == 6, 0.0, F_new)
    return np.maximum(F_new, 0.0), t_up_new


def simulate_batch(
    s_of_t: np.ndarray,
    dt: float,
    params: dict,
    L: float = 1200.0,
    d_rod: float = 0.0222,
    damping: float = 2.5,
    rho_fluid: float = 900.0,
    n_nodes: int = 24,
    n_warmup: int = 1,
) -> dict:
    """
    Batched forward solve.

    params : dict of per-candidate arrays, each shape (n_cand,):
             code, Fo, phi, tv_rate, sv_rate, gas_c, tag_force, tag_frac, rod_frac
    Returns PRL of shape (n_cand, n_t).
    """
    n_cand = len(params["code"])
    A_rod = np.pi * d_rod**2 / 4.0
    EA = E_STEEL * A_rod
    g_eff = G * (1.0 - rho_fluid / RHO_STEEL)

    # Parted rods shorten the effective string. Solve the batch at the SHORTEST
    # effective length grid by rescaling dx per candidate (dx varies, r2 varies).
    L_eff = L * params["rod_frac"]  # (n_cand,)
    dx = L_eff / n_nodes  # (n_cand,)

    # CFL target of 0.85 (not 1.0): the viscous damping term shrinks the
    # stability region of the explicit scheme. dx.min() is used because the
    # shortest candidate string sets the limit for the whole batch.

    cfl = V_SOUND * dt / dx.min()
    substeps = max(1, int(np.ceil(cfl / _rs.CFL_TARGET)))
    dt_sub = dt / substeps
    r2 = (V_SOUND * dt_sub / dx) ** 2  # (n_cand,)
    r2 = r2[:, None]

    stroke = float(s_of_t.max() - s_of_t.min())
    n_t = len(s_of_t)
    u_surf = s_of_t.max() - s_of_t

    # static initialisation per candidate
    xi = np.linspace(0.0, 1.0, n_nodes + 1)[None, :] * L_eff[:, None]
    F0 = np.where(params["code"] == 6, 0.0, params["Fo"])[:, None]
    C1 = F0 / EA + RHO_STEEL * g_eff * L_eff[:, None] / E_STEEL
    u_static = -RHO_STEEL * g_eff * xi**2 / (2.0 * E_STEEL) + C1 * xi
    u_prev = u_surf[0] + u_static
    u_curr = u_prev.copy()

    F = F0[:, 0].copy()
    t_up = np.zeros(n_cand)

    prl = np.empty((n_cand, n_t))
    xpl_min = np.full(n_cand, np.inf)
    xpl_max = np.full(n_cand, -np.inf)

    dx_c = dx[:, None]
    for cyc in range(n_warmup + 1):
        record = cyc == n_warmup
        for i in range(n_t):
            u_a, u_b = u_surf[i], u_surf[(i + 1) % n_t]
            for k in range(substeps):
                u_top = u_a + (u_b - u_a) * (k + 1) / substeps
                x_pl = u_curr[:, -1]
                v_pl = (u_curr[:, -1] - u_prev[:, -1]) / dt_sub
                F, t_up = pump_force_batch(
                    x_pl, v_pl, stroke, F, t_up, params, dt_sub
                )

                u_next = np.empty_like(u_curr)
                u_next[:, 1:-1] = (
                    2.0 * u_curr[:, 1:-1]
                    - u_prev[:, 1:-1]
                    + r2 * (u_curr[:, 2:] - 2.0 * u_curr[:, 1:-1] + u_curr[:, :-2])
                    + g_eff * dt_sub**2
                    - damping * dt_sub * (u_curr[:, 1:-1] - u_prev[:, 1:-1])
                )
                u_next[:, 0] = u_top
                ghost = u_curr[:, -2] + 2.0 * dx_c[:, 0] * F / EA
                u_next[:, -1] = (
                    2.0 * u_curr[:, -1]
                    - u_prev[:, -1]
                    + r2[:, 0] * (ghost - 2.0 * u_curr[:, -1] + u_curr[:, -2])
                    + g_eff * dt_sub**2
                    - damping * dt_sub * (u_curr[:, -1] - u_prev[:, -1])
                )
                u_prev, u_curr = u_curr, u_next

            if record:
                dudx0 = (
                    -3.0 * u_curr[:, 0] + 4.0 * u_curr[:, 1] - u_curr[:, 2]
                ) / (2.0 * dx_c[:, 0])
                prl[:, i] = EA * dudx0
                xpl_min = np.minimum(xpl_min, u_curr[:, -1])
                xpl_max = np.maximum(xpl_max, u_curr[:, -1])

    # Guard: a diverged candidate must never poison the least-squares solve.
    # Replace non-finite / absurd results with a large finite sentinel so the
    # candidate simply scores badly instead of raising.
    bad = ~np.isfinite(prl).all(axis=1) | (np.abs(prl).max(axis=1) > 5e6)
    if bad.any():
        prl[bad] = 1e7
    return {"PRL": prl, "net_stroke": xpl_max - xpl_min, "diverged": bad}


def make_params(n: int, **kw) -> dict:
    """Build a batch parameter dict with sensible defaults, broadcasting scalars."""
    d = {
        "code": 0,
        "Fo": 16000.0,
        "phi": 1.0,
        "tv_rate": 0.0,
        "sv_rate": 0.0,
        "gas_c": 0.15,
        "tag_force": 0.0,
        "tag_frac": 0.03,
        "rod_frac": 1.0,
    }
    d.update(kw)
    out = {}
    for k, v in d.items():
        arr = np.broadcast_to(np.asarray(v, dtype=float), (n,)).astype(float).copy()
        out[k] = arr
    out["code"] = out["code"].astype(int)
    return out
