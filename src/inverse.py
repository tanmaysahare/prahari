"""
PRAHARI - Blind self-calibrating inversion.  << THE CORE CONTRIBUTION >>

PROBLEM
-------
Every published method that infers rod-pump condition from motor electrical
data requires the maximum counterbalance moment M and the structural unbalance
SU as INPUTS. In a brownfield unit these are:
  - not recorded (counterweights get moved and never logged),
  - physically hidden (auxiliary weights sit inside the master-weight pocket),
  - obtainable only by a counterbalance test, which requires SHUTTING THE WELL IN.

That single dependency is why a 14-year-old technique is not deployed at scale
on marginal wells.

APPROACH - VARIABLE PROJECTION
------------------------------
Write the observed motor power as

    P(theta) = [ kappa*g(theta)*(PRL(theta) - SU) - M*sin(theta+tau) ] * omega(theta)
               / efficiency  +  P_noload

where g(theta) = dpsi/dtheta is measured EXACTLY by the beam IMU, and
omega(theta) is measured EXACTLY by differentiating the IMU crank timing.

Define the torque proxy  z(theta) = P_meas(theta) * e(theta) / omega(theta).
Then the model is LINEAR in the calibration parameters:

    z = X @ beta,
    X    = [ g*PRL ,  -g ,  -sin(theta) ,  -cos(theta) ,  -e/omega ]
    beta = [ kappa , kappa*SU , M*cos(tau) , M*sin(tau) , P_noload ]

    =>  M = hypot(beta2, beta3),  tau = atan2(beta3, beta2),
        SU = beta1/beta0,         kappa = beta0

So for ANY candidate pump-health vector we can solve the four calibration
unknowns in CLOSED FORM by least squares. The nonlinear search therefore
collapses onto the 1-4 dimensional pump-health manifold only. This is the
classical variable-projection (VarPro) separation, and it is what makes blind
calibration tractable.

IDENTIFIABILITY
---------------
The counterbalance is a PURE first harmonic in theta. Could the fit absorb it
into kappa*g*PRL? Only if PRL were unconstrained. It is not: PRL is restricted
to the manifold of physically admissible damped-wave-equation responses,
spanned by at most four pump-health parameters. That physical constraint is
what breaks the degeneracy - and we measure the residual degeneracy empirically
in Experiment 2 rather than asserting it.

FAULT CLASSIFICATION IS A BY-PRODUCT
------------------------------------
We fit every fault class and select by residual. That makes the classifier a
PHYSICAL MODEL-SELECTION procedure, not a black box: the answer comes with a
fitted physical parameter set and a residual we can report as a confidence.
"""

from __future__ import annotations

import numpy as np

from batch_sim import CODE_FAULT, FAULT_CODE, make_params, simulate_batch
from kinematics import UnitGeometry, polished_rod_position, torque_factor

ETA_COMB = 0.855  # gearbox x motor, combined (nominal; error absorbed by kappa)

# Search ranges per fault class: (param name -> (lo, hi))
CLASS_SEARCH = {
    "normal": {"Fo": (4e3, 34e3)},
    "fluid_pound": {"Fo": (4e3, 34e3), "phi": (0.25, 0.75)},
    "gas_interference": {"Fo": (4e3, 34e3), "phi": (0.40, 0.85), "gas_c": (0.05, 0.30)},
    "tv_leak": {"Fo": (4e3, 34e3), "tv_rate": (0.25, 1.40)},
    "sv_leak": {"Fo": (4e3, 34e3), "sv_rate": (0.25, 1.40)},
    "worn_pump": {
        "Fo": (4e3, 34e3), "phi": (0.65, 1.0),
        "tv_rate": (0.35, 1.10), "sv_rate": (0.35, 1.10),
    },
    "rod_part": {"Fo": (4e3, 34e3), "rod_frac": (0.30, 0.85)},
    "tagging": {"Fo": (4e3, 34e3), "tag_force": (1e3, 18e3)},
}


def bandlimit(x: np.ndarray, n_harm: int) -> np.ndarray:
    """
    Keep only the first `n_harm` crank harmonics.

    Physically justified, not a convenience: the counterbalance signature is a
    PURE first harmonic and the pump-health signature lives in the low-order
    harmonics, whereas rod-string ringing sits at the string's natural modes
    (~10-20x crank frequency). A real clamp-on CT feeding an edge ADC is
    anti-alias filtered anyway. Band-limiting before the linear solve stops
    un-modelled high-frequency ringing from biasing the calibration estimate.
    """
    X = np.fft.rfft(x, axis=-1)
    X[..., n_harm + 1:] = 0.0
    return np.fft.irfft(X, n=x.shape[-1], axis=-1)


def _varpro(z: np.ndarray, gth: np.ndarray, PRL: np.ndarray,
            theta: np.ndarray, e_over_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form calibration solve for a BATCH of candidate PRL curves.

    PRL : (n_cand, n_t)
    Returns (beta (n_cand,5), residual_rms (n_cand,))
    """
    n_cand, n_t = PRL.shape
    # Columns that do NOT depend on the candidate
    base = np.stack([-gth, -np.sin(theta), -np.cos(theta), -e_over_w], axis=1)  # (n_t,4)
    a = gth[None, :] * PRL  # (n_cand, n_t) -- the only candidate-dependent column

    # Batched normal equations. Only the first row/column of X^T X varies with
    # the candidate, so the fixed 4x4 block and the fixed 4-vector are computed
    # once. This replaces n_cand separate lstsq calls with a single batched
    # 5x5 solve.
    CtC = base.T @ base  # (4,4) fixed
    Ctz = base.T @ z  # (4,)  fixed
    aC = a @ base  # (n_cand,4)
    aa = np.einsum("ij,ij->i", a, a)  # (n_cand,)
    az = a @ z  # (n_cand,)

    XtX = np.empty((n_cand, 5, 5))
    XtX[:, 0, 0] = aa
    XtX[:, 0, 1:] = aC
    XtX[:, 1:, 0] = aC
    XtX[:, 1:, 1:] = CtC
    Xtz = np.empty((n_cand, 5))
    Xtz[:, 0] = az
    Xtz[:, 1:] = Ctz

    # RELATIVE ridge only. An absolute ridge scaled to the largest diagonal
    # entry would swamp the smaller columns (the columns differ by ~9 orders of
    # magnitude because PRL is in newtons), corrupting the calibration solve.
    d = np.arange(5)
    XtX[:, d, d] *= 1.0 + 1e-10
    try:
        beta = np.linalg.solve(XtX, Xtz[..., None])[..., 0]  # numpy>=2 batched RHS
    except np.linalg.LinAlgError:
        # A degenerate candidate can make one 5x5 block singular. Fall back to
        # the pseudo-inverse so that candidate scores badly rather than
        # aborting the whole well.
        beta = (np.linalg.pinv(XtX) @ Xtz[..., None])[..., 0]
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

    # residual^2 = (z.z - 2 b.Xtz + b.XtX.b) / n_t
    zz = float(z @ z)
    quad = np.einsum("ij,ijk,ik->i", beta, XtX, beta)
    lin = np.einsum("ij,ij->i", beta, Xtz)
    res = np.sqrt(np.maximum((zz - 2.0 * lin + quad) / n_t, 0.0))
    return beta, res


def _sample(rng: np.random.Generator, fault: str, n: int,
            centre: dict | None = None, width: float = 1.0) -> dict:
    """Sample n candidate parameter sets for one fault class."""
    spec = CLASS_SEARCH[fault]
    kw = {"code": FAULT_CODE[fault]}
    for name, (lo, hi) in spec.items():
        if centre is None:
            kw[name] = rng.uniform(lo, hi, n)
        else:
            span = (hi - lo) * width
            kw[name] = np.clip(
                rng.normal(centre[name], span / 2.0, n), lo, hi
            )
    return make_params(n, **kw)


def _cat(dicts: list[dict]) -> dict:
    keys = dicts[0].keys()
    return {k: np.concatenate([d[k] for d in dicts]) for k in keys}


def invert_well(
    P_meas: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
    geom: UnitGeometry,
    spm: float,
    n_per_class: int = 96,
    n_refine: int = 24,
    n_nodes: int = 24,
    n_warmup: int = 1,
    seed: int = 0,
    n_harm: int = 12,
    n_samp: int = 180,
    classes: tuple[str, ...] | None = None,
) -> dict:
    """
    Blind inversion of one well from motor power + beam IMU only.

    Returns the best fit across all fault classes: fault label, physical health
    parameters, recovered calibration (M_cb, tau_cb, SU), reconstructed PRL,
    and a normalised residual used as a confidence measure.
    """
    rng = np.random.default_rng(seed)
    classes = classes or tuple(CLASS_SEARCH.keys())

    # Decimate to `n_samp` crank samples. Nyquist for the harmonics we fit is
    # 2*n_harm; 180 samples is far above that, and it halves solver cost.
    if len(theta) > n_samp:
        k = len(theta) // n_samp
        theta, P_meas, omega = theta[::k], P_meas[::k], omega[::k]

    # --- quantities the sensor kit measures directly ---
    s = polished_rod_position(theta, geom)
    gth = torque_factor(theta, geom)  # ds/dtheta, from the IMU
    e = np.where(P_meas >= 0.0, ETA_COMB, 1.0 / ETA_COMB)
    z = P_meas * e / omega
    e_over_w = e / omega
    dt = (60.0 / spm) / len(theta)

    zb = bandlimit(z, n_harm)

    def evaluate(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The CFL sub-step count is set by the SHORTEST candidate rod string, so
        # a single parted-rod candidate would otherwise slow the whole batch by
        # ~3x. Solve short and full-length candidates as separate batches.
        short = params["rod_frac"] < 0.95
        PRL = np.empty((len(params["code"]), len(s)))
        for mask in (~short, short):
            if not mask.any():
                continue
            sub = {k: v[mask] for k, v in params.items()}
            PRL[mask] = simulate_batch(
                s, dt, sub, n_nodes=n_nodes, n_warmup=n_warmup
            )["PRL"]
        beta, res = _varpro(zb, gth, bandlimit(PRL, n_harm), theta, e_over_w)
        return beta, res, PRL

    # ---- STAGE 1: broad random search over every fault class, one batch ----
    parts = [_sample(rng, f, n_per_class) for f in classes]
    P1 = _cat(parts)
    beta1, res1, PRL1 = evaluate(P1)

    # ---- STAGE 2: local refinement around the best candidate of each class ----
    refine_parts = []
    for f in classes:
        m = P1["code"] == FAULT_CODE[f]
        idx = np.where(m)[0]
        best = idx[np.argmin(res1[idx])]
        centre = {k: P1[k][best] for k in CLASS_SEARCH[f]}
        refine_parts.append(_sample(rng, f, n_refine, centre=centre, width=0.22))
    P2 = _cat(refine_parts)
    beta2, res2, PRL2 = evaluate(P2)

    # ---- combine and select ----
    Pall = _cat([P1, P2])
    beta = np.concatenate([beta1, beta2])
    res = np.concatenate([res1, res2])
    PRL = np.concatenate([PRL1, PRL2])

    b_i = int(np.argmin(res))
    bb = beta[b_i]
    kappa = bb[0]
    SU = bb[1] / bb[0] if abs(bb[0]) > 1e-9 else 0.0
    M = float(np.hypot(bb[2], bb[3]))
    tau = float(np.arctan2(bb[3], bb[2]))

    # per-class best residual -> soft confidence over fault classes
    class_res = {}
    for f in classes:
        m = Pall["code"] == FAULT_CODE[f]
        class_res[f] = float(res[m].min())
    order = sorted(class_res, key=class_res.get)
    best_f, second_f = order[0], order[1]
    # separation margin: how much better the winner is than the runner-up
    margin = (class_res[second_f] - class_res[best_f]) / max(class_res[best_f], 1e-9)

    return {
        "fault": CODE_FAULT[int(Pall["code"][b_i])],
        "Fo": float(Pall["Fo"][b_i]),
        "fillage": float(Pall["phi"][b_i]),
        "tv_rate": float(Pall["tv_rate"][b_i]),
        "sv_rate": float(Pall["sv_rate"][b_i]),
        "rod_frac": float(Pall["rod_frac"][b_i]),
        "tag_force": float(Pall["tag_force"][b_i]),
        "M_cb": M,
        "tau_cb": tau,
        "SU": float(SU),
        "kappa": float(kappa),
        "P_noload": float(bb[4]),
        "PRL_hat": PRL[b_i],
        "residual": float(res[b_i]),
        "residual_norm": float(res[b_i] / (np.std(zb) + 1e-9)),
        "class_residuals": class_res,
        "margin": float(margin),
        "runner_up": second_f,
    }


def invert_fixed_calibration(
    P_meas: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
    geom: UnitGeometry,
    spm: float,
    M_assumed: float,
    tau_assumed: float,
    SU_assumed: float,
    **kw,
) -> dict:
    """
    ABLATION BASELINE - the published approach.

    Represents every prior motor-power method (PA-1..PA-4 in the prior-art
    matrix): the counterbalance is taken as a KNOWN input rather than
    identified. We feed it a nameplate/stale value, which is exactly what an
    operator would have available without a shut-in counterbalance test.
    """
    rng = np.random.default_rng(kw.get("seed", 0))
    classes = kw.get("classes") or tuple(CLASS_SEARCH.keys())
    n_per_class = kw.get("n_per_class", 96)
    n_refine = kw.get("n_refine", 24)

    s = polished_rod_position(theta, geom)
    gth = torque_factor(theta, geom)
    e = np.where(P_meas >= 0.0, ETA_COMB, 1.0 / ETA_COMB)
    z = P_meas * e / omega
    e_over_w = e / omega
    dt = (60.0 / spm) / len(theta)

    # Counterbalance is SUBTRACTED using the assumed value, then only
    # (kappa, kappa*SU, P_noload) are fitted.
    z_corr = bandlimit(z + M_assumed * np.sin(theta + tau_assumed), kw.get("n_harm", 12))
    base = np.stack([-gth, -e_over_w], axis=1)

    def evaluate(params):
        sim = simulate_batch(s, dt, params, n_nodes=kw.get("n_nodes", 24),
                             n_warmup=kw.get("n_warmup", 1))
        PRL = sim["PRL"]
        PRLb = bandlimit(PRL, kw.get("n_harm", 12))
        n = PRL.shape[0]
        res = np.empty(n)
        for i in range(n):
            X = np.concatenate([(gth * PRLb[i])[:, None], base], axis=1)
            b, *_ = np.linalg.lstsq(X, z_corr, rcond=None)
            res[i] = np.sqrt(np.mean((X @ b - z_corr) ** 2))
        return res, PRL

    parts = [_sample(rng, f, n_per_class) for f in classes]
    P1 = _cat(parts)
    res1, PRL1 = evaluate(P1)
    refine_parts = []
    for f in classes:
        idx = np.where(P1["code"] == FAULT_CODE[f])[0]
        best = idx[np.argmin(res1[idx])]
        centre = {k: P1[k][best] for k in CLASS_SEARCH[f]}
        refine_parts.append(_sample(rng, f, n_refine, centre=centre, width=0.22))
    P2 = _cat(refine_parts)
    res2, PRL2 = evaluate(P2)

    Pall = _cat([P1, P2])
    res = np.concatenate([res1, res2])
    PRL = np.concatenate([PRL1, PRL2])
    b_i = int(np.argmin(res))
    return {
        "fault": CODE_FAULT[int(Pall["code"][b_i])],
        "Fo": float(Pall["Fo"][b_i]),
        "fillage": float(Pall["phi"][b_i]),
        "rod_frac": float(Pall["rod_frac"][b_i]),
        "PRL_hat": PRL[b_i],
        "residual": float(res[b_i]),
    }
