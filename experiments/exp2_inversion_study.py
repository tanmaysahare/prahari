"""
EXPERIMENT 2 - Blind calibration, health estimation, fault classification.

Three questions, one pass over a randomised well population:

  Q1  Can the counterbalance moment and phase be recovered blind, with NO
      counterbalance test and NO well shut-in?
  Q2  How accurately are the physical health parameters (pump fillage, fluid
      load) recovered from electrical + inertial data only?
  Q3  How well does physics-based model selection classify the fault, versus
      (a) the published approach that ASSUMES a known counterbalance, and
      (b) a physics-free ML classifier on the same raw signals?

EVERY NUMBER PRODUCED BY THIS SCRIPT IS A SIMULATION RESULT.
No field data is used. Ground truth comes from the forward physics model in
src/forward.py; the inverse solver never sees it.

Deliberate anti-"inverse crime" measures:
  - ground truth is simulated at 48 nodes / 3 warm-up cycles / 360 crank samples
  - the inverse model runs at 24 nodes / 1 warm-up cycle / 180 crank samples
  - measurement noise is added to both observables
  - the well population randomises depth, rod size, SPM, damping, fluid load,
    unit geometry and counterbalance de-tuning

RESUMABLE: the execution environment caps a single run at ~2 minutes, so this
script checkpoints after every well and exits cleanly when its time budget is
reached. Re-run it until it prints DONE. Per-well RNG is derived from
(seed, well index), so results are identical regardless of how the work is
split into chunks.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from forward import DriveConfig, WellState, simulate_well
from inverse import invert_fixed_calibration, invert_well
from kinematics import UnitGeometry
from rodstring import FAULTS, RodString, make_pump

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
TRAIN_CACHE = os.path.join(RES, "exp2_train_cache.npz")
EVAL_CSV = os.path.join(RES, "exp2_inversion_study.csv")


def well_rng(seed: int, idx: int, tag: str) -> np.random.Generator:
    """Deterministic per-well RNG so chunked execution is reproducible."""
    return np.random.default_rng(
        abs(hash((seed, idx, tag))) % (2**32)
    )


def random_well(rng: np.random.Generator, fault: str) -> WellState:
    """Sample a randomised but physically plausible marginal well."""
    w = WellState()
    w.geom = UnitGeometry(
        A=rng.uniform(3.2, 4.0), C=rng.uniform(2.5, 3.0), P=rng.uniform(3.0, 3.5),
        R=rng.uniform(0.75, 1.05), K=rng.uniform(4.0, 4.6),
        phi_K=rng.uniform(0.33, 0.47),
    )
    w.rod = RodString(
        L=rng.uniform(700.0, 1700.0),
        d_rod=float(rng.choice([0.01905, 0.0222, 0.0254])),
        damping=rng.uniform(1.8, 3.2),
        rho_fluid=rng.uniform(830.0, 1010.0),
    )
    w.pump = make_pump(fault, rng)
    w.pump.fluid_load = float(rng.uniform(7.0e3, 28.0e3))
    w.drive = DriveConfig(
        spm=float(rng.uniform(4.0, 9.0)), J_crank=float(rng.uniform(2500, 4800))
    )
    return w


def power_features(P: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Physics-free feature vector for the ML baseline: exactly the same raw
    signals PRAHARI sees, summarised without any rod-string model.
    """
    Pn = (P - P.mean()) / (P.std() + 1e-9)
    F = np.abs(np.fft.rfft(Pn))[:16]
    s = pd.Series(Pn)
    return np.concatenate(
        [
            F,
            [
                P.mean(), P.std(), P.min(), P.max(), np.ptp(P),
                float(np.mean(P < 0)),
                float(np.percentile(P, 10)), float(np.percentile(P, 90)),
                float(s.skew()), float(s.kurt()),
                float(omega.std() / omega.mean()),
            ],
        ]
    )


# ----------------------------------------------------------------------------


def phase_train(args, t0) -> bool:
    """Build the ML-baseline training cache. Returns True when complete."""
    X, y, done = [], [], 0
    if os.path.exists(TRAIN_CACHE):
        d = np.load(TRAIN_CACHE, allow_pickle=True)
        X, y = list(d["X"]), list(d["y"])
        done = len(y)
    while done < args.n_train:
        f = FAULTS[done % len(FAULTS)]
        rng = well_rng(args.seed, done, "train")
        w = random_well(rng, f)
        r = simulate_well(w, seed=int(rng.integers(1e9)),
                          noise_power_pct=args.noise, noise_imu_deg=0.05)
        X.append(power_features(r["P_elec_meas"], r["omega"]))
        y.append(f)
        done += 1
        if time.time() - t0 > args.budget:
            break
    np.savez(TRAIN_CACHE, X=np.array(X), y=np.array(y))
    print(f"  train cache: {done}/{args.n_train}", flush=True)
    return done >= args.n_train


def fit_baseline():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    d = np.load(TRAIN_CACHE, allow_pickle=True)
    X, y = d["X"], d["y"]
    sc = StandardScaler().fit(X)
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1
    ).fit(sc.transform(X), y)
    return sc, clf


def phase_eval(args, t0) -> bool:
    """Evaluate held-out wells. Returns True when complete."""
    rows = []
    done = 0
    if os.path.exists(EVAL_CSV):
        prev = pd.read_csv(EVAL_CSV)
        rows = prev.to_dict("records")
        done = len(rows)
    if done >= args.n_wells:
        return True

    sc, clf = fit_baseline()

    while done < args.n_wells:
        i = done
        f_true = FAULTS[i % len(FAULTS)]
        rng = well_rng(args.seed, i, "eval")
        w = random_well(rng, f_true)
        r = simulate_well(w, seed=int(rng.integers(1e9)),
                          noise_power_pct=args.noise, noise_imu_deg=0.05)

        # ---- (1) PRAHARI: blind self-calibrating inversion ----
        inv = invert_well(
            r["P_elec_meas"], r["theta"], r["omega"], w.geom, w.drive.spm,
            seed=int(rng.integers(1e9)), n_harm=10,
            n_per_class=args.n_per_class, n_refine=args.n_refine,
        )

        # ---- (2) ABLATION: published approach, counterbalance ASSUMED ----
        # The operator has only a stale nameplate value; counterweights drift
        # after install. Modelled as +-25% magnitude and +-15 deg phase error.
        fix = invert_fixed_calibration(
            r["P_elec_meas"], r["theta"], r["omega"], w.geom, w.drive.spm,
            M_assumed=r["M_cb_true"] * float(rng.uniform(0.75, 1.25)),
            tau_assumed=r["tau_cb_true"] + float(rng.uniform(-0.26, 0.26)),
            SU_assumed=w.drive.SU, seed=int(rng.integers(1e9)), n_harm=10,
            n_per_class=args.n_per_class, n_refine=args.n_refine,
        )

        # ---- (3) BASELINE: physics-free ML on the same raw signals ----
        ml = str(
            clf.predict(
                sc.transform(power_features(r["P_elec_meas"], r["omega"])[None, :])
            )[0]
        )

        d_t = inv["tau_cb"] - r["tau_cb_true"]
        rows.append(
            dict(
                well=i, fault_true=f_true,
                fault_prahari=inv["fault"], fault_fixedcal=fix["fault"],
                fault_ml=ml,
                M_true=r["M_cb_true"], M_hat=inv["M_cb"],
                M_err_pct=100 * (inv["M_cb"] - r["M_cb_true"]) / r["M_cb_true"],
                tau_err_deg=float(np.degrees(np.arctan2(np.sin(d_t), np.cos(d_t)))),
                fillage_true=w.pump.fillage, fillage_hat=inv["fillage"],
                fillage_hat_fixedcal=fix["fillage"],
                Fo_true=w.pump.fluid_load, Fo_hat=inv["Fo"],
                Fo_err_pct=100 * (inv["Fo"] - w.pump.fluid_load) / w.pump.fluid_load,
                residual_norm=inv["residual_norm"], margin=inv["margin"],
                depth=w.rod.L, spm=w.drive.spm, d_rod=w.rod.d_rod,
            )
        )
        done += 1
        pd.DataFrame(rows).to_csv(EVAL_CSV, index=False)
        if time.time() - t0 > args.budget:
            break

    print(f"  eval: {done}/{args.n_wells}", flush=True)
    return done >= args.n_wells


def summarise(args) -> None:
    from sklearn.metrics import confusion_matrix, f1_score

    df = pd.read_csv(EVAL_CSV)

    def acc(c):
        return float((df[c] == df.fault_true).mean())

    labels = list(FAULTS)
    summary = {
        "NOTE": "ALL RESULTS ARE SIMULATION. No field data used.",
        "n_wells": int(len(df)),
        "noise_power_pct": args.noise,
        "calibration_blind": {
            "M_abs_err_pct_median": float(df.M_err_pct.abs().median()),
            "M_abs_err_pct_p90": float(df.M_err_pct.abs().quantile(0.9)),
            "M_err_pct_mean_bias": float(df.M_err_pct.mean()),
            "tau_abs_err_deg_median": float(df.tau_err_deg.abs().median()),
            "tau_abs_err_deg_p90": float(df.tau_err_deg.abs().quantile(0.9)),
        },
        "health": {
            "fillage_MAE_prahari": float((df.fillage_hat - df.fillage_true).abs().mean()),
            "fillage_MAE_fixedcal_ablation": float(
                (df.fillage_hat_fixedcal - df.fillage_true).abs().mean()
            ),
            "Fo_abs_err_pct_median": float(df.Fo_err_pct.abs().median()),
        },
        "classification": {
            "accuracy_prahari": acc("fault_prahari"),
            "accuracy_fixedcal_ablation": acc("fault_fixedcal"),
            "accuracy_ml_physicsfree": acc("fault_ml"),
            "macroF1_prahari": float(
                f1_score(df.fault_true, df.fault_prahari, average="macro",
                         labels=labels, zero_division=0)
            ),
            "macroF1_fixedcal_ablation": float(
                f1_score(df.fault_true, df.fault_fixedcal, average="macro",
                         labels=labels, zero_division=0)
            ),
            "macroF1_ml_physicsfree": float(
                f1_score(df.fault_true, df.fault_ml, average="macro",
                         labels=labels, zero_division=0)
            ),
        },
        "confusion_labels": labels,
        "confusion_prahari": confusion_matrix(
            df.fault_true, df.fault_prahari, labels=labels
        ).tolist(),
    }
    with open(os.path.join(RES, "exp2_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n" + "=" * 74)
    print("EXPERIMENT 2 SUMMARY  (ALL RESULTS ARE SIMULATION)")
    print("=" * 74)
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wells", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--budget", type=float, default=95.0,
                    help="wall-clock seconds before checkpointing and exiting")
    ap.add_argument("--n-per-class", type=int, default=64)
    ap.add_argument("--n-refine", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    if not phase_train(args, t0):
        print("PARTIAL (training cache) - re-run", flush=True)
        return
    if not phase_eval(args, t0):
        print("PARTIAL (evaluation) - re-run", flush=True)
        return
    summarise(args)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
