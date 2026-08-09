"""
EXPERIMENT 4 - Fleet prognostics and workover-rig allocation.

THIS IS THE EXPERIMENT THAT PRODUCES THE MONEY NUMBER.

Motivation (published industry figures, sourced in docs/REFERENCES.md):
  - In ONGC's Mehsana asset, >25% of sucker-rod-pumped wells required a
    well-servicing job within a year, and up to 40% of ALL workover jobs were
    rod-pump repairs -- which yield the LEAST oil gain of any job type.
  - Reported MTBF in an Indian mature field: as low as 4-5 months.
  => Workover rig-days, not capital, are the binding constraint, and today they
     are allocated to the lowest-value jobs.

The key physical insight this experiment encodes:
  A rod pump rarely fails abruptly. It DEGRADES -- fillage falls, valves leak,
  the plunger wears -- and the well keeps producing at a quietly reduced rate
  for weeks before anything breaks. On a marginal well that is never tested
  individually, nobody sees that loss. PRAHARI sees it, because pump fillage is
  exactly what the inversion estimates.

  So the value is NOT mainly "predict the failure". It is "find the silent
  production loss that nobody is measuring, and spend the scarce rig-days on the
  wells where recovery per rig-day is highest."

Discrete-event simulation, N wells over one year, R rigs.
Policies compared:
  reactive         run-to-failure, then queue for a rig (today's practice)
  calendar         fixed servicing interval, condition-blind
  rate_first       react to failures, prioritise by well rate
  prahari          rank by EXPECTED DEFERRED BARRELS RECOVERED PER RIG-DAY,
                   using the noisy health estimate, intervening pre-emptively

ALL RESULTS ARE SIMULATION. The health-estimate noise is set from the measured
fillage MAE of Experiment 2, so the scheduler is not given a perfect sensor.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# --- job durations, rig-days (ESTIMATES, stated as such) ---
JOB_DAYS_PREEMPTIVE = 2.0   # planned pump change-out
JOB_DAYS_FAILURE = 4.5      # unplanned: fishing, parted rod, rig waiting
HEALTH_FAIL = 0.25          # below this the well is effectively dead


def make_fleet(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    A marginal-well fleet. Rates follow a lognormal: a few good wells, a long
    tail of 1-3 t/d wells -- the shape of a mature Indian onshore asset.
    """
    q0 = np.clip(rng.lognormal(mean=np.log(2.2), sigma=0.75, size=n), 0.4, 22.0)
    return pd.DataFrame(
        {
            "q0": q0,
            # Per-day degradation rate, heterogeneous (sand, deviation, fluid).
            # CALIBRATED so that the reactive baseline services ~25-30% of wells
            # per year, matching the reported ONGC Mehsana figure of >25% of SRP
            # wells requiring a servicing job within a year.
            "deg": rng.gamma(shape=2.0, scale=0.00066, size=n),
            # probability per day of a sudden catastrophic event (rod part)
            "p_sudden": rng.uniform(0.0000, 0.00025, size=n),
            "health": rng.uniform(0.80, 1.0, size=n),
        }
    )


def efficiency(h: np.ndarray) -> np.ndarray:
    """
    Production efficiency vs pump health. Health here IS essentially pump
    fillage: a pump at 60% fillage lifts ~60% of the fluid. Below HEALTH_FAIL
    the well is down.
    """
    return np.where(h <= HEALTH_FAIL, 0.0, np.clip(h, 0.0, 1.0))


def run_policy(
    policy: str,
    fleet: pd.DataFrame,
    days: int,
    n_rigs: int,
    seed: int,
    est_sigma: float,
    survey_every: int = 7,
    calendar_interval: int = 150,
    hurdle: float = 12.0,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(fleet)
    h = fleet.health.to_numpy().copy()
    q0 = fleet.q0.to_numpy()
    deg = fleet.deg.to_numpy()
    p_sud = fleet.p_sudden.to_numpy()

    busy_until = np.zeros(n)          # day index the well is back online
    rig_free_at = np.zeros(n_rigs)
    last_service = np.zeros(n)
    h_est = h.copy()

    produced = 0.0
    potential = 0.0
    n_failures = 0
    n_jobs = 0
    rig_days_used = 0.0
    downtime_days = 0.0
    hist = []

    for d in range(days):
        online = busy_until <= d

        # ---- degradation ----
        h_prev = h.copy()
        h = np.where(online, h - deg, h)
        sudden = online & (rng.random(n) < p_sud)
        h = np.where(sudden, 0.05, h)
        h = np.clip(h, 0.0, 1.0)

        # Count TRANSITIONS into the failed state, not days spent failed.
        newly_failed = online & (h <= HEALTH_FAIL) & (h_prev > HEALTH_FAIL)
        n_failures += int(newly_failed.sum())

        # ---- production ----
        eff = efficiency(h) * online
        produced += float((q0 * eff).sum())
        potential += float(q0.sum())
        downtime_days += float((~online).sum())

        # ---- surveillance (PRAHARI only) ----
        if policy == "prahari" and d % survey_every == 0:
            # noisy health estimate; sigma comes from the measured fillage MAE
            h_est = np.clip(h + rng.normal(0.0, est_sigma, n), 0.0, 1.0)

        # ---- rig dispatch ----
        for r in range(n_rigs):
            if rig_free_at[r] > d:
                continue
            cand = np.where(busy_until <= d)[0]
            if len(cand) == 0:
                continue

            if policy == "do_nothing":
                continue

            if policy == "reactive":
                sick = cand[h[cand] <= HEALTH_FAIL]
                if len(sick) == 0:
                    continue
                pick = sick[0]  # first-come-first-served queue
                dur = JOB_DAYS_FAILURE

            elif policy == "rate_first":
                sick = cand[h[cand] <= HEALTH_FAIL]
                if len(sick) == 0:
                    continue
                pick = sick[np.argmax(q0[sick])]
                dur = JOB_DAYS_FAILURE

            elif policy == "calendar":
                due = cand[(d - last_service[cand]) >= calendar_interval]
                sick = cand[h[cand] <= HEALTH_FAIL]
                if len(sick) > 0:
                    pick, dur = sick[0], JOB_DAYS_FAILURE
                elif len(due) > 0:
                    pick, dur = due[0], JOB_DAYS_PREEMPTIVE
                else:
                    continue

            elif policy == "prahari":
                # Expected barrels recovered per rig-day if we service NOW.
                # Two components:
                #   (a) restore the CURRENT efficiency shortfall, and
                #   (b) avoid the deferred production of an imminent failure.
                he = h_est[cand]
                failed = he <= HEALTH_FAIL
                # days until predicted failure at the observed degradation rate
                rul = np.where(
                    deg[cand] > 1e-9, (he - HEALTH_FAIL) / deg[cand], 1e6
                )
                horizon = 90.0
                # barrels lost over the horizon if we do nothing
                loss_rate_now = q0[cand] * (1.0 - efficiency(he))
                exposure = np.minimum(rul, horizon)
                loss = loss_rate_now * horizon + q0[cand] * np.maximum(
                    horizon - exposure, 0.0
                ) * 0.5
                dur_c = np.where(failed, JOB_DAYS_FAILURE, JOB_DAYS_PREEMPTIVE)
                value = loss / dur_c  # tonnes recovered per rig-day
                # ABSOLUTE economic hurdle, not a percentile: a rig is only
                # dispatched if the expected recovery clears `hurdle` tonnes per
                # rig-day. Without this the scheduler would always keep rigs
                # busy regardless of whether the work was worth doing.
                worth = value > hurdle
                if not worth.any():
                    continue
                j = int(np.argmax(np.where(worth, value, -np.inf)))
                pick, dur = cand[j], float(dur_c[j])
            else:
                raise ValueError(policy)

            busy_until[pick] = d + dur
            rig_free_at[r] = d + dur
            last_service[pick] = d + dur
            h[pick] = rng.uniform(0.94, 1.0)  # restored, not perfect
            h_est[pick] = h[pick]
            n_jobs += 1
            rig_days_used += dur

        hist.append(produced)

    return {
        "policy": policy,
        "produced_t": produced,
        "potential_t": potential,
        "recovery_pct": 100.0 * produced / potential,
        "deferred_t": potential - produced,
        "n_failures": n_failures,
        "n_jobs": n_jobs,
        "rig_days_used": rig_days_used,
        "rig_utilisation_pct": 100.0 * rig_days_used / (n_rigs * days),
        "t_per_rig_day": produced / max(rig_days_used, 1e-9),
        "well_down_days": downtime_days,
        "history": hist,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wells", type=int, default=400)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--n-rigs", type=int, default=3)
    ap.add_argument("--hurdle", type=float, default=12.0,
                    help="min expected tonnes recovered per rig-day to dispatch")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--est-sigma", type=float, default=0.043,
                    help="health-estimate noise = measured fillage MAE from Exp 2")
    args = ap.parse_args()

    policies = ["do_nothing", "reactive", "calendar", "rate_first", "prahari"]
    rows = []
    for rep in range(args.reps):
        rng = np.random.default_rng(1000 + rep)
        fleet = make_fleet(args.n_wells, rng)
        for p in policies:
            # SAME fleet and SAME random seed for every policy -> paired comparison
            out = run_policy(p, fleet, args.days, args.n_rigs,
                             seed=5000 + rep, est_sigma=args.est_sigma,
                             hurdle=args.hurdle)
            out.pop("history")
            out["rep"] = rep
            rows.append(out)

    df = pd.DataFrame(rows)
    # Marginal tonnes recovered per rig-day, measured against a do-nothing
    # reference within the SAME replicate. This is the metric an asset manager
    # actually optimises: what do I get for each scarce rig-day I spend?
    ref = df[df.policy == "do_nothing"].set_index("rep").produced_t
    df["marginal_t"] = df.produced_t - df.rep.map(ref)
    df["marginal_t_per_rig_day"] = df.marginal_t / df.rig_days_used.replace(0, np.nan)
    df.to_csv(os.path.join(RES, "exp4_fleet_scheduling.csv"), index=False)

    agg = df.groupby("policy").agg(["mean", "std"]).round(2)
    base = df[df.policy == "reactive"].groupby("rep").produced_t.first()
    pra = df[df.policy == "prahari"].groupby("rep").produced_t.first()
    uplift = 100.0 * (pra - base) / base

    # paired t-test across replicates
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(pra, base)

    summary = {
        "NOTE": "ALL RESULTS ARE SIMULATION (discrete-event fleet model).",
        "config": {
            "n_wells": args.n_wells, "days": args.days, "n_rigs": args.n_rigs,
            "reps": args.reps,
            "health_estimate_sigma": args.est_sigma,
            "job_days_preemptive": JOB_DAYS_PREEMPTIVE,
            "job_days_failure": JOB_DAYS_FAILURE,
        },
        "mean_by_policy": {
            p: {
                "produced_t": float(df[df.policy == p].produced_t.mean()),
                "recovery_pct": float(df[df.policy == p].recovery_pct.mean()),
                "n_failures": float(df[df.policy == p].n_failures.mean()),
                "n_jobs": float(df[df.policy == p].n_jobs.mean()),
                "marginal_t_per_rig_day": float(df[df.policy == p].marginal_t_per_rig_day.mean()),
                "marginal_t": float(df[df.policy == p].marginal_t.mean()),
                "rig_utilisation_pct": float(
                    df[df.policy == p].rig_utilisation_pct.mean()
                ),
            }
            for p in policies
        },
        "prahari_vs_reactive": {
            "production_uplift_pct_mean": float(uplift.mean()),
            "production_uplift_pct_std": float(uplift.std()),
            "extra_tonnes_per_year_mean": float((pra - base).mean()),
            "paired_t_stat": float(t_stat),
            "paired_p_value": float(p_val),
            "failures_avoided_pct": float(
                100 * (1 - df[df.policy == "prahari"].n_failures.mean()
                       / df[df.policy == "reactive"].n_failures.mean())
            ),
            "marginal_t_per_rig_day_prahari": float(
                df[df.policy == "prahari"].marginal_t_per_rig_day.mean()
            ),
            "marginal_t_per_rig_day_reactive": float(
                df[df.policy == "reactive"].marginal_t_per_rig_day.mean()
            ),
            "marginal_t_per_rig_day_gain_pct": float(
                100 * (df[df.policy == "prahari"].marginal_t_per_rig_day.mean()
                       / df[df.policy == "reactive"].marginal_t_per_rig_day.mean() - 1)
            ),
            "wells_serviced_pct_reactive": float(
                100 * df[df.policy == "reactive"].n_jobs.mean() / args.n_wells
            ),
        },
    }
    # ---- FAIREST COMPARISON: vs the CALENDAR policy ----
    # Comparing against `reactive` alone would be a strawman, because reactive
    # leaves rig capacity idle. `calendar` is the realistic incumbent: operators
    # do schedule preventive work, they just do it condition-blind.
    cal = df[df.policy == "calendar"].groupby("rep").produced_t.first()
    up_cal = 100.0 * (pra - cal) / cal
    t_cal, p_cal = stats.ttest_rel(pra, cal)
    summary["prahari_vs_calendar"] = {
        "production_uplift_pct_mean": float(up_cal.mean()),
        "production_uplift_pct_std": float(up_cal.std()),
        "paired_t_stat": float(t_cal),
        "paired_p_value": float(p_cal),
        "marginal_t_per_rig_day_gain_pct": float(
            100 * (df[df.policy == "prahari"].marginal_t_per_rig_day.mean()
                   / df[df.policy == "calendar"].marginal_t_per_rig_day.mean() - 1)
        ),
    }

    # ---- SENSITIVITY: how good does the sensor actually need to be? ----
    sens = []
    rng0 = np.random.default_rng(77)
    fleet0 = make_fleet(args.n_wells, rng0)
    for sig in [0.0, 0.02, 0.043, 0.08, 0.15, 0.25]:
        o = run_policy("prahari", fleet0, args.days, args.n_rigs, seed=99,
                       est_sigma=sig, hurdle=args.hurdle)
        sens.append({"est_sigma": sig, "produced_t": o["produced_t"],
                     "recovery_pct": o["recovery_pct"],
                     "n_failures": o["n_failures"]})
    base_cal = run_policy("calendar", fleet0, args.days, args.n_rigs, seed=99,
                          est_sigma=0.0, hurdle=args.hurdle)
    for r_ in sens:
        r_["uplift_vs_calendar_pct"] = 100 * (
            r_["produced_t"] / base_cal["produced_t"] - 1)
    summary["sensitivity_to_health_estimate_noise"] = sens

    with open(os.path.join(RES, "exp4_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 74)
    print("EXPERIMENT 4 - FLEET SCHEDULING (SIMULATION)")
    print("=" * 74)
    print(agg[[("produced_t", "mean"), ("recovery_pct", "mean"),
               ("n_failures", "mean"), ("n_jobs", "mean"),
               ("marginal_t_per_rig_day", "mean")]].to_string())
    print()
    print("vs REACTIVE :", json.dumps(summary["prahari_vs_reactive"], indent=2))
    print("vs CALENDAR :", json.dumps(summary["prahari_vs_calendar"], indent=2))
    print("SENSITIVITY :", json.dumps(summary["sensitivity_to_health_estimate_noise"],
                                      indent=2))

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    order = ["reactive", "calendar", "rate_first", "prahari"]
    lbl = {"reactive": "Reactive\n(today)", "calendar": "Calendar",
           "rate_first": "Rate-first", "prahari": "PRAHARI"}
    cols = ["#8a8a8a", "#8a8a8a", "#8a8a8a", "#1b3a5c"]

    for ax, (metric, title, unit) in zip(
        axes,
        [("recovery_pct", "Production recovery", "% of potential"),
         ("n_failures", "Unplanned failures per year", "count"),
         ("marginal_t_per_rig_day", "MARGINAL tonnes recovered\nper rig-day spent", "t / rig-day")],
    ):
        m = [df[df.policy == p][metric].mean() for p in order]
        e = [df[df.policy == p][metric].std() for p in order]
        ax.bar([lbl[p] for p in order], m, yerr=e, capsize=4, color=cols)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(
        f"PRAHARI fleet simulation - {args.n_wells} wells, {args.n_rigs} rigs, "
        f"{args.days} days, {args.reps} paired replicates (SIMULATION)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_fleet_scheduling.png"), dpi=150)
    print(f"\nFigure -> {os.path.join(FIG, 'fig4_fleet_scheduling.png')}")


if __name__ == "__main__":
    main()
