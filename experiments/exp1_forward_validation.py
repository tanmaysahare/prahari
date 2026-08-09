"""
EXPERIMENT 1 - Forward-model validation.

Purpose: demonstrate that the PRAHARI physics engine produces surface
dynamometer cards whose shapes are the textbook signatures of each fault mode,
and that the static limits agree with closed-form theory.

ALL RESULTS ARE SIMULATION. No field data is used in this experiment.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forward import WellState, card_features, simulate_well
from rodstring import FAULTS, make_pump

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

TITLES = {
    "normal": "Normal (full pump)",
    "fluid_pound": "Fluid pound (low fillage)",
    "gas_interference": "Gas interference",
    "tv_leak": "Travelling-valve leak",
    "sv_leak": "Standing-valve leak",
    "worn_pump": "Worn pump (both valves)",
    "rod_part": "Parted rod string",
    "tagging": "Pump tagging bottom",
}


def main() -> None:
    rng = np.random.default_rng(7)
    rows = []

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.6))
    fig2, axes2 = plt.subplots(2, 4, figsize=(19, 8.6))

    for ax, ax2, f in zip(axes.ravel(), axes2.ravel(), FAULTS):
        w = WellState()
        w.pump = make_pump(f, rng)
        r = simulate_well(w, seed=11)
        cf = card_features(r["PRL"], r["s"])

        # --- surface dynamometer card ---
        ax.plot(r["s"], r["PRL"] / 1000.0, lw=1.7, color="#1b3a5c")
        ax.axhline(
            w.rod.weight_buoyed / 1000.0, ls="--", lw=0.9, color="#999",
            label="buoyed rod weight",
        )
        ax.axhline(
            (w.rod.weight_buoyed + w.pump.fluid_load) / 1000.0, ls=":", lw=0.9,
            color="#c1440e", label="rod weight + $F_o$",
        )
        ax.set_title(TITLES[f], fontsize=11, fontweight="bold")
        ax.set_xlabel("Polished rod position (m)")
        ax.set_ylabel("Polished rod load (kN)")
        ax.grid(alpha=0.25)
        if f == "normal":
            ax.legend(fontsize=7, loc="lower right")

        # --- observable electrical power ---
        ax2.plot(np.degrees(r["theta"]), r["P_elec_true"] / 1000.0, lw=1.6,
                 color="#7a1f3d")
        ax2.set_title(TITLES[f], fontsize=11, fontweight="bold")
        ax2.set_xlabel("Crank angle (deg)")
        ax2.set_ylabel("Motor electrical power (kW)")
        ax2.grid(alpha=0.25)
        ax2.axhline(0, color="k", lw=0.6)

        rows.append(
            dict(
                fault=f,
                PRL_min_kN=cf["PRL_min"] / 1e3,
                PRL_max_kN=cf["PRL_max"] / 1e3,
                PRL_range_kN=cf["PRL_range"] / 1e3,
                card_area_kJ=cf["card_area_J"] / 1e3,
                net_stroke_m=r["net_stroke"],
                surface_stroke_m=r["stroke"],
                P_peak_kW=r["P_elec_true"].max() / 1e3,
                P_mean_kW=r["P_elec_true"].mean() / 1e3,
                speed_ripple_pct=100 * np.ptp(r["omega"]) / r["omega"].mean(),
                fillage=w.pump.fillage,
            )
        )

    fig.suptitle(
        "PRAHARI forward model - simulated surface dynamometer cards by fault mode "
        "(SIMULATION, not field data)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_cards_by_fault.png"), dpi=150)

    fig2.suptitle(
        "PRAHARI - the ONLY electrical signal PRAHARI actually observes "
        "(clamp-on CT). SIMULATION.",
        fontsize=13, fontweight="bold",
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG, "fig2_power_by_fault.png"), dpi=150)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp1_forward_validation.csv"), index=False)

    # --- closed-form sanity checks ---
    w = WellState()
    Wr = w.rod.weight_buoyed / 1e3
    Fo = w.pump.fluid_load / 1e3
    print("=" * 78)
    print("EXPERIMENT 1 - FORWARD MODEL VALIDATION (all simulation)")
    print("=" * 78)
    print(f"Closed-form static buoyed rod weight      : {Wr:6.2f} kN")
    print(f"Closed-form peak static load (Wr + Fo)    : {Wr + Fo:6.2f} kN")
    print(f"Theoretical hydraulic power @ 6 SPM       : "
          f"{w.pump.fluid_load * 2.4 * 6 / 60 / 1000:6.2f} kW")
    print()
    print(df.round(3).to_string(index=False))
    print()
    nrm = df[df.fault == "normal"].iloc[0]
    print("CHECKS")
    print(f"  normal card brackets the static limits   : "
          f"{nrm.PRL_min_kN:.1f} < {Wr:.1f} .. {Wr + Fo:.1f} < {nrm.PRL_max_kN:.1f}  -> "
          f"{'PASS' if nrm.PRL_min_kN < Wr and nrm.PRL_max_kN > Wr + Fo else 'FAIL'}")
    rp = df[df.fault == "rod_part"].iloc[0]
    print(f"  parted rod collapses the load range      : "
          f"{rp.PRL_range_kN:.1f} kN vs normal {nrm.PRL_range_kN:.1f} kN -> "
          f"{'PASS' if rp.PRL_range_kN < 0.4 * nrm.PRL_range_kN else 'FAIL'}")
    fp = df[df.fault == "fluid_pound"].iloc[0]
    print(f"  fluid pound shortens net plunger stroke  : "
          f"{fp.net_stroke_m:.3f} m vs normal {nrm.net_stroke_m:.3f} m -> "
          f"{'PASS' if fp.net_stroke_m < nrm.net_stroke_m else 'FAIL'}")
    print(f"  speed ripple within NEMA-D range (1-12%) : "
          f"{df.speed_ripple_pct.min():.1f}-{df.speed_ripple_pct.max():.1f}% -> "
          f"{'PASS' if df.speed_ripple_pct.max() < 12 else 'FAIL'}")
    print(f"\nFigures -> {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
