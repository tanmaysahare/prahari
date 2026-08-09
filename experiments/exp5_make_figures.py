"""
Publication figures from the Experiment 2 and 4 result files.
ALL FIGURES DEPICT SIMULATION RESULTS.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

NAVY, RUST, GREY = "#1b3a5c", "#c1440e", "#9a9a9a"

df = pd.read_csv(os.path.join(RES, "exp2_inversion_study.csv"))
S = json.load(open(os.path.join(RES, "exp2_summary.json")))

# =====================================================================
# FIG 3 - the core inversion result panel
# =====================================================================
fig = plt.figure(figsize=(18, 9.5))
gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

# --- (a) blind counterbalance recovery ---
ax = fig.add_subplot(gs[0, 0])
ax.scatter(df.M_true / 1e3, df.M_hat / 1e3, s=34, color=NAVY, alpha=0.75,
           edgecolor="w", linewidth=0.5)
lim = [0, max(df.M_true.max(), df.M_hat.max()) / 1e3 * 1.05]
ax.plot(lim, lim, "--", color="k", lw=1, label="perfect")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("True counterbalance moment $M$ (kN·m)")
ax.set_ylabel("Blind estimate $\\hat{M}$ (kN·m)")
ax.set_title("(a) Counterbalance recovered BLIND\n"
             "no counterbalance test, no shut-in", fontweight="bold", fontsize=11)
ax.text(0.04, 0.94,
        f"median |err| = {S['calibration_blind']['M_abs_err_pct_median']:.1f}%\n"
        f"p90 |err| = {S['calibration_blind']['M_abs_err_pct_p90']:.0f}%",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(fc="white", ec=GREY, alpha=0.9))
ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")

# --- (b) counterbalance phase error ---
ax = fig.add_subplot(gs[0, 1])
ax.hist(df.tau_err_deg, bins=22, color=NAVY, alpha=0.85, edgecolor="w")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Counterbalance phase error (deg)")
ax.set_ylabel("wells")
ax.set_title("(b) Phase $\\tau$ is recovered almost exactly",
             fontweight="bold", fontsize=11)
ax.text(0.04, 0.94,
        f"median |err| = {S['calibration_blind']['tau_abs_err_deg_median']:.2f}°\n"
        f"p90 |err| = {S['calibration_blind']['tau_abs_err_deg_p90']:.2f}°",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(fc="white", ec=GREY, alpha=0.9))
ax.grid(alpha=0.25)

# --- (c) pump fillage: the quantity that drives the economics ---
ax = fig.add_subplot(gs[0, 2])
ax.scatter(df.fillage_true, df.fillage_hat, s=34, color=NAVY, alpha=0.8,
           edgecolor="w", linewidth=0.5, label="PRAHARI (self-calibrating)")
ax.scatter(df.fillage_true, df.fillage_hat_fixedcal, s=26, color=RUST,
           alpha=0.55, marker="^", label="assumed calibration (prior art)")
ax.plot([0.2, 1.02], [0.2, 1.02], "--", color="k", lw=1)
ax.set_xlabel("True pump fillage"); ax.set_ylabel("Estimated pump fillage")
ax.set_title("(c) Pump fillage — the driver of the money\n"
             "MAE 0.047 vs 0.084 for the prior-art ablation",
             fontweight="bold", fontsize=11)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

# --- (d) confusion matrix ---
ax = fig.add_subplot(gs[1, 0])
labels = S["confusion_labels"]
C = np.array(S["confusion_prahari"], dtype=float)
Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
short = [l.replace("_", "\n") for l in labels]
ax.set_xticks(range(len(labels))); ax.set_xticklabels(short, fontsize=7, rotation=45,
                                                      ha="right")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(short, fontsize=7)
for i in range(len(labels)):
    for j in range(len(labels)):
        if C[i, j]:
            ax.text(j, i, int(C[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if Cn[i, j] > 0.5 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("(d) Fault classification by physical\nmodel selection",
             fontweight="bold", fontsize=11)

# --- (e) THE ABLATION: does self-calibration matter? ---
ax = fig.add_subplot(gs[1, 1])
cls = S["classification"]
names = ["Assumed\ncalibration\n(prior art)", "Physics-free\nML\n(needs labels)",
         "PRAHARI\n(self-calibrating)"]
vals = [cls["macroF1_fixedcal_ablation"], cls["macroF1_ml_physicsfree"],
        cls["macroF1_prahari"]]
bars = ax.bar(names, vals, color=[RUST, GREY, NAVY])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.0); ax.set_ylabel("macro F1")
ax.set_title("(e) ABLATION — removing self-calibration\ncosts 0.26 macro-F1",
             fontweight="bold", fontsize=11)
ax.grid(alpha=0.25, axis="y")

# --- (f) residual as a usable confidence signal ---
ax = fig.add_subplot(gs[1, 2])
correct = df.fault_prahari == df.fault_true
ax.scatter(df.residual_norm[correct], df.margin[correct], s=34, color=NAVY,
           alpha=0.75, label="correct")
ax.scatter(df.residual_norm[~correct], df.margin[~correct], s=44, color=RUST,
           alpha=0.85, marker="x", label="misclassified")
ax.set_xlabel("normalised fit residual")
ax.set_ylabel("margin over runner-up model")
ax.set_yscale("log")
ax.set_title("(f) The model knows when it is unsure\n"
             "low margin ⇒ defer to a human", fontweight="bold", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.25)

fig.suptitle(
    "PRAHARI — blind inversion of rod-pump condition from motor power + beam IMU only "
    f"({S['n_wells']} randomised wells, {S['noise_power_pct']}% power noise)   "
    "· ALL RESULTS ARE SIMULATION ·",
    fontsize=13, fontweight="bold", y=0.985,
)
fig.savefig(os.path.join(FIG, "fig3_inversion_results.png"), dpi=150,
            bbox_inches="tight")
print("wrote fig3_inversion_results.png")

# =====================================================================
# FIG 5 - economic sensitivity to sensor quality
# =====================================================================
S4 = json.load(open(os.path.join(RES, "exp4_summary.json")))
sens = pd.DataFrame(S4["sensitivity_to_health_estimate_noise"])
fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.plot(sens.est_sigma, sens.uplift_vs_calendar_pct, "o-", color=NAVY, lw=2.2,
        ms=7)
ours = float(S["health"]["fillage_MAE_prahari"])
ax.axvline(ours, color=RUST, ls="--", lw=1.8)
ax.annotate(
    f"PRAHARI measured accuracy\n(fillage MAE = {ours:.3f})\n"
    f"captures ~96% of the value\na perfect sensor would give",
    xy=(ours, np.interp(ours, sens.est_sigma, sens.uplift_vs_calendar_pct)),
    xytext=(0.11, 10.4), fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color=RUST, lw=1.4),
    bbox=dict(fc="#fff4ef", ec=RUST, alpha=0.95),
)
ax.set_xlabel("Health-estimate error (std. dev. of fillage estimate)")
ax.set_ylabel("Production uplift vs calendar policy (%)")
ax.set_title("Value is driven by HAVING an estimate, not by precision\n"
             "— which is why a ₹5,500 kit is sufficient (SIMULATION)",
             fontweight="bold")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_sensor_quality_sensitivity.png"), dpi=150)
print("wrote fig5_sensor_quality_sensitivity.png")
