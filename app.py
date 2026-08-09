"""
PRAHARI - live technology demonstrator.

    streamlit run app.py

This is NOT a marketing landing page. It runs the real forward physics and the
real blind inversion live, on whatever well you configure, and shows the answer
next to the hidden ground truth so the result can be assessed honestly.

Three tabs:
  1. LIVE INVERSION   configure a well -> simulate -> invert -> compare
  2. FLEET & RIG PLAN what the asset manager actually receives
  3. METHOD           the identifiability argument, in plain terms
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from forward import DriveConfig, WellState, simulate_well
from inverse import invert_well
from kinematics import UnitGeometry
from rodstring import FAULTS, RodString, make_pump

NAVY, RUST, GREY = "#1b3a5c", "#c1440e", "#9a9a9a"

st.set_page_config(page_title="PRAHARI", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .stApp {background:#fbfbfa;}
      h1,h2,h3 {color:#1b3a5c;}
      .lbl {font-size:0.78rem;color:#666;letter-spacing:.06em;text-transform:uppercase;}
      .sim-badge {background:#fff4ef;border:1px solid #c1440e;color:#c1440e;
                  padding:3px 10px;border-radius:3px;font-size:0.75rem;
                  font-weight:600;letter-spacing:.05em;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PRAHARI")
st.markdown(
    "**Predictive Rod-lift Asset Health And Rig Intelligence** &nbsp;&nbsp;"
    "<span class='sim-badge'>SIMULATED WELL — GROUND TRUTH KNOWN</span>",
    unsafe_allow_html=True,
)
st.caption(
    "Estimating downhole rod-pump condition from a clamp-on motor current "
    "sensor and a walking-beam IMU only — no polished-rod load cell, no "
    "counterbalance test, no well shut-in."
)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Well configuration")
    fault = st.selectbox("True downhole condition (hidden from the solver)",
                         FAULTS, index=1)
    depth = st.slider("Pump setting depth (m)", 600, 1800, 1200, 50)
    spm = st.slider("Strokes per minute", 3.0, 10.0, 6.0, 0.5)
    Fo = st.slider("Fluid load $F_o$ (kN)", 5.0, 30.0, 16.0, 0.5)
    st.divider()
    st.header("Sensor quality")
    noise = st.slider("Motor-power noise (%)", 0.0, 8.0, 1.5, 0.5)
    imu_noise = st.slider("IMU angle noise (deg)", 0.0, 0.5, 0.05, 0.01)
    st.divider()
    st.header("Solver effort")
    n_per_class = st.select_slider("Candidates per fault class",
                                   [32, 64, 96, 128], value=64)
    run = st.button("Run inversion", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(
    ["1 · Live inversion", "2 · Fleet & rig plan", "3 · Method"]
)

# ================================================================== TAB 1
with tab1:
    if not run:
        st.info("Configure a well in the sidebar and press **Run inversion**. "
                "The solver never sees the true condition, the counterbalance, "
                "or the downhole card.")
    else:
        rng = np.random.default_rng(0)
        w = WellState()
        w.geom = UnitGeometry()
        w.rod = RodString(L=float(depth), damping=2.5)
        w.pump = make_pump(fault, rng)
        w.pump.fluid_load = Fo * 1000.0
        w.drive = DriveConfig(spm=float(spm))

        with st.spinner("Running forward physics (damped wave equation) ..."):
            r = simulate_well(w, seed=7, noise_power_pct=noise,
                              noise_imu_deg=imu_noise)
        with st.spinner("Blind self-calibrating inversion ..."):
            inv = invert_well(r["P_elec_meas"], r["theta"], r["omega"],
                              w.geom, w.drive.spm, seed=3,
                              n_per_class=int(n_per_class), n_harm=10)

        # ---------------- what the sensor sees ----------------
        st.subheader("Step 1 — what a ₹5,500 kit actually measures")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6.4, 3.1))
            ax.plot(np.degrees(r["theta"]), r["P_elec_meas"] / 1e3, lw=1.4,
                    color=RUST)
            ax.set_xlabel("crank angle (deg)"); ax.set_ylabel("motor power (kW)")
            ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25)
            ax.set_title("Clamp-on CT — motor electrical power", fontsize=10,
                         fontweight="bold")
            st.pyplot(fig, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6.4, 3.1))
            ax.plot(np.degrees(r["theta"]), np.degrees(r["psi_meas"]), lw=1.4,
                    color=NAVY)
            ax.set_xlabel("crank angle (deg)"); ax.set_ylabel("beam angle (deg)")
            ax.grid(alpha=.25)
            ax.set_title("Beam IMU — walking-beam rotation", fontsize=10,
                         fontweight="bold")
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Step 2 — inferred surface card vs hidden ground truth")
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.plot(r["s"], r["PRL"] / 1e3, lw=2.0, color=GREY,
                    label="TRUE card (hidden from solver)")
            n = min(len(inv["PRL_hat"]), len(r["s"]))
            s_ds = r["s"][:: max(1, len(r["s"]) // n)][:n]
            ax.plot(s_ds, inv["PRL_hat"][:n] / 1e3, lw=1.8, color=NAVY,
                    label="PRAHARI reconstruction")
            ax.set_xlabel("polished rod position (m)")
            ax.set_ylabel("polished rod load (kN)")
            ax.legend(fontsize=9); ax.grid(alpha=.25)
            ax.set_title("Surface dynamometer card", fontweight="bold")
            st.pyplot(fig, use_container_width=True)
        with c2:
            st.markdown("<div class='lbl'>Diagnosis</div>", unsafe_allow_html=True)
            ok = inv["fault"] == fault
            st.markdown(
                f"### {'✅' if ok else '⚠️'} {inv['fault'].replace('_',' ')}"
            )
            st.caption(f"true condition: **{fault.replace('_',' ')}** · "
                       f"runner-up model: {inv['runner_up'].replace('_',' ')}")
            st.metric("Pump fillage (estimated)", f"{inv['fillage']:.3f}",
                      f"{inv['fillage'] - w.pump.fillage:+.3f} vs true")
            st.metric("Counterbalance moment $M$ — recovered BLIND",
                      f"{inv['M_cb']/1e3:.1f} kN·m",
                      f"{100*(inv['M_cb']-r['M_cb_true'])/r['M_cb_true']:+.1f}% vs true")
            d = inv["tau_cb"] - r["tau_cb_true"]
            st.metric("Counterbalance phase τ",
                      f"{np.degrees(inv['tau_cb']):.1f}°",
                      f"{np.degrees(np.arctan2(np.sin(d), np.cos(d))):+.2f}° vs true")
            conf = "HIGH" if inv["margin"] > 0.15 else (
                "MEDIUM" if inv["margin"] > 0.05 else "LOW — refer to a human")
            st.markdown(f"<div class='lbl'>Confidence</div>**{conf}**  "
                        f"<br><small>margin over runner-up = "
                        f"{inv['margin']:.3f}, residual = "
                        f"{inv['residual_norm']:.3f}</small>",
                        unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Step 3 — how every candidate physical model scored")
        cr = pd.Series(inv["class_residuals"]).sort_values()
        fig, ax = plt.subplots(figsize=(11, 2.9))
        cols = [NAVY if i == fault else (RUST if i == inv["fault"] else GREY)
                for i in cr.index]
        ax.bar([i.replace("_", "\n") for i in cr.index], cr.values, color=cols)
        ax.set_ylabel("fit residual\n(lower = better)")
        ax.set_title("Fault classification is PHYSICAL MODEL SELECTION — "
                     "not a black-box label", fontsize=10, fontweight="bold")
        ax.grid(alpha=.25, axis="y")
        st.pyplot(fig, use_container_width=True)
        st.caption("Navy = true condition. Every bar is a full damped-wave-equation "
                   "solve with its own closed-form calibration. The winner comes "
                   "with a physical parameter set, not just a label.")

# ================================================================== TAB 2
with tab2:
    st.subheader("What the asset manager receives on Monday morning")
    st.caption("Illustrative fleet generated from the Experiment 4 model. "
               "SIMULATION.")
    rng = np.random.default_rng(11)
    n = 40
    q0 = np.clip(rng.lognormal(np.log(2.2), 0.75, n), 0.4, 20)
    fill = np.clip(rng.beta(5, 2, n), 0.2, 1.0)
    deg = rng.gamma(2.0, 0.00066, n)
    rul = np.clip((fill - 0.25) / deg, 0, 999)
    loss = q0 * (1 - fill)
    dur = np.where(fill <= 0.25, 4.5, 2.0)
    value = (loss * 90 + q0 * np.maximum(90 - np.minimum(rul, 90), 0) * .5) / dur
    df = pd.DataFrame({
        "Well": [f"DUL-{i:03d}" for i in range(n)],
        "Rate (t/d)": q0.round(2),
        "Fillage (est.)": fill.round(3),
        "Est. RUL (days)": rul.round(0),
        "Silent loss (t/d)": loss.round(2),
        "Job (rig-days)": dur,
        "Value (t/rig-day)": value.round(1),
    }).sort_values("Value (t/rig-day)", ascending=False).reset_index(drop=True)

    rigs = st.slider("Workover rigs available this week", 1, 6, 3)
    cap = rigs * 7.0
    cum = df["Job (rig-days)"].cumsum()
    df["This week"] = np.where(cum <= cap, "▶ DISPATCH", "")
    st.dataframe(df, use_container_width=True, height=430)
    sel = df[df["This week"] != ""]
    c1, c2, c3 = st.columns(3)
    c1.metric("Wells dispatched", len(sel))
    c2.metric("Rig-days committed", f"{sel['Job (rig-days)'].sum():.0f} / {cap:.0f}")
    c3.metric("Silent loss addressed", f"{sel['Silent loss (t/d)'].sum():.1f} t/d")
    st.info("The ranking column is **tonnes recovered per rig-day** — not fault "
            "severity, and not well rate. That is the objective an asset "
            "manager is actually measured on.")

# ================================================================== TAB 3
with tab3:
    st.subheader("Why this is not just 'AI on a pump'")
    st.markdown(
        """
**The observation model.** The motor sees

$$P(\\theta)=\\Big[\\underbrace{\\kappa\\,g(\\theta)\\,(\\mathrm{PRL}(\\theta)-SU)}_{\\text{rod torque}}
-\\underbrace{M\\sin(\\theta+\\tau)}_{\\text{counterbalance}}\\Big]\\,\\omega(\\theta)/\\eta+P_0$$

where \\(g(\\theta)=ds/d\\theta\\) and \\(\\omega(\\theta)\\) are measured exactly by
the beam IMU.

**The blocker in the prior art.** Every published motor-power method needs
\\(M\\) and \\(SU\\) as *inputs*. On a 40-year-old brownfield unit those are not
recorded, are physically hidden inside the master-weight pocket, and can only be
measured by a counterbalance test that requires **shutting the well in**. That
single dependency is why a 14-year-old technique is not deployed at scale.

**What we do instead — variable projection.** For any candidate pump-health
vector, the model above is *linear* in
\\((\\kappa,\\ \\kappa SU,\\ M\\cos\\tau,\\ M\\sin\\tau,\\ P_0)\\).
So we solve the calibration in **closed form** by least squares and search only
over the 1–4 dimensional pump-health manifold.

**Why it is identifiable.** The counterbalance is a *pure first harmonic*. It
could be absorbed into the rod-torque term only if PRL were unconstrained — but
PRL must be a physically admissible solution of the damped wave equation,
spanned by a handful of parameters. That physical constraint breaks the
degeneracy. We *measure* the residual degeneracy (Experiment 2) rather than
asserting it.

---
**Honest limitations**
- Validated in simulation only. TRL 3–4. Field validation is Stage 2.
- Fluid-load *magnitude* \\(F_o\\) is poorly identified (median error ~34%)
  because it is multiplicatively confounded with \\(\\kappa\\). Pump *fillage* —
  a ratio, and the quantity that drives the economics — is identified well
  (MAE ≈ 0.047).
- Conventional crank-balanced units only. Mark-II and air-balanced units need a
  different kinematic model.
- We do **not** claim to have invented electrical dynamometry. It is prior art
  and we cite it. Our claim is the self-calibration and the decision layer.
        """
    )
