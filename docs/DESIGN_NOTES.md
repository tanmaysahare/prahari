# Design Notes, Modelling Decisions and Limitations

Why the model is built the way it is, what was tried, what broke, and where the approach is weak.

---

## 1. Physics

### 1.1 Rod-string model

A one-dimensional damped wave equation (Gibbs, 1963), integrated with an explicit second-order finite-difference scheme:

```
d²u/dt² = (E/ρ)·d²u/dx² + g_eff − c·du/dt
```

Boundary conditions are mixed: prescribed displacement at the surface (from the unit kinematics) and a force condition at the pump, applied through a ghost node.

**Why a wave equation rather than a lumped model.** Fluid pound releases the fluid load abruptly at the pump. Integrating the real wave equation makes the resulting high-frequency ringing in the surface card emerge from the physics, rather than being drawn in by hand. This matters because the same mechanism produces the differences between fault modes that the inversion relies on.

**Known simplifications:**
- **Uniform (single-taper) string.** Real strings are tapered; correct treatment requires segment-wise impedance matching at each taper change.
- **Viscous damping only.** No Coulomb friction term, so **deviated wells are not modelled**. This is the most significant physical gap, and it matters because rod/tubing wear in deviated wells is a dominant failure mechanism in some assets.
- **Conventional crank-balanced units only.** Mark-II and air-balanced units have different kinematics and, in the air-balanced case, entirely different counterbalance dynamics.

### 1.2 Pumping-unit kinematics

The four-bar linkage is solved exactly (two applications of the law of cosines), with no small-angle approximation. The torque factor is obtained by differentiating the kinematic solution rather than from tabulated values:

```
TF(θ) = ds/dθ
```

This is exact by virtual work, and it removes a lookup table and a source of error.

### 1.3 Numerical stability

The scheme is sub-stepped to a Courant number of **0.85, not 1.0**. Runs landing at CFL ≈ 0.997 were observed to diverge for the combination of deep string, thick rod and high stroke rate — the viscous damping term erodes the stability region of the explicit scheme below the bare Courant limit. With the 0.85 target, 0 of 120 randomised forward runs diverged.

The batched solver in `src/batch_sim.py` reproduces the scalar reference in `src/rodstring.py` to 0.000 % across all eight fault modes, and is roughly 39× faster per candidate. The speed-up comes from amortising numpy call overhead across candidates; on a 25-element state vector that overhead otherwise dominates the arithmetic.

### 1.4 Drive train

The crank speed is **not** assumed constant. An induction motor slips under load and the crank/counterweight inertia smooths torque, so `ω(θ)` varies by a few percent through the stroke. This is integrated explicitly.

The motor torque–speed characteristic is stiff: a 1 % speed change swings torque by tens of kN·m. Explicit integration of the flywheel equation diverges. A backward-Euler update is used instead, which is unconditionally stable and recovers the physically correct behaviour (a few percent of slip ripple rather than runaway).

---

## 2. The inverse problem

### 2.1 Formulation

The observed motor power is

```
P(θ) = [ κ·g(θ)·(PRL(θ) − SU) − M·sin(θ+τ) ] · ω(θ)/η + P₀
```

where `g(θ) = ds/dθ` and `ω(θ)` are measured directly by the beam IMU.

For any candidate pump-health vector, this is **linear** in `(κ, κ·SU, M·cosτ, M·sinτ, P₀)`. The calibration is therefore solved in closed form by least squares, and the nonlinear search is confined to the 1–4 dimensional pump-health manifold. This is the classical variable-projection (VarPro) separation.

### 2.2 Identifiability

The counterbalance term is a pure first harmonic in crank angle. It could be absorbed into the rod-torque term only if `PRL` were unconstrained. It is not: `PRL` must be an admissible solution of the damped wave equation, spanned by at most four pump-health parameters. That physical constraint is what breaks the degeneracy.

This is an argument, not a proof. The residual degeneracy is measured empirically rather than asserted — see the 8.1 % median / 34.9 % p90 counterbalance error in the results.

### 2.3 Band-limiting

The fit uses only the first ten crank harmonics. This is a physical choice, not a convenience: the counterbalance signature is a pure first harmonic and the pump-health signature lives in low-order harmonics, whereas rod-string ringing sits at the string's natural modes, roughly 10–20× the crank frequency. A real clamp-on CT feeding an edge ADC would be anti-alias filtered in any case. Without band-limiting, un-modelled high-frequency ringing biased the calibration estimate badly.

### 2.4 Fault classification as model selection

Each fault class is fitted independently and selected by residual. The classifier is therefore a physical model-selection procedure: the winning answer arrives with a fitted parameter set and a residual, not just a label. The margin between the best and second-best model is used as a confidence measure, and it separates most misclassifications.

### 2.5 Avoiding an inverse crime

Ground truth is generated at 48 nodes, 3 warm-up cycles and 360 crank samples. The inverse model runs at 24 nodes, 1 warm-up cycle and 180 samples, with measurement noise added to both observables. The solver is deliberately a coarser, mismatched model.

The residual weakness is that it remains the *same model family*. Only field data resolves that.

---

## 3. Frequently asked technical questions

**Why an IMU as well as a current clamp? Can't position be recovered from the power waveform alone?**
It can be approximated, and published methods do exactly that. But it makes crank angle an estimated quantity coupled to the same unknowns being identified. An IMU costs a few hundred rupees and makes `g(θ)` and `ω(θ)` measured rather than inferred, which is what keeps the variable-projection separation clean.

**Why is the fluid load `Fo` estimated so poorly (~34 % median error)?**
`Fo` is multiplicatively confounded with the geometry scale `κ` — only the product is identified. Pump fillage is a *ratio*, so it is scale-free, and it is recovered well (MAE 0.047). Fillage is also the quantity that drives the economics, so the confound is tolerable. It is reported rather than hidden.

**A physics-free classifier matches the classification accuracy. Why use physics?**
Three reasons. The classifier required 240 *labelled* wells; in the field a label comes from pulling the pump, so building that training set means 240 workovers. It transfers to a new field only if the new field resembles the training field. And it outputs a label and nothing else — no fillage, no calibration, no card — whereas the fleet analysis shows the value comes from fillage. The physics model needs no training data and extrapolates by construction.

**Isn't the fleet simulation circular — your own model validating your own method?**
Partly, and it is the fairest criticism of this work. The one non-circular check available is that the reactive baseline independently reproduces a 28.6 % annual well-servicing rate against a published figure of ">25 % within a year", without being tuned to it. That is one point of external contact, not a validation.

**What does the p-value mean here?**
The paired t-test across replicates shows the effect is not sampling noise within the simulation. It says nothing about whether the model is right.

**Real motor power has supply harmonics, VFD switching and voltage dips. None of that is modelled.**
Correct. Band-limiting removes most high-frequency content, and the counterbalance signature is a first harmonic, so the information used is robust to it. But the real spectrum has to be measured. This is the single largest simulation-to-field risk.

**Do real pumps have exactly one fault at a time?**
No. Real wells have concurrent faults and the model-selection step picks one. The mitigation is that a mixed condition tends to produce a low margin over the runner-up model, which is exactly the signature the confidence measure is designed to surface.

---

## 4. Limitations

- **Simulation only.** No hardware has been built and no field data has been used. TRL 3–4.
- Fluid-load magnitude is poorly identified (see above).
- Conventional crank-balanced units only.
- Deviated wells are not modelled (no Coulomb friction term).
- Variable-speed drives are untested; the pump physics also changes with variable stroke rate.
- The p90 counterbalance error of ~35 % indicates a tail of difficult wells. The confidence signal separates most but not all of them.
- 64 evaluation wells give stable medians but not reliable tail statistics.
- The incumbent-cost comparison is an order-of-magnitude estimate, not a quotation.
- Hazardous-area certification of a powered device near the wellhead is unresolved. Mounting the electronics inside the existing motor starter panel is proposed but unproven.

---

## 5. Next experiment

Five wells that already carry a dynamometer. That gives true surface cards as ground truth on day one, and converts every simulated result here into a measured one — or falsifies it. It is the cheapest decisive test available and it is the natural next step.
