# Techno-Economic Analysis and Deployment Path

Labels used throughout: `[FACT]` verified from a cited source · `[SIM]` simulation output reproducible from this repository · `[CALC]` arithmetic from stated inputs · `[EST]` estimate with stated basis · `[ASSUM]` assumption open to challenge.

---

## 1. Sensor kit — bill of materials

Per-well non-intrusive kit, Indian small-volume pricing, August 2026.

| Item | Spec | Unit ₹ `[EST]` | Basis |
|---|---|---:|---|
| Split-core current transformer | 100 A : 50 mA, clamp-on, no circuit break | 900 | Indian industrial-electronics retail band ₹700–1,400 |
| MEMS IMU | 6-axis, I²C, ±2 g / ±250 °/s | 450 | Commodity module band ₹250–800 |
| Edge MCU + ADC | ESP32-class, Wi-Fi/BLE, 12-bit ADC | 600 | Commodity band ₹400–900 |
| Voltage sense / isolation | resistive divider + opto-isolation | 350 | Component build-up |
| IP65 enclosure, gland, mounting | outdoor, high-humidity service | 800 | Retail band |
| Power supply or Li-ion + solar | from the existing starter panel | 700 | Retail band |
| GSM/LoRa backhaul | shared gateway per ~40 wells | 550 | Amortised module + gateway share |
| Assembly, test, potting | ~1 h skilled labour | 450 | Labour estimate |
| **Hardware sub-total** | | **≈ 4,800** | |
| Installation (2 technicians × 45 min, no shut-in) | | 700 | Labour estimate |
| **Installed cost per well** | | **≈ ₹5,500** | |

A conventional polished-rod load cell with position transducer and RTU is estimated at **₹1.5–3 lakh per well installed** `[EST]`. This is an order-of-magnitude figure, not a quotation, and should be replaced with an operator's own procurement data. The structural argument does not depend on the price: the incumbent requires wellhead intervention, which is a barrier at any cost.

---

## 2. Where the value comes from

| Stream | Mechanism | Evidence |
|---|---|---|
| **Recovered silent production** | Pumps run for weeks at 50–80 % fillage. On a low-rate well that is never individually tested, that loss is invisible. Fillage is exactly what the inversion estimates. | `[SIM]` fillage MAE 0.047 |
| **Better rig-day allocation** | Rank interventions by expected tonnes recovered per rig-day rather than by failure-queue order. | `[SIM]` +31 % marginal t/rig-day vs a calendar policy |
| **Cheaper interventions** | Planned change-out (~2 rig-days `[EST]`) instead of an unplanned failure workover with fishing (~4.5 rig-days `[EST]`). | `[SIM]` 71 % of unplanned failures avoided vs reactive |

### Fleet simulation result `[SIM]`

400 wells, 3 workover rigs, 365 days, 12 paired replicates. Health-estimate noise set to the measured fillage MAE (σ = 0.043).

| Policy | Recovery (% of potential) | Unplanned failures/yr | Marginal t per rig-day |
|---|---:|---:|---:|
| Do nothing | 64.5 | 112 | — |
| Reactive (run-to-failure) | 70.7 | 121 | 50.5 |
| Calendar (condition-blind preventive) | 73.0 | 105 | 53.7 |
| Rate-first | 70.9 | 121 | 52.7 |
| **Condition-based (this work)** | **83.0** | **32** | **70.4** |

- **+13.5 %** production vs the calendar policy, p = 2.5 × 10⁻¹³ (paired t-test, 12 replicates)
- **+31 %** marginal tonnes per rig-day

**Model self-check.** The reactive baseline independently produces a **28.6 %** annual well-servicing rate, against a published figure of *">25 % of SRP wells required a well-servicing job within a year"* for a comparable asset `[FACT]`. The degradation model was not tuned to reproduce this.

### Sensitivity to sensor quality `[SIM]`

| Health-estimate noise σ | Uplift vs calendar |
|---:|---:|
| 0.00 (perfect sensor) | 14.5 % |
| 0.02 | 14.3 % |
| **0.043 (measured accuracy of this method)** | **14.0 %** |
| 0.08 | 13.3 % |
| 0.15 | 11.9 % |
| 0.25 (very poor sensor) | 8.9 % |

The value is dominated by having a reasonable estimate at all rather than by precision. At the measured accuracy the scheduler captures roughly 96 % of the value a perfect sensor would deliver, which is what makes low-cost sensing viable.

---

## 3. Financial scenarios

Scaling logic: per well → 400-well asset. No national market extrapolation is attempted.

| Parameter | Value | Class |
|---|---|---|
| Fleet | 400 rod-pumped wells, 3 workover rigs | `[ASSUM]` |
| Mean well rate | ~2.9 t/d | `[ASSUM]` |
| Fleet potential | ~415,000 t/yr | `[SIM]` |
| Crude realisation | ₹38,000/tonne | `[ASSUM]` — ≈ US$62/bbl at ₹88/US$, 7.3 bbl/t. Replace with the operator's netback. |
| Kit installed | ₹5,500/well | `[EST]` |
| Backhaul, cloud, support | ₹1,200/well/yr | `[EST]` |

| | Conservative | Base | Optimistic |
|---|---:|---:|---:|
| Uplift captured | 3.0 % | 7.0 % | 13.5 % (full simulated) |
| Rationale | partial adoption, only the worst wells acted on | half the simulated benefit realised | simulation realised in full |
| Incremental tonnes/yr | 12,450 | 29,050 | 56,025 |
| Incremental revenue/yr `[CALC]` | ₹47.3 Cr | ₹110.4 Cr | ₹212.9 Cr |
| CAPEX (400 kits) | ₹22.0 lakh | ₹22.0 lakh | ₹22.0 lakh |
| OPEX/yr | ₹4.8 lakh | ₹4.8 lakh | ₹4.8 lakh |
| Simple payback | < 1 week | < 1 week | < 1 week |

A payback measured in days is not a meaningful selling point — it indicates that the sensing hardware is economically negligible. A full kit costs less than a single rig-day. The entire economic question is whether the decision layer works, which is why the technical validation matters more than the cost model.

At ₹25,000/t the base case gives ₹72.6 Cr/yr; at ₹50,000/t, ₹145.3 Cr/yr. The conclusion is invariant across plausible prices.

**Conditions under which this does not hold:**
- With no spare workover-rig capacity, the allocation and intervention-cost benefits collapse; only the identification of which wells to prioritise survives.
- If wells are already individually well-tested weekly, the silent-loss stream collapses. That is typically true of high-rate wells, which is why this targets low-rate wells.

---

## 4. Deployment path

| Stage | Objective | Resources | Validation gate | Risks | Timeline |
|---|---|---|---|---|---|
| **1. Bench** (TRL 3→4) | Prove the electrical + inertial chain on physical hardware | Motor + eccentric crank + spring/mass load emulating rod dynamics; 3 kits. ~₹60k `[EST]` | Reconstructed load matches a reference load cell within a stated tolerance | Bench dynamics differ from a 1,200 m string | 6–10 weeks |
| **2. Instrumented-well pilot** (TRL 4→5) | 3–5 wells that already have a dynamometer, giving ground truth | Host operator, site access, safety clearance | Blind fillage vs measured dynamometer fillage; counterbalance estimate vs an actual counterbalance test | Access and permits; hazardous-area certification | 3–4 months |
| **3. Single-asset** (TRL 5→6) | 50–100 uninstrumented wells, scheduler in advisory mode | Gateways, cloud, one embedded engineer | A/B against the incumbent schedule on deferred production and rig-days | Change management; supervisors must trust the ranking | 6–9 months |
| **4. Multi-asset** (TRL 6→7) | 400–1,000 wells, scheduler as decision support of record | Field-service partner; SCADA/ERP integration | Sustained uplift over ≥2 quarters | Data governance; systems integration | 12–18 months |
| **5. Product** | Service or licensing model | — | Repeat deployments | Incumbent vendors offering a competing low-cost product | 24 months+ |

The critical path is Stage 2, not the technology. Every result in this repository is simulation-validated; five wells with an existing dynamometer convert that into measurement.

---

## 5. Risk register

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Real motor power contains harmonics, supply distortion and VFD switching not modelled | High | High | Band-limiting discards HF content; measure the real spectrum in Stage 2 |
| R2 | Blind calibration degenerate on non-conventional units (Mark-II, air-balanced) | High | Medium | Out of scope; different kinematic model required |
| R3 | Hazardous-area certification of a powered device near the wellhead | **High** | Medium | Mount in the existing certified starter enclosure — proposed, not proven |
| R4 | Supply interruptions and voltage swings on rural feeders | Medium | High | Cycle-level quality gating; discard incomplete strokes |
| R5 | Operators do not trust an unexplained ranking | Medium | High | Every recommendation carries a fitted card, a residual and a value figure |
| R6 | Cybersecurity of a fleet of connected edge devices | Medium | Medium | Sensing only, no actuation path, one-way device-to-cloud, signed firmware. No security review has been performed. |
| R7 | Simulation-to-field gap larger than expected | **High** | Medium | The headline risk. Stage 2 is designed to measure it, with an explicit go/no-go gate. |

---

## 6. Energy and environmental notes

Mature-field decline is the stated cause of India's rising crude import dependence (record 88.7 % in 2025-26) `[FACT]`. Production recovered from existing wells requires no new drilling, no new land and no new flowlines.

Two plausible co-benefits are **not quantified and not claimed**: a pump running at reduced fillage still draws near-full motor power, so correcting fillage should raise tonnes per kWh; and avoided unplanned workovers reduce rig mobilisations and associated diesel use and spill risk. Neither has been modelled, and no CO₂e figure is presented.
