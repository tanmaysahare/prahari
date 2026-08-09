# References and Claim Provenance

Every factual and quantitative claim made in this repository, with its source and evidence class.

| Class | Meaning |
|---|---|
| `FACT` | Verified from a cited external source |
| `SIM` | Simulation output, reproducible from this repository |
| `CALC` | Arithmetic derived from stated inputs |
| `EST` | Estimate, with the basis stated |
| `ASSUM` | Assumption open to challenge |
| `INFER` | Reasoning, not a measurement |

---

## 1. Context and market

| Claim | Class | Source |
|---|---|---|
| India's crude oil import dependence reached a record 88.7 % in 2025-26 | `FACT` | PPAC data, reported by ThePrint; corroborated by OilPrice.com and EY Economy Watch |
| Import dependence rose from 85.5 % (2021-22) to 88.7 % (2025-26) | `FACT` | Same |
| Domestic crude production fell 29.7 → 28.0 MMT over the same period | `FACT` | Same |
| The decline is attributed *"primarily owing to natural decline in mature and ageing oil and gas fields"* | `FACT` | Government statement reported via ThePrint |

## 2. Operational problem

| Claim | Class | Source |
|---|---|---|
| More than 25 % of sucker-rod-pumped wells in a mature Indian onshore asset required a well-servicing job within a year | `FACT` | SPE / OnePetro literature on an Indian western oil field |
| Up to 40 % of all workover jobs are rod-pump repairs, and *"the oil gain from these jobs is the least amongst all kinds of workover jobs"* | `FACT` | Same |
| Mean time between failures as low as 4–5 months in rod-pumped wells of a mature field | `FACT` | SPE Journal 28(03):1481. Applies to a problem subset, not an asset-wide average. |
| Sucker rod pump is *"the most popular form of artificial lift"* in the asset studied; suited to *"low volume operations… quite common in… mature fields in India"* | `FACT` | SPE OGIC / OnePetro |
| Rod and tubing wear is driven by rubbing in deviated wells and rod buckling below the neutral point | `FACT` | SPE Journal 28(03):1481. **Deviated wells are not modelled in this work.** |

## 3. Prior art

Catalogued in full in [`PRIOR_ART.md`](PRIOR_ART.md). Key entries:

| Claim | Class | Source |
|---|---|---|
| Dynamometer cards can be constructed from motor power consumption | `FACT` | IEEE, 2012 |
| Fault detection from motor power is published | `FACT` | Control Engineering Practice, 2019 |
| Hybrid indirect dynamometer-card measurement from motor power | `FACT` | IEEE, 2022 |
| Card estimation from driving-motor terminal quantities | `FACT` | IEEE, 2023 |
| ML and transfer-learning classification of dynamometer cards | `FACT` | Sensors 20(19):5659, 2020; SPE-194949 |
| Accelerometer-based rod position sensing, integrated into a load cell | `FACT` | Patent literature (US 11,060,392 and related) |
| Counterweight *"weights and centre of gravities… are not known"* and may be *"hidden in a pocket in the master weight"* | `FACT` | Southwestern Petroleum Short Course |
| Counterbalance adjustment *"requir[es] unit shut-down and restraint"* | `FACT` | US 9,157,431, background |
| No prior art found combining blind counterbalance identification with rig-day allocation | `INFER` | Literature search, 9 Aug 2026. Not a freedom-to-operate opinion. |

## 4. Results produced by this repository

| Claim | Class | Produced by |
|---|---|---|
| Forward model reproduces textbook card signatures for 8 fault modes; static limits match closed-form theory | `SIM` | `experiments/exp1_forward_validation.py` |
| Batched solver agrees with the scalar reference to 0.000 % on all 8 faults | `SIM` | Verification run |
| Counterbalance moment recovered blind: 8.1 % median absolute error, 34.9 % p90 | `SIM` | `exp2`, n = 64 |
| Counterbalance phase: 0.40° median absolute error, 2.33° p90 | `SIM` | `exp2` |
| Pump fillage MAE 0.047 | `SIM` | `exp2` |
| Fluid load `Fo`: 33.7 % median absolute error — poorly identified | `SIM` | `exp2` |
| Fault classification 73.4 % accuracy, macro-F1 0.723 | `SIM` | `exp2` |
| Ablation with assumed calibration: macro-F1 0.463, fillage MAE 0.084 | `SIM` | `exp2` |
| Physics-free ML baseline: macro-F1 0.719 | `SIM` | `exp2` |
| Fleet recovery 83.0 % vs calendar 73.0 % → +13.5 % | `SIM` | `exp4`, 12 paired replicates |
| Marginal tonnes per rig-day 50.5 / 53.7 / 70.4 → +31 % vs calendar | `SIM` | `exp4` |
| Unplanned failures reduced 121 → 32 vs reactive | `SIM` | `exp4` |
| Paired t-test vs calendar: p = 2.5 × 10⁻¹³ | `SIM` | `exp4`. Measures effect stability within the simulation, not real-world validity. |
| Reactive baseline yields a 28.6 % annual servicing rate, against a published ">25 %" | `SIM` + `FACT` | `exp4` vs §2. The only external point of contact. |
| At σ = 0.043 the uplift is 14.0 % vs 14.5 % for a perfect sensor (~96 % of achievable value) | `SIM` | `exp4` sensitivity sweep |

## 5. Costs

| Claim | Class | Basis |
|---|---|---|
| Sensor kit BOM ≈ ₹4,800; installed ≈ ₹5,500 | `EST` | Component build-up, [`ECONOMICS.md`](ECONOMICS.md) §1 |
| Incumbent load-cell system ≈ ₹1.5–3 lakh/well installed | `EST` | Order-of-magnitude only, not a quotation |
| Planned change-out ≈ 2 rig-days; failure workover ≈ 4.5 rig-days | `EST` | Assumption in `exp4` |
| Crude realisation ₹38,000/tonne | `ASSUM` | ≈ US$62/bbl at ₹88/US$, 7.3 bbl/t |
| Base-case incremental revenue ₹110 Cr/yr on a 400-well asset | `CALC` | Fully dependent on the realisation assumption above |

## 6. Explicitly not claimed

- No CO₂e or energy-efficiency figure. Neither has been modelled.
- No claim of being first, unique or unprecedented. Prior art is dense and cited.
- No market-size extrapolation beyond a single asset.
- No field validation, pilot deployment, partnership or revenue. None exist.
- No hardware has been built or tested.

---

## 7. Bibliography

- Gibbs, S. G. (1963). *Predicting the behavior of sucker-rod pumping systems*. Journal of Petroleum Technology.
- API Spec 11E — *Specification for Pumping Units*.
- Everitt, T. A., & Jennings, J. W. — finite-difference computation of downhole dynamometer cards.
- *Tubing and Rod Failure Analysis in Rod Pumped Wells in an Indian Western Oil Field*. SPE Journal 28(03):1481.
- *Ways to Obtain Optimum Power Efficiency of Artificial Lift Installations*. SPE-126544-MS.
- *Beam Pump Dynamometer Card Classification Using Machine Learning*. SPE-194949.
- *Automatic Recognition of Sucker-Rod Pumping System Working Conditions Using Dynamometer Cards with Transfer Learning and SVM*. Sensors 20(19):5659, 2020.
- *Fault detection for sucker rod pump based on motor power*. Control Engineering Practice, 2019.
- *Building the dynamometer card of a sucker rod pump using power consumption of the electric motor*. IEEE, 2012.
- *A novel hybrid method for indirect measurement of the dynamometer card using measured motor power*. IEEE, 2022.
- *Improving the estimation of a sucker-rod-pumping dynamometer card based on the terminal quantities of the driving motor*. IEEE, 2023.
- US 9,157,431; US 10,107,282; US 10,815,770; US 11,060,392 — USPTO.
- Southwestern Petroleum Short Course — *Best Method to Balance Torque of a Pumping Unit Gearbox*.
- PPAC / Ministry of Petroleum & Natural Gas — petroleum import and production statistics.

Sources were retrieved on 9 August 2026. Citations should be independently verified before being relied upon.
