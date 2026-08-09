# Prior Art Review

A survey of published work on rod-pump condition inference, and a precise statement of what this project does and does not claim as new.

Search performed 9 August 2026 across Google Patents, USPTO, IEEE Xplore, ScienceDirect and SPE/OnePetro. This is a literature review, not a professional freedom-to-operate opinion.

---

## 1. What is already established

Three ideas that are sometimes presented as novel are not, and this project does not claim them.

| Idea | Status |
|---|---|
| Inferring a dynamometer card from motor electrical data | **Established, ~14 years old.** See PA-1…PA-4 below. |
| Machine-learning classification of dynamometer cards | **Established and commercialised.** See PA-5…PA-8. |
| Accelerometer-based polished-rod position sensing | **Published and patented.** See PA-10, PA-11. |

---

## 2. Prior-art matrix

| ID | Work | Technology | Strength | Limitation for brownfield marginal wells | Difference here |
|---|---|---|---|---|---|
| PA-1 | *Building the dynamometer card of a sucker rod pump using power consumption of the electric motor* (IEEE, 2012) | Motor wattmeter card → dynamometer card | Established the electrical-inference principle | Assumes known unit geometry and counterbalance; no uncertainty; no prognostics | Calibration parameters treated as **unknowns to be identified** |
| PA-2 | *Fault detection for sucker rod pump based on motor power* (Control Engineering Practice, 2019) | Motor power → fault detection | Avoids a load cell | Detection only — present state, not time-to-failure | Outputs a health state and a remaining-useful-life estimate |
| PA-3 | *A novel hybrid method for indirect measurement of the dynamometer card using measured motor power* (IEEE, 2022) | Hybrid model | Strong reconstruction | Requires calibrated unit parameters | Blind self-calibration |
| PA-4 | *Improving the estimation of a sucker-rod-pumping dynamometer card based on the terminal quantities of the driving motor* (IEEE, 2023) | Induction-motor equivalent circuit + beam unit model | Physically rigorous | Same calibration dependency; single-well scope | Fleet-scale, coupled to an allocation decision |
| PA-5 | *Automatic recognition of sucker-rod pumping system working conditions using dynamometer cards with transfer learning and SVM* (Sensors 20(19):5659, 2020) | AlexNet transfer learning + ECOC-SVM | High accuracy | Requires a real dynamometer; classification only | No dynamometer required; estimates physical parameters rather than labels |
| PA-6 | *Automated dynamometer chart pattern recognition of sucker rod pumps using a CNN approach* (SPE Eastern Regional, 2025) | CNN on cards | Field-scale data | Same | Same |
| PA-7 | *Beam pump dynamometer card classification using machine learning* (SPE-194949) | ML on 6,385 field cards | Real field data | Requires instrumented wells | Targets uninstrumented wells |
| PA-8 | Commercial rod-lift optimisation systems (load cell + position transducer + RTU + analytics) | Full condition monitoring and control | Mature, proven, field-hardened | Capital cost per well and wellhead intervention; not deployed on low-rate wells | Different economic segment — not a competing accuracy claim |
| PA-9 | US 10,815,770 — *Method and device for measuring surface dynamometer cards and operation diagnosis in sucker-rod pumped oil wells* | Surface card measurement + diagnosis | Granted patent | Measurement-centric | Inference without dedicated load measurement, plus an allocation layer |
| PA-10 | US 11,060,392 — *Wireless load position sensor* | Wireless load + position sensing | Granted patent | Still a load sensor | No load sensor in the loop |
| PA-11 | Accelerometer for rod position **integrated into the load cell** (patent literature) | Accelerometer position transduction | Economical position sensing | Presupposes a load cell | IMU used for kinematics, motor power for load |
| PA-12 | US 9,157,431 — *Counterbalance system for pumping units*; US 10,107,282 — *Articulated reciprocating counterweight* | Mechanical counterbalance hardware | — | Hardware, not estimation | Orthogonal — we estimate counterbalance rather than re-engineer it |
| PA-13 | API torque-factor method | Infers gearbox torque from a surface card and a known counterbalance | Industry standard | **Requires the counterbalance moment to be known** | The quantity identified blind here |
| PA-14 | Gibbs (1963), damped wave equation; Everitt & Jennings finite-difference downhole card | Surface card → downhole card | Foundational, universally used | Requires an accurate surface card as input | Used as the forward operator inside an inverse problem |

---

## 3. The gap

Every published motor-power method (PA-1…PA-4) requires the maximum counterbalance moment `M` and the structural unbalance `SU` as inputs. In an ageing brownfield unit these quantities are difficult or impossible to obtain:

> *"A common problem in determining existing counterbalance moment is that the weights and centre of gravities for particular crank types are not known, and the method cannot be accurately used if the location of weights from the end of the crank is unknown or was not recorded… some types can be hidden in a pocket in the master weight."*
> — Southwestern Petroleum Short Course, *Best Method to Balance Torque of a Pumping Unit Gearbox*

> *"Counterbalance adjustments on existing beam unit designs are performed manually by repositioning, adding or removing counterweights in an equipment- and labour-intensive process requiring unit shut-down and restraint."*
> — US 9,157,431, background section

The counterbalance is therefore unrecorded, physically hidden, and measurable only by a test that requires taking the well off production. On a low-rate well that cost is rarely justified, which is a plausible reason why a well-established technique has not been deployed at scale on marginal wells.

---

## 4. Contribution

This project claims a three-part combination for which no prior art was found in the sources searched:

1. **Blind self-calibration.** Joint identification of the counterbalance moment `M`, its phase `τ`, the structural unbalance `SU` and an effective geometry scale directly from operating data, with no counterbalance test and no shut-in.
2. **Calibrated-uncertainty health inference.** The output is a set of physically meaningful pump parameters (fillage, leak rates, net stroke) with a residual-based confidence measure, rather than a hard class label — so the system can decline to answer.
3. **Decision-coupled objective.** The optimisation target is tonnes recovered per workover-rig-day across a fleet, not per-well classification accuracy.

**What is explicitly not claimed:** electrical dynamometry, dynamometer-card classification, accelerometer position sensing, or any assertion of being first, unique or unprecedented. All are prior art and are cited above.

---

## 5. Intellectual property notes

These are engineering observations, not legal advice.

| Element | Assessment |
|---|---|
| Blind counterbalance/geometry identification | Strongest candidate for protection — a concrete technical method with a technical effect (measurement without shut-in) |
| Sensor-kit architecture (CT + beam IMU, no load cell) | Possible device claim; must be assessed carefully against PA-9, PA-10 and PA-11 |
| Uncertainty-aware health estimate coupled to rig scheduling | Likely to attract a §3(k) *"algorithm per se"* objection under the Indian Patents Act unless claimed as part of the sensing system |

The specific regularisation and identifiability construction underlying the blind calibration is not documented in this repository.
