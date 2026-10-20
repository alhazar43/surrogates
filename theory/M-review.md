# Review of theory/M.md

Written by the RevM checker on 19 October 2026 for unit notes/units/RevM.md. theory/M.md was read line by line and not edited. Each derivation was redone by hand or in a shell. The calibration script was rerun with python -I. Every source of section 6 was opened in its primary text. The peer numbers were checked against C:/Users/YuanW/PLMS/irt4llm/notes/state.md (row S1), notes/blueprint.md and results/Q2a/personfit.csv, which were not changed. Shell checks ran from the scratch folder C:/Users/YuanW/AppData/Local/Temp/revm (chk/checks.py, chk/power1500.py, rerun.json, src/*.txt for the fetched sources). No project file was written except this one and the two unit files.

Severity. Critical changes a prediction, a calibration number, or a claim of a source that a derivation, a prediction or the Disagreement uses. Major is a wrong derivation, a contradiction or a missing condition that changes no prediction. Minor is wording, rung or a number that enters nothing.

## Critical errors

C1. P4(b) misstates panel B3 Proposition 3, and with it the SFT half of the tier 1 prediction loses its only derivation (M.md:66, 139). M.md says that under a constant bank to training kernel "every route gives a pure person change". The panel says the outcome parts of both routes give a pure person change, and that the teacher route adds the import term J_B G, which "is item specific in general" (panel-2026-10-19.md:372 to 382; "Only the import term J_B G separates the routes under a Rasch kernel"). Under M.md's own wording, SFT on teacher text under a Rasch kernel keeps the order, which contradicts the prediction "Each SFT state rejects". M.md gives no other derivation of an SFT reorder. Corrected statement. Under a constant kernel the outcome part of every route is a pure person change; the teacher route adds an import term that is item specific in general, and that term is the derived reason for an SFT reorder.

C2. The tier 1 prediction is calibrated at a design it does not use, and at its own design it tests a sharp null at power near 1 (M.md:122, 138 to 141). Section 4 calibrates J = 200 items at k = 16 and calls 16 "the peer's number of draws". The tier 1 material has J = 1,500 items, 16 draws per option order and two orders averaged, so 32 draws per item (irt4llm state.md row S1, 48,000 rows = 1,500 x 16 x 2), and a split half null at k = 8. With the unchanged test at J = 1,500 and k = 32 over 40 synthetic learners (chk/power1500.py), the rejection counts are 40 of 40 for a move of 1 logit along 2PL slopes, 10 of 40 for a move of 0.3 logit, 40 of 40 for an item specific change of SD 0.25 logit, 16 of 40 for SD 0.1 logit, and 0 of 40 with no effect at k = 32 and at k = 8. So "GRPO does not reject" holds only if the GRPO change on the bank is in C_mono to within about 0.1 logit per item. M.md's own P5 says that a person change under the population 2PL, the peer's model of these states, reorders. The prediction does not state the assumption that would make GRPO avoid P5 (item blind geometry on the bank, panel B3 Proposition 2), and it gives no tolerance. The two option orders also leave P3b open, since P3b's control needs every option order (M.md:62). Corrected statement. Recalibrate at J = 1,500, k = 32 and k = 8; state the prediction with a tolerance on the residual SD or on the population axis move; name the item blind assumption; or predict rejection for GRPO under P5 and read the residual after C_pop(a).

C3. The Jain et al. claim "the wrapper is tied to the spurious attribute" is stated without its condition (M.md:162, 210, and the Disagreement of TM.md). In Jain et al. (arXiv 2311.12786v2, section 5.1) the fine-tuned model depends on the spurious attribute when the pretraining prior of the fine-tuning operand is low; when that prior is raised, "it starts to perform well on the test data regardless of the presence of the spurious attribute" (src/jain.txt:268). The wrapper over a weakly relevant capability is stated as a hypothesis ("We hypothesize that, at a mechanistic level, the model exploits the weakly relevant capability by learning a wrapper over it", src/jain.txt:269), supported by pruning and probing. Proposed edit 2 turns the claim into a prediction for a mixed task bank in general. Corrected statement. When the capability is weakly relevant (low pretraining prior) the learned wrapper exploits the spurious attribute; with a strongly relevant capability the fine-tuned model works without it. A reorder by task on a mixed bank is an idea, not a derivation from Jain et al.

## Major errors

M1. The identification condition of P5 is wrong in both directions (M.md:69, and the table row at M.md:85). The move eta0 + delta a keeps the order exactly when delta (a_j - a_i) < eta0(i) - eta0(j) for every pair with eta0(i) > eta0(j), which is P5's own crossing rule. A strictly decreasing a with a small delta keeps the order (rho 1.000 at delta 5 with a = exp(-0.1 eta0), chk/checks.py), so "not increasing" does not give identification. An increasing a with delta < 0 reorders (rho -0.845 at delta -5 with a = exp(0.3 eta0)), so "increasing" does not give non-identification. "a delta is of the form g(eta0)" does not put it in C_mono; P4(a) needs g' > -1. Corrected statement. The move is told apart from C_mono if and only if it reorders at least one pair, that is if delta exceeds min over discordant pairs of (eta0(i) - eta0(j)) / (a_j - a_i), with the sign of delta taken into account.

M2. "The null C_mono cannot represent a capability change that is a function of the base log odds alone" states the reverse of P4(a) two lines above (M.md:67). C_mono contains every such change with g' > -1, which is why a failure to reject says nothing about capability. The conclusion of the sentence holds; its premise is false as written.

M3. P2(d) drops the step condition of its source (M.md:58). Panel B3 Proposition 2 is a flow result with a Lipschitz right side (panel:368). Panel B4 point 4 states that a discrete step p to p + eta f(p) keeps the order only while eta f'(p) > -1 (panel:442). The same condition is P4(a)'s g' > -1. "The whole training path keeps the order" holds for the gradient flow, not for the training algorithm with a finite step.

M4. P3a is called a change of use without an argument (M.md:60 to 61). The research program's "use" is idea 1, holding a capability against using it, the wrapper (research-program.md:30, 56). A sharpening of the option logits is neither a wrapper nor a change of the mixture weight u of P2(b), and as a decoding temperature it breaks A3. P3a shows that a change of no capability can reorder; it does not show that a change of use reorders. P3b does show that, under a failure of A5.

M5. Proposed edit 1 says "A change that depends on more than the base log odds reorders items" (M.md:209). A change that depends on more than eta0 reorders only if it crosses a pair (M1); a small item specific change or a move along slopes that are not increasing in eta0 can keep the order. Corrected wording. "can reorder items".

M6. The tier 1 prediction has two parts without a refutation condition (M.md:139, 141). "A fitted relink slope c > 1" has no estimator at tier 1 and no refutation rule. "Each SFT state rejects" is refuted only "if no SFT seed rejects", so one rejection in three is read as consistent with "each".

M7. The second tier 2 prediction is already observed for the six S1 states (M.md:147). Under the 2PL, a shift s of all b is a(theta - b - s) = a((theta - s) - b), so one common shift of b is absorbed into theta, and lz at the fitted theta is unchanged up to the prior. Q2a already gives lz +2.03, +2.18, +2.52 for the three GRPO states (results/Q2a/personfit.csv). Only the 12 unseen Q1 states make it a prediction. A value below -1.645 is not covered by the refutation rule.

M8. The tier 2 deviance prediction reads an exceedance as curvature (M.md:144 to 145). An exceedance also follows from an item specific residual that tier 1 failed to detect (C2 gives the power), and from error in the base log odds if the bootstrap holds eta0_hat fixed. The statistic is not calibrated, as M.md states.

M9. The reading map M chooses among five sets, but only the C_mono boundary has a defined and calibrated test (M.md:29, 90 to 122). "Within the noise" is not defined for C_shift and C_aff. The project requirement is a calibration before use.

M10. P5's number 0.963 is not reproduced (M.md:69). The script gives a median noise free rho of 0.9586 at k = 16 and 0.9597 at k = 64 (calibration.json, population(1.0)), and section 4's table gives 0.959. Five further seeds of 200 banks give medians 0.9589 to 0.9606 (chk/checks.py). No file produces 0.963.

M11. The Maia Polo et al. row states as a finding what the source states as an expectation (M.md:164). The paper says the 40 specialized models "are likely to have unique strengths" and "their correctness patterns might be different" (src/tiny.txt:383); it shows only the estimation error (Figure 5). The setup is still usable; the claim is narrowed.

M12. "k = 16, the peer's number of draws" (M.md:122) narrows to 16 per option order; the tier 1 reading averages two orders, 32 draws per item (part of C2).

## Minor errors

m1. P3a's 0.987 (M.md:61) differs from the table's 0.986 (M.md:115; calibration.json 0.98637). Three further seeds give 0.9863 to 0.9867.
m2. P2's "rho = 1 under C_mono" needs h strictly increasing; a weakly increasing h makes ties (M.md:15, 54).
m3. P4(a) calls a common shift "the same observation as a use shift"; under P2(b) a use shift is a curved map, and only P2(a) gives a common shift (M.md:65).
m4. P6 marks a definition, r = eta1 - h_hat(eta0), as a derivation (M.md:71).
m5. P7's rate 1/sqrt(m k) omits the bias from the error in eta0_hat, which does not shrink with m; the same paragraph names that attenuation later (M.md:73).
m6. Stocking and Lord give two reasons against regression, error in the regressor and asymmetry, and rank asymmetry first (src/umn2.txt:71 to 77). M.md gives only the first, and the script's relink is the regression the source rejects, with no sentence on why the asymmetry objection does not apply here (calibrate.py:138 to 144).
m7. P8 calls the LLRA test of generalizability over item domains "the student form of the order test". The source describes equality of treatment effects across ability domains (src/umn1.txt:527 to 536), which is a test of a common shift against domain shifts, not of C_mono.
m8. The method of moments prior removes m(1 - m)/k from the variance; the exact removal divides by 1 - 1/k as well, so Var(p) is low by 1/k, 6.25 percent at k = 16 (calibrate.py:44; chk/checks.py, 0.0469 against 0.05).
m9. "The test holds its level" narrows to "does not exceed it". 3 of 200 rejections in the stretch and use weight rows has P(X <= 3) = 0.009 at level 0.05, so the test is conservative there (M.md:122).
m10. Section 7 names dependence between items as the risk to the false positive rate (M.md:175). For a fixed model the sampled answers are independent given p; the untested risk is a misfit of the Beta prior to the real distribution of p.
m11. "Half of each kind" is a Bernoulli(0.5) draw per item (M.md:61; calibrate.py:76).
m12. M.md writes Jain et al. 2024; the research program writes 2023 (research-program.md:30). arXiv 2023, ICLR 2024.
m13. "Equipercentile linking, the classical name of the C_mono relink" is stated from unopened texts (M.md:166).
m14. Tier 3 "the reorder rate is the same" needs a test and a tolerance (M.md:150).
m15. Tier 1 says "the order of items under each route was not seen". It holds, but the cited row S1 holds a per item mode change reading with a split half floor at k = 8 that M.md should name (irt4llm state.md row S1).
m16. The Nelder-Mead history has no artifact (M.md:134).
m17. The separation table row "population slopes known and not increasing in eta0" inherits M1 (M.md:85).
m18. Proposed edit 2 states an idea as a prediction (M.md:210; see C3).

## On the Disagreement with the research program

The research program says a change of use predicts a shift that a small relink repairs and a change of capability predicts a reorder (research-program.md:56). M.md says this is wrong in both directions. On the merits, the claim holds in substance at the rung of derivation, with three narrowings.
- Capability to reorder fails. P4(a) is correct, and under the Rasch model a general ability rise is a common shift. P5 shows that a general ability change under the 2PL does reorder, so whether a capability change reorders is set by the item side (slopes, dimensions), not by capability as such.
- Use to shift fails. P2(b) gives a curved increasing map that a two parameter relink does not repair for single items (P7), and P3b gives a reorder from a change of use when the fallback differs by item. P3a is not a change of use (M4).
- The Jain et al. support holds only for a weakly relevant capability (C3).
The replacement in proposed edit 1 is right except for "reorders", which must read "can reorder" (M5).

## Full table

| # | item | line | outcome | evidence | severity |
|---|---|---|---|---|---|
| 1 | Nesting C_0 in C_shift in C_aff in C_mono in C_all | 12 to 17 | holds | c = 1 gives C_shift in C_aff; c > 0 affine is increasing | none |
| 2 | C_shift is the Rasch reading with b_j = -eta0(j) | 32 | holds | theta - b_j = theta + eta0(j) | none |
| 3 | Reading map M over five sets | 29 | narrowed | only the C_mono boundary has a test (M.md:90 to 122); "within the noise" undefined | major (M9) |
| 4 | P1 Rasch non-identification | 50 | holds | (theta1 + t, b1 + t 1) gives the same eta1 | none |
| 5 | P1 2PL equivalence, orthogonal part identified | 50 | holds | eta0 + a(delta + t) + d - t a = eta0 + a delta + d | none |
| 6 | P1 Fischer and Formann p. 406 to 407 | 50 | holds | src/umn1.txt:389 to 390, "can also be postulated that the item difficulty ... changes"; title, authors, APM 6, 397 to 416, 1982 match | none |
| 7 | P1 Strobl et al. 2021, anchors untestable a priori | 50 | holds | src/strobl.txt, "neither assumption can be approved or empirically tested a priori"; APM 45(3) 214 to 230 | none |
| 8 | P2 rho = 1 and tau = 1 under C_mono | 54 | narrowed | needs h strictly increasing | minor (m2) |
| 9 | P2(a) KL optimum raises logit by 1/beta | 55 | holds | Rafailov et al. Eq. 4, src/dpo.txt:200 to 205; chk/checks.py logit difference 1.4286 = 1/0.7 | none |
| 10 | P2(b) mixture map increasing, in C_mono, outside C_aff | 56 | holds | chk/checks.py, slope of logit p1 on logit p0 varies 1.50 to 4.21 | none |
| 11 | P2(c) temperature on two outcomes is eta0 / tau | 57 | holds | binary softmax; panel D9 (panel:899 to 903) | none |
| 12 | P2(d) whole training path keeps the order | 58 | narrowed | flow only; step needs eta f' > -1 (panel:368, 442) | major (M3) |
| 13 | P3a derivation, p_B < p_A for tau < 1 | 61 | holds | chk/checks.py, equal 0.4259 at tau 1; 0.623 against 0.355 at tau 0.5 | none |
| 14 | P3a number 0.987 | 61 | contradicted | table 0.986 (M.md:115), calibration.json 0.98637 | minor (m1) |
| 15 | P3a is a change of use | 60 to 61 | narrowed | research-program.md:30, 56; A3 at M.md:42 | major (M4) |
| 16 | P3b item specific fallback reorders | 62 | holds | p1 = (u1/u0) p0 + (1 - u1/u0) g_j depends on g_j | none |
| 17 | P3b control (every option order) against the tier 1 design (two orders) | 62, 139 | contradicted | irt4llm state.md row S1, 2 orders | major (part of C2) |
| 18 | P4(a) Delta = g(eta0) with g' > -1 is in C_mono | 65 | holds | d(eta0 + g)/d eta0 = 1 + g' > 0 | none |
| 19 | P4(a) common shift "the same observation as a use shift" | 65 | narrowed | P2(b) use change is curved | minor (m3) |
| 20 | P4(b) "every route gives a pure person change" | 66 | contradicted | panel:372 to 382, outcome parts only; import term item specific | critical (C1) |
| 21 | "C_mono cannot represent" a function of eta0 | 67 | contradicted | P4(a) at M.md:65 | major (M2) |
| 22 | P5 crossing threshold | 69 | holds | eta1(i) - eta1(j) = eta0(i) - eta0(j) - delta(a_j - a_i); panel:378 | none |
| 23 | P5 number 0.963 | 69 | contradicted | calibration.json 0.9586 and 0.9597; chk/checks.py 0.9589 to 0.9606 | major (M10) |
| 24 | P5 identified iff a not increasing in eta0 | 69 | contradicted | chk/checks.py, decreasing a rho 1.000 at delta 5; increasing a rho -0.845 at delta -5 | major (M1) |
| 25 | P6 decomposition at rung derivation | 71 | narrowed | a definition | minor (m4) |
| 26 | P7 error of order 1/sqrt(m k) | 73 | narrowed | omits the eta0_hat error bias; RMSE 0.037 to 0.015 for m 10 to 50, ratio 2.47 against sqrt 5 = 2.24 | minor (m5) |
| 27 | P7 single item bias under C_mono not C_aff | 73 | holds | misspecified link, bias does not depend on m | none |
| 28 | P7 bank mean nearly unbiased under r with random anchors | 73 | holds | calibration.json bias -0.0063 to +0.0007 | none |
| 29 | P7 Stocking and Lord reject regression | 73 | narrowed | src/umn2.txt:71 to 77, two reasons; script uses regression (calibrate.py:138 to 144) | minor (m6) |
| 30 | P8 cohort ability against common drift not identified | 75 | holds | P1 applied to group means | none |
| 31 | P8 LLRA, theta_ai and theta_ai + delta_a | 75 | holds | src/umn1.txt:480 to 495 | none |
| 32 | P8 LLRA generalization test is the order test | 75 | narrowed | src/umn1.txt:527 to 536, equality of effects across domains | minor (m7) |
| 33 | Separation table, row 5 condition | 85 | narrowed | inherits row 24 | minor (m17) |
| 34 | Separation table, other rows | 81 to 86 | holds | rows follow P1, P3a, P3b, P4, section 4 | none |
| 35 | Null procedure steps 1 to 4 match code | 90 to 94 | holds | calibrate.py:49 to 62 | none |
| 36 | Method of moments removes binomial variance | 91 | narrowed | calibrate.py:44 omits 1/(1 - 1/k); chk/checks.py 0.0469 against 0.05 | minor (m8) |
| 37 | Synthetic learner settings | 96 to 100 | holds | calibrate.py:65 to 112 | none |
| 38 | "Half of each kind" | 61, 99 | narrowed | Bernoulli(0.5) per item, calibrate.py:76 | minor (m11) |
| 39 | Rejection rate table, 22 cells and noise free rho | 108 to 120 | holds | rerun with python -I in 4 min 1 s gives 159 of 159 values equal to calibration.json (rerun.json) | none |
| 40 | Standard errors sqrt(rate(1 - rate)/200) | 102 | holds | calibrate.py:130; 0.04 gives 0.0139 | none |
| 41 | "The test holds its level" | 122 | narrowed | 3 of 200, P(X <= 3) = 0.009 at 0.05; 0 of 40 at J = 1,500 (chk/power1500.py) | minor (m9) |
| 42 | "k = 16, the peer's number of draws" | 122 | narrowed | 16 per order, 2 orders, state.md row S1 | major (M12) |
| 43 | Relink table, 15 cells | 126 to 132 | holds | calibration.json linking_k0_64_k1_16, rerun equal | none |
| 44 | Relink wins 15 of 15, bias within 0.007 | 134 | holds | largest absolute bias 0.0063 | none |
| 45 | Nelder-Mead history | 134 | unchecked | no artifact of the first run | minor (m16) |
| 46 | Tier 1 material, 7 states, 1,500 items, k 16, 2 orders | 138 | holds | state.md row S1, 48,000 rows per state | none |
| 47 | Seen numbers 0.269, 0.797 to 0.836, lz +2.0 to +2.5 | 140 | holds | state.md row S1; personfit.csv 2.034, 2.182, 2.518 | none |
| 48 | Order of items not seen | 140 | holds | state.md row S1 holds a mode change reading, not item order | minor (m15) |
| 49 | Tier 1 prediction, GRPO keeps the order | 139, 141 | contradicted | chk/power1500.py 40 of 40 for SD 0.25 and for 1 logit along a; P5 at M.md:69 | critical (C2) |
| 50 | Tier 1 refutation rules for c > 1 and SFT | 139, 141 | narrowed | no estimator or rule for c; "each" against "none" | major (M6) |
| 51 | Tier 1 SFT reorder has a derivation | 139 | contradicted | none in M.md; P4(b) implies the opposite | major (part of C1) |
| 52 | Tier 2 deviance prediction and its reading | 144 to 145 | narrowed | other causes of exceedance; uncalibrated | major (M8) |
| 53 | Tier 2 Q2b lz prediction written before data | 147 | contradicted | a common b shift is absorbed into theta; Q2a lz seen | major (M7) |
| 54 | Tier 3 prediction and refutation | 149 to 152 | narrowed | no test or tolerance for "the same rate" | minor (m14) |
| 55 | Gini anchor point and criterion plot of Strobl et al. | 149, 160 | holds | src/strobl.txt abstract and section on the criterion plot | none |
| 56 | Fischer and Formann row | 158 | holds | as rows 6 and 31 | none |
| 57 | Stocking and Lord row and setups | 159 | holds | src/umn2.txt:103 to 122, 269 to 273; APM 7(2) 201 to 210, 1983 | none |
| 58 | Rafailov et al. row | 161 | holds | title, six authors, arXiv 2305.18290, 2023 | none |
| 59 | Jain et al. row, wrapper tied to the spurious attribute | 162 | narrowed | src/jain.txt:268 to 269 | critical (C3) |
| 60 | Jain et al. year | 162 | contradicted | research-program.md:30 gives 2023; arXiv 2023, ICLR 2024 | minor (m12) |
| 61 | Hofmann et al. row | 163 | holds | src/fluid.txt:328 (item parameters fixed), 561 (61 to 94 checkpoints, 102 models); COLM 2025 | none |
| 62 | Maia Polo et al. row | 164 | narrowed | src/tiny.txt:383, "likely", "might be different" | major (M11) |
| 63 | Unopened sources and equipercentile name | 166 | narrowed | stated without opening | minor (m13) |
| 64 | Section 7 risk named as item dependence | 175 | narrowed | answers independent given p for a fixed model | minor (m10) |
| 65 | Decision 1 rationale | 185 | holds | panel D4 row F34 (panel:483), D9 | none |
| 66 | Decision 3 cost 12 min 27 s, 48,000 answers | 198 | holds | state.md row S1 | none |
| 67 | Disagreement, capability does not imply reorder | TM Result | holds | P4(a), row 18; P5 bounds it | none |
| 68 | Disagreement, use does not imply shift | TM Result | narrowed | P2(b) and P3b hold; P3a not use (row 15) | none beyond M4 |
| 69 | Proposed edit 1 "reorders" | 209 | contradicted | row 24 counterexample | major (M5) |
| 70 | Proposed edit 2 | 210 | narrowed | idea stated as prediction; row 59 | minor (m18) |
| 71 | Proposed edits 3 to 5 | 211 to 213 | holds | follow P5, P6, P8 | none |
| 72 | File within 500 lines | all | holds | 213 lines | none |

Counts. 72 rows. 37 hold, 23 narrowed, 11 contradicted, 1 unchecked. By row, 3 critical, 14 major (12 distinct errors; rows 17 and 51 are parts of C2 and C1), 18 minor, 37 none.
