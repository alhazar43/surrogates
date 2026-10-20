# TM The theory end of surrogate M, Measurement under learning

## Plan
- Target: Give surrogate M of C:/Users/YuanW/PLMS/surrogates/program/research-program.md section 4 its theory, so each tier has a defined object, a derived prediction and a calibration with known truth.
- Goal: Write theory/M.md, the theory end of surrogate M.
- Establishes: The object, its assumptions, what can be derived, the calibration of its tier 1 reading, and a prediction per tier.
- Cannot establish: Any empirical result. Nothing is trained or fitted on real data.
- Inputs: C:/Users/YuanW/PLMS/surrogates/program/research-program.md (all); C:/Users/YuanW/claude-thesis-docs/exploration/signature/goodfire-reading.md section 7; C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md (the synthesis sections S1 to S15 first, member sections as needed); C:/Users/YuanW/claude-thesis-docs/exploration/signature/three-questions.md Part E.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/M.md; new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md; any tiny synthetic script under C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_m/.
- Check: theory/M.md holds the eight sections of the Specification, each derivation marks its rung, and every cited source has a row in sources/reading.md.
- Open choices: the formal object, the derivations and the calibration design are the builder's; the choices that change scope go to section 8 as options for the author.
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 1 (fix after RevM, theory/M-review.md)
- If pass: the lead reads section 8 and folds accepted edits into the research program.
- If fail: the missing sections are named and one attempt fixes them.
- Resolution: none

### Specification
The surrogate answers one part of "what is learning": what learning changes and what it leaves fixed, read through whether a calibration made before a change still reads the learner after. The theory file has these sections.
1. The object. Define it formally, as an equation or a signature, before any prose. Say which part of the question it answers and which parts it leaves.
2. The assumptions. List each so that it can fail, and say what fails with it.
3. What can be derived. Each result with its conditions and its rung (idea, derivation, checked numerically, source). Include the cases where two explanations cannot be told apart, and the conditions under which they can.
4. The calibration of the tier 1 reading. A case with no effect and a case with a planted effect of known size, on a learner whose truth is known by construction. Say how the false positive rate and the power are computed. If it is cheap, compute them on a tiny synthetic case and give the numbers with the script path.
5. One prediction per tier, written before any data. Say what refutes it and what each outcome changes in the tiers above.
6. The literature. Open the primary text of every source you rely on; for a detail, open the full text, not the abstract. Add one row per source to sources/reading.md with what it gives and its rung. Prefer setups that can be copied over inventing new ones, and name the setup to copy.
7. What the theory cannot show.
8. Choices for the author and proposed edits to the research program. Each choice gives options, mechanism, cost, what each shows, and your pick.
Rules. Plain English, short sentences, technical terms as their field defines them, no jargon of your own. The object comes first, then one sentence of reading. Do not edit the research program. Run no cluster jobs and no heavy computation; a local script runs on tiny synthetic records only, with python -I. Do not open C:/Users/YuanW/PLMS/learner-state, C:/Users/YuanW/PLMS/learner-models or any docs/conversation folder. No commits. At most 500 lines.

## Result
theory/M.md exists with all eight sections, and the tier 1 order test is calibrated on synthetic learners. Missing are a calibration of the tier 2 single item statistic, any tier 3 calibration, and four primary texts (Bechger and Maris 2015, Embretson 1991, Andersen 1985, Meredith 1993), which were behind a paywall.
Effect: 8 of 8 sections written, ratio 1.0

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/M.md (213 lines)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_m/calibrate.py (run with python -I, about 7 min on the PC, synthetic only)
- C:/Users/YuanW/PLMS/surrogates/results/theory_m/calibration.json and calibration.log
- 7 new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md (Jain et al., Rafailov et al., Stocking and Lord, Fischer and Formann, Strobl et al., Hofmann et al., Maia Polo et al.), each opened in full text

Requirements
- Not met. Every reading calibrated before use. Tier 1 order test met (false positive rate 0.015 to 0.060 at level 0.05, power 0.775 at item specific SD 0.5 logit, k = 16, J = 200). The tier 2 single item deviance and the tier 3 cohort reorder are stated but not calibrated. Neither is used yet.
- Met. Sources cited only after the primary text was opened; the unopened four are named in section 6 as not relied on.
- Met. Nothing from learner-state or learner-models opened; no docs/conversation opened.
- Met. No commits, no cluster jobs, local script on synthetic records with python -I, at most 500 lines, research program not edited.
- Met. The peer project irt4llm was read and not changed.

What was tried
- The object is a chain of nested sets of after states, C_0, C_shift, C_aff, C_mono, C_all, plus C_pop(A) for population slopes. The reading is the smallest set that holds the after state.
- Statistic, Spearman rho of base and after right answers, with a co-monotone posterior bootstrap null. An isotonic likelihood ratio and a split half disattenuated correlation were the alternatives; the bootstrap was kept because it needs no fit of the map.
- The first four option temperature design gave noise free rho 0.999, too small to show P3a; the design was rebuilt so that the two item kinds match in p at temperature 1 (rho 0.986).
- The tier 2 relink first used Nelder-Mead and showed a bias of -0.013 under a pure shift that an exact base did not remove. A BFGS fit with the exact gradient removed it; the whole script was rerun and the order rates are unchanged (same seeds).
- The lead's message renamed the bible to the research program; the file uses the new name.

Disagreement: The research program's line "a change of use predicts a shift that a small relinking repairs, a change of capability predicts a reordering" is wrong in both directions. Evidence, theory/M.md P3a (temperature on four options reorders, rejected at 0.320 at k = 16, calibration.json order_test_k16 temperature4(0.5)), P3b, P4 (a capability change that is a function of the base log odds keeps the order), and Jain et al. 2024 section 5, where the wrapper is tied to the spurious attribute. Rung, derivation and checked numerically. Complied; the corrected line is proposed edit 1 of section 8, and the research program is not edited.

Lesson: none

### Attempt 1
theory/M.md is fixed on all 3 critical and 12 major review items and 18 minor ones, and tier 1 is recalibrated at 1,500 items. Not met: the order test exceeds its level at that size, and tiers 2 and 3 stay uncalibrated.
Effect: 15 of 15 critical and major items fixed, ratio 1.0

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/M.md (301 lines, fix log at the end)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_m/calibrate.py (rewritten, python -I, 4 to 5 workers, 915 s) and prior_check.py
- C:/Users/YuanW/PLMS/surrogates/results/theory_m/calibration.json, calibration.log, prior_check.json, calibration_logitnormal_only.log (first run, stopped), calibration_attempt0.json (attempt 0 output, kept)
- C:/Users/YuanW/PLMS/surrogates/sources/reading.md, the Jain et al. and Maia Polo et al. rows of TM narrowed

Requirements
- Not met. Every reading calibrated before use. At J = 1,500 the order test has a false positive rate of 0.065 to 0.140 under order preserving changes (rule of both priors), so it does not decide. sigma_hat at threshold 0.22 has a false positive rate of at most about 0.05 in 5 of 6 such cells, but not for a U shaped after state (95 percent quantile 0.344). Its power is at least about 0.95 at an item specific SD of 0.25. The tier 2 deviance and the tier 3 test are uncalibrated.
- Met. Sources cited only after the primary text was opened. Nothing from learner-state or learner-models opened. No commits, no cluster jobs, synthetic records only, at most 500 lines. The research program, the review and irt4llm were not edited.

What was tried
- The Beta prior with the exact variance was anti-conservative at J = 1,500 (0.330 under a shift). A logit normal prior by marginal ML gave the smallest sigma_hat bias of four priors, but alone it reached 0.115 with no effect. The kept rule rejects only if both priors reject.
- The 90 percent interval of sigma_hat by test inversion covers the truth in 0.19 to 0.21 of learners at SD 0.25 and 0.5, so it is computed and not used.
- GRPO is no longer predicted to keep the order. T1.2 gives a tolerance of 0.35 logit, under the assumption of a sharpening plus an item blind kernel part.
- The calibration output of attempt 0 was overwritten by the first run and restored from the identical review rerun as calibration_attempt0.json.

Disagreement: none. The attempt 0 Disagreement with the research program stands as narrowed by the review (P3a is not a change of use). Proposed edit 1 now reads "can reorder".

Lesson: none

## Audit
Copied from theory/M-review.md by the RevM checker, 19 October 2026. Critical rows 20, 49, 59.

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
