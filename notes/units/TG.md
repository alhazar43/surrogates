# TG The theory end of surrogate G, Guided learning

## Plan
- Target: Give surrogate G of C:/Users/YuanW/PLMS/surrogates/program/research-program.md section 4 its theory, so each tier has a defined object, a derived prediction and a calibration with known truth.
- Goal: Write theory/G.md, the theory end of surrogate G.
- Establishes: The object, its assumptions, what can be derived, the calibration of its tier 1 reading, and a prediction per tier.
- Cannot establish: Any empirical result. Nothing is trained or fitted on real data.
- Inputs: C:/Users/YuanW/PLMS/surrogates/program/research-program.md (all); C:/Users/YuanW/claude-thesis-docs/exploration/signature/goodfire-reading.md section 7; C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md (the synthesis sections S1 to S15 first, member sections as needed); C:/Users/YuanW/claude-thesis-docs/exploration/signature/three-questions.md Part E.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/G.md; new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md; any tiny synthetic script under C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_g/.
- Check: theory/G.md holds the eight sections of the Specification, each derivation marks its rung, and every cited source has a row in sources/reading.md.
- Open choices: the formal object, the derivations and the calibration design are the builder's; the choices that change scope go to section 8 as options for the author.
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 1 (fix after RevG, theory/G-review.md)
- If pass: the lead reads section 8 and folds accepted edits into the research program.
- If fail: the missing sections are named and one attempt fixes them.
- Resolution: none

### Specification
The surrogate answers one part of "what is learning": whether a learner's measured state tells which experience will change it most. The theory file has these sections.
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
theory/G.md exists with all eight sections and a tier 1 calibration computed on a synthetic learner; missing are the permutation-test reading of R7 (derived, not computed) and the full text of Foster and Foerster 2025 (PDF unreadable).
Effect: 8 of 8 sections written, ratio 8/8 = 1.0

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/G.md (219 lines; sections 1 to 8; results R1 to R10, each with its rung)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_g/calibrate_tier1.py (python -I, tiny synthetic records, about 4 min on the PC)
- C:/Users/YuanW/PLMS/surrogates/results/theory_g_calibration.json (the numbers of G.md section 4)
- C:/Users/YuanW/PLMS/surrogates/sources/reading.md, 11 new rows (10 sources and the peer blueprint as data)

Requirements
- Not met. Every reading calibrated before use. The blueprint rule and the t tests over 3 and 10 seeds are calibrated; the permutation test across weightings that G.md proposes for tier 1 is not.
- Not met in part. Full text for a detail. Foster and Foerster 2025 and Wang et al. 2020 were opened at the abstract only; G.md uses no detail from them beyond the abstract and marks the rung.
- Met. Eight sections, rungs marked, every cited source has a row in sources/reading.md.
- Met. A null case and planted cases on a learner with known truth, with false positive rate and power (G.md section 4).
- Met. The research program not edited; the irt4llm project not changed; learner-state, learner-models and docs/conversation not opened; no cluster jobs; no commits; at most 500 lines.

What was tried
- The object is the one-step value V(phi, x) of an item on a fixed bank and the guidance gain of a selection rule at equal cost. At first order V is one column of the learner's logit kernel, the object of panel S11.
- Derived. Maximum information, plain value, SD-normalized value and learnability rank differently (R1) and coincide only for a Rasch learner with constant a (R2). Population discrimination helps only through the learner's kernel alignment (R3). A precision learner has its optimum at pass rate 0.8240 and none at 0.5 (R4, checked numerically).
- The critic arms of the peer's Q1 change variance, not the expected step (R5), so they are not a tier of G. The I1 weight is a^2 sqrt(p(1-p)), not the information that the blueprint states (R6). The peer was not told, since the unit allows no change there.
- Calibration on a 2PL learner, 400 experiments per case. The blueprint rule had false positive rate 0.0025 to 0.0125 and power 0.56 to 0.61 at a planted gain of +0.018 bank accuracy. A 10-seed t test had power near 1.0 but false positive rate 0.105, from the pool-conditional effect of a fixed weight (R7).
- Online discrimination from one learner's answers has SE 0.61 to 1.22 at 16 answers per item over 10 checkpoints (R9). Tier 2 therefore predicts from logit velocity.
- The first calibration claimed that one permuted arm controls R7. That claim is wrong, since arm I keeps its own fixed subset. It was replaced by a permutation test across several fixed weightings before the file closed.

Disagreement: none with the Plan. The research program's tier 1 wording mixes the indistinguishable case of R2 with the case that can differ; G.md section 8, edit 1, proposes the change.

Lesson: none

### Attempt 1
theory/G.md now has all 12 critical and major review items fixed, R6 withdrawn and the permutation test calibrated. Still not met: the full text of Foster et al. 2025 and Wang et al. 2020, and a calibration of the second tier 1 reading.
Effect: 12 of 12 critical and major items fixed, ratio 12/12 = 1.0

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/G.md (284 lines; fix log at lines 254 to 284, one row per review item C1 to C3, M1 to M9, m1 to m15)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_g/calibrate_permutation.py (new; PT with K = 19, 200 experiments per case, 4 min 50 s, python -I)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_g/derived_numbers.py (new; the GRPO shape, R9 SEs and answer counts, guessing floor, c'_B, net pool SD, t10 rate from the ratio, BP parts)
- C:/Users/YuanW/PLMS/surrogates/results/theory_g_permutation.json, results/theory_g_derived.json (new)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_g/calibrate_tier1.py (docstring only; numbers unchanged, not rerun)
- C:/Users/YuanW/PLMS/surrogates/sources/reading.md (rows Mindermann, Foster, Peer blueprint corrected)

Requirements
- Not met. A detail from full text. Foster et al. 2025 and Wang et al. 2020 stay at abstract; G.md uses nothing beyond the abstract.
- Not met in part. Every reading calibrated before use. PT is calibrated (false positive rate 0.040 to 0.060, power 0.665 to 0.795 with 1 seed, 0.965 to 0.995 with 3, at +0.018). The four-arm shape reading of tier 1 is not, and G.md says so (section 5, choice 3).
- Met. Eight sections, rungs marked, at most 500 lines, every cited source has a row.
- Met. The research program, the review and irt4llm not edited; learner-state, learner-models and docs/conversation not opened; no cluster jobs; no commits.

What was tried
- C1. R4 now carries c'_B. The sharpening peak lies at 0.824 or 0.176 by the sign of c'_B. With a guessing floor of 0.25 the midpoint is 0.625 and the peaks are 0.868 and 0.382. The tier 1 shape and tier 3 predictions depend on that sign.
- C2. Velocity and a_hat are one estimate. Reliability 0.8 needs k = 79 (spread 0.5 logit) or 314 (spread 0.25) answers per item per checkpoint at C = 10. Tier 2 reads a probe set above that floor.
- C3. R6 withdrawn as the v2.1 record.
- M1, M2. The net pool SD 0.0024 against the read SE 0.0026 predicts a T10 rate of 0.104 (observed 0.105), 0.067 for N1 (0.060), 0.080 for N2 with run noise (0.0725).
- M3, M4. K = 19 is computed: 20 runs, 72 to 76 h (1 seed), or 60 runs, 216 to 228 h, at the current 3.6 to 3.8 h per run (blueprint.md:105, changed from the 3.8 to 5.3 h of attempt 0).
- M5 to M9. GRPO at G = 5 ranks near p(1-p). R5 is restated as a second moment, with b* = 1 - p on one bit. A8 adds the optimizer. The shape reading gets four narrow band arms. Choice 1 uses G's own arms.
- m14 relays the checker's reading of Mindermann Eq. (3); this attempt did not reopen the PDF.

Disagreement: none.

Lesson: none

## Audit
| item | outcome | evidence |
|---|---|---|
| 1. Object signatures, V, G_T (theory/G.md:7-18) | holds | definitions are typed and consistent; severity none |
| 2. First-order V and grad F (theory/G.md:27-28) | holds | differentiating mean_j sigmoid(l_j); severity none |
| 3. Leave-in group mean gives a prompt-dependent factor (theory/G.md:40) | narrowed | sum_i (r_i - rbar)(r_i - p) = k(G - k)/G; the leave-in mean alone gives (G - 1)/G; severity minor |
| 4. Blueprint names three differences for arm G (theory/G.md:40) | holds | blueprint.md:55 says three and lists four; severity minor |
| 5. R1 V = eta c_B alpha p (1 - p), c_B > 0 (theory/G.md:57) | holds | rederived under A1 to A3; c_B > 0 needs alpha_j > 0, unstated; severity minor |
| 6. R1 SD-divided step equals GRPO in expectation (theory/G.md:65) | contradicted | G = 5, E sqrt(q(1-q)) / sqrt(p(1-p)) = 0.57, 0.73, 0.86 at p = 0.1, 0.2, 0.5 (checks.py); severity major |
| 7. R1 information row, Hofmann Eq. 4 and 5 (theory/G.md:66) | holds | arXiv 2509.11106 HTML, Eq. 4 a^2 p(1-p), Eq. 5 argmax information; severity none |
| 8. R1 learnability and band rows (theory/G.md:67) | holds | Foster et al. abstract "variance of success"; Bae et al. Section 3.4 band 0.2 to 0.8; severity none |
| 9. Value is information divided by a (theory/G.md:69) | holds | alpha p(1-p) = a^2 p(1-p) / a at a = alpha; severity none |
| 10. R2 four rules rank by p(1 - p) (theory/G.md:71) | narrowed | a band is a uniform draw, not a ranking; constant alpha with varying a omitted; severity minor |
| 11. R2 restates blueprint section 1 (theory/G.md:71) | holds | blueprint.md:14; severity none |
| 12. R3 covariance condition and alpha_eff (theory/G.md:76) | holds | E_w V = mean V + Cov(w, V)/mean w; V_x = eta p_x(1-p_x) <grad F, grad l_x>; severity none |
| 13. R3 loss through fewer distinct items (theory/G.md:79) | narrowed | not derived; theory_g_calibration.json N1 -0.00002 (SE 0.00003), N2 -0.00002 (SE 0.00005); severity minor |
| 14. R4 l tanh(l/2) = 1, l* = 1.5434, p* = 0.8240 (theory/G.md:84) | holds | brentq 1.54340, grid argmax 1.54338, sigmoid 0.82396 (checks.py); severity none |
| 15. R4 Wilson 0.1587, 0.1839, logistic between (theory/G.md:87) | holds | PMC6831579 Eq. 30, 31; no logistic case; logistic error 0.1760; severity none |
| 16. R4 bank gain peaks near 0.82 under sharpening; tier 1 second and tier 3 predictions (theory/G.md:87, 153, 157) | contradicted | V_x = eta c'_B delta_x p_x(1-p_x); c'_B = -0.018 at bank accuracy 0.465, +0.019 at 0.536 (checks.py); severity critical |
| 17. R4 differs from the logistic peak of S1.2 (theory/G.md:87) | holds | panel-2026-10-19.md:1428; severity none |
| 18. R5 expected step unchanged by an answer-independent b (theory/G.md:89) | holds | E[grad log pi] = 0; severity none |
| 19. R5 variance p(1-p) + (p - b)^2; critic helps only through MSE of b (theory/G.md:89) | narrowed | second moment, not variance; one-bit optimal b = 1 - p, 0.700 at p = 0.3 (checks.py); severity major |
| 20. R5 arms C and I0 differ only in b (theory/G.md:89) | holds | blueprint.md:55 now; false for v2.1 per blueprint.md:34; severity none |
| 21. R5 zero advantage 0.328, 0.591 at G = 5 (theory/G.md:89) | holds | 0.2^5 + 0.8^5 = 0.328; 0.1^5 + 0.9^5 = 0.5905; severity none |
| 22. R6 I1 weight a^2 sqrt(p(1-p)), not the stated information; clip; R6 corrects the blueprint (theory/G.md:91-97, 65, 184) | contradicted | blueprint.md:29 no divisor, no clip; :32 a^2 p(1-p); :34 divisor removed; blueprint.md saved 11:08:17, G.md 11:14:26; severity critical |
| 23. R6 clip factor 4.6 (theory/G.md:97) | holds | 1/sqrt(0.0475) = 4.588; clip absent from current blueprint; severity none |
| 24. R7 one permuted arm insufficient; weighting as unit (theory/G.md:99) | holds | arm I keeps its subset; weightings exchangeable under the null; severity none |
| 25. R8 second-order term with K(x, y), negative entries lower gain (theory/G.md:101) | narrowed | term not sign-definite; K(x, y) enters with (1 - 2 p_y); severity minor |
| 26. R8 Huang Theorem 3 interference (theory/G.md:101) | narrowed | arXiv 2605.29548v2 Theorem 3 utility order; interference in Section 3.2; severity minor |
| 27. R9 information k C 0.2 Delta^2 / 12 and table (theory/G.md:103-109) | holds | arithmetic exact; evenly spaced points give SE 0.554, 1.108, 0.206 (checks.py); severity minor |
| 28. R9 conclusion, velocity over a_hat; tier 2 prediction (theory/G.md:111, 155) | contradicted | velocity = a_j dtheta/dt, same relative SE as a_hat; logit difference SE 0.79 at k = 16; severity critical |
| 29. R10 Park Definition 3, Section 4.1 (theory/G.md:113) | holds | arXiv 2406.19370v4 HTML, Definition 3, Section 4.1 heading; severity none |
| 30. R10 information as sensitivity (theory/G.md:113) | holds | I = (dp/dtheta)^2 / (p(1-p)) = a^2 p(1-p); severity none |
| 31. Script output equals the result file (theory/G.md:145) | holds | rerun python -I, JSON identical after key sort, 4 min 12 s; severity none |
| 32. Script step and its expectation (theory/G.md:119) | holds | calibrate_tier1.py:88; E[(r - ph)(r - p)] alpha = alpha p(1-p); severity none |
| 33. Script BP rule equals the blueprint rule (theory/G.md:131) | holds | calibrate_tier1.py:100-103; blueprint.md:48; severity none |
| 34. Table values against the JSON (theory/G.md:134-141) | holds | every cell rounds from theory_g_calibration.json; severity none |
| 35. Column SD of the pool-conditional effect (theory/G.md:134-141) | contradicted | calibrate_tier1.py:113 includes seed noise; net 0.0024, 0.0025 in run-noise rows, stated 0.0035, 0.0036; severity major |
| 36. N1, N2 zero by construction (theory/G.md:123-124, 140) | narrowed | first order only; N2 run noise +0.0003 (SE 0.00014); severity minor |
| 37. Binomial SE 0.011; 0.105 is 5 SE above 0.05 (theory/G.md:143, 147) | holds | sqrt(0.05 x 0.95 / 400) = 0.0109; 0.055 / 0.0109 = 5.05; severity none |
| 38. Bank accuracy 0.659 to 0.670; band 0.585 to 0.594 (theory/G.md:143) | holds | JSON bank_acc_B_mean, band_share_mean; severity none |
| 39. Pool SD 0.0024 about 5 times SE 0.0005 explains 0.105 (theory/G.md:147) | contradicted | read SE 0.0026, ratio 0.91, simulated rate 0.104; ratio 4.8 gives 0.36; severity major |
| 40. BP strict because of 2 of 3 seeds and the seed mean (theory/G.md:147) | narrowed | per-seed test binds, about 0.59 per seed against 0.96 for the seed mean (approximation); severity minor |
| 41. BP false positive rate rises past item SE (theory/G.md:149) | narrowed | no row with run SD above item SE; largest 0.0087 against 0.0081; severity minor |
| 42. Script runtime (calibrate_tier1.py:26; theory/G.md:145) | contradicted | docstring a few seconds; measured 4 min 12 s; severity minor |
| 43. Tier 1 refutation under the calibrated rule (theory/G.md:153) | contradicted | G.md:99 not run; G.md:210 not yet computed; TG Result requirement not met; severity major |
| 44. K = 2 permuted weightings, 34 to 48 h (theory/G.md:194, 210) | contradicted | smallest p = 1/(K + 1) = 0.33; K = 19 needs 60 runs, 228 to 318 h; severity major |
| 45. Tier 1 second reading measurable (theory/G.md:153, 204) | unchecked | no estimator for gain per training item; band [0.2, 0.8] excludes 0.824; severity major |
| 46. Optimizer assumption (theory/G.md:24-50, 153) | contradicted | none named; panel-2026-10-19.md:1429 requires it; severity major |
| 47. Choice 1 option B and pick C use the band arm C (theory/G.md:181, 194, 196) | contradicted | blueprint.md:43 arm C is a critic; band arm in archive v2.0:32; severity major |
| 48. I1 equals drawing proportional to a_j in expectation (theory/G.md:184) | holds | blueprint.md:45 a_j scaled to mean 1; first order; severity none |
| 49. C against I0 tests the critic, not a tier of G (theory/G.md:184) | holds | blueprint.md:31, 55; severity none |
| 50. Tier 3 information clause separates learners (theory/G.md:157) | narrowed | also holds under location with varying alpha (R1); severity minor |
| 51. Post-training mostly sharpens, S1.4 (theory/G.md:153) | narrowed | panel-2026-10-19.md:1430 decoding temperature; no "mostly"; severity minor |
| 52. Consistency with panel S11 and S14 (theory/G.md:32, 42) | holds | panel-2026-10-19.md:2857, 2877; severity none |
| 53. Option B cost arithmetic (theory/G.md:194) | holds | 9 x 3.8 = 34.2 h, 9 x 5.3 = 47.7 h; blueprint.md:97; severity none |
| 54. Bae et al. abstract against Proposition 3.1 (theory/G.md:176) | holds | arXiv 2504.03380v2 HTML, abstract and Proposition 3.1; severity none |
| 55. Weinshall et al. abstract against Theorems 1 and 2 (theory/G.md:176) | holds | arXiv 1802.03796v4 PDF, abstract "increasing", Theorem 1 "decreasing", Theorem 2 "increasing with the loss"; severity none |
| 56. Clement et al. pooled tests, four groups, randomization not stated (theory/G.md:173, 176, 182) | holds | arXiv 1310.3174v2 PDF, Section 6, Eq. 1; severity none |
| 57. Mindermann et al. Section 3 (theory/G.md:167) | holds | arXiv 2206.07137v3 PDF Eq. (1) to (3); reading.md says Eq. 6; severity minor |
| 58. Huang et al. Section 3 setup (theory/G.md:179) | holds | arXiv 2605.29548v2 HTML, K = 32, power law, width N, Adam; severity none |
| 59. Hofmann et al. 2PL, 102 models, MMLU (theory/G.md:171) | holds | arXiv 2509.11106 HTML Section 3 and 4.1; severity none |
| 60. Foster and Foerster 2025 record (theory/G.md:67, 169) | narrowed | arXiv lists five authors, NeurIPS 2025 lists four; severity minor |
| 61. Wang et al. 2020 measurement objective (theory/G.md:174) | holds | arXiv 2007.12061 abstract, third task predicts the student's answers; severity none |
| 62. Park et al. 2024 record (theory/G.md:165) | holds | arXiv 2406.19370, NeurIPS 2024; severity none |
| 63. Wilson et al. 2019 record (theory/G.md:166) | holds | Nat. Commun. 10, 4646, PMC6831579; severity none |
| 64. Consistency with research program tier 3 (theory/G.md:157, 217) | contradicted | research-program.md section 4 G tier 3 claim is the opposite; edit 4 leaves it; severity minor |

Audit by the RevG checker, 19 October 2026. Full review with the critical, major and minor lists: theory/G-review.md. 3 critical (rows 16, 22, 28), 9 major, 17 minor of 64 rows.
