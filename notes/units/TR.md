# TR The theory end of surrogate R, Reach, shortcut against rule

## Plan
- Target: Give surrogate R of C:/Users/YuanW/PLMS/surrogates/program/research-program.md section 4 its theory, so each tier has a defined object, a derived prediction and a calibration with known truth.
- Goal: Write theory/R.md, the theory end of surrogate R.
- Establishes: The object, its assumptions, what can be derived, the calibration of its tier 1 reading, and a prediction per tier.
- Cannot establish: Any empirical result. Nothing is trained or fitted on real data.
- Inputs: C:/Users/YuanW/PLMS/surrogates/program/research-program.md (all); C:/Users/YuanW/claude-thesis-docs/exploration/signature/goodfire-reading.md section 7; C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md (the synthesis sections S1 to S15 first, member sections as needed); C:/Users/YuanW/claude-thesis-docs/exploration/signature/three-questions.md Part E.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/R.md; new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md; any tiny synthetic script under C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_r/.
- Check: theory/R.md holds the eight sections of the Specification, each derivation marks its rung, and every cited source has a row in sources/reading.md.
- Open choices: the formal object, the derivations and the calibration design are the builder's; the choices that change scope go to section 8 as options for the author.
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 1 (fix after RevR, theory/R-review.md)
- If pass: the lead reads section 8 and folds accepted edits into the research program.
- If fail: the missing sections are named and one attempt fixes them.
- Resolution: none

### Specification
The surrogate answers one part of "what is learning": whether a change follows the surface of the practiced problems or their structure. The theory file has these sections.
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
theory/R.md exists with all eight sections and a tier 1 calibration on written learners; the calibration on trained learners (the seed spread sigma_delta of real MLPs) is missing.
Effect: 8 of 8 sections written, ratio 1.0

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/R.md (221 lines)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_r/calibrate.py and calibrate_output.txt (python -I, numpy and scipy, 3.3 minutes on the PC)
- C:/Users/YuanW/PLMS/surrogates/sources/reading.md, 7 rows appended (Lubana 2023, Hermann 2023, Shah 2020, Geirhos 2020, Raventos 2023, Lubana 2024, Merullo 2025). Jain 2023 and Park 2024 already had rows from other units; this unit opened both in full text as well.

Requirements
- Not met. Calibration of the reading on trained learners. Only written learners are calibrated; the seed spread of trained MLPs sets the power and is the first tier 1 build item (theory/R.md section 4, last paragraph).
- Met. Calibration on a case with no effect and planted effects of known size, with false positive rate and power (section 4 table).
- Met. Every source relied on was opened in full text (PDF text extracted with pdftotext); rows added.
- Met. Nothing opened from learner-state, learner-models or docs/conversation. No commits. Research program not edited. No cluster jobs. Under 500 lines.

What was tried
- Object: Lambda_s and Lambda_c, invariance to resampling the surface or the structure, measured on outputs over the crossed grid (Lubana 2023 Definitions 2 to 4 in the output form of Shah 2020). Alternative considered: Lubana's loss form; kept as a remark, since the output form gives the reference values of the calibration learners.
- Derived: the design null E Lambda_s = 0 for any learner invariant to the surface, and Lambda_s = lam for a cue mixture (checked: T_s mean -0.0001, 0.0499, 0.1001 at lam 0, 0.05, 0.1).
- Calibration result: in the binary product grid with a seed level surface dependence (sigma_delta = 0.3), the item level sign test rejects in 0.305 of null experiments; the seed level t test holds at 0.053 with power 0.893 at lam = 0.05 over 10 seeds. The K option design keeps the item level test at 0.030.
- Derived: threshold in surfaces per structure for a kernel learner, rule fraction m / (m + kappa_s / kappa_c), checked against kernel regression and gradient descent to four decimals. It turns the tier 3 threshold into a prediction computed from the base model, with departures marking moved features.
- Literature: 9 papers opened in full; setups named (Hermann 2023 for tier 1, Jain 2023 PCFG Counter with Lubana 2023 linear paths for tier 2, Raventos 2023 sweep design for tier 3).
- Scratch PDFs were downloaded to a temporary folder in the project and deleted after reading.

Disagreement: none.

Lesson: none.

### Attempt 1
theory/R.md is fixed against all 13 critical and major review items and 26 of 27 minor items; the calibration on trained learners is still missing.
Effect: 13 of 13 critical and major items fixed, ratio 1.0

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/R.md (365 lines, with "Fix log of attempt 1", one row per review item)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_r/fix_checks.py and fix_checks_output.txt (python -I, numpy and scipy, 21 s on the PC). calibrate.py unchanged, not rerun.

Requirements
- Not met. Calibration of the reading on trained learners. Unchanged from attempt 0.
- Not met. Review item 72 (run time of calibrate.py) is not checked. No number in R.md depends on it.
- Met. Every critical and major item fixed, each with its line in the fix log.
- Met. Tier 3 threshold restated on argmax answers. The closed form holds only in the symmetric design. The share of sampled answers and the asymmetric case under cross entropy are withdrawn as closed forms.
- Met. The tier 1 estimand is stated, bar Lambda, the mean over seeds (R.md lines 19 and 24).
- Met. Lambda_c calibrated, false positive rate and power beside it (R.md section 4).
- Met, with a note. No source was reopened in this attempt. Four locations and one reading come from the review and are marked "from the review" in R.md.
- Met. Research program and review not edited. No cluster jobs. Synthetic records only. At most 500 lines. No commits.

What was tried
- R5a, derived. Under the symmetry group of the design, full batch gradient descent from zero keeps the coefficients in the form a Y + b (1 - Y). The argmax is then that of S_j(x) for any class symmetric loss at every step. The result is a step at m* = kappa_s / kappa_c. Part D checks it under cross entropy at 200 and 5,000 steps, with exact ties at m = 4.
- Part D, asymmetric kappa_s. The cross entropy answers change with the step count, 0.333 after 200 and 0.417 after 5,000 steps at m = 3, against 0.500 under squared loss. This supports the withdrawal of a closed form there.
- Tier 3 now compares LoRA answers with the linearized base model (R9, the kernel at B = 0 through dF/dB = (dF/dW) A^T). Choice 6 (SGD against AdamW) is added, since R9 does not cover AdamW.
- The tier 2 refutation is now a barrier between seeds of one condition. The reward and keys comparison has no prediction. The Jain prediction is comparative and marked idea.
- Powers by noncentral t (Part E). E[w] formula and power against eps (Part F). Lambda_c calibration (Part G). Rerun on Hermann's grid (Part H).
- My cross entropy softmax shares differ from the review's 0.94 and 0.99 because my step size differs. The review's point holds: shares grow with steps and argmax does not.

Disagreement: none.

Lesson: none.

## Audit
| item | outcome | evidence |
|---|---|---|
| 1. [minor] Reference values (1,0), (0,1), (0,0) of rule, cue and constant learners (R.md 23) | narrowed | Rederived. Lambda_c(rule) = (1 - P_c'(r(c') = r(c))) / (1 - 1/K), which is 1 only if r(c') is uniform over K options. Lambda_s(cue) = 1 needs the same of h. Neither balance is stated |
| 2. [minor] Mixture learner has Lambda_s = lam exactly (R.md 23, 54 to 58) | narrowed | Rederived. The difference of terms is lam (1 - P_s'(h(s') = h(s))). This equals lam (1 - 1/K) only if h(s') is uniform for every s. A2 states a different balance, over wrong options within a structure |
| 3. [minor] E Lambda_s = 0 on a balanced grid for an A_s invariant learner (R.md 52 to 53, 58) | narrowed | Both terms equal q_{c,n}(h(s)), so Lambda_s = 0 exactly on any grid. Balance enters only the tests |
| 4. [none] Lambda_s = 0 for every A_s invariant learner, and loss and output forms agree on pure learners (R.md 25) | holds | Rederived. A pure cue learner changes its loss under A_s and a pure rule learner does not |
| 5. [none] Lubana et al. 2023, mechanism as invariance set (Definitions 2 to 4) and invariance on the loss (Definition 3) (R.md 25) | holds | arXiv 2211.08422 Section 3. Definition 3 says counterfactuals "do not increase its loss". Definition 4 says similar if I(theta_1) = I(theta_2) |
| 6. [minor] Shah et al. measure the same thing as S randomized accuracy (R.md 25) | narrowed | arXiv 2006.07710 Definition 1 is E[1{f(x) = y}] under S randomization, accuracy against the label. Lambda_s is the excess rate of the cue's answer. They agree at the extremes only |
| 7. [minor] Lambda_s is the first term of Hermann et al.'s reliance (R.md 25, 163) | narrowed | arXiv 2310.16228 Section 3. Reliance is a prefactor times (sum of y_hat sign(z_s) minus sum of y_hat sign(z_c)). It is defined in Section 3, not Section 4. It holds up to the constant |
| 8. [major] The item level test's 0.305 is a false positive rate with no cue use (R.md 38, 101, 134) | narrowed | Exact E[T_s given delta] on the grid of calibrate.py is 0.0250, 0.0500 and 0.1000 at delta 0.15, 0.3 and 0.6. Lambda_s(f) of Section 1 is a per learner quantity, so these learners follow the surface. The null holds only for E over seeds of Lambda_s, which is not stated |
| 9. [none] A1, the gauge and the third feature (panel B19, last paragraph) (R.md 35) | holds | panel 1735 |
| 10. [minor] A2, the null of the tier 1 test is exact only under balance (R.md 36) | narrowed | calibrate_output.txt Part B, the item rejection is 0.030 at alpha 0.05. Dirichlet(0.3) errors make the hit count Poisson binomial, which is conservative. The sign test of Part A is exact |
| 11. [major] A2, h fixed by the design before training, is enough for the K option item test (R.md 36, 109) | narrowed | calibrate.py line 96 draws u independently of h by construction. A trained learner's surface dependence is independent of h only if h is randomized independently of the surface features. Choice 1b rests on this |
| 12. [minor] A3, test cells new (panel S7 point 8) (R.md 37) | narrowed | panel 2147 holds. A3 fails for the binary design of Choice 1a, since the probe grid lies inside the Gaussian training support (Hermann et al. Section 3) |
| 13. [minor] A4, rejection 0.305 with no cue use (R.md 38) | narrowed | It depends on sigma_delta = 0.3, a chosen value that was not measured, and on the estimand of row 8 |
| 14. [none] A5, held and unused (panel 7A2 point 1) (R.md 39) | holds | panel 2226 |
| 15. [minor] A6, Theorem 1 conditions, and the linear system example in Section 8 (R.md 40) | holds | arXiv 2211.08422 Appendix F.2 (1 hidden layer ReLU, interpolating minimizers), Lemma 3 (perfectly predictive), Section 8 ("any two solutions of the linear system ... can be interpolated"). The proof uses Lemma 3, which is not proved there |
| 16. [minor] A7, a single simple cue sits in high curvature (R.md 41) | narrowed | This is the heuristic of R7 stated as a fact, with no rung mark |
| 17. [none] A8, the kernel regime with squared loss and minimum norm (R.md 42) | holds | It is consistent with R5. Tier 3 violates it (row 59) |
| 18. [none] A9, independence for the item test (R.md 43) | holds | It is stated as an assumption |
| 19. [none] R1, practiced cells identify nothing, and the Geirhos et al. quote in Section 3 (R.md 47) | holds | arXiv 2004.07780 Section 3, "As long as tests are performed only on i.i.d. data, it is impossible to distinguish between these" |
| 20. [minor] R1, a reading uses only the conflict cells (R.md 47) | narrowed | T_s averages all 900 probes, agree probes included (calibrate.py line 55). There is no bias, only added variance |
| 21. [minor] R3, the sum of y_hat sign(z_s) over the grid is zero for each z_c (R.md 60) | narrowed | It is zero in expectation |
| 22. [none] R3, null 1/(K - 1) and log likelihood ratio log(0.9 / 0.033) = 3.3 (R.md 60) | holds | ln 27 = 3.296. Panel C15 (panel 1834) computes it between two rules. The same arithmetic holds for a pure cue learner against a pure rule learner on an answer equal to h(s) |
| 23. [none] R3, unit of replication in the two designs (R.md 60) | holds | It is consistent with calibrate.py |
| 24. [minor] R4, the data give no push on agree cells and only the learner's bias decides (panel C16) (R.md 62) | narrowed | The loss does not separate the two solutions. Availability, an amplification in the data, decides in Hermann et al. (line 65 of R.md), and panel B26 point 3 shows a push toward the stronger feature |
| 25. [none] R4, Shah et al. theorem on LSN (R.md 63) | holds | arXiv 2006.07710 Section 4, Theorem 1. A one hidden layer ReLU network trained by mini batch gradient descent relies only on the linear coordinate |
| 26. [minor] R4, LMS-5 randomized AUC 0.50 and 1.00 in Table 1 (R.md 63) | narrowed | Table 1 is the decision template. The measured values are in Section 4, Figure 3(b) and Table 4 |
| 27. [minor] R4, Lubana et al. Lemma 3, proved in their simplified setting (R.md 64) | narrowed | arXiv 2211.08422 Appendix F. Lemma 3 is introduced as a rephrasing of Shah et al. and has no proof |
| 28. [none] R4, Hermann et al. predictivity, availability, Theorems 3 and 4, Figure 3A (R.md 65) | holds | arXiv 2310.16228 Sections 3, 4 and 6. Figure 3A uses ratio 64 and predictivity 0.85. Theorem 4 concerns a two layer ReLU network |
| 29. [none] R4, Park et al. Definition 3 and Section 4.1, and the link of concept signal to amplification (R.md 66) | holds | arXiv 2406.19370 Definition 3. With Hermann's e_i = alpha_i w_i z_i and a unit w_i, the norm of dx/dz_i is alpha_i |
| 30. [none] R5 closed form scores and eigenspace (R.md 72 to 82) | holds | Rederived. G = (kappa_s + kappa_cs) I + kappa_c B, and Y lies in the block eigenspace. Part C rerun is identical to four decimals |
| 31. [major] R5 consequence 1, logistic in log m, no sharp transition, midpoint the availability ratio (R.md 83) | narrowed | It holds for the score ratio. Argmax answers give cue at m = 3 (0.400 against 0.533) and rule at m = 5 (0.526 against 0.421), a step at m* = 4. "Availability ratio in kernel units" is an idea marked as a derivation |
| 32. [minor] R5 consequence 2, repetitions leave the rule fraction unchanged (R.md 84) | narrowed | It holds for squared loss, with or without ridge. Under cross entropy on fixed features, the share at m = 8 is 0.83 after 200 steps and 0.93 after 5,000 steps |
| 33. [none] R5 consequence 3, any kappa_c > 0 raises new surface cells (R.md 85) | holds | Rederived, the score is m kappa_c / lambda |
| 34. [none] Raventos et al. D = 8, K = 16, noise 0.25, threshold between 2^14 and 2^15, reversal (R.md 87) | holds | arXiv 2306.15063 Section 3.1, Figure 2 |
| 35. [minor] Raventos et al. weight decay lowers the threshold, layers and width raise it (R.md 87) | narrowed | Section 3.3, Figure 7. Width has no effect on the base model. Width and depth raise the threshold for the small model only |
| 36. [none] Lubana et al. 2024 p_c about 1/sqrt(abs(E) abs(K)), sqrt scaling, unexplained abs(K)^(3/2) (R.md 87) | holds | arXiv 2408.12578 Section 7.2, Figure 7 and the paragraph after it |
| 37. [critical] R5 conclusion, a sharper rise or a reversal of repetitions marks moved features (R.md 87) | contradicted | Argmax gives a step at m*. Cross entropy on fixed features gives a rule share of 0.94 (200 steps) and 0.99 (5,000 steps) at m = 16, against 0.80. More steps change the share |
| 38. [none] R6, Lubana et al. Definition 1, Conjecture 1, Figures 4 and 5, Table 2 (R.md 89) | holds | arXiv 2211.08422 Sections 2 and 5, Figure 5 caption, Table 2 caption |
| 39. [none] R7, curvature ratio about m (R.md 91) | holds | The arithmetic is right. It is marked as a heuristic |
| 40. [none] R7, Merullo et al. numbers (R.md 91) | holds | arXiv 2510.24256 Section 3.3 (OLMo-2 7B), Section 4 (23.1 percent), Section 6 (95 to 106, 66 to 74, 3 to 16 percent) |
| 41. [minor] R8, kernel learner at m = m* mixes both at 0.5 (R.md 103) | contradicted | A deterministic learner has tied scores at m*. It does not mix |
| 42. [none] R8, other rows (R.md 97 to 102) | holds | They are consistent with R1, A5, A6, R7 and A1 |
| 43. [none] Section 4 design text against code (R.md 107 to 114) | holds | calibrate.py lines 31 to 37, 77 to 103, 47 to 64 |
| 44. [none] Section 4 table, 11 rows (R.md 118 to 130) | holds | The rerun is identical to calibrate_output.txt |
| 45. [none] Spread of the estimates (R.md 132) | holds | calibrate_output.txt rows 0.3 0.00, 0.0 0.00 and Part B 0.00 |
| 46. [minor] Six times alpha, 1.6 standard errors, largest sign test size 0.011 (R.md 134) | holds | 0.305 / 0.05 = 6.1, 0.008 / 0.00487 = 1.64, 11/1024 = 0.0107. The 0.007 lies below 0.0107 because T_s = 0 counts as not positive (calibrate.py line 64) |
| 47. [minor] The 30 x 30 probe grid of Hermann et al. over [-3, 3] (R.md 108) | contradicted | arXiv 2310.16228 Section 4. The grid spans plus and minus 3 mu_i, mu_c = 1.2816. The rerun on that grid with 1,000 experiments gives 0.299, 0.055 and 0.883, against 0.305, 0.053 and 0.893 |
| 48. [major] The K option statistic w is the reading, and its power is 0.791 at lam 0.1 (R.md 113, 134, 190) | narrowed | E[w] = 0.1765 / 0.3295 = 0.536 at eps 0.2 and eta 0.1, matching 0.5360. w is not Lambda_s, and its power depends on the slip rate |
| 49. [major] Lambda_c is part of the reading (R.md 17, 217) | narrowed | No calibration of Lambda_c exists in section 4 or in the script. The project requirement (CLAUDE.md line 21) is not met for it |
| 50. [none] The calibration does not test trained learners (R.md 136) | holds | It is stated |
| 51. [none] Tier 1 setup, Figure 3B (R.md 141) | holds | arXiv 2310.16228 Figure 3B, ReLU, depth 2, width 128, predictivity 0.9, ratio 32 |
| 52. [major] Tier 1 refutation (R.md 142) | narrowed | The null band is the rule learners' own spread, so "inside" cannot fail. "Above 0.9" has no refutation. The stated condition tests power |
| 53. [minor] Tier 1, no difference means Lambda_s and reliance disagree (R.md 142) | narrowed | It can also be a failed reproduction, unless reliance is computed on the same models |
| 54. [none] Tier 2 setup of Jain et al. (R.md 145, 171) | holds | arXiv 2311.12786 Section 4 (count of b one more than a, C_Tr, C_Te, priors), Appendix C.2 (minGPT, about 3 million parameters, 6 blocks, 6 heads, 192, AdamW 10^-3, M 10^-5, S 10^-6, 10K iterations, batch 96) |
| 55. [major] Tier 2, answer count(a) + 1 on C_Te = 0, so Lambda_s above 0.9 (R.md 146) | narrowed | Jain et al. Figures 5 and 8 show low accuracy and a readable count(a), not the answer. P(count(a') + 1 = count(a) + 1) is not 1/K, so R2's scale does not apply |
| 56. [critical] Tier 2 refutation, no barrier means Conjecture 1 fails (R.md 147) | contradicted | arXiv 2211.08422 Section 5, Conjecture 1 runs from no linear connectivity to dissimilarity. Table 2 and Section 8 allow connected dissimilar models. R.md lines 40 and 89 say so. A6 is not checked for count(a) + 1 against count(b) |
| 57. [critical] Tier 2, reward equals keys within the seed spread (R4, panel B26 point 3) (R.md 146 to 147) | contradicted | panel 2446 to 2448, rho falls from -0.385 to -0.793 when the cue is stronger. Panel 2442 gives reward a p (1 - p) factor against 1 - p for keys. No derivation of equality remains |
| 58. [none] Tier 2, curvature not predicted on PCFG Counter (R.md 148) | holds | It is consistent with A7 and R7 |
| 59. [critical] Tier 3, the rule share follows m / (m + m*) under LoRA (R.md 151 to 153) | contradicted | R5 is a ratio of squared loss scores. Answers are argmax or samples. LoRA uses cross entropy, which A8 excludes. The check is the same as row 37 |
| 60. [major] Tier 3, m* from the base model's gradient inner products (R.md 151) | narrowed | The parameter set is unstated. At LoRA init B = 0, so dL/dA = 0. The fit with rest(b, q) not zero and the map from K outputs to one kappa are unstated |
| 61. [minor] Bibliographic records of the nine sources (R.md 161 to 169) | holds | arXiv first pages match titles, authors and venues. Jain et al. is 2023 here and 2024 in reading.md line 7 |
| 62. [none] Setups to reproduce (R.md 171) | holds | Hermann et al. Section 4, Jain et al. Appendix C.2, Lubana et al. 2023 Section 5, Raventos et al. Sections 3.1 to 3.3 |
| 63. [none] Jain et al. reverse fine-tuning 0.1K to 1K against 4.5K (R.md 162) | holds | arXiv 2311.12786 Figure 9 caption. The weakly relevant case takes 3K |
| 64. [none] Choice 1, powers 0.893 and 0.791 (R.md 189 to 190) | holds | calibrate_output.txt |
| 65. [critical] Choice 2, 20 seeds would detect lam = 0.02 at a spread of 0.3 (R.md 197) | contradicted | Noncentral t, sd 0.0524, n 20, alpha 0.05 one sided, gives power 0.50 |
| 66. [minor] Choice 2, the 5 seed power is not computed (R.md 195) | narrowed | It is computable, 0.57 at lam 0.05 with sd 0.0506 |
| 67. [minor] The tier 1 prediction uses the K option extension (R.md 140) | narrowed | That extension is open Choice 1b |
| 68. [minor] Proposed edit 4 against the research program sections 5 and 6 (R.md 220) | narrowed | program/research-program.md lines 96 and 102 have R and E share tier 1 ground and E feed R tier 1. The edit does not address this |
| 69. [none] Other program consistency (part answered, never answers, tiers) (R.md 27, 216 to 221) | holds | program/research-program.md lines 62 to 68 |
| 70. [minor] R5 gives the lazy prediction for tier 3 (R.md 181) | narrowed | It holds under squared loss only (row 59) |
| 71. [none] TR Result, 7 rows appended, sources opened (TR.md Result) | holds | sources/reading.md lines 25 to 31. All nine papers open and carry the cited items, except rows 26, 27 and 35 |
| 72. [minor] Run time about 3.5 minutes (calibrate.py line 14) | unchecked | The time of the rerun was not captured |
