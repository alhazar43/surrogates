# Review of theory/R.md

Unit RevR, 19 October 2026. The reviewer did not write theory/R.md and did not edit it. Line numbers refer to C:/Users/YuanW/PLMS/surrogates/theory/R.md as read on 19 October 2026. Sources were opened as arXiv PDFs and read through pdftotext. The panel is C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md, cited as panel with its line.

## How the checks were run

- The script calibrate.py was rerun with python -I. Its output is identical to calibrate_output.txt, line for line (diff empty, seed 20261019).
- Three small checks were run on synthetic records only, in a scratch folder outside the project.
  1. The argmax answer of the R5 kernel learner on a conflict cell, at kappa_s / kappa_c = 4, kappa_cs = 0.5.
  2. Cross entropy gradient descent on the same fixed features as Part C, 200 and 5,000 steps, learning rate 0.5.
  3. Part A of calibrate.py on the probe grid that Hermann et al. actually use, plus the exact E[T_s | delta] of a written null learner.
- Powers for other seed counts use the noncentral t distribution with the seed spread of calibrate_output.txt. The formula reproduces the simulated powers 0.301 (10 seeds, lam 0.02) and 0.893 (10 seeds, lam 0.05) as 0.299 and 0.891.

## Critical errors

Five rows change a prediction, a calibration number or a claim of a source. Rows 37 and 59 share one root.

1. Row 37, line 87. "A rise sharper than m / (m + m*), or a reversal of the effect of repetitions, marks a learner whose features moved." This is false. R5 gives the ratio of two squared loss scores, not a share of answers. The same kernel learner answers by argmax, and its answers switch from cue to rule in one step at m = m*. With fixed features and cross entropy, the rule share on the conflict cell is 0.94 at m = 16 after 200 steps and 0.99 after 5,000 steps, against 0.80 from the formula. At m = 8 it is 0.83 after 200 steps and 0.93 after 5,000 steps, so more training changes it. A sharper rise and an effect of repetitions therefore occur with no feature movement. The midpoint m* stays correct in this symmetric design.
2. Row 59, lines 151 to 153. The tier 3 prediction says the share of conflict cells answered by the rule follows m / (m + m*) under LoRA. LoRA fine-tuning uses cross entropy, which A8 (line 42) excludes. Answers are argmax or samples, not squared loss scores. The prediction, its refutation and the reading "features moved" all rest on row 37. A corrected statement says this. Under squared loss and the minimum norm solution, the ratio of the scores of r(c) and r(c') is m / (m + m*), and the argmax answer changes at m = m*. No closed form for cross entropy training is derived.
3. Row 56, line 147. "No barrier while Lambda_s differs. Then Conjecture 1 fails in this setting." Conjecture 1 of Lubana et al. 2023 says that two minimizers that are not linearly connected are mechanistically dissimilar (Section 5). A linear connection between dissimilar models does not contradict it. Lubana et al. say such connections are possible (Table 2, Section 8). R.md says the same at line 89 and at line 40. The refutation condition contradicts the file's own R6. A6's condition, a cue of lower complexity than the rule, is also not checked for count(a) + 1 against count(b).
4. Row 57, line 146. "Fine-tuning by a correctness reward on C_Tr = 1 data gives the same Lambda_s as fine-tuning on the keys, ... since an outcome signal on agree cells gives no push between cue and rule (R4, panel B26 point 3)." Panel B26 point 3 says the opposite for the general case (panel 2446 to 2449). On agree data, outcome signals keep rho at its start only when cue and rule have equal strength. When the cue is stronger, rho falls from -0.385 to -0.793. Panel B26 point 1 gives reward a factor p (1 - p) against 1 - p for keys (panel 2442). So the two routes weight the training examples differently, and nothing derived makes them reach the same solution. The equality is a conjecture, and its stated support is misreported.
5. Row 65, line 197. "20 seeds ... would detect lam = 0.02 under a seed spread of 0.3." The power of the one sided seed level t test at 20 seeds, lam 0.02 and T_s standard deviation 0.0524 is 0.50. That is not detection by the 0.8 standard of line 142.

## Major errors

- Row 8, lines 16, 38, 101, 134. The estimand changes between sections. Section 1 defines Lambda_s per learner. A written null learner with delta = 0.3 has E[T_s | delta] = 0.0500, the size of the planted lam = 0.05. By Section 1 this learner follows the surface. The 0.305 is a false positive rate only for the mean of Lambda_s over seeds, an estimand the file uses and never states.
- Row 11, lines 36, 109. The K option item level test holds because the surface dependence of Part B is drawn independently of the cue map (calibrate.py line 96). A trained learner meets this only if h is randomized independently of every surface feature the learner reads. A2 says only that h is fixed before training. This is the stated advantage of Choice 1b.
- Row 31, lines 83, 103. "No sharp transition occurs" holds for the score ratio and fails for answers (row 37).
- Row 48, lines 113, 134, 190. The K option statistic w is not an estimate of Lambda_s. Its expectation depends on the rule's slip rate and the cue blind noise. At lam 0.1, eps 0.2 and eta 0.1, E[w] = 0.1765 / 0.3295 = 0.536, which matches the simulated 0.5360. The power 0.791 holds only at these values of the nuisance parameters, and the file does not say so.
- Row 49, lines 17, 217. Lambda_c is never calibrated. Proposed edit 1 makes Lambda_c part of the reading. The project requirement says a reading is calibrated before use.
- Row 52, line 142. The tier 1 refutation does not refute the prediction. "Inside the null band" cannot fail if the band is the rule learners' own seed spread. "Cue learners above 0.9" has no refutation condition. The listed condition tests the power of the reading.
- Row 55, line 146. Jain et al. report low accuracy on C_Te = 0 strings and a probe that still reads count(a) (Figures 5 and 8). They do not report that the answer equals count(a) + 1. The threshold 0.9 also assumes P(h(s') = h(s)) = 1/K, which a count offset does not satisfy, so 0.9 is not on the scale of R2.
- Row 60, line 151. The way to compute m* from the base model is not stated. At LoRA initialization B = 0, so the gradient with respect to A is zero. The fit of kappa_s and kappa_c is also unstated when rest(b, q) of panel B19 is not zero, and so is the map from a kernel with K outputs to one kappa.

## Minor errors

- Row 1, line 23. The rule learner has Lambda_c = 1 only if r(c') is uniform over the K options. This assumption is not stated.
- Row 2, lines 54 to 58. Lambda_s = lam needs P(h(s') = h(s)) = 1/K for every s. That is a balance of h over all of S, not the per structure balance of A2.
- Row 3, line 53. An A_s invariant learner has Lambda_s = 0 exactly, on any grid. Balance is needed by the tests, not by Lambda_s.
- Row 6, line 25. S randomized accuracy is accuracy against the label. Lambda_s is the excess rate of the cue's answer. They agree at the two pure learners only, so "the same thing" is too strong.
- Row 7, lines 25, 163. Hermann et al. define reliance in Section 3, with a prefactor. Lambda_s equals the first term only up to that constant.
- Row 10, line 36. Under balance, the K option binomial null is conservative (0.030 at alpha 0.05), not exact.
- Row 12, line 37. A3 fails for the binary design. The probe grid lies inside the support of the Gaussian training latents.
- Row 13, line 38. The rate 0.305 depends on sigma_delta = 0.3, a chosen value that was not measured.
- Row 15, line 40. Theorem 1 of Lubana et al. uses Lemma 3, which they restate from Shah et al. without proof.
- Row 16, line 41. A7 states the heuristic of R7 as a fact.
- Row 20, line 47. "A reading uses only the conflict cells" does not match T_s, which averages all 900 probes, agree probes included.
- Row 21, line 60. The sum of y_hat sign(z_s) is zero in expectation, not in each sample.
- Row 24, line 62. "The data give no push" conflicts with Hermann et al., where availability, a property of the data, decides.
- Row 26, line 63. Shah et al.'s Table 1 is the decision template. The measured AUCs are in Section 4, Figure 3(b) and Table 4.
- Row 27, line 64. Lubana et al.'s Lemma 3 has no proof in Appendix F.
- Row 32, line 84. Repetitions leave the score ratio unchanged under squared loss only. Under cross entropy more steps raise the rule share.
- Row 35, line 87. Raventos et al. find that width has no effect on the threshold of the base model. Width and depth raise it only for the small model.
- Row 41, line 103. A deterministic kernel learner at m = m* gives a tie, not a mix.
- Row 46, line 134. The sign test rejects at 0.007, below 0.0107, because T_s = 0 counts as not positive (calibrate.py line 64).
- Row 47, line 108. Hermann et al.'s probe grid spans plus and minus 3 mu, with mu_c = 1.2816, not [-3, 3]. On the true grid the rates are 0.299, 0.055 and 0.883 against 0.305, 0.053 and 0.893, so no conclusion changes.
- Row 53, line 142. "Do not differ" can also mean a failed reproduction.
- Row 61, line 162. Jain et al. is dated 2023 here and 2024 in reading.md line 7.
- Row 66, line 195. The 5 seed power is computable, 0.57 at lam 0.05.
- Row 67, line 140. The tier 1 prediction depends on an open author choice.
- Row 68, line 220. Proposed edit 4 does not address the program's shared ground with E.
- Row 70, line 181. "Lazy prediction" holds under squared loss only.
- Row 72, line 14 of calibrate.py. The run time is unchecked.

## Full table

| # | item | line | outcome | evidence | severity |
|---|---|---|---|---|---|
| 1 | Reference values (1,0), (0,1), (0,0) of rule, cue and constant learners | 23 | narrowed | Rederived. Lambda_c(rule) = (1 - P_c'(r(c') = r(c))) / (1 - 1/K), which is 1 only if r(c') is uniform over K options. Lambda_s(cue) = 1 needs the same of h. Neither balance is stated | minor |
| 2 | Mixture learner has Lambda_s = lam exactly | 23, 54 to 58 | narrowed | Rederived. The difference of terms is lam (1 - P_s'(h(s') = h(s))). This equals lam (1 - 1/K) only if h(s') is uniform for every s. A2 states a different balance, over wrong options within a structure | minor |
| 3 | E Lambda_s = 0 on a balanced grid for an A_s invariant learner | 52 to 53, 58 | narrowed | Both terms equal q_{c,n}(h(s)), so Lambda_s = 0 exactly on any grid. Balance enters only the tests | minor |
| 4 | Lambda_s = 0 for every A_s invariant learner, and loss and output forms agree on pure learners | 25 | holds | Rederived. A pure cue learner changes its loss under A_s and a pure rule learner does not | none |
| 5 | Lubana et al. 2023, mechanism as invariance set (Definitions 2 to 4) and invariance on the loss (Definition 3) | 25 | holds | arXiv 2211.08422 Section 3. Definition 3 says counterfactuals "do not increase its loss". Definition 4 says similar if I(theta_1) = I(theta_2) | none |
| 6 | Shah et al. measure the same thing as S randomized accuracy | 25 | narrowed | arXiv 2006.07710 Definition 1 is E[1{f(x) = y}] under S randomization, accuracy against the label. Lambda_s is the excess rate of the cue's answer. They agree at the extremes only | minor |
| 7 | Lambda_s is the first term of Hermann et al.'s reliance | 25, 163 | narrowed | arXiv 2310.16228 Section 3. Reliance is a prefactor times (sum of y_hat sign(z_s) minus sum of y_hat sign(z_c)). It is defined in Section 3, not Section 4. It holds up to the constant | minor |
| 8 | The item level test's 0.305 is a false positive rate with no cue use | 38, 101, 134 | narrowed | Exact E[T_s given delta] on the grid of calibrate.py is 0.0250, 0.0500 and 0.1000 at delta 0.15, 0.3 and 0.6. Lambda_s(f) of Section 1 is a per learner quantity, so these learners follow the surface. The null holds only for E over seeds of Lambda_s, which is not stated | major |
| 9 | A1, the gauge and the third feature (panel B19, last paragraph) | 35 | holds | panel 1735 | none |
| 10 | A2, the null of the tier 1 test is exact only under balance | 36 | narrowed | calibrate_output.txt Part B, the item rejection is 0.030 at alpha 0.05. Dirichlet(0.3) errors make the hit count Poisson binomial, which is conservative. The sign test of Part A is exact | minor |
| 11 | A2, h fixed by the design before training, is enough for the K option item test | 36, 109 | narrowed | calibrate.py line 96 draws u independently of h by construction. A trained learner's surface dependence is independent of h only if h is randomized independently of the surface features. Choice 1b rests on this | major |
| 12 | A3, test cells new (panel S7 point 8) | 37 | narrowed | panel 2147 holds. A3 fails for the binary design of Choice 1a, since the probe grid lies inside the Gaussian training support (Hermann et al. Section 3) | minor |
| 13 | A4, rejection 0.305 with no cue use | 38 | narrowed | It depends on sigma_delta = 0.3, a chosen value that was not measured, and on the estimand of row 8 | minor |
| 14 | A5, held and unused (panel 7A2 point 1) | 39 | holds | panel 2226 | none |
| 15 | A6, Theorem 1 conditions, and the linear system example in Section 8 | 40 | holds | arXiv 2211.08422 Appendix F.2 (1 hidden layer ReLU, interpolating minimizers), Lemma 3 (perfectly predictive), Section 8 ("any two solutions of the linear system ... can be interpolated"). The proof uses Lemma 3, which is not proved there | minor |
| 16 | A7, a single simple cue sits in high curvature | 41 | narrowed | This is the heuristic of R7 stated as a fact, with no rung mark | minor |
| 17 | A8, the kernel regime with squared loss and minimum norm | 42 | holds | It is consistent with R5. Tier 3 violates it (row 59) | none |
| 18 | A9, independence for the item test | 43 | holds | It is stated as an assumption | none |
| 19 | R1, practiced cells identify nothing, and the Geirhos et al. quote in Section 3 | 47 | holds | arXiv 2004.07780 Section 3, "As long as tests are performed only on i.i.d. data, it is impossible to distinguish between these" | none |
| 20 | R1, a reading uses only the conflict cells | 47 | narrowed | T_s averages all 900 probes, agree probes included (calibrate.py line 55). There is no bias, only added variance | minor |
| 21 | R3, the sum of y_hat sign(z_s) over the grid is zero for each z_c | 60 | narrowed | It is zero in expectation | minor |
| 22 | R3, null 1/(K - 1) and log likelihood ratio log(0.9 / 0.033) = 3.3 | 60 | holds | ln 27 = 3.296. Panel C15 (panel 1834) computes it between two rules. The same arithmetic holds for a pure cue learner against a pure rule learner on an answer equal to h(s) | none |
| 23 | R3, unit of replication in the two designs | 60 | holds | It is consistent with calibrate.py | none |
| 24 | R4, the data give no push on agree cells and only the learner's bias decides (panel C16) | 62 | narrowed | The loss does not separate the two solutions. Availability, an amplification in the data, decides in Hermann et al. (line 65 of R.md), and panel B26 point 3 shows a push toward the stronger feature | minor |
| 25 | R4, Shah et al. theorem on LSN | 63 | holds | arXiv 2006.07710 Section 4, Theorem 1. A one hidden layer ReLU network trained by mini batch gradient descent relies only on the linear coordinate | none |
| 26 | R4, LMS-5 randomized AUC 0.50 and 1.00 in Table 1 | 63 | narrowed | Table 1 is the decision template. The measured values are in Section 4, Figure 3(b) and Table 4 | minor |
| 27 | R4, Lubana et al. Lemma 3, proved in their simplified setting | 64 | narrowed | arXiv 2211.08422 Appendix F. Lemma 3 is introduced as a rephrasing of Shah et al. and has no proof | minor |
| 28 | R4, Hermann et al. predictivity, availability, Theorems 3 and 4, Figure 3A | 65 | holds | arXiv 2310.16228 Sections 3, 4 and 6. Figure 3A uses ratio 64 and predictivity 0.85. Theorem 4 concerns a two layer ReLU network | none |
| 29 | R4, Park et al. Definition 3 and Section 4.1, and the link of concept signal to amplification | 66 | holds | arXiv 2406.19370 Definition 3. With Hermann's e_i = alpha_i w_i z_i and a unit w_i, the norm of dx/dz_i is alpha_i | none |
| 30 | R5 closed form scores and eigenspace | 72 to 82 | holds | Rederived. G = (kappa_s + kappa_cs) I + kappa_c B, and Y lies in the block eigenspace. Part C rerun is identical to four decimals | none |
| 31 | R5 consequence 1, logistic in log m, no sharp transition, midpoint the availability ratio | 83 | narrowed | It holds for the score ratio. Argmax answers give cue at m = 3 (0.400 against 0.533) and rule at m = 5 (0.526 against 0.421), a step at m* = 4. "Availability ratio in kernel units" is an idea marked as a derivation | major |
| 32 | R5 consequence 2, repetitions leave the rule fraction unchanged | 84 | narrowed | It holds for squared loss, with or without ridge. Under cross entropy on fixed features, the share at m = 8 is 0.83 after 200 steps and 0.93 after 5,000 steps | minor |
| 33 | R5 consequence 3, any kappa_c > 0 raises new surface cells | 85 | holds | Rederived, the score is m kappa_c / lambda | none |
| 34 | Raventos et al. D = 8, K = 16, noise 0.25, threshold between 2^14 and 2^15, reversal | 87 | holds | arXiv 2306.15063 Section 3.1, Figure 2 | none |
| 35 | Raventos et al. weight decay lowers the threshold, layers and width raise it | 87 | narrowed | Section 3.3, Figure 7. Width has no effect on the base model. Width and depth raise the threshold for the small model only | minor |
| 36 | Lubana et al. 2024 p_c about 1/sqrt(abs(E) abs(K)), sqrt scaling, unexplained abs(K)^(3/2) | 87 | holds | arXiv 2408.12578 Section 7.2, Figure 7 and the paragraph after it | none |
| 37 | R5 conclusion, a sharper rise or a reversal of repetitions marks moved features | 87 | contradicted | Argmax gives a step at m*. Cross entropy on fixed features gives a rule share of 0.94 (200 steps) and 0.99 (5,000 steps) at m = 16, against 0.80. More steps change the share | critical |
| 38 | R6, Lubana et al. Definition 1, Conjecture 1, Figures 4 and 5, Table 2 | 89 | holds | arXiv 2211.08422 Sections 2 and 5, Figure 5 caption, Table 2 caption | none |
| 39 | R7, curvature ratio about m | 91 | holds | The arithmetic is right. It is marked as a heuristic | none |
| 40 | R7, Merullo et al. numbers | 91 | holds | arXiv 2510.24256 Section 3.3 (OLMo-2 7B), Section 4 (23.1 percent), Section 6 (95 to 106, 66 to 74, 3 to 16 percent) | none |
| 41 | R8, kernel learner at m = m* mixes both at 0.5 | 103 | contradicted | A deterministic learner has tied scores at m*. It does not mix | minor |
| 42 | R8, other rows | 97 to 102 | holds | They are consistent with R1, A5, A6, R7 and A1 | none |
| 43 | Section 4 design text against code | 107 to 114 | holds | calibrate.py lines 31 to 37, 77 to 103, 47 to 64 | none |
| 44 | Section 4 table, 11 rows | 118 to 130 | holds | The rerun is identical to calibrate_output.txt | none |
| 45 | Spread of the estimates | 132 | holds | calibrate_output.txt rows 0.3 0.00, 0.0 0.00 and Part B 0.00 | none |
| 46 | Six times alpha, 1.6 standard errors, largest sign test size 0.011 | 134 | holds | 0.305 / 0.05 = 6.1, 0.008 / 0.00487 = 1.64, 11/1024 = 0.0107. The 0.007 lies below 0.0107 because T_s = 0 counts as not positive (calibrate.py line 64) | minor |
| 47 | The 30 x 30 probe grid of Hermann et al. over [-3, 3] | 108 | contradicted | arXiv 2310.16228 Section 4. The grid spans plus and minus 3 mu_i, mu_c = 1.2816. The rerun on that grid with 1,000 experiments gives 0.299, 0.055 and 0.883, against 0.305, 0.053 and 0.893 | minor |
| 48 | The K option statistic w is the reading, and its power is 0.791 at lam 0.1 | 113, 134, 190 | narrowed | E[w] = 0.1765 / 0.3295 = 0.536 at eps 0.2 and eta 0.1, matching 0.5360. w is not Lambda_s, and its power depends on the slip rate | major |
| 49 | Lambda_c is part of the reading | 17, 217 | narrowed | No calibration of Lambda_c exists in section 4 or in the script. The project requirement (CLAUDE.md line 21) is not met for it | major |
| 50 | The calibration does not test trained learners | 136 | holds | It is stated | none |
| 51 | Tier 1 setup, Figure 3B | 141 | holds | arXiv 2310.16228 Figure 3B, ReLU, depth 2, width 128, predictivity 0.9, ratio 32 | none |
| 52 | Tier 1 refutation | 142 | narrowed | The null band is the rule learners' own spread, so "inside" cannot fail. "Above 0.9" has no refutation. The stated condition tests power | major |
| 53 | Tier 1, no difference means Lambda_s and reliance disagree | 142 | narrowed | It can also be a failed reproduction, unless reliance is computed on the same models | minor |
| 54 | Tier 2 setup of Jain et al. | 145, 171 | holds | arXiv 2311.12786 Section 4 (count of b one more than a, C_Tr, C_Te, priors), Appendix C.2 (minGPT, about 3 million parameters, 6 blocks, 6 heads, 192, AdamW 10^-3, M 10^-5, S 10^-6, 10K iterations, batch 96) | none |
| 55 | Tier 2, answer count(a) + 1 on C_Te = 0, so Lambda_s above 0.9 | 146 | narrowed | Jain et al. Figures 5 and 8 show low accuracy and a readable count(a), not the answer. P(count(a') + 1 = count(a) + 1) is not 1/K, so R2's scale does not apply | major |
| 56 | Tier 2 refutation, no barrier means Conjecture 1 fails | 147 | contradicted | arXiv 2211.08422 Section 5, Conjecture 1 runs from no linear connectivity to dissimilarity. Table 2 and Section 8 allow connected dissimilar models. R.md lines 40 and 89 say so. A6 is not checked for count(a) + 1 against count(b) | critical |
| 57 | Tier 2, reward equals keys within the seed spread (R4, panel B26 point 3) | 146 to 147 | contradicted | panel 2446 to 2448, rho falls from -0.385 to -0.793 when the cue is stronger. Panel 2442 gives reward a p (1 - p) factor against 1 - p for keys. No derivation of equality remains | critical |
| 58 | Tier 2, curvature not predicted on PCFG Counter | 148 | holds | It is consistent with A7 and R7 | none |
| 59 | Tier 3, the rule share follows m / (m + m*) under LoRA | 151 to 153 | contradicted | R5 is a ratio of squared loss scores. Answers are argmax or samples. LoRA uses cross entropy, which A8 excludes. The check is the same as row 37 | critical |
| 60 | Tier 3, m* from the base model's gradient inner products | 151 | narrowed | The parameter set is unstated. At LoRA init B = 0, so dL/dA = 0. The fit with rest(b, q) not zero and the map from K outputs to one kappa are unstated | major |
| 61 | Bibliographic records of the nine sources | 161 to 169 | holds | arXiv first pages match titles, authors and venues. Jain et al. is 2023 here and 2024 in reading.md line 7 | minor |
| 62 | Setups to reproduce | 171 | holds | Hermann et al. Section 4, Jain et al. Appendix C.2, Lubana et al. 2023 Section 5, Raventos et al. Sections 3.1 to 3.3 | none |
| 63 | Jain et al. reverse fine-tuning 0.1K to 1K against 4.5K | 162 | holds | arXiv 2311.12786 Figure 9 caption. The weakly relevant case takes 3K | none |
| 64 | Choice 1, powers 0.893 and 0.791 | 189 to 190 | holds | calibrate_output.txt | none |
| 65 | Choice 2, 20 seeds would detect lam = 0.02 at a spread of 0.3 | 197 | contradicted | Noncentral t, sd 0.0524, n 20, alpha 0.05 one sided, gives power 0.50 | critical |
| 66 | Choice 2, the 5 seed power is not computed | 195 | narrowed | It is computable, 0.57 at lam 0.05 with sd 0.0506 | minor |
| 67 | The tier 1 prediction uses the K option extension | 140 | narrowed | That extension is open Choice 1b | minor |
| 68 | Proposed edit 4 against the research program sections 5 and 6 | 220 | narrowed | program/research-program.md lines 96 and 102 have R and E share tier 1 ground and E feed R tier 1. The edit does not address this | minor |
| 69 | Other program consistency (part answered, never answers, tiers) | 27, 216 to 221 | holds | program/research-program.md lines 62 to 68 | none |
| 70 | R5 gives the lazy prediction for tier 3 | 181 | narrowed | It holds under squared loss only (row 59) | minor |
| 71 | TR Result, 7 rows appended, sources opened | TR.md Result | holds | sources/reading.md lines 25 to 31. All nine papers open and carry the cited items, except rows 26, 27 and 35 | none |
| 72 | Run time about 3.5 minutes | calibrate.py line 14 | unchecked | The time of the rerun was not captured | minor |
