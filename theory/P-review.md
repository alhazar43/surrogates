# Review of theory/P.md

Reviewer for unit RevP, 19 October 2026. I did not write P.md and did not edit it. Every number below was recomputed or rerun. Every source was opened as its arXiv HTML full text. The extracted texts are in C:/Users/YuanW/PLMS/surrogates/notes/units/revp_scratch/dl/ (file `<arXiv id>.txt`, cited below as `<id>.txt:line`). My scripts are in notes/units/revp_scratch/tools/ and their outputs in notes/units/revp_scratch/run/. Line numbers of the form P.md:n refer to theory/P.md as of this review.

## Critical errors

These change a prediction, a calibration number, or a claim of a source.

1. GRPO is not inside the class of D5 (row R27). P.md D5, the tier 2 prediction and proposed edit 4 put GRPO among the rewards whose fixed point keeps the runs inside each outcome (P.md:97, 179, 234). The GRPO objective, opened in Shao et al. 2024, arXiv 2402.03300, eq. 3 (2402.03300.txt:724-727), has two terms that break this. The first is the 1/|o_i| weight, which P.md itself names as a failure (P.md:97). The second is that GRPO adds a per-token KL estimate directly to the loss, not to the reward. With the gradient not passing through the prefix distribution, the stationary step rule at prefix h is

   ```
   exact per-token KL in the loss:  pi(a | h) proportional to pi_ref(a | h) exp( c q(h, a) / beta )
   k3 estimator in the loss (GRPO): pi(a | h) = beta pi_ref(a | h) / ( beta - c (q(h, a) - q(h)) )
   q(h, a) = probability of the key after prefix h and token a, under the trained policy
   ```

   The factor depends on the success probability of every prefix the run passes, so its product along a run is not a function of R alone. Right runs through prefixes of low success probability, such as a planted wrong step that is later repaired, lose mass against right runs that never pass such a prefix. Checked numerically on 200 random token trees, depth 3, 3 tokens per step, outcome-only reward, equal run lengths (tools/grpo_fast.py, converged to residual below 1e-12). At beta = 0.5 and c = 1 the exact per-token form moves the within-right distribution by total variation 0.0585 in mean and 0.121 at most, and a random run property by 0.022 in mean and 0.105 at most, against 0.0000 for the sequence-level tilt at nearly equal accuracy (0.847 against 0.859). At beta = 2 the k3 form moves them by 0.0078 and 0.022 at most. Corrected statement. D5 holds for a sequence-level KL penalty and a run-length-free loss. It does not hold for GRPO as published, and the tier 2 prediction "under GRPO ... match the tilt prediction" has no derivation. The direction of the per-token departure predicts a fall of the repair rate and a shift of step count inside right runs with no change of method, which are two of the four declared properties.

2. The planted effect of part C is half its label (row R52). P.md:163 says "a share 0.05 / 0.10 / 0.20 of right runs after training gains phi". The code sets phi to 1 on a share delta of right runs, and about half of those runs already had phi (calibration.py:220-221). The expected change of the within-right rate is delta (1 - a_1), with mean a_1 = 0.494 over the generator's items (tools/checks.py), so 0.025, 0.051 and 0.101. The power 0.512 belongs to a change of the within-right rate of about 0.025, and 0.966 to about 0.051.

3. Prystawski et al. do not prove a bias toward the marginal (row R55). P.md:169 says a direct estimate of an unseen pair "is biased toward the marginal, by their theorem for a chain". Their Theorem 3.1 gives the bias of the direct estimator as |p(y_i | y_j) - 1/|X||, a bias toward the uniform distribution (2304.03843.txt:168, 590). The pull toward the marginal is their empirical finding (2304.03843.txt:174), and they write "Our theoretical results do not explain the phenomenon observed in practice that transformers revert to the marginal distributions" (2304.03843.txt:284).

4. The tier 3 prediction misstates Boppana et al. (row R65). P.md:184 predicts that inflection words co-occur with belief shifts "per question and not within a window of steps, as in Boppana et al. 2026 Table 2 and section 6.2". Table 2 compares responses whose probe stays at 0.9 or above from the first step with all other responses (2603.05488.txt:205). It does not compare responses with and without belief shifts. Section 6.2 and appendix I report a within-window relation on MMLU for both models. For DeepSeek-R1, reconsiderations occur twice as often within 10 steps after a 20 percent probe shift (2603.05488.txt:213). For GPT-OSS, reconsiderations follow a probe shift in 59 percent of windows against 35 percent without one (2603.05488.txt:817). On GPQA-Diamond there is no trend for R1 and the reverse for GPT-OSS. The second sentence of the prediction also has no refutation condition.

5. The tier 3 refutation misstates Ye et al. (row R66). P.md:185 says "the model plans the value in one pass, as Ye et al. 2024 found for the necessary parameters in iGSM (Result 4)". Result 4 is that the model knows which parameters are necessary, the probe nece(A), by the end of the problem statement (2407.20311.txt:97, 285). It does not report that the values are computed in one pass. The model writes the values step by step in its solution. Corrected statement. A probe confident at the first token on needed items would match Ye's finding only if the probe reads which intermediates are needed, not the key.

## What Boppana et al. 2026 report, by level

The writer of P.md disagrees with the program line "Belief shifts at backtracking steps on hard items" (research-program.md:80). I opened arXiv 2603.05488v4 in full (sections 1 to 9, appendices A to I).

Whole responses. Section 6.1 labels a DeepSeek-R1 MMLU-Redux response High Confidence when the attention probe (layer 60) starts at 0.9 or above on its final answer and never falls below it. 215 of 522 responses qualify (2603.05488.txt:205). Inflections are labeled by a Gemini-2.5-Flash monitor per paragraph step.

| inflections per step, MMLU | High Confidence | other responses |
|---|---|---|
| DeepSeek-R1, reconsideration | 0.015 | 0.033 |
| DeepSeek-R1, realization | 0.004 | 0.009 |
| DeepSeek-R1, backtrack | 0.001 | 0.003 |
| DeepSeek-R1, total (Table 2) | 0.020 | 0.045 |
| GPT-OSS 120B, backtrack | 0.007 | 0.016 |
| GPT-OSS 120B, total (Table 7, 124 of 514 high) | 0.053 | 0.096 |

Sources 2603.05488.txt:182-202 and 795-815. No such table exists for GPQA-Diamond. The abstract and the introduction state that inflections "occur almost exclusively in responses where probes show large belief shifts" (2603.05488.txt:91, 101). The tables show a ratio near 2, with inflections still present in High Confidence responses, and the comparison group is "not High Confidence", not "large belief shift". The abstract claims more than its Table 2 shows.

Single steps. Section 6.2 defines a probe shift as a change of 0.2 in the top answer probability between consecutive steps and uses a window of 10 steps (2603.05488.txt:211). Results:
- DeepSeek-R1, MMLU. Reconsiderations follow probe shifts twice as often as in windows without shifts. The reverse direction shows little or no difference (2603.05488.txt:213). Appendix I puts the strongest effect at threshold 0.5 and windows of 2 to 10 steps (2603.05488.txt:822).
- DeepSeek-R1, GPQA-Diamond. No significant trend for any inflection type (Figure 5 caption, 2603.05488.txt:208).
- GPT-OSS. Section 6.2 says inflections precede probe shifts on both datasets. Appendix I says probe shifts precede reconsiderations on MMLU (59 against 35 percent) and that GPQA-Diamond shows the opposite trend (2603.05488.txt:213, 817). The two passages disagree with each other.
- All results change substantially with window size and threshold, and the authors conclude that local windowed co-occurrence does not capture the relation (2603.05488.txt:214).

So the paper does not report belief shifts at backtracking steps on hard items, at either level. Backtracking is the rarest inflection type (0.001 to 0.003 per step for R1). The response-level result exists only on MMLU, which the paper calls recall-based. The step-level result on GPQA-Diamond is null for R1 and reversed for GPT-OSS. The writer's disagreement holds. The replacement line P.md proposes (P.md:231) is narrowed. It omits the GPT-OSS replication, the positive step-level result on MMLU, and the abstract's "almost exclusively", which is the likely source of the program line. The program's second clause, "idle chains on recall items", is supported by section 5 (Table 1, forced against monitor 0.505 on MMLU-Redux against 0.010 on GPQA-Diamond for R1).

Two smaller inconsistencies inside the paper. Section 6.1 gives reconsiderations as 1.4 and 3.2 percent of steps where Table 2 gives 0.015 and 0.033. Appendix I gives an overall R1 rate of 0.038 per step.

## Major errors

- R8. A3 says a temperature other than the one of training breaks the martingale. The tower rule needs only that the path tokens and the continuations come from the same sampling kernel. The training temperature does not enter.
- R17. The D2 check confirms a different equation. P.md:62 states E sum ||o_{t+1} - o_t||^2 = 1 - sum_a o_0(a)^2, which is 2 o_0 (1 - o_0) = 0.385 for two outcomes. The script compares with o_0 (1 - o_0) = 0.1927 (calibration.py:45), the one-coordinate form. Both forms are correct with their norms, but the stated check does not check the stated equation.
- R23. "The filler control rules out hidden computation for a model trained on ordinary chains" (P.md:88) exceeds Pfau et al. They show no transfer from instance-adaptive chains to filler tokens on one task, 3SUM, with a 4-layer 34M Llama (2404.15758.txt:152, 183). A null filler result rules out computation carried by filler tokens, not computation hidden in the representations of chain tokens.
- R34. D8 and the tier 2 refutation attribute a within-outcome rise to "the judge of panel D17 reading validity" (P.md:109, 180). By D5, and by panel D17 point 1 (panel-2026-10-19.md:1621), no judge changes within-outcome rates at the KL fixed point, whatever it reads. A rise means A6 failed, through finite training, shared weights or a reward term, and P.md:209 says nothing bounds the shared-weight route. The refutation therefore cannot name the judge. Panel D33 point 2 (panel-2026-10-19.md:2752) makes the same identification, so the panel synthesis carries the same gap.
- R38 and R74. The D9 summary and edit 1 omit the positive step-level result on MMLU and the GPT-OSS replication (see the section above).
- R53. In part C, narrowing with phi drawn at random gives rejection 0.054 (P.md:164). H_freq is false in that case (D6, mean absolute change 0.035), so 0.054 is power against a true H_proc, not a level. The t-test of the mean per-item change detects only changes with a common direction across items. P.md does not say so.
- R58. The main tier 1 refutation is close to empty. b_k is the position where the chain has written v_k = a*(x) (P.md:8, 10). F_k then reads the chain's own written answer, and T at the last cut tracks the completion's accuracy minus F_0. "T not positive while the normal completion beats F_0" can occur only when the forced suffix fails to copy a written answer, which is A1. The intermediate cuts, where the information lies, do not enter T.
- R59. The tier 1 statistic "more than half of the squared movement of o within one step of a cut" is not calibrated. At S continuations each estimated adjacent increment carries sampling noise of expected size 2 (1 - sum_a o(a)^2) / S. For the base distribution of part A this is 2 x 0.585 / 30 = 0.039 per pair at S = 30, so 100 positions add about 3.9 of noise against a true total of at most 0.585. The project requires a case with no effect and a planted effect before any reading is used (CLAUDE.md, Requirements). The threshold 0.5 is marked as chosen.
- R60. Choice 1a budgets o at S = 30 on the 9 cuts only (P.md:220). With o at the cuts only, neither movement "within one step of a cut" (P.md:173) nor movement at the first tokens (P.md:175) can be located.
- R62. The side prediction from Lanham and Boppana (P.md:177) needs a capability floor. Lanham et al. find the most faithful model at 13B on 6 of 8 tasks and little gain from chains below 13B (2307.13702.txt:329). Boppana et al. attribute the weak forced answers of small distilled models to off-policy prompting (2603.05488.txt:179). A 0.5B to 1.5B learner (Choice 1a) is below that floor.
- R63. The tier 2 prediction and refutation use "the spread between seeds" (P.md:179-180). Part C calibrates a t-test over items, not a seed-spread rule. The number of seeds and the rule are not stated, so the decisive comparison has no level and no power.

## Minor errors

- R3. tau_text and tau_F are defined (P.md:16-17) and enter no prediction or calibration, while the program asks for "the point where the answer first appears" (research-program.md:81).
- R4. H_freq is undefined on items with p_0(x) = 0 or 1, where one conditional does not exist.
- R1. F_j is a softmax over an answer set. The built problems need a declared answer set when the answer is a free number, as in Lanham's addition tasks (2307.13702.txt:367).
- R6. A1 reports Boppana appendix F as "small distilled models answer badly". The appendix shows forced answering below probe accuracy for 1.5B and 7B and gives the cause with "likely" (2603.05488.txt:779-780). For 14B and 32B forced answering drops in the second half of reasoning (2603.05488.txt:781-782).
- R7. A2 cites a false positive rate of 0.78 to 1.00. The scale failure, which A2 also names, gives 0.445 to 0.997.
- R13. The SPACE upper bound of Merrill and Sabharwal is Theorem 4, not Theorems 1 and 2 (2310.07923.txt:361-362).
- R14. "A fixed-depth network can store any finite map" needs width that grows with the map.
- R19. Forking Fast resamples only alternative tokens with probability at least 0.05 (2608.19611.txt:78), a second departure from the on-policy outcome distribution beside the greedy path.
- R24. reading.md:41 says the Pfau section 4.3 claim was taken from the introduction, while P.md cites section 4.3 as source. Section 4.3 supports the claim (2404.15758.txt:180-183).
- R42. Part B uses only cuts 0 and k (calibration.py:112-121). The step count k = 8 and capacity c = 2 do not enter the statistic.
- R43. Part B calls the planted learner one that "uses its tape", while T measures dependence of F on the prefix (D3).
- R49. Forking Fast draws S continuations per alternative token, so its per-position sample exceeds S. The simulation's single multinomial of size S understates their sample.
- R50. The pooled-window chi-square is not "the test form of their segment pooling". Their estimator is PELT with an exact multinomial cost and kernel pooling within segments (2608.19611.txt:92, 241-245).
- R57. The Prystawski reasoning gap on the transformer is measured, depends on training amount and data structure, and vanishes with fully observed data (2304.03843.txt:241). "Known by construction" holds for the data, not for the trained network's need.
- R61. "Followed more often than at w = 0" (P.md:173) names a parameter of the reference learner. A real model has no w. The baseline follow rate is undefined.
- R78. The docstring says the script runs in seconds (calibration.py:12). The result file records more than 10 minutes.

## Full table

| item | line | outcome | evidence | severity |
|---|---|---|---|---|
| R1 F_j over an answer set | P.md:13 | narrowed | matches Boppana 3.3, softmax over four letters (2603.05488.txt:133); free numeric answers need a declared set (2307.13702.txt:367) | minor |
| R2 o_j definition | P.md:15 | holds | definition | none |
| R3 tau_text, tau_F unused | P.md:16-17 | narrowed | no later use in P.md; research-program.md:81 asks for it | minor |
| R4 H_freq, H_proc | P.md:25-27 | narrowed | conditional undefined at p_0(x) in {0, 1} | minor |
| R5 rate_1 formula | P.md:34 | holds | total probability under H_freq; panel-2026-10-19.md:2745-2748 | none |
| R6 A1 and Boppana appendix F | P.md:45 | narrowed | 2603.05488.txt:779-782 | minor |
| R7 A2 false positive range | P.md:46 | narrowed | results/theory_p_calibration.txt, B prob rows 0.445 to 0.997 | minor |
| R8 A3 training temperature | P.md:47 | contradicted | tower rule needs one kernel for path and continuations; calibration.py:30-46 uses no temperature | major |
| R9 A4 multinomial counts | P.md:48 | holds | 2608.19611.txt:86-87 | none |
| R10 A6 statement | P.md:50 | holds | stated as an assumption; GRPO fails it (R27) | none |
| R11 A8 per-item rates | P.md:52 | holds | D7 numbers (R33) | none |
| R12 D1 complexity results | P.md:56 | holds | 2310.07923.txt:70, 98-102, 116, 123, 241-247 | none |
| R13 D1 theorem numbers | P.md:56 | narrowed | upper bound is Theorem 4, 2310.07923.txt:361-362 | minor |
| R14 D1 finite map at fixed depth | P.md:56 | narrowed | needs width growing with the map | minor |
| R15 D2 martingale | P.md:61 | holds | tower rule, re-derived | none |
| R16 D2 total squared movement | P.md:62 | holds | orthogonal increments and o_T = e_A, re-derived | none |
| R17 D2 check against L62 | P.md:65 | contradicted | calibration.py:45 uses o0(1-o0) = 0.19273; L62 gives 0.38546 (tools/checks.py) | major |
| R18 D2 numbers 0.739, -0.0003, 0.261 | P.md:65 | holds | rerun identical; P(Bin(12, 0.55) >= 6) = 0.73931 | none |
| R19 D2 Forking Fast o_t | P.md:69 | narrowed | 2608.19611.txt:78, truncation at p >= 0.05 | minor |
| R20 D3 necessity, dependence, use | P.md:72-75 | holds | 2307.13702.txt:171, 244-256 | none |
| R21 D3 recompute in text, panel D20 | P.md:75 | holds | panel-2026-10-19.md:1756 | none |
| R22 D4 filler 2.5, paraphrase 2.6 | P.md:83-84 | holds | 2307.13702.txt:30-33, 288, 311 | none |
| R23 D4 filler rules out hidden computation | P.md:88 | narrowed | 2404.15758.txt:152, 183 | major |
| R24 Pfau rung | P.md:88; reading.md:41 | narrowed | 2404.15758.txt:180-183 supports the claim; rung stated above what was opened | minor |
| R25 D5 tilt keeps runs inside outcome | P.md:92-95 | holds | factor depends on R only, re-derived | none |
| R26 D5 fixed advantage c(x) = 1/sd | P.md:97 | holds | holds for a sequence-level reverse KL with advantage fixed | none |
| R27 D5, tier 2, edit 4, GRPO included | P.md:97, 179, 234 | contradicted | 2402.03300.txt:724-727 (weight 1 over the length of o_i, and per-token KL in the loss); tools/grpo_fast.py within-right TVD 0.0585 mean, 0.121 max | critical |
| R28 D5 length weight recalled | P.md:97 | holds | 2402.03300.txt:724 | none |
| R29 D5 exact tilt change at most 1e-4 | P.md:97 | holds | exact_tilt_max_abs_change 0.0000 | none |
| R30 D6 logit flow | P.md:99 | holds | dJ/dl_y = pi_y (R_y - p), re-derived | none |
| R31 D6 numbers 0.069, 0.065, 0.000, 0.035 | P.md:99 | holds | rerun 0.0687, 0.0648, 0.0000, 0.0354 | none |
| R32 D6 no mean direction | P.md:99 | holds | not printed by the script; tools/checks.py signed mean -0.0014, t = -1.10 over 2000 items | none |
| R33 D7 0.32 to 0.44 | P.md:101 | holds | (0.16+0.16)/1.0 and (0.48+0.18)/1.5 | none |
| R34 D8 and tier 2 refutation name the D17 judge | P.md:109, 180 | contradicted | D5; panel-2026-10-19.md:1621, 2752; P.md:209 | major |
| R35 D9 Lanham numbers | P.md:114 | holds | 2307.13702.txt:121, 159, Table 2 at 166-232, 293, 318, 328 | none |
| R36 D9 performativity 0.505, 0.010 | P.md:115 | holds | 2603.05488.txt:147-166 | none |
| R37 D9 inflections 0.020, 0.045 | P.md:115 | holds | 2603.05488.txt:182-202 | none |
| R38 D9 no reliable local co-occurrence | P.md:115 | narrowed | 2603.05488.txt:213, 817, 822 | major |
| R39 D9 linear probe near chance, 88 percent | P.md:115 | holds | 2603.05488.txt:724-740 (31.85 against 87.98) | none |
| R40 D9 Forking Fast numbers | P.md:116 | holds | 2608.19611.txt:106, 118, 217-218, 230, 256 | none |
| R41 B equations against code | P.md:124-129 | holds | calibration.py:100-126 | none |
| R42 B k and c unused | P.md:122 | narrowed | calibration.py:112-121 | minor |
| R43 B planted learner called use | P.md:132 | narrowed | T is a dependence statistic, P.md:73 | minor |
| R44 B table | P.md:134-139 | holds | rerun identical; header rises mix n rows (0.120 at n = 100, w = 0.2) | none |
| R45 B power 0.77 at 100 items | P.md:141 | holds | 0.769 | none |
| R46 B probability scale 0.445 to 0.997 | P.md:142 | holds | result file B prob rows | none |
| R47 B unequal drift 0.778 to 1.000 | P.md:143 | holds | result file B rows drift (1.0, 0.5) | none |
| R48 A table | P.md:150-154 | holds | rerun identical | none |
| R49 A raw rule of appendix D | P.md:156 | narrowed | 2608.19611.txt:78, 228 | minor |
| R50 A test form of segment pooling | P.md:156 | narrowed | 2608.19611.txt:92, 241-245 | minor |
| R51 C table | P.md:160-165 | holds | rerun identical | none |
| R52 C planted share label | P.md:163 | contradicted | calibration.py:220-221; tools/checks.py effective 0.025, 0.051, 0.101 | critical |
| R53 C narrowing with random phi, 0.054 | P.md:164 | narrowed | H_freq false there, P.md:99 | major |
| R54 C reading of level and power | P.md:167 | holds | 0.042 at the tilt; power subject to R52 | none |
| R55 Prystawski bias toward the marginal by theorem | P.md:169 | contradicted | 2304.03843.txt:168, 174, 284, 590 | critical |
| R56 Prystawski 32M, 10 layers, 4.5M | P.md:169 | holds | 2304.03843.txt:736, 767 | none |
| R57 Prystawski need known by construction | P.md:169 | narrowed | 2304.03843.txt:241 | minor |
| R58 tier 1 main refutation | P.md:173-174 | narrowed | P.md:8, 10; F_k reads the written key | major |
| R59 tier 1 share of squared movement | P.md:173, 175 | narrowed | noise 2(1 - sum o^2)/S = 0.039 per pair at S = 30; no calibration | major |
| R60 Choice 1a o only at cuts | P.md:220 against 173, 175 | contradicted | 9 cuts cannot locate movement near cuts or at t = 1 | major |
| R61 follow rate against w = 0 | P.md:173 | narrowed | w exists only in the reference learner | minor |
| R62 side prediction, smaller model larger T | P.md:177 | narrowed | 2307.13702.txt:329; 2603.05488.txt:179 | major |
| R63 tier 2 seed spread | P.md:179-180 | narrowed | part C calibrates a t-test over items, P.md:158 | major |
| R64 tier 2 positive control SFT | P.md:182 | holds | cross entropy fixed point at the teacher's runs, panel-2026-10-19.md D18 | none |
| R65 tier 3 not within a window, as in Boppana | P.md:184 | contradicted | 2603.05488.txt:205, 213, 817 | critical |
| R66 tier 3 refutation via Ye Result 4 | P.md:185 | contradicted | 2407.20311.txt:97, 285 | critical |
| R67 tier 3 DeepSeek route | P.md:186 | holds | 2603.05488.txt:567, 583-592 | none |
| R68 section 6 ids, titles, authors, years | P.md:194-200 | holds | arXiv abstract titles of all seven ids; author lists 2608.19611.txt:46-52, 2603.05488.txt:71-87 | none |
| R69 Boppana forced answer setup | P.md:195 | holds | 2603.05488.txt:583-587 | none |
| R70 Ye pretraining hardware | P.md:200, 222 | holds | 2407.20311.txt:831 | none |
| R71 Prystawski cost 20 h, 300,000 steps | P.md:221 | holds | 2304.03843.txt:206, 738 | none |
| R72 Choice 1a counts 3,600 and 108,000 | P.md:220 | holds | 200 x 2 x 9 = 3,600; x 30 = 108,000 | none |
| R73 Choice 2 monitor needs a second model | P.md:226 | holds | 2603.05488.txt:140 | none |
| R74 edit 1 replacement line | P.md:231 | narrowed | 2603.05488.txt:91, 101, 213, 795-817 | major |
| R75 edit 1, no backtracking shifts on hard items | P.md:231 | holds | 2603.05488.txt:197-199, 208, 213 | none |
| R76 section 7 limits | P.md:206-212 | holds | consistent with D1, D3, D6 | none |
| R77 script reproduces the result file | calibration.py | holds | python -I rerun, run/rerun.txt identical to the result file except line endings | none |
| R78 docstring runtime | calibration.py:12 | contradicted | result file header, above 10 minutes | minor |

Counts. 78 rows. Critical 5 (R27, R52, R55, R65, R66). Major 12 (R8, R17, R23, R34, R38, R53, R58, R59, R60, R62, R63, R74). Minor 16. Rows that hold with no error 45. Unchecked 0.

What this review does not check. The proofs of Merrill and Sabharwal and of Prystawski et al. were not re-derived; only their statements were read. The per-token GRPO result is a tabular tree computation, not a network, and it ignores clipping and the old-policy ratio. No source outside the seven cited and DeepSeekMath was opened.
