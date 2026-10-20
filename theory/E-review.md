# Review of theory/E.md

Unit RevE, 19 October 2026. Reviewer did not write theory/E.md and did not edit it. Line numbers refer to theory/E.md as read on this date (244 lines). Panel sections refer to C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md.

What was done.
- Every derivation was derived again by hand. The revival claim of E10 was integrated numerically (notes/units/reve_tools/checks.py).
- calibrate.py was run again with python -I. The output is identical to calibrate_output.txt line by line (notes/units/RevE-rerun.txt). The run took 3 min 41 s on the PC.
- The Clopper-Pearson intervals, the power interpolation and the true differences in rho were computed again (checks.py, truediff.py with 20000 learners per arm).
- Both sources were downloaded from arXiv (HTML v2 and the abstract pages) into notes/units/reve_src/ and read in the sections cited.

## Critical errors

1. E10 reading (line 111) and section 7, bullet 5 (line 192). The file says the one-step response predicts revival time only for a held capability, and that the difference between held and absent "sits in the ln(1/epsilon^2) term, which no first order term contains". In the toy of E10 this is false. The first order quantity is the initial kernel K on phi_v, with K = kappa^2 for the held rule and K = 2 epsilon^2 for the absent one. The absent time is (1/(2t)) ln(2t/K - 1), so ln(1/epsilon^2) is ln(2/K), a function of the first order term. The held time, once kappa^2 is below t, is also logarithmic in 1/K. At t = 1 and equal K, the time to t/2 is 1.821 against 1.473 at K = 0.1, 2.645 against 2.299 at K = 0.02, and 3.802 against 3.455 at K = 0.002 (held against absent). The gap stays near ln 2 / (2t) = 0.35 while both times grow. Corrected statement. In the toy, revival time is a decreasing function of the initial kernel for both held and absent rules, up to a bounded gap, so the rank of the one-step response predicts the rank of revival time in both cases. The error enters where the kappa^(-2) law, valid only for kappa^2 much larger than t, is compared with the logistic law. This changes the stated reason behind the tier 2 refutation (line 166) and removes the claim of section 7, bullet 5.

## Major errors

2. Calibration coverage (lines 153, 240). The two-layer learner of calibrate.py is z = a . (W x) with a dense hidden layer. Its kernel is (W x_b).(W x_e) + norm(a)^2 x_b.x_e (calibrate.py:59 to 60). It is not the diagonal learner of E4 that E11 names as L2 and L7. In E4's learner, rho of the rule absent learner is -1 and the true difference at kappa = 1 is 0.400. In the script they are -0.296 and 0.166. The false positive rates hold. The power 0.958 at 0.163 and the 0.8 point near 0.13 hold for the script's construction only. Proposed edit 2 carries these numbers into the research program without that condition.
3. E11 row 2 (lines 120, 127). L2 against L5 is one of the two rows where the response adds to the answers. It holds only if L5's cue weight P_u receives no gradient from the test step. If P_u is a parameter, L5's kernel on fresh problems is u(b) u(e), so rho = -1, which equals L2's rho in E4's learner. The assumption is not stated, and it decides the row.
4. E5 consequence 2 (line 82). The file calls a second consuming head "the structure of Jain et al. 2023". In Jain et al. fine-tuning trains one task and operand pair only (Appendix C.2), so the pretraining task gives no gradient during fine-tuning, and the minGPT has one unembedding. The identification is an idea and is labeled derivation.
5. Panel results said to be derived again (lines 3, 184) against E9 (line 105), which states that B31 point 1 was not checked again. The two statements contradict each other.
6. Jain et al. Appendix E.2 and Choice 3a (lines 179, 222). The source recovers counting a with extra task family tokens T_J1 and T_J2, which are paired only with a in pretraining, and it does so when the pretraining prior of b, the fine-tuning target, is high. The file says "under a high pretraining prior" in a section whose condition axis is the prior of a. The tokens need their own pretraining data, so "inside the setup, no new build" is false. A task token is not a context that holds conflict cases with keys, so Choice 3a does not measure R_ctx of section 1.
7. Scr.+FT as the absent condition (lines 110, 165, 179). In the source, Scr.+FT starts from a model pretrained to count O_FT and then fine-tuned to count a. Its counting circuit exists for another operand, so it is not the absent state of E10. The Scr. baseline of Appendix E.4, trained from scratch on a, is closer. Scr.+FT reaches perfect accuracy only at the larger rate eta_M (Figure 9), so its T_rev is censored at smaller rates.
8. Tier 2 second prediction (line 165). It is derived from E5, whose conservation law holds for plain gradient flow and fails under Adam (A4, line 41). Jain et al. do not state a weight decay value. The prediction has no refutation condition.
9. Tier 3 prediction (lines 171 to 172). "b moves" has no sign. E9 implies b rises by 1/beta toward the key. The E8 reading (line 100) says entrenchment in this delivery is a more negative b. A fall of b, which is entrenchment, passes the prediction. Conditioning on prompts with equal accuracy selects prompts where the tilt of E9, which raises every prompt's log odds, did not act. alpha has no refutation condition.

## Minor errors

10. A2 failure row (line 39). Cross entropy on a softmax is -log p_key = log(1 + exp(-z)) with z the log odds of the key against the rest, a loss of z alone. A2 then holds, and the stated failure applies only to a loss that is not a function of z.
11. A3 (line 40) cites "B19, last paragraph". The statement is in B19's gauge paragraph (panel line 1735). The last paragraph treats levels of computation.
12. A6 (line 43) states 0.690 for cases as replicates in general. It holds for N1. On N0 the same test rejects in 0.050 (block 3).
13. A1 (line 38) names a check by halving eta. calibrate.py uses one eta (line 15). The check was not run.
14. E4 reading (line 75) holds for the diagonal learner only. In the dense learner of the script the use part norm(a)^2 x_b.x_e is shared by all inputs.
15. E5 (line 79). The rate -2 lambda holds for a penalty (lambda/2)norm(theta)^2. The convention is not stated. E7's lasso agrees with it under a loss (1/2)(residual)^2.
16. E6 (line 89). The orbit (c a, s/c) is a symmetry inside one architecture. Two learners of one architecture that compute one function can have different rho. One architecture is necessary for comparison and not sufficient.
17. E7 (line 94). B26 point 3 states the start, -0.385, and a no-penalty end of -0.793. The file says B26 does not state its start and does not mention the different no-penalty end, against the promise of line 3.
18. E10 (line 109). The initial slope of the revival loss is -eta g^T K g. The sign is missing.
19. E11 row 4 (lines 122, 127). L4 has no gradient rho; the entry is rho_ctx. The no-feedback and repeated-case responses separate L4 from L7 at equal answers, so "two rows only" omits this row.
20. E11 row 7 (line 125). The value 0.163 is the script's. In E4's learner it is 0.400.
21. True differences (lines 135, 143 to 147) are Monte Carlo means over 50 learners per arm, standard error about 0.004. With 20000 learners per arm they are 0.000, 0.012, 0.046, 0.099 and 0.166, against 0.003, 0.007, 0.044, 0.099 and 0.163 reported. Line 135 says the N1 truth is 0 and the table says +0.003.
22. Run time (line 151). The rerun took 3 min 41 s. The docstring says about a minute (calibrate.py:8).
23. Section 4 design (line 133) omits the random part of the linear learners (standard deviation 0.1, calibrate.py:72) and the noise on the case x_e (calibrate.py:91).
24. Jain et al. Section 5.1 (line 179). The learning rate statement comes from Tracr attention maps (Figure 10), not from the PCFG setup the file copies.
25. Tier 1 prediction (lines 159 to 160). Refutation covers the N0 rate and the power at kappa = 1. The other rows of E11 and the L4 part have none. L4's values are exact by construction.
26. Tier 2 reading (line 165). The threshold r_s >= 0.6 over 19 conditions has no null distribution or power. Section 1 defines T_rev in plain steps (line 28) while the copied reverse fine-tuning uses AdamW.
27. E8 ceiling (line 100) against the tier 3 fit form (line 171). The fit form has no ceiling. A constant tilt still shifts only the fitted b, since b is a free intercept, so E9's prediction survives. The reading of gamma as a per case evidence weight does not.
28. Research program section 3 says one case "read once" shows a rule held and unused. E2's condition (line 59) needs several cases. The proposed edits (lines 237 to 244) do not revise that line.
29. The group Cu (line 11) is defined and never used in rho, in the calibration or in E11. Panel D28 uses that group to separate a learner that copies surface.

## Full table

| # | item | line | outcome | evidence | severity |
|---|---|---|---|---|---|
| 1 | g_e = y_e (1 - p_e) for cross entropy on the shown key | 13 | holds | derivative of softplus(-y z); calibrate.py:36, 94 | none |
| 2 | E1, R = eta g_e y_b K + O(eta^2), k = y_b y_e K | 51 | holds | chain rule; panel B25 Proposition 6 (panel:2393) | none |
| 3 | E1 check, one step matches weights within 0.026 | 52 | holds | RevE-rerun.txt block 1, largest gap 0.026 (linear mixed) | none |
| 4 | rho reading -1, 0, +1 | 30 | holds | rho = (alpha_v - alpha_u)/(alpha_v + alpha_u) when features are +1 or -1 and K(N, e) = 0 | none |
| 5 | E2 invariance to eta, size and sign of g_e, loss | 56 | holds | first order, k does not contain g_e | none |
| 6 | E2 no-feedback rho within 0.044 | 57 | holds | block 1, largest printed gap 0.043 (held, -0.127 against -0.170) | none |
| 7 | E2 raw sibling move +5.87 and -0.82 | 57 | holds | block 1; ratio 7.2 against (1 - p)/p = 7.26 at z = 1.98 | none |
| 8 | E2 per case division, 0.120 against 0.050 | 59 | holds | blocks 2 and 3b; the sign of the bias derived, abs(g_e) rises with y_e n_e1 for the cue learner and falls for the mixed one, mean read difference -0.033 | none |
| 9 | E3 linear kernel independent of state | 65 | holds | K = phi(b).phi(e); block 1 and block 2 N0 | none |
| 10 | E4 kernel and rho formula, z(CS)/z(Pr) | 70 to 74 | holds | gradients s_k phi_k and a_k phi_k; a^2 + s^2 = 2 abs(P) when balanced | none |
| 11 | E4 reading, rho adds nothing in a balanced learner | 75 | narrowed | holds for the diagonal learner only; dense learner kernel calibrate.py:60 | minor |
| 12 | E5 conservation and decay rate -2 lambda | 79 | narrowed | holds for penalty (lambda/2)norm(theta)^2; -4 lambda for lambdanorm(theta)^2; convention unstated | minor |
| 13 | E5.1 held unused rule unreachable from a balanced start | 81 | holds | delta conserved at 0 forces abs(a) = abs(s) | none |
| 14 | E5.2 second consumer is the structure of Jain et al. | 82 | contradicted | Jain C.2, fine-tuning on a single task and operand pair; one unembedding; labeled derivation | major |
| 15 | E5.3 s_v decays as exp(-lambda t) | 83 | holds | gradient of s_v carries a_v = 0 | none |
| 16 | E6 rescaling orbit and Jacobian rank | 88 | holds | z unchanged; rank preserved under invertible reparameterization | none |
| 17 | E6 conclusion, compare within one architecture | 89 | narrowed | the orbit lies inside one architecture; necessary, not sufficient | minor |
| 18 | E7 dynamics, exponent r_u/r_v, lasso | 93 | holds | d ln P_k/dt = 2 r_k residual; lasso end q_u = 1 - lambda/r_u = 0.967, P_u = 0.644 | none |
| 19 | E7 numbers 1.500, -0.105, -0.655, 0.967, 0.0002, -0.999 | 94 | holds | RevE-rerun.txt block 4 | none |
| 20 | E7 comparison with B26 | 94 | narrowed | B26 states start -0.385 and no-penalty end -0.793 (panel:2448); difference not stated | minor |
| 21 | E8 likelihood ratio 27, posteriors 0.214, 0.750, 0.920 | 98 to 99 | holds | 0.9 x 3 / 0.1 = 27; checks.py; block 5; panel C23, C24 | none |
| 22 | E8 rho_ctx = 0 for every prior | 98 | holds | practiced problems have equal likelihood under both rules | none |
| 23 | E8 N* = (-b/gamma)^(1/(1 - alpha)) | 98 | holds | Bigelow Eq. 10 at m = 0 (bigelow.txt:231) | none |
| 24 | E8 ceiling log 9 = 2.20 | 100 | holds | log(0.9/0.1) = 2.197 | none |
| 25 | E9 tilt adds 1/beta, R_ctx unchanged | 104 | holds | ratio of tilted odds; panel D27 (panel:2337) | none |
| 26 | E9 B31 point 1 not checked again | 105 | contradicted | lines 3 and 184 say every panel result was derived again | major |
| 27 | E10 held time scales as kappa^(-2) | 109 | narrowed | needs kappa^2 much larger than t; at t = 1, K = 0.02, held time 2.645 against kappa^(-2) ln 2 = 34.7 (checks.py) | major |
| 28 | E10 absent time (1/(2t)) ln(t/epsilon^2) | 109 | holds | integrated time matches the logistic law within 0.002 (checks.py) | none |
| 29 | E10 initial slope eta g^T K g | 109 | narrowed | sign is minus | minor |
| 30 | E10 reading, response predicts revival only when held | 111 | contradicted | at equal K, held 1.821, 2.645, 3.802 against absent 1.473, 2.299, 3.455 (checks.py) | critical |
| 31 | Section 7 bullet 5, absent case not predicted | 192 | contradicted | same computation as row 30 | critical, counted with row 30 |
| 32 | Jain revival 0.1K to 1K, 3K, 4.5K at eta_M | 110, 179 | holds | Figure 9 caption and Figure 26 (jain2.txt, Section 5.1 and E.3) | none |
| 33 | E11 row 1, L1 against L5 | 119 | holds | table with no shared feature moves nothing off the case | none |
| 34 | E11 row 2, L2 against L5 | 120 | narrowed | needs P_u of L5 frozen; with P_u trainable rho(L5) = -1 = rho(L2) in E4's learner | major |
| 35 | E11 row 3, L2 against L3 | 121 | holds | rho -1 against +1 in E4's learner | none |
| 36 | E11 row 4, L4 against L7 | 122 | narrowed | L4 has rho_ctx, not gradient rho; the no-feedback branch separates them, so "two rows only" (line 127) omits it | minor |
| 37 | E11 rows 5 and 6 | 123 to 124 | holds | E3, E4 | none |
| 38 | E11 row 7, absent against held, 0.163 | 125 | narrowed | 0.166 with 20000 learners in the script; 0.400 in E4's learner | minor |
| 39 | A2 failure row for a softmax | 39 | contradicted | cross entropy -log p_key = log(1 + exp(-z)) is a loss of z | minor |
| 40 | A3 citation B19 last paragraph | 40 | narrowed | the statement is the gauge paragraph, panel:1735 | minor |
| 41 | A6, 0.690 in general | 43 | narrowed | N0 case-level test rejects 0.050 (block 3) | minor |
| 42 | A1 check by halving eta | 38 | unchecked | one eta in calibrate.py:15 | minor |
| 43 | Section 4 table, all eight rows | 142 to 149 | holds | rerun identical to calibrate_output.txt; intervals recomputed (checks.py) | none |
| 44 | True differences in rho | 135, 143 to 147 | narrowed | Monte Carlo of 50 learners per arm, se 0.004; 20000 learners give 0.000, 0.012, 0.046, 0.099, 0.166 (truediff.py) | minor |
| 45 | Power 0.8 at about 0.13 | 153 | holds | linear interpolation 0.135; normal model with se 0.045 gives 0.125 | none |
| 46 | Run time about 2 minutes | 151 | contradicted | rerun 3 min 41 s; docstring says about a minute | minor |
| 47 | Calibration covers L2, L6, L7 | 153 | narrowed | dense learner calibrate.py:59 to 68, not E4's learner; rho absent -0.296 against -1 | major |
| 48 | Proposed edit 2 carries power numbers | 240 | narrowed | numbers hold for the script's construction only, see row 47 | major, counted with row 47 |
| 49 | Section 4 design description | 133 | narrowed | omits linear random weights, sd 0.1 (calibrate.py:72), and noise on x_e (calibrate.py:91) | minor |
| 50 | Jain record, title, authors, arXiv 2311.12786 v2, 2023 | 179 | holds | reve_src/jain_abs.html citation metadata, dated 2023-11-21, v2 2024-08-21 | none |
| 51 | Jain Section 3, linear readout and wrapper g composed with C | 179 | holds | Definitions 1 and 2 | none |
| 52 | Jain, fine-tuning rarely alters capabilities | 179 | holds | abstract | none |
| 53 | Jain Section 5.1, larger rate alters computation only when not weakly relevant | 179 | narrowed | Tracr attention maps (Figure 10), not PCFG | minor |
| 54 | Jain E.2, task tokens under a high prior | 179 | narrowed | the high prior is that of b; tokens T_J1, T_J2 need their own pretraining | major |
| 55 | Jain H.1, inhibitor under the large rate | 179 | holds | H.1 captions, inhibitor with strongly relevant capability and eta_L | none |
| 56 | Jain setup details | 179 | holds | C.2, 3M, 6 blocks, 6 heads, 192; AdamW 1e-3 cosine; 10K to 200K; priors; 1e-4, 1e-5, 1e-6; batch 96; 10K; C_Tr in {0, 0.5, 0.8, 1} | none |
| 57 | Scr.+FT as pretrained on the other token, absent condition | 110, 165, 179 | narrowed | source Section 5.1, pretrained to count O_FT then fine-tuned on a; Scr. of E.4 is closer; censored below eta_M | major |
| 58 | Bigelow record, title, authors, arXiv 2511.00617 v2, 2025 | 180 | holds | reve_src/big_abs.html metadata, dated 2025-11-01 | none |
| 59 | Bigelow Eq. 3, 4, 5, A.1, 7, 8, 9, 10 | 180 | holds | bigelow.txt:146 to 231, 659 to 712 | none |
| 60 | Bigelow r = 0.98, and 0.97 to 0.98 on two other models | 180 | holds | bigelow.txt:144, 766 | none |
| 61 | Bigelow flipped labels need more shots | 180 | holds | Appendix F, bigelow.txt:788 | none |
| 62 | Bigelow setup, 128 shots, L-BFGS-B, basin hopping, 10-fold | 180 | holds | Appendix H, bigelow.txt:801 to 807 | none |
| 63 | A7, fit on 7B to 70B models | 44 | holds | Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B, Llama-3.1-70B | none |
| 64 | Sign of evidence term differs in Eq. 3 and Eq. 5 | 182 | holds | sigma(log o) against sigma(-log o) | none |
| 65 | Tier 1 prediction and refutation | 159 to 160 | narrowed | no refutation for the other rows or for L4 | minor |
| 66 | Tier 2 conditions, 18 plus Scr.+FT | 165 | holds | 3 x 3 x 2 = 18; priors of a 0.999, 0.9, 0.5 match P^L, P^M, P^H | none |
| 67 | Tier 2 second prediction from E5 under AdamW | 165 | narrowed | E5 needs plain flow; weight decay not in source; no refutation | major |
| 68 | Tier 2 threshold r_s 0.6, T_rev definition | 165, 28 | unchecked | no null or power for r_s; T_rev in plain steps against AdamW reverse fine-tuning | minor |
| 69 | Tier 3 prediction, b moves | 171 to 172 | narrowed | no sign; E8 entrenchment is b falling; equal accuracy conditioning; alpha unrefuted | major |
| 70 | E8 ceiling against tier 3 fit form | 100, 171 | narrowed | free intercept keeps E9's prediction; reading of gamma not kept | minor |
| 71 | Research program "read once" line not revised | 237 to 244 | narrowed | E2 condition line 59 | minor |
| 72 | Group Cu unused | 11 | narrowed | not in rho, calibration or E11; panel D28 uses it | minor |
| 73 | Choice 3a measures the context delivery | 222 | contradicted | a task token is not a context of conflict cases; see row 54 | major, counted with row 54 |
| 74 | reading.md rows for both sources | 182 | holds | sources/reading.md:43 to 44 match the file | none |
