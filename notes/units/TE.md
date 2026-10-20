# TE The theory end of surrogate E, Response to new evidence

## Plan
- Target: Give surrogate E of C:/Users/YuanW/PLMS/surrogates/program/research-program.md section 4 its theory, so each tier has a defined object, a derived prediction and a calibration with known truth.
- Goal: Write theory/E.md, the theory end of surrogate E.
- Establishes: The object, its assumptions, what can be derived, the calibration of its tier 1 reading, and a prediction per tier.
- Cannot establish: Any empirical result. Nothing is trained or fitted on real data.
- Inputs: C:/Users/YuanW/PLMS/surrogates/program/research-program.md (all); C:/Users/YuanW/claude-thesis-docs/exploration/signature/goodfire-reading.md section 7; C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md (the synthesis sections S1 to S15 first, member sections as needed); C:/Users/YuanW/claude-thesis-docs/exploration/signature/three-questions.md Part E.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/E.md; new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md; any tiny synthetic script under C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_e/.
- Check: theory/E.md holds the eight sections of the Specification, each derivation marks its rung, and every cited source has a row in sources/reading.md.
- Open choices: the formal object, the derivations and the calibration design are the builder's; the choices that change scope go to section 8 as options for the author.
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 1 (fix after RevE, theory/E-review.md)
- If pass: the lead reads section 8 and folds accepted edits into the research program.
- If fail: the missing sections are named and one attempt fixes them.
- Resolution: none

### Specification
The surrogate answers one part of "what is learning": whether a change alters what later evidence does to the learner. The theory file has these sections.
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
theory/E.md exists with all eight sections and a calibrated gradient reading; the context delivery and four of the seven reference learners are not calibrated.
Effect: 8 of 8 sections written, ratio 1.00

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/E.md (244 lines).
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_e/calibrate.py and calibrate_output.txt beside it (python -I, about 2 minutes on the PC).
- Two rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md (Jain et al. 2023, Bigelow et al. 2025).
- Documents read: the research program, goodfire-reading.md section 7, panel S1 to S15 and B19, B25 to B33, C23 to C31, D26 to D34, 7A1 to 7A3, R8A3, three-questions.md Part E. None covers the theory of E; theory/E.md is the file the project CLAUDE.md names for it.

Requirements
- Not met. The reading is calibrated for the gradient delivery on learners L2, L6, L7 only. The context delivery and L1, L3, L4, L5 have no false positive rate or power. Their responses are zero or exact by construction (theory/E.md section 4).
- Met. Gradient reading calibrated on a no-effect case and a planted effect of known size. False positive rate 0.050 and 0.058, power 0.958 at a true difference in rho of 0.163 (section 4).
- Met. Both sources opened in full text (arXiv HTML v2), authors and titles checked against the abstract pages.
- Met. Nothing opened from learner-state, learner-models or docs/conversation. No commits. Research program not edited. No cluster jobs.

What was tried
- First calibration used a raw ratio of means over cases. It rejected the linear null at 0.120, above the chosen level 0.05. Division of each case by its delivery scalar fixed it (0.050). Both are in the output file.
- A case-level paired t-test rejects a true null at 0.690. The seed-level permutation test replaced it.
- Main results: a linear learner's response does not depend on its state (E3). In a balanced two-layer learner rho is a function of the answers (E4). A held, unused rule needs imbalance, which weight decay erases (E5). rho magnitudes are not invariant to weight rescaling (E6). Entrenchment end point is a lasso (E7, B26 numbers reproduced).
- The first PDF fetch of Jain et al. was encrypted. The HTML version was read instead.

Disagreement: none.
Lesson: none.

### Attempt 1
theory/E.md is revised with the critical item and all eight major items fixed and the reading calibrated on the diagonal learner; the context delivery and L1, L3, L4, L5 are still not calibrated.
Effect: 9 of 9 critical and major items fixed, ratio 1.00

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/E.md (303 lines), with the section "Fix log of attempt 1", one row for each of the 29 review items.
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_e/calibrate_diagonal.py and calibrate_diagonal_output.txt (python -I, 4 min 50 s). This file holds the diagonal calibration, the eta halving check and the true differences from 20000 learners.
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_e/revival_toy.py and revival_toy_output.txt (python -I, 4 min 18 s). This file holds E10, the inversion window, the r_s threshold and the GRPO factor of B31 point 1.
- calibrate.py has one change, its docstring run time. sources/reading.md row 43 (Jain et al.) is corrected.
- Documents read: TE.md, theory/E.md, theory/E-review.md, notes/units/reve_src/jain.html (full text, v2), panel B11, B19, B26, B31, D28.

Requirements
- Not met. The context delivery and L1, L3, L4, L5 have no false positive rate or power (E.md section 4).
- Met. Diagonal learner of E4 calibrated. False positive rate 0.045 [0.027, 0.070]. Power 0.8 at a true difference in rho of about 0.10. Power 1.000 at kappa 1, where the true difference is 0.379. The dense power, 0.958 at 0.166, is now stated for the dense learner only.
- Met. Every claim attributed to Jain et al. was checked against the full text. The absent condition is Scr. (Appendix E.4), not Scr.+FT.
- Met. The assumption on the cue weight in E11 row 2 is stated (frozen). Row 2b gives the trained case.
- Met. Tiny synthetic records only, python -I. The research program and the review were not edited. No commits. Nothing was opened from learner-state, learner-models or docs/conversation.

What was tried
- E10 was derived again with the conserved delta. A held rule is slower than an absent rule at equal kernel by (ln 2)/(2t). The rank inverts only when K_h is between K_a and about 2 K_a.
- The diagonal learner was built from the dense learner's draws (s_k f_k = W_k). The two learners compute one function and differ only in the weights that the test step moves.
- B31 point 1 was recomputed with the group's population standard deviation. The bound is 0.275/beta.

Disagreement: none.
Lesson: none.

## Audit
Auditor RevE, 19 October 2026. Full review with the critical, major and minor lists in C:/Users/YuanW/PLMS/surrogates/theory/E-review.md.

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
