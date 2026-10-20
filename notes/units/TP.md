# TP The theory end of surrogate P, The process of solving

## Plan
- Target: Give surrogate P of C:/Users/YuanW/PLMS/surrogates/program/research-program.md section 4 its theory, so each tier has a defined object, a derived prediction and a calibration with known truth.
- Goal: Write theory/P.md, the theory end of surrogate P.
- Establishes: The object, its assumptions, what can be derived, the calibration of its tier 1 reading, and a prediction per tier.
- Cannot establish: Any empirical result. Nothing is trained or fitted on real data.
- Inputs: C:/Users/YuanW/PLMS/surrogates/program/research-program.md (all); C:/Users/YuanW/claude-thesis-docs/exploration/signature/goodfire-reading.md section 7; C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md (the synthesis sections S1 to S15 first, member sections as needed); C:/Users/YuanW/claude-thesis-docs/exploration/signature/three-questions.md Part E.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/P.md; new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md; any tiny synthetic script under C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_p/.
- Check: theory/P.md holds the eight sections of the Specification, each derivation marks its rung, and every cited source has a row in sources/reading.md.
- Open choices: the formal object, the derivations and the calibration design are the builder's; the choices that change scope go to section 8 as options for the author.
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 1 (fix after RevP, theory/P-review.md)
- If pass: the lead reads section 8 and folds accepted edits into the research program.
- If fail: the missing sections are named and one attempt fixes them.
- Resolution: accepted for the toy trees. results/theory_p_grpo_departure.txt gives a rise of the repair rate inside right runs (mean +0.0228 at beta 0.5, positive in 0.950 of 179 trees; k3 +0.0050 at beta 2, all 179), from a derivation checked to 3e-11. The review stated the direction without a computation. The rise holds for depth 3, three tokens per step and equal run lengths; it is not shown for a trained model, and its size at beta 2 is small.

### Specification
The surrogate answers one part of "what is learning": whether training changes how a learner solves, or only how often it is right. The theory file has these sections.
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
theory/P.md exists with all eight sections and a computed calibration on synthetic reference learners; no calibration on a trained network exists, and nothing ran on a real model.
Effect: 8 of 8 sections written, ratio 1.0

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/P.md (235 lines)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_p/calibration.py (numpy and scipy, python -I, seed 20261019)
- C:/Users/YuanW/PLMS/surrogates/results/theory_p_calibration.txt (its output; wall time above 10 minutes on one PC core)
- seven new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md

Requirements
- met: eight sections, each derivation with its rung (idea, derivation, checked numerically, source).
- met: calibration with a case with no effect and a planted effect of known size, false positive rate and power computed (P.md section 4).
- met: every cited paper opened as primary text; rows added; Forking Fast and Reasoning Theater read in full. One statement (the 1/|y| weight of the GRPO loss) is marked recalled, not opened, and carries no derivation.
- met: research program not edited; learner-state, learner-models and docs/conversation not opened; no cluster job; no commit; under 500 lines; downloads deleted.

What was tried
- Part B, forced answer after cuts. The logit difference in differences against items with the answer given holds its level (0.045 to 0.059) and has power 0.77 for a key probability rise of 0.06 at 100 items per kind. It fails when taken on the probability scale (false positive rate 0.45 to 1.00) and when drift differs between item kinds (0.78 to 1.00).
- Part A, forks in o. The raw threshold rule gives false forks at 0.11 of pairs at S = 200. Pooled windows of 10 positions detect a jump of 0.2 at S = 30 with power 0.94 and any-false-fork rate 0.022.
- Part C, tilt null per item. False positive rate 0.042; power 0.97 for a 0.10 planted share. The test also flags plain gradient narrowing with rate 0.996 when the property goes with the modal right method, so a rejection does not show a new method.
- Derived: the outcome distribution per prefix is a martingale on sampled chains (checked, 0.1921 against 0.1927). The tilt null extends to every outcome-only reward at a KL-regularized fixed point, GRPO included. Pooling across items breaks the null (0.32 to 0.44 under the exact tilt).

Disagreement: the research program's Theory line for P says "belief shifts at backtracking steps on hard items" (Boppana et al. 2026). The source finds inflections at 0.020 against 0.045 per step by probe confidence on MMLU, no reliable local co-occurrence with belief shifts, and no significant trend on GPQA-Diamond (its section 6.2 and Table 2). Rung: source. I complied, left the program unedited, and proposed the corrected line in P.md section 8, edit 1.

Lesson: none

### Attempt 1
theory/P.md is revised with all 5 critical and 12 major review items fixed; five readings named in the predictions still have no calibration and wait for choice 5.
Effect: 17 of 17 critical and major items fixed, ratio 1.0
Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/P.md (308 lines, fix log at the end); calibration.py (parts seeded A 20261019, B 20261020, C 20261021) and results/theory_p_calibration.txt (A1, B, C rerun; A2 stream checked by notes/units/tp_fix_scratch/stream_check.py)
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_p/grpo_departure.py and results/theory_p_grpo_departure.txt (new); six rows of sources/reading.md corrected or added (Shao et al. 2024)
Requirements
- not met: calibration before use for five readings (early decision, forks near cuts, follow rate, tier 3 probe shift, tau statistic); each is marked in P.md and listed as choice 5.
- met: python -I, tiny synthetic records; research program and review unedited; learner-state, learner-models, docs/conversation not opened; no commit; under 500 lines.
What was tried
- Tier 1 T moved to cut k-1: level 0.051, power 0.781 at a key-probability rise of 0.06, 100 items per kind. Part C planted change is 0.025, 0.050, 0.098, 0.197 in the within-right rate, power 0.516, 0.974, 1.000, 1.000.
- Per-token KL identity: log pi/pi_ref = (c/beta)(R - q(x)) + sum of per-prefix KL, error 3.2e-11 on 200 trees.
Disagreement: the review (P-review.md:17) says the per-token departure predicts a fall of the repair rate inside right runs. The identity (P.md:103) and results/theory_p_grpo_departure.txt give a rise, mean 0.0228 at beta 0.5 (positive in 0.950 of 179 trees) and 0.0050 for k3 at beta 2 (all 179). Rung: derivation, checked numerically. I complied with the restatement and wrote the rise.
Lesson: none

## Audit
| item | outcome | evidence |
|---|---|---|
| R1 F_j over an answer set (P.md:13; severity minor) | narrowed | matches Boppana 3.3, softmax over four letters (2603.05488.txt:133); free numeric answers need a declared set (2307.13702.txt:367) |
| R2 o_j definition (P.md:15; severity none) | holds | definition |
| R3 tau_text, tau_F unused (P.md:16-17; severity minor) | narrowed | no later use in P.md; research-program.md:81 asks for it |
| R4 H_freq, H_proc (P.md:25-27; severity minor) | narrowed | conditional undefined at p_0(x) in {0, 1} |
| R5 rate_1 formula (P.md:34; severity none) | holds | total probability under H_freq; panel-2026-10-19.md:2745-2748 |
| R6 A1 and Boppana appendix F (P.md:45; severity minor) | narrowed | 2603.05488.txt:779-782 |
| R7 A2 false positive range (P.md:46; severity minor) | narrowed | results/theory_p_calibration.txt, B prob rows 0.445 to 0.997 |
| R8 A3 training temperature (P.md:47; severity major) | contradicted | tower rule needs one kernel for path and continuations; calibration.py:30-46 uses no temperature |
| R9 A4 multinomial counts (P.md:48; severity none) | holds | 2608.19611.txt:86-87 |
| R10 A6 statement (P.md:50; severity none) | holds | stated as an assumption; GRPO fails it (R27) |
| R11 A8 per-item rates (P.md:52; severity none) | holds | D7 numbers (R33) |
| R12 D1 complexity results (P.md:56; severity none) | holds | 2310.07923.txt:70, 98-102, 116, 123, 241-247 |
| R13 D1 theorem numbers (P.md:56; severity minor) | narrowed | upper bound is Theorem 4, 2310.07923.txt:361-362 |
| R14 D1 finite map at fixed depth (P.md:56; severity minor) | narrowed | needs width growing with the map |
| R15 D2 martingale (P.md:61; severity none) | holds | tower rule, re-derived |
| R16 D2 total squared movement (P.md:62; severity none) | holds | orthogonal increments and o_T = e_A, re-derived |
| R17 D2 check against L62 (P.md:65; severity major) | contradicted | calibration.py:45 uses o0(1-o0) = 0.19273; L62 gives 0.38546 (tools/checks.py) |
| R18 D2 numbers 0.739, -0.0003, 0.261 (P.md:65; severity none) | holds | rerun identical; P(Bin(12, 0.55) >= 6) = 0.73931 |
| R19 D2 Forking Fast o_t (P.md:69; severity minor) | narrowed | 2608.19611.txt:78, truncation at p >= 0.05 |
| R20 D3 necessity, dependence, use (P.md:72-75; severity none) | holds | 2307.13702.txt:171, 244-256 |
| R21 D3 recompute in text, panel D20 (P.md:75; severity none) | holds | panel-2026-10-19.md:1756 |
| R22 D4 filler 2.5, paraphrase 2.6 (P.md:83-84; severity none) | holds | 2307.13702.txt:30-33, 288, 311 |
| R23 D4 filler rules out hidden computation (P.md:88; severity major) | narrowed | 2404.15758.txt:152, 183 |
| R24 Pfau rung (P.md:88; reading.md:41; severity minor) | narrowed | 2404.15758.txt:180-183 supports the claim; rung stated above what was opened |
| R25 D5 tilt keeps runs inside outcome (P.md:92-95; severity none) | holds | factor depends on R only, re-derived |
| R26 D5 fixed advantage c(x) = 1/sd (P.md:97; severity none) | holds | holds for a sequence-level reverse KL with advantage fixed |
| R27 D5, tier 2, edit 4, GRPO included (P.md:97, 179, 234; severity critical) | contradicted | 2402.03300.txt:724-727 (weight 1 over the length of o_i, and per-token KL in the loss); tools/grpo_fast.py within-right TVD 0.0585 mean, 0.121 max |
| R28 D5 length weight recalled (P.md:97; severity none) | holds | 2402.03300.txt:724 |
| R29 D5 exact tilt change at most 1e-4 (P.md:97; severity none) | holds | exact_tilt_max_abs_change 0.0000 |
| R30 D6 logit flow (P.md:99; severity none) | holds | dJ/dl_y = pi_y (R_y - p), re-derived |
| R31 D6 numbers 0.069, 0.065, 0.000, 0.035 (P.md:99; severity none) | holds | rerun 0.0687, 0.0648, 0.0000, 0.0354 |
| R32 D6 no mean direction (P.md:99; severity none) | holds | not printed by the script; tools/checks.py signed mean -0.0014, t = -1.10 over 2000 items |
| R33 D7 0.32 to 0.44 (P.md:101; severity none) | holds | (0.16+0.16)/1.0 and (0.48+0.18)/1.5 |
| R34 D8 and tier 2 refutation name the D17 judge (P.md:109, 180; severity major) | contradicted | D5; panel-2026-10-19.md:1621, 2752; P.md:209 |
| R35 D9 Lanham numbers (P.md:114; severity none) | holds | 2307.13702.txt:121, 159, Table 2 at 166-232, 293, 318, 328 |
| R36 D9 performativity 0.505, 0.010 (P.md:115; severity none) | holds | 2603.05488.txt:147-166 |
| R37 D9 inflections 0.020, 0.045 (P.md:115; severity none) | holds | 2603.05488.txt:182-202 |
| R38 D9 no reliable local co-occurrence (P.md:115; severity major) | narrowed | 2603.05488.txt:213, 817, 822 |
| R39 D9 linear probe near chance, 88 percent (P.md:115; severity none) | holds | 2603.05488.txt:724-740 (31.85 against 87.98) |
| R40 D9 Forking Fast numbers (P.md:116; severity none) | holds | 2608.19611.txt:106, 118, 217-218, 230, 256 |
| R41 B equations against code (P.md:124-129; severity none) | holds | calibration.py:100-126 |
| R42 B k and c unused (P.md:122; severity minor) | narrowed | calibration.py:112-121 |
| R43 B planted learner called use (P.md:132; severity minor) | narrowed | T is a dependence statistic, P.md:73 |
| R44 B table (P.md:134-139; severity none) | holds | rerun identical; header rises mix n rows (0.120 at n = 100, w = 0.2) |
| R45 B power 0.77 at 100 items (P.md:141; severity none) | holds | 0.769 |
| R46 B probability scale 0.445 to 0.997 (P.md:142; severity none) | holds | result file B prob rows |
| R47 B unequal drift 0.778 to 1.000 (P.md:143; severity none) | holds | result file B rows drift (1.0, 0.5) |
| R48 A table (P.md:150-154; severity none) | holds | rerun identical |
| R49 A raw rule of appendix D (P.md:156; severity minor) | narrowed | 2608.19611.txt:78, 228 |
| R50 A test form of segment pooling (P.md:156; severity minor) | narrowed | 2608.19611.txt:92, 241-245 |
| R51 C table (P.md:160-165; severity none) | holds | rerun identical |
| R52 C planted share label (P.md:163; severity critical) | contradicted | calibration.py:220-221; tools/checks.py effective 0.025, 0.051, 0.101 |
| R53 C narrowing with random phi, 0.054 (P.md:164; severity major) | narrowed | H_freq false there, P.md:99 |
| R54 C reading of level and power (P.md:167; severity none) | holds | 0.042 at the tilt; power subject to R52 |
| R55 Prystawski bias toward the marginal by theorem (P.md:169; severity critical) | contradicted | 2304.03843.txt:168, 174, 284, 590 |
| R56 Prystawski 32M, 10 layers, 4.5M (P.md:169; severity none) | holds | 2304.03843.txt:736, 767 |
| R57 Prystawski need known by construction (P.md:169; severity minor) | narrowed | 2304.03843.txt:241 |
| R58 tier 1 main refutation (P.md:173-174; severity major) | narrowed | P.md:8, 10; F_k reads the written key |
| R59 tier 1 share of squared movement (P.md:173, 175; severity major) | narrowed | noise 2(1 - sum o^2)/S = 0.039 per pair at S = 30; no calibration |
| R60 Choice 1a o only at cuts (P.md:220 against 173, 175; severity major) | contradicted | 9 cuts cannot locate movement near cuts or at t = 1 |
| R61 follow rate against w = 0 (P.md:173; severity minor) | narrowed | w exists only in the reference learner |
| R62 side prediction, smaller model larger T (P.md:177; severity major) | narrowed | 2307.13702.txt:329; 2603.05488.txt:179 |
| R63 tier 2 seed spread (P.md:179-180; severity major) | narrowed | part C calibrates a t-test over items, P.md:158 |
| R64 tier 2 positive control SFT (P.md:182; severity none) | holds | cross entropy fixed point at the teacher's runs, panel-2026-10-19.md D18 |
| R65 tier 3 not within a window, as in Boppana (P.md:184; severity critical) | contradicted | 2603.05488.txt:205, 213, 817 |
| R66 tier 3 refutation via Ye Result 4 (P.md:185; severity critical) | contradicted | 2407.20311.txt:97, 285 |
| R67 tier 3 DeepSeek route (P.md:186; severity none) | holds | 2603.05488.txt:567, 583-592 |
| R68 section 6 ids, titles, authors, years (P.md:194-200; severity none) | holds | arXiv abstract titles of all seven ids; author lists 2608.19611.txt:46-52, 2603.05488.txt:71-87 |
| R69 Boppana forced answer setup (P.md:195; severity none) | holds | 2603.05488.txt:583-587 |
| R70 Ye pretraining hardware (P.md:200, 222; severity none) | holds | 2407.20311.txt:831 |
| R71 Prystawski cost 20 h, 300,000 steps (P.md:221; severity none) | holds | 2304.03843.txt:206, 738 |
| R72 Choice 1a counts 3,600 and 108,000 (P.md:220; severity none) | holds | 200 x 2 x 9 = 3,600; x 30 = 108,000 |
| R73 Choice 2 monitor needs a second model (P.md:226; severity none) | holds | 2603.05488.txt:140 |
| R74 edit 1 replacement line (P.md:231; severity major) | narrowed | 2603.05488.txt:91, 101, 213, 795-817 |
| R75 edit 1, no backtracking shifts on hard items (P.md:231; severity none) | holds | 2603.05488.txt:197-199, 208, 213 |
| R76 section 7 limits (P.md:206-212; severity none) | holds | consistent with D1, D3, D6 |
| R77 script reproduces the result file (calibration.py; severity none) | holds | python -I rerun, run/rerun.txt identical to the result file except line endings |
| R78 docstring runtime (calibration.py:12; severity minor) | contradicted | result file header, above 10 minutes |

Audit by the RevP checker, 19 October 2026. Full review with critical, major and minor lists in C:/Users/YuanW/PLMS/surrogates/theory/P-review.md. Critical rows R27, R52, R55, R65, R66.
