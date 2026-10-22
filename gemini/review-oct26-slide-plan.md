# Review of the 26 October slide plan and its figures

Reader check of notes/OCT26-SLIDE-PLAN.md (the plan, identical to gemini/OCT26-SLIDE-PLAN.md), gemini/make_oct26_figures.py (the script) and gemini/figures/F1 to F8. Each number was checked against results/, notes/units/*.md, results/report-q1q2.md and results/closure.md. Recomputations read the committed CSVs with pandas. src/irt4llm/verify_master_plan.py was run read-only with the repository as its root. No model was loaded and no job was run.

The script reads C:\Users\steph\...\item-params-llm\results (script:24). The same file names exist under irt4llm/results, and every script computation that reads a file reproduces on them. The hard-coded arrays do not come from a file.

Verdicts are holds, wrong, no source, and withdrawn or narrowed. "Seen" means that the 7 S1 states were seen when the Q2 rules were written (report-q1q2.md:35).

## Wrong

| Item | Location | Source | Verdict |
|---|---|---|---|
| 600 Eedi items | plan:17, :43; script:71, :77 | results/E1/difficulty.csv has 704 rows with both b; notes/units/E1.md:84 (704) | wrong, 704 |
| Human 2PL "reliability 0.985" | plan:43; script:78 | results/E1/summary.md:48. 0.985 is the intercept and theta recovery Pearson of nrm_fit (nominal) on synthetic data under the student truth. It is not a reliability of the 2PL student scores | wrong statistic |
| Population a orthogonal to the within-model RL change (F1 B, "Cross-sectional 2PL ≠ intra-model trajectory") | plan:44; script:110 (title) | Recomputed from T9r/items_wide.csv and W0/items_bank.csv on 1,472 items. Pearson of a_pop with the 3-seed mean ΔL of GRPO step 666 is +0.392 (R² 15.4%). The script prints the same 15.39% on the panel (script:95) | wrong, r = +0.39 |
| GRPO fit "R² = 50.1%" (F2 A) | script:130 | verify_master_plan.py eps 1e-4, seed 0 R²(ΔL on L_base) 48.65% (seeds 48.65, 49.43, 51.36) | wrong, 48.7% for the plotted seed 0 |
| SFT "τ = 0.43" beside the seed 0 scatter and fit (F2 A) | script:125, :131 | The plotted seed 0 fit has slope −0.456, so τ = 0.544. 0.43 is the mean over 3 seeds (0.544, 0.354, 0.394) | wrong for the plotted data |
| "Jeffreys K = 32 (S1 GRPO)" r 0.908, σ ratio 1.525, τ 1.38 | plan:18, :51; script:152-153 (hard-coded) | Recomputed from Q2g/s1/residuals.csv, p_tilde over 32 draws per item. GRPO r 0.731 to 0.746, σ ratio 1.91 to 1.99, τ 1.43 to 1.46 per seed (pooled counts r 0.765, ratio 2.21) | wrong |
| SFT "r = 0.733 in K = 32 rollouts" | plan:52, :118 | Same file. SFT r 0.592 to 0.611 per seed, 0.629 on pooled counts | wrong |
| "σ_s/σ_base = 1.53 to 2.09" for GRPO | plan:18, :51 | 1.53 has no file. Token-0 reads give 2.07 to 2.13 (verify_master_plan.py); K = 32 draws give 1.91 to 1.99 | wrong |
| "48.7% to 87.3%" and "49% to 69% for GRPO" of item change governed by own L_base | plan:18, :48; script:140 | verify_master_plan.py, token-0 forced reads only. GRPO R² is 48.7 to 51.4% at clip eps 1e-4 and 63.5 to 68.5% at eps 1e-6, so "49 to 69" joins two clip settings. On the K = 32 draws that F3, F4 and Q2-A use, R²(ΔL on L_base) is 0.27 to 0.29 for SFT and 0.095 to 0.101 for GRPO | wrong as stated; holds only for token-0 reads at one eps |
| "All 15 Base to RL runs hit the 0.479 wall by update 83"; "0.469 to 0.479 wall" | plan:26, :108, :113; script:605, :612 | Q1/trajectory.csv has update 83 for seed 0 only, 4 runs. S1 GRPO has no update 83 state. Final mean@16 per run is 0.459 to 0.480 (Q1/stats.csv) and 0.473 to 0.486 (S1/stats.csv) | wrong |
| "84.4% to 100% of the gain by update 83" | plan:26, :111; script:603 | Q1/trajectory.csv seed 0, share of the final gain reached at update 83. grpo 83.7%, ctt 91.5%, pop0 93.9%, pop1 94.5% | wrong, 83.7% to 94.5% in one seed |
| RL mean@16 "bounded by Base maj@16 0.414 / 0.459*" | plan:26, :27, :112; script:598 | Every Q1 and S1 GRPO final mean@16 (0.459 to 0.486) is above 0.414, and the four arm means (0.4685 to 0.4748) are above 0.459 (Q1/stats.csv, S1/stats.csv, rewrite1_parsed.csv) | wrong, the data contradict the bound |
| Initial W1 G = 5 zero-variance share 35.0% | plan:26; script:625 | W1/summary.json wasted_share_g5_raw 0.438, shrunk 0.280; Q1 grpo traces, updates 0 to 9, mean 0.369 to 0.438 | wrong, 0.438 raw |
| Static critics starve because zero variance times a_pop is zero | plan:111; script:633-636 | pop0, pop1 and ctt are G = 1 critics with advantage r − p̂, which is not zero (report-q1q2.md:11-14). zero_adv_share is empty in their traces (Q1/traces/pop0_*.csv, pop1_*.csv, ctt_*.csv). Only the grpo arm has a zero-advantage share, and it takes no a_pop | wrong mechanism |
| CAT RMSE "1.40x to 1.65x worse across all k = 16 cells" | plan:101; script:569-570 | notes/units/Q2a.md:73. 1.650, 1.528, 1.403 are means over states; classical is at most adaptive in 19 of 21 cells | wrong, 19 of 21 |
| Ceiling band "0.470 to 0.489" (F4 B) | script:400 | Drawn span is 0.466 to 0.491. Per-run maj@16 is 0.467 to 0.496; q1_ctt_1 at 0.496 lies outside (Q1/stats.csv) | wrong |
| ρ −0.17 and +0.14 named Cov(a, Δp_SFT) and Cov(a, Δp_GRPO) | plan:24, :89; script:467-468 | Q2g/s1/flags.csv A3_spearman_c_a; Q2f-notes.md:39. The statistic is the Spearman of the IRT residual c, after a_j Δθ is removed, with a. Values hold (SFT −0.160 to −0.184, GRPO +0.134 to +0.149) | wrong statistic name |
| "p < 10^-9" beside ρ(c_SFT, a) | script:467 | No p value in flags.csv or Q2f/Q2g notes | no source |
| F5 B x axis "a_j quintiles across MMLU bank (J = 1,472)" | script:458, :473 | The code cuts the 15 subject means into 5 groups of 3 subjects | wrong label |
| F8 B "Base to RL (15 runs)" bars 0.474, 0.481, 0.490 | script:690-692 (hard-coded) | 15-run mean of mean@16 is 0.4727. 0.481 and 0.490 are the 3 S1 GRPO runs only | wrong |
| F8 B "Unit M1 (SFT to RL)" bars 0.479, 0.504, 0.555 | script:690-692 (hard-coded) | results/M1 holds jobs.txt only; state.md:46 M1 running. The bars copy the base-to-RL and SFT values and present them as M1 data | no source, placeholder drawn as data |
| "0.63% band" | plan:109; script:643 | Width 0.0063 of accuracy, which is 0.63 percentage points or 1.3% of 0.47 | wrong unit |
| "100% existing committed data on Slides 1 to 10" | plan:7, :13 | The script reads another machine's path (script:24-25), hard-codes arrays (script:152-153, :690-692) and cites numbers with no file in this repository (rows below) | wrong |

## No source found

| Item | Location | Source | Verdict |
|---|---|---|---|
| Corr(a_pop, a_traj) = 0.04 on MMLU, "signature F5" | plan:17, :44, :110; script:103 | No file in results/ or notes/units. signature/notes/findings.md:33 F5 is "the theory check on the computed vocabulary". CROSS-REPO-INDEX.md:101 describes a 102-model against 165-checkpoint comparison, so a_traj would be a pretraining trajectory, not the RL trajectory of one 1B model that plan:44 names | no source |
| 16x binomial variance, z_sd 4.02, "signature F6 to F8" | plan:100 | No irt4llm file. signature/notes/findings.md:40-60 F6 to F8 are U1f, U1c and a theory check | no source |
| ΔR² ≤ 1.04% (SFT) and ≤ 0.06% (GRPO) from z_pop | plan:18, :50; script:139 | verify_master_plan.py prints γ and R², not ΔR². No unit | no source |
| Tables T1 to T5 from verify_master_plan.py | plan:19, :21, :23, :28 | The script exists (src/irt4llm/verify_master_plan.py) with no unit file, no Audit and no committed output. Its Part 1 and Part 4 values reproduce when run | no audited source |
| CAT "+0.042 to +0.092 asymptotic bias" (script "+0.04 to +0.09") | plan:101; script:568 | Q2c-notes.md:44 adaptive bias +0.042 to +0.058 (two bands, checkpoints); review-2026-10-19/K.md:106 selection effect +0.049 to +0.095 (S1 states, k = 1). The range joins two units, and no n above 100 is tested | no source for the range or "asymptotic" |
| maj@16 0.529 "random tie" (GRPO 0.485) | plan:22, :76, :117 | No file. It is the midpoint of maj_tie_wrong 0.504 and maj_tie_correct 0.555 (S1/rewrite1_parsed.csv) | no source, derivable |
| "85.4% in T9r" zero variance | script:635 | No file | no source |
| "Theorem 1", Δθ ∝ mean Δp + Cov(a, Δp)/ā | plan:87-89; script:475 | No derivation in any unit. A first-order step of the 2PL score equation weights Δy_j by a_j over Σ a_j² p_j q_j | no source, idea rung |
| "Theorem 2", E[SSR] formula, lz "tracking β_s monotonically" | plan:25, :94 | No unit; no fit of β_s exists | no source |
| ∂M_maj/∂β = 0, ∂M_1/∂β > 0, "Corollary 1" | plan:74; script:375 | No unit; no fit of the (μ, β) respondent law | no source |
| Cosine-NTK ρ 0.772, OOD +10.4 to +11.4 (SFT), 0 OOD and −0.7 to −1.1 on ARC-C, GPQA, GSM8K (GRPO) | plan:67, :68 | Not in irt4llm (signature T6, T8). The OOD set holds GSM8K | no source here; GSM8K part removed 2026-10-22 |
| p_SFT,j pass rates and M1a at T = 0.2 "ready today" | plan:7, :27, :120 | results/M1 holds jobs.txt only. M1-notes.md:52 says a_0 completed on the cluster, not copied | no source |
| "RL is pure gain modulation", "SFT acquires procedural schemas" | plan:66-68, :112; script:288-289 | No unit establishes either. The reorder split failed against the symmetric null (state.md:32) | no source |
| "accepted JEDM 2026 paper", "Nov 20 40th IRT Workshop" | plan:120, :124 | Not in irt4llm | no source |

## Withdrawn or narrowed

| Item | Location | Source | Verdict |
|---|---|---|---|
| GSM8K Corr = −0.22 | plan:17, :44; script:104 (F1 B box) | CLAUDE-HANDOFF.md:29; M1.md:16 fix (6) | withdrawn 2026-10-22, still on F1 |
| 4.96x RL amplification (4.962 against 1.356) of option-order noise | plan:100 | T9r.md:169; T9r/summary.md:7, :23. The values are the order noise of trained token-0 forced reads over the base read under T9r Rewrite 1, not K = 32 draws and not order variance over binomial variance. The scale reading of Rewrite 1 was withdrawn (state.md:25) | narrowed, value holds, reading withdrawn |
| Raw mode change 0.560 to 0.564 (SFT) against 0.398 to 0.436 (GRPO) read as SFT shifting procedural modes | plan:76 | Values hold (S1.md:83). The reorder verdict is withdrawn (state.md:32; S1.md:84), because raw change scales with spread. The values cover all 1,500 items, not procedural ones | withdrawn reading |
| Attribution 60/40 (accuracy) to 21/79 (θ), +0.058 / +0.039 and +0.164 / +0.620 | plan:24, :85-86, :128; script:433-436 | Q2g/s1/personfit.csv recomputes 59.5/40.5 and 20.9/79.1. S1 states only, seen. report-q1q2.md:79 withdraws the S1 reading of rule 2 as a finding. Parsed variant 47/53 and 15/85; SFT θ share per seed 0.141 to 0.255. On the 12 unseen Q1 finals, θ is sharper than accuracy against the base in 3 of 12 (Q2g/q1/precision.csv, all draws) | narrowed |
| θ "losing 0/3 on base vs SFT, winning 9/9 on SFT vs GRPO" | plan:89 | Q2g/s1/precision.csv, all variant. Holds on seen S1 pairs | narrowed, seen |
| lz(1) = +2.24 to lz(32) = −482 (S1) and −577 (Q1 pop0) "sign flip" | plan:25, :92-93, :128; script:529-533 | Values hold. Q2a/personfit.csv GRPO 2.03, 2.18, 2.52 (mean 2.24); Q2g/s1 −477, −498, −472 (mean −482.3); Q2g/q1 pop0 mean −576.5. The two lz are different statistics (one order-0 draw at the EAP θ, Q2a.md:46, against 32 trials at the Q2g θ), so the flip compares two estimators. Base and SFT are negative at both (−3.7, −4.3 to −8.4; −212, −218 to −231). report-q1q2.md:46 reads all trained models as misfit | narrowed |
| R² = 76.9% (r +0.8771, p 1.8e-5), R² 64.2% (r −0.8015, p 0.0003), r(a, b) −0.8841, r(a, ΔL_GRPO) +0.7464 (p 0.0014) | plan:20, :60, :67, :69, :127; script:242, :273, :280, :295 | All values recompute. n = 15 subject means of seen S1 states. Subject b_pop correlates −0.94 with the subject mean base log odds; after the base log odds is removed, the partial r of b is 0.17 (all draws) and 0.25 (parsed). Item level r(b, ΔL_SFT − ΔL_GRPO) is 0.42 on 1,472 items (R² 17.7%). The prior scaling route (τ_SFT < 1, τ_GRPO > 1) gives the same subject pattern. Unseen Q1 arms give r(a, ΔL) 0.64 to 0.72 (verify_master_plan.py Part 4), description only | narrowed |
| "SFT wins 5/6 procedural, GRPO wins 8/8 declarative" | plan:21, :67-68, :127; script:277-279 | Counts hold at a ±0.1 logit threshold. Statistics (−0.077) is a tie and Microeconomics (−0.084, a 0.61) is in neither group. The threshold and the group bounds 0.28 to 0.55 and 0.69 to 1.45 appear after the result | narrowed |
| "SFT overtakes on maj@16" (0.504 against 0.481) | plan:22, :76 | S1.md:78. Rule 3 fires in 2 of 3 seeds (ratios −2.77, −0.75, −2.58); on order 1 alone no rule fires | narrowed |
| "Algebraic identity M7", unweighted 1PL equals the ctt band | plan:110 | review-2026-10-19/M.md:97 proves it for the Rasch critic in expectation. pop0 uses fixed Fluid a_j (report-q1q2.md:13), so M7 does not cover pop0 | narrowed |
| Human against LLM b, r −0.013, ρ 0.089 | plan:17, :43; script:74 | closure.md:27 holds. closure.md:29 says the LLM b rest on the rejected 61-respondent calibration, so only the rank order reads. F1 A shows LLM b from about −1,570 to +920 | narrowed, Spearman only |
| Q2c n = 100 "0.063 against 0.050" | plan:101 | Q2c-notes.md:23 (0.0626, 0.0497) holds. In the same band IRT adaptive beats classical at n = 25 and 50 (0.064 against 0.098, 0.064 against 0.071) | holds, narrowed |

## Holds

| Item | Location | Source | Verdict |
|---|---|---|---|
| Q1 band 0.4685 to 0.4748 (grpo, ctt, pop0, pop1) | plan:26, :109; script:643 | report-q1q2.md:11-14; Q1.md:95 | holds |
| "0.479" as S1 GRPO mean@16 at 666 updates | plan:22, :109 | S1/stats.csv pooled, mean of 3 seeds 0.4790 | holds |
| mean@16 0.479 against 0.439; maj@16 0.504 / 0.555*, GRPO 0.481 / 0.490*; pass@16 0.966 against 0.656 | plan:22, :76, :117; script:325-344 | S1/stats.csv, S1/rewrite1_parsed.csv. Base pass@16 is 0.930, so SFT adds 0.036 over the base | holds |
| Entropy 0.963 (base), 0.281 (GRPO), 0.943 (SFT) | plan:75-76 | S1/stats.csv pooled | holds |
| maj@16 − mean@16 GRPO +0.002 / +0.011*; SFT +0.065 / +0.116* | plan:22, :75, :117 | S1/stats.csv, rewrite1_parsed.csv. The "Condorcet" name has no source | holds, label unsupported |
| GRPO zero_adv_share after update 83 63.1%, peak 84% to 91% | plan:26, :111; script:626 | Q1/traces/grpo_0..2.csv, means 0.638, 0.614, 0.642 (mean 0.631), maxima 0.906, 0.844, 0.906 | holds |
| stable_12 pop0 0.942, grpo 0.757, critics 0.866 to 0.942 | plan:26; script:638 | Q1/stats.csv pooled | holds |
| L0 = 78.98 | script:637 | W1/summary.json mean_tokens_mean 78.976 | holds |
| 0.382 to 0.455 to 0.474 by update 83 | plan:111 | report-q1q2.md:28 (seed 0) | holds, seed 0 only |
| Q2-A rule 1, 0/6 (S1) and 1/12 (Q1) | plan:48 | report-q1q2.md:43-44 | holds |
| 61/61 LLMs below −1.96 | plan:17, :43; script:80 | E1/llm_scores.csv zh_2pl, 61 of 61 below −1.96. The statistic is Zh (closure.md:25) | holds |
| 4,918 students, 61 LLMs, 102 leaderboard models | plan:6, :38 | closure.md:25; report-q1q2.md:16 | holds |
| 1.48M rollouts | plan:38 | S1 7 x 48,000 + Q1 24 x 48,000 = 1,488,000 | holds |
| Sociology 40% to 49% of CAT choices, mean a 1.63 to 2.08 | plan:25, :101 | review-2026-10-19/K.md:47-51 (S1 states, k = 1) | holds |
| Subject values Econometrics −0.575, Macro −0.192, Physics −0.035, Elem Math +1.491, HS Math +0.772, SFT +1.20 to +1.65 | plan:68, :117 | verify_master_plan.py Part 4; recomputed from Q2g/s1/residuals.csv | holds |
| Subject a 0.28 to 1.45, b ranges −0.99 to +0.72 and +0.25 to +2.30; "5x" (1.450 / 0.277 = 5.2) | plan:21, :61-62, :89 | W0/items_bank.csv subject means | holds |
| Free text r 0.289, σ_s 7.44 against σ_b 3.17, ratio 2.35, τ 0.68 | plan:52; script:152-153, :175-176 | Recomputed from T9g/s3/matched.csv per seed: r 0.275 to 0.305, σ_s 7.20 to 7.64, σ_b 3.17 to 3.19, ratio 2.26 to 2.39 (2.11 to 2.56 over clip eps), τ 0.62 to 0.73. No unit or Audit states these | holds, unaudited |
| Token-0 SFT r 0.822, σ ratio 0.523, τ 0.43; GRPO r 0.904, 2.091, 1.89; SFT R² 59.2% (seed 0) | script:124-131, :152-153 | verify_master_plan.py eps 1e-4, means of 3 seeds. The committed T9r τ differs by definition (SFT 0.342 to 0.585, rl 2.089 to 2.170, T9r/summary.md) | holds, unaudited |
| F5 SE ranges 8.9 to 10.5, 3.8 to 5.9, 2.9 to 5.1, 8.6 to 10.5 | script:446-450 | Q2g/s1/precision.csv | holds |
| W1 interior 2 to 6 of 8 51.7%, exterior 48.3% | script:672-682 | W1.md:51 (5,518 of 10,682) | holds |

## Counts

| Verdict | Rows |
|---|---|
| wrong | 21 |
| no source (14 in its table, 2 in the wrong table) | 16 |
| withdrawn or narrowed | 12 |
| holds | 20 |

## Figure issues

1. Research rule 25. F1 A is an agreement plot with one point per item, but it has no identity line and no printed slope (fitted slope −0.76). F1 B prints R² and no slope. F3 A plots 15 subject means, not items, with no printed slope. F5 B plots 5 groups of 3 subject means under an item axis label.
2. Mechanism stated as fact over 15 subject means or with no fit. F3 title "Neuro-Cognitive Double Dissociation"; F3 B legend "Procedural Schema" and "Neural Gain"; F4 title "Generative Respondent Duality", axes "Higher Neural Gain β_s" and "Modal Location Boundary"; F4 A and F8 B "Condorcet"; F5 "Theorem 1", "Mathematical Mechanism", "Procedural Schema", "Declarative Gain"; F6 "Theorem 2"; F7 "Base Modal Boundary"; F8 "Immovable Wall" and "Operator Composition Law".
3. F1 contradicts itself. The F1 B box says Corr = +0.04, and the panel's own fit gives R² 15.39% (r +0.39). The suptitle says "orthogonal". The box keeps the withdrawn GSM8K −0.22.
4. F1 A has "J = 600" and "600 items" on 704 points. The y axis runs to about −1,570 and +920, so a few degenerate LLM b values set the Pearson.
5. F2 A plots seed 0 and labels it with the 3-seed τ_SFT 0.43, while its own fitted line gives 0.54. The GRPO points form a straight diagonal band from items clipped at p = 1 − 1e-4, so the fit and the R² depend on the clip.
6. F2 B bar "Jeffreys K = 32 (S1 GRPO)" 0.91, 1.52, 1.38 does not reproduce (0.73 to 0.75, 1.91 to 1.99, 1.43 to 1.46).
7. F4 B title says "19 post-trained states"; one of the 19 points is the base. Its band label (0.470 to 0.489) disagrees with the drawn band (0.466 to 0.491), and a Q1 ctt point sits at 0.496.
8. F4 A shows base pass@16 0.930 beside SFT 0.966. The plan text reads the SFT value as a new ceiling, and the base is 0.036 below it.
9. F6 A title calls a sign flip, but the panel shows lz(32) only, and base and SFT are negative at k = 1 too. F6 B puts the red label "0.064 (bias +0.056)" for adaptive CAT at n = 50 below the green irt_random point.
10. F7 A title "All 15 runs hit the 0.469 to 0.479 wall by update 83" shows 4 seed-0 traces, and every trace sits above the "Base Modal Boundary" line at 0.414. F7 B draws a 35.0% line with no source.
11. F8 B draws "Unit M1" bars (0.479, 0.504, 0.555) for a unit with no result, and its "15 runs" bars mix 15-run and 3-run values. The F8 suptitle names p_SFT,j, and panel A plots base pass rates from W1.
12. The script writes to D:\surrogates and reads C:\Users\steph (script:24-25), so it does not run on this PC as committed.

## What this check does not test

It does not open the signature checkout beyond its findings.md headings, the M1 cluster files, or any number of the MASTER-PLAN that the slide plan and the script do not repeat. The partial correlation and the item-level correlation above are descriptive recomputations, not a test with a null.
