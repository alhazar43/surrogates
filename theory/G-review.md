# Review of theory/G.md

Written by the RevG checker on 19 October 2026 for unit notes/units/RevG.md. theory/G.md was read line by line and not edited. Each derivation was redone by hand or in a shell, the calibration script was rerun with python -I, every cited source was opened, and the two claims about the peer design (R5, R6) were checked against C:/Users/YuanW/PLMS/irt4llm/notes/blueprint.md as it stands now, which was not changed. Shell checks ran from a scratch folder outside the project (checks.py, rerun.json); no file of the project was written except this one and the two unit files.

## Critical errors

C1. The second tier 1 prediction and the tier 3 prediction rest on an unstated sign condition (G.md:87, 153, 157). Under pure sharpening, l_j = beta delta_j, F = mean_j sigmoid(beta delta_j), and the bank gain from one score-function step on item x is

```
V_x = eta c'_B delta_x p_x (1 - p_x),     c'_B = dF/dbeta = (1/|B|) sum_j delta_j p_j (1 - p_j)
```

R4 derives only the item factor delta_x p_x (1 - p_x). The bank factor c'_B has the sign of the p(1-p)-weighted mean logit of the bank. With bank logits drawn from N(mu, 1.5^2) at beta = 1, c'_B is -0.018 at a bank accuracy of 0.465 and +0.019 at 0.536 (shell). If c'_B < 0, a sharpening learner's bank gain is negative for every training item above 0.5 and peaks at p = 0.176, so "larger in the bins above 0.5 than at 0.5" fails under sharpening and the refutation rule reads that failure as evidence against sharpening. Wilson et al. 2019 do not face this, since every item there has error F(-beta Delta) of at most 0.5 (Methods Eq. 24). The binary logistic form centered at 0.5 is also assumed for four-option MMLU items without being stated. Corrected statement. A sharpening learner's gain curve is zero at 0.5 and peaks at 0.824 if the bank's weighted mean logit is positive, and at 0.176 if it is negative; the prediction must state the bank accuracy of the base model and the condition.

C2. R9's conclusion and the tier 2 prediction contradict R9's own model (G.md:111, 155). In l_jt = a_j theta_t - c_j the logit velocity of item j is a_j d theta/dt. With theta known, the velocity estimate is a_j times a known factor and has the same relative standard error as a_hat_j, 0.61 at k = 16, C = 10, Delta = 1. Without theta known, a two-checkpoint logit difference has SE sqrt(2 / (k p (1 - p))) = 0.79 logit at k = 16, p (1 - p) = 0.2. So the argument "a_hat does not beat p (1 - p) since its SE exceeds the spread of a" applies with equal force to the velocity predictor that tier 2 predicts will win. The refutation rule then reads a velocity failure as "the kernel column turns over" when R9 predicts the same failure from sampling noise alone. Velocity is also the kernel row contracted with the training distribution, not "the kernel column itself". Corrected statement. At the answer counts of R9 neither predictor is identified; tier 2 must state the answer counts at which velocity is estimated and attribute a failure to turnover only above that noise floor.

C3. R6 is false against the blueprint as it stands (G.md:91 to 97, also 65 and 184). blueprint.md:29 now reads "A_j = 2 a_j^gamma (r - p_hat_j), with no divisor and no clip". blueprint.md:32 states the gamma = 1 weight a_j^2 p_j (1 - p_j), which is the item information under one ability and a = alpha. blueprint.md:34 records that the divisor was removed after an external review that found the same a_j^2 sqrt(p_j (1 - p_j)) weight R6 derives. R6's derivation is correct for the v2.1 form G.md read (reading.md row "Peer blueprint"), but blueprint.md was saved at 11:08:17 and G.md at 11:14:26 on 19 October 2026, so the claim was stale when G.md closed. The statements "not the information that blueprint section 3 states", "the clip of p_hat to [0.05, 0.95] bounds that factor at 4.6", "blueprint critic arms" in the R1 table and "R6 corrects its stated weight" in section 6 all fail now. Corrected statement. Under the current blueprint the expected step of arm I1 is 2 a_j alpha_j p_j (1 - p_j) v, the information when a = alpha; R6 should be withdrawn or kept only as the record of v2.1.

## Major errors

M1. The reason given for the 0.105 false positive rate of the 10-seed t test has the wrong size (G.md:147). It compares the pool SD 0.0024 with the within-pool SE of the exact difference, 0.0005. The t test sees the sampled bank read, whose seed SE is sqrt(0.0016^2 + 0.0081^2) / sqrt(10) = 0.0026, so the ratio is 0.91, not 5. A simulation with ratio 0.91 gives a false positive rate of 0.104, matching the file; a ratio of 4.8 would give 0.36. The mechanism (R7) holds; the comparison does not.

M2. The column "SD of the pool-conditional effect" is the SD over experiments of the 10-seed mean, which includes seed noise (script:113; G.md:134, 140, 141). Net of seed noise it is 0.0024 for N2 and 0.0025 for P1 with run noise 0.15, not 0.0035 and 0.0036. The rows without run noise are near their net values (N2 0.00238 against 0.0024).

M3. Two permuted weightings cannot reject at 0.05 (G.md:194, 210). A permutation test whose unit is the weighting has a smallest p value of 1 / (K + 1), 0.33 at K = 2. K = 19 is the smallest for 0.05, which is 20 arms by 3 seeds, 60 runs, 228 to 318 h of runs at 3.8 to 5.3 h each, against the 34 to 48 h stated.

M4. The tier 1 refutation calls on "the calibrated rule" (G.md:153), but the permutation rule is "not run in section 4" (G.md:99) and "derived, not yet computed" (G.md:210). The project requirement is a calibration before use.

M5. "GRPO in expectation" does not rank by alpha sqrt(p (1 - p)) at the recipe's group of 5 (G.md:65). The expected GRPO step per answer is E[sqrt(q (1 - q))] grad l with q = k/5, k ~ Binomial(5, p). Its ratio to sqrt(p (1 - p)) is 0.57 at p = 0.1, 0.73 at 0.2 and 0.86 at 0.5, so its shape is close to p (1 - p) (0.68 against 0.64 at p = 0.2, relative to p = 0.5).

M6. R5's variance statement is narrowed (G.md:89). p (1 - p) + (p - b)^2 is the second moment of the advantage, not its variance and not the gradient variance. The constant baseline that minimizes gradient variance is E[r |s|^2] / E[|s|^2], with s the score vector, which is p only if |s|^2 is uncorrelated with r. On the one-bit learner of section 4 it is 1 - p (0.700 at p = 0.3, shell). So "C against I0 asks which estimator of p is closer" holds only under that uncorrelation.

M7. G.md names no optimizer (A1, A2, R1, R4, G.md:153), while panel S1.3 (panel-2026-10-19.md:1429) states that no gain shape claim stands without the optimizer named. The derivations assume a plain gradient step; the tier 1 recipe is GRPO with the Princeton optimizer.

M8. The second tier 1 reading has no estimator (G.md:153, 204). The bank gain is measured per run, not per training item; attributing it to items binned by first-use pass rate needs a stated regression or a design. The band [0.2, 0.8] on base pass rate also excludes the 0.824 peak, so "a peak near 0.82" is not observable in that design.

M9. Choice 1 option B and the pick C rely on peer arms that no longer exist (G.md:181, 194, 196). The band arm C is in the archived v2.0 (archive/blueprint-v2.0-2026-10-19.md:32). In the current blueprint arm C is a critic over all prompts with 160 prompts by 1 answer (blueprint.md:43), so the Q1 seed spread is for different arms and is not supplied "at no extra cost" for G's design.

## Minor errors

m1. A leave-in group mean alone changes the expected step by the constant (G - 1)/G; the prompt-dependent factor comes from the SD divisor (G.md:40). The blueprint lists four differences for arm G under "three things" (blueprint.md:55); G.md repeats "three".
m2. c_B > 0 needs alpha_j > 0 on the bank, unstated (G.md:57).
m3. R2 treats balanced filtering as a ranking; it is a uniform draw within a band. The "only if" list omits the case of constant alpha and varying a (G.md:71).
m4. R3's "a loss through fewer distinct items" is not derived and the calibration shows none, -0.00002 (SE 0.00003) in N1 and -0.00002 (SE 0.00005) in N2 (G.md:79).
m5. R8's second-order term is not sign-definite. K(x, y) enters through the change of p_y with the factor (1 - 2 p_y); a simultaneous batch gives K(x, j) K(y, j) through the bank (G.md:101). Interference loss is Huang et al. Section 3.2, not Theorem 3.
m6. R9 uses the continuous uniform variance. C evenly spaced checkpoints give the factor (C + 1)/(C - 1), so the SEs are 0.554, 1.108 and 0.206 instead of 0.612, 1.225 and 0.217 (G.md:103 to 109). The conclusion of the table is unchanged.
m7. "Zero by construction" holds at first order only; arm I draws fewer distinct items and the step is nonlinear in theta. N2 with run noise shows +0.0003 (SE 0.00014), 2.2 SE above zero, without comment (G.md:123, 140).
m8. "It is strict because it needs two of three seeds and the seed mean" (G.md:147). From the JSON SEs, the per-seed 2 SE test passes with probability near 0.59 at +0.018 and the seed-mean test near 0.96, so the per-seed part binds (approximation, not simulated).
m9. "The false positive rate of BP rises as the run SD grows past the item SE" (G.md:149) is stated, not shown; no calibrated row has run SD above the item SE (largest 0.0087 against 0.0081).
m10. The script docstring says "a few seconds" (calibrate_tier1.py:26); the rerun took 4 min 12 s.
m11. Tier 3's clause "the item of maximum Fisher information is not the item of maximum gain" also holds for a location learner whenever alpha varies (R1), so it does not separate location from precision (G.md:157).
m12. "Post-training mostly sharpens (panel S1.4)" (G.md:153). S1.4 states that the scale term on the GRPO states equals a decoding temperature (panel-2026-10-19.md:1430); "mostly" is not in it.
m13. "Foster and Foerster 2025" matches neither record. arXiv 2502.12272 lists Foster, Sims, Forkel, Fellows, Foerster; NeurIPS 2025 LILO lists Foster, Sims, Forkel, Foerster (G.md:67, 169).
m14. reading.md cites Mindermann et al. Eq. 6 for the reducible holdout loss; it is Eq. (3), from the selection function Eq. (2).
m15. The research program's tier 3 claim "practice on high information items raises later learning more" (research-program.md, section 4 G) is the opposite of G's tier 3 prediction, and proposed edit 4 (G.md:217) does not reword it.

## Full table

| # | item | line | outcome | evidence | severity |
|---|---|---|---|---|---|
| 1 | Object signatures, V, G_T | 7-18 | holds | definitions are typed and consistent | none |
| 2 | First-order V and grad F | 27-28 | holds | differentiating mean_j sigmoid(l_j) | none |
| 3 | Leave-in group mean gives a prompt-dependent factor | 40 | narrowed | sum_i (r_i - rbar)(r_i - p) = k(G - k)/G; the leave-in mean alone gives (G - 1)/G | minor |
| 4 | Blueprint names three differences for arm G | 40 | holds | blueprint.md:55 says three and lists four | minor |
| 5 | R1 V = eta c_B alpha p (1 - p), c_B > 0 | 57 | holds | rederived under A1 to A3; c_B > 0 needs alpha_j > 0, unstated | minor |
| 6 | R1 SD-divided step equals GRPO in expectation | 65 | contradicted | G = 5, E sqrt(q(1-q)) / sqrt(p(1-p)) = 0.57, 0.73, 0.86 at p = 0.1, 0.2, 0.5 (checks.py) | major |
| 7 | R1 information row, Hofmann Eq. 4 and 5 | 66 | holds | arXiv 2509.11106 HTML, Eq. 4 a^2 p(1-p), Eq. 5 argmax information | none |
| 8 | R1 learnability and band rows | 67 | holds | Foster et al. abstract "variance of success"; Bae et al. Section 3.4 band 0.2 to 0.8 | none |
| 9 | Value is information divided by a | 69 | holds | alpha p(1-p) = a^2 p(1-p) / a at a = alpha | none |
| 10 | R2 four rules rank by p(1 - p) | 71 | narrowed | a band is a uniform draw, not a ranking; constant alpha with varying a omitted | minor |
| 11 | R2 restates blueprint section 1 | 71 | holds | blueprint.md:14 | none |
| 12 | R3 covariance condition and alpha_eff | 76 | holds | E_w V = mean V + Cov(w, V)/mean w; V_x = eta p_x(1-p_x) <grad F, grad l_x> | none |
| 13 | R3 loss through fewer distinct items | 79 | narrowed | not derived; theory_g_calibration.json N1 -0.00002 (SE 0.00003), N2 -0.00002 (SE 0.00005) | minor |
| 14 | R4 l tanh(l/2) = 1, l* = 1.5434, p* = 0.8240 | 84 | holds | brentq 1.54340, grid argmax 1.54338, sigmoid 0.82396 (checks.py) | none |
| 15 | R4 Wilson 0.1587, 0.1839, logistic between | 87 | holds | PMC6831579 Eq. 30, 31; no logistic case; logistic error 0.1760 | none |
| 16 | R4 bank gain peaks near 0.82 under sharpening; tier 1 second and tier 3 predictions | 87, 153, 157 | contradicted | V_x = eta c'_B delta_x p_x(1-p_x); c'_B = -0.018 at bank accuracy 0.465, +0.019 at 0.536 (checks.py) | critical |
| 17 | R4 differs from the logistic peak of S1.2 | 87 | holds | panel-2026-10-19.md:1428 | none |
| 18 | R5 expected step unchanged by an answer-independent b | 89 | holds | E[grad log pi] = 0 | none |
| 19 | R5 variance p(1-p) + (p - b)^2; critic helps only through MSE of b | 89 | narrowed | second moment, not variance; one-bit optimal b = 1 - p, 0.700 at p = 0.3 (checks.py) | major |
| 20 | R5 arms C and I0 differ only in b | 89 | holds | blueprint.md:55 now; false for v2.1 per blueprint.md:34 | none |
| 21 | R5 zero advantage 0.328, 0.591 at G = 5 | 89 | holds | 0.2^5 + 0.8^5 = 0.328; 0.1^5 + 0.9^5 = 0.5905 | none |
| 22 | R6 I1 weight a^2 sqrt(p(1-p)), not the stated information; clip; R6 corrects the blueprint | 91-97, 65, 184 | contradicted | blueprint.md:29 no divisor, no clip; :32 a^2 p(1-p); :34 divisor removed; blueprint.md saved 11:08:17, G.md 11:14:26 | critical |
| 23 | R6 clip factor 4.6 | 97 | holds | 1/sqrt(0.0475) = 4.588; clip absent from current blueprint | none |
| 24 | R7 one permuted arm insufficient; weighting as unit | 99 | holds | arm I keeps its subset; weightings exchangeable under the null | none |
| 25 | R8 second-order term with K(x, y), negative entries lower gain | 101 | narrowed | term not sign-definite; K(x, y) enters with (1 - 2 p_y) | minor |
| 26 | R8 Huang Theorem 3 interference | 101 | narrowed | arXiv 2605.29548v2 Theorem 3 utility order; interference in Section 3.2 | minor |
| 27 | R9 information k C 0.2 Delta^2 / 12 and table | 103-109 | holds | arithmetic exact; evenly spaced points give SE 0.554, 1.108, 0.206 (checks.py) | minor |
| 28 | R9 conclusion, velocity over a_hat; tier 2 prediction | 111, 155 | contradicted | velocity = a_j dtheta/dt, same relative SE as a_hat; logit difference SE 0.79 at k = 16 | critical |
| 29 | R10 Park Definition 3, Section 4.1 | 113 | holds | arXiv 2406.19370v4 HTML, Definition 3, Section 4.1 heading | none |
| 30 | R10 information as sensitivity | 113 | holds | I = (dp/dtheta)^2 / (p(1-p)) = a^2 p(1-p) | none |
| 31 | Script output equals the result file | 145 | holds | rerun python -I, JSON identical after key sort, 4 min 12 s | none |
| 32 | Script step and its expectation | 119 | holds | calibrate_tier1.py:88; E[(r - ph)(r - p)] alpha = alpha p(1-p) | none |
| 33 | Script BP rule equals the blueprint rule | 131 | holds | calibrate_tier1.py:100-103; blueprint.md:48 | none |
| 34 | Table values against the JSON | 134-141 | holds | every cell rounds from theory_g_calibration.json | none |
| 35 | Column SD of the pool-conditional effect | 134-141 | contradicted | calibrate_tier1.py:113 includes seed noise; net 0.0024, 0.0025 in run-noise rows, stated 0.0035, 0.0036 | major |
| 36 | N1, N2 zero by construction | 123-124, 140 | narrowed | first order only; N2 run noise +0.0003 (SE 0.00014) | minor |
| 37 | Binomial SE 0.011; 0.105 is 5 SE above 0.05 | 143, 147 | holds | sqrt(0.05 x 0.95 / 400) = 0.0109; 0.055 / 0.0109 = 5.05 | none |
| 38 | Bank accuracy 0.659 to 0.670; band 0.585 to 0.594 | 143 | holds | JSON bank_acc_B_mean, band_share_mean | none |
| 39 | Pool SD 0.0024 about 5 times SE 0.0005 explains 0.105 | 147 | contradicted | read SE 0.0026, ratio 0.91, simulated rate 0.104; ratio 4.8 gives 0.36 | major |
| 40 | BP strict because of 2 of 3 seeds and the seed mean | 147 | narrowed | per-seed test binds, about 0.59 per seed against 0.96 for the seed mean (approximation) | minor |
| 41 | BP false positive rate rises past item SE | 149 | narrowed | no row with run SD above item SE; largest 0.0087 against 0.0081 | minor |
| 42 | Script runtime | script:26, 145 | contradicted | docstring a few seconds; measured 4 min 12 s | minor |
| 43 | Tier 1 refutation under the calibrated rule | 153 | contradicted | G.md:99 not run; G.md:210 not yet computed; TG Result requirement not met | major |
| 44 | K = 2 permuted weightings, 34 to 48 h | 194, 210 | contradicted | smallest p = 1/(K + 1) = 0.33; K = 19 needs 60 runs, 228 to 318 h | major |
| 45 | Tier 1 second reading measurable | 153, 204 | unchecked | no estimator for gain per training item; band [0.2, 0.8] excludes 0.824 | major |
| 46 | Optimizer assumption | 24-50, 153 | contradicted | none named; panel-2026-10-19.md:1429 requires it | major |
| 47 | Choice 1 option B and pick C use the band arm C | 181, 194, 196 | contradicted | blueprint.md:43 arm C is a critic; band arm in archive v2.0:32 | major |
| 48 | I1 equals drawing proportional to a_j in expectation | 184 | holds | blueprint.md:45 a_j scaled to mean 1; first order | none |
| 49 | C against I0 tests the critic, not a tier of G | 184 | holds | blueprint.md:31, 55 | none |
| 50 | Tier 3 information clause separates learners | 157 | narrowed | also holds under location with varying alpha (R1) | minor |
| 51 | Post-training mostly sharpens, S1.4 | 153 | narrowed | panel-2026-10-19.md:1430 decoding temperature; no "mostly" | minor |
| 52 | Consistency with panel S11 and S14 | 32, 42 | holds | panel-2026-10-19.md:2857, 2877 | none |
| 53 | Option B cost arithmetic | 194 | holds | 9 x 3.8 = 34.2 h, 9 x 5.3 = 47.7 h; blueprint.md:97 | none |
| 54 | Bae et al. abstract against Proposition 3.1 | 176 | holds | arXiv 2504.03380v2 HTML, abstract and Proposition 3.1 | none |
| 55 | Weinshall et al. abstract against Theorems 1 and 2 | 176 | holds | arXiv 1802.03796v4 PDF, abstract "increasing", Theorem 1 "decreasing", Theorem 2 "increasing with the loss" | none |
| 56 | Clement et al. pooled tests, four groups, randomization not stated | 173, 176, 182 | holds | arXiv 1310.3174v2 PDF, Section 6, Eq. 1 | none |
| 57 | Mindermann et al. Section 3 | 167 | holds | arXiv 2206.07137v3 PDF Eq. (1) to (3); reading.md says Eq. 6 | minor |
| 58 | Huang et al. Section 3 setup | 179 | holds | arXiv 2605.29548v2 HTML, K = 32, power law, width N, Adam | none |
| 59 | Hofmann et al. 2PL, 102 models, MMLU | 171 | holds | arXiv 2509.11106 HTML Section 3 and 4.1 | none |
| 60 | Foster and Foerster 2025 record | 67, 169 | narrowed | arXiv lists five authors, NeurIPS 2025 lists four | minor |
| 61 | Wang et al. 2020 measurement objective | 174 | holds | arXiv 2007.12061 abstract, third task predicts the student's answers | none |
| 62 | Park et al. 2024 record | 165 | holds | arXiv 2406.19370, NeurIPS 2024 | none |
| 63 | Wilson et al. 2019 record | 166 | holds | Nat. Commun. 10, 4646, PMC6831579 | none |
| 64 | Consistency with research program tier 3 | 157, 217 | contradicted | research-program.md section 4 G tier 3 claim is the opposite; edit 4 leaves it | minor |

Counts. 64 rows. 3 critical, 9 major (rows 6, 19, 35, 39, 43, 44, 45, 46, 47), 17 minor. 39 rows hold, 12 narrowed, 12 contradicted, 1 unchecked. Rows marked holds with a minor severity carry a defect beside a correct main claim.

What this review does not show. It does not show whether surrogate G is worth pursuing. It does not check goodfire-reading.md section 7 or three-questions.md Part E, which G.md names as inputs but RevG does not list. Sources were opened through arXiv HTML, extracted PDF text and Europe PMC XML; pages beyond the first 100k characters of the Bae, Park, Huang and Wilson texts were not read.
