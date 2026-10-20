# G. Guided learning, the theory end

Written by the TG builder on 19 October 2026 for unit notes/units/TG.md, and revised in fix attempt 1 after the review theory/G-review.md (fix log at the end). The inputs are the research program (program/research-program.md, section 4 G), goodfire-reading.md section 7, the panel syntheses S1 to S15 and three-questions.md Part E. Rungs are idea, derivation, checked numerically, source. Nothing here is trained or fitted on real data.

## 1. The object

```
learner state        phi_t in Phi                    (weights of a model; a student's knowledge)
pool of experience   X, finite                       (training prompts; practice items)
update               U : Phi x X x Xi -> Phi          (one step on item x; Xi = sampled answers and feedback)
target               F : Phi -> [0, 1]                (mean accuracy on a fixed bank B, B and X disjoint)
measured state       m : Phi -> M                     (answers so far, an ability estimate theta_hat, pass rates p_hat)
item side            s : X -> S                       (difficulty b, discrimination a; from a population or online)

value of item x      V(phi, x) = E_Xi[ F(U(phi, x, Xi)) ] - F(phi)
selection rule       pi : M x S^X -> Delta(X),  cost c answers per step
guidance gain        G_T(pi, pi_0) = E[ F(phi_T^pi) ] - E[ F(phi_T^pi_0) ]   at equal c and T
```

G asks whether a rule pi that reads the measured state m and the item side s has G_T(pi, pi_band) > 0, where pi_band reads only each item's own pass rate.

It answers one part of "what is learning". It asks whether the state a measurement shows tells which experience changes the learner most. It leaves what the change installed (surrogates R and E), whether a calibration made before the change still reads the learner after it (M), how the learner solves (P), and the student's inner process (S).

The first-order form carries every derivation below. For a step U = phi + eta P g_x(phi, Xi), with step size eta, update direction g_x and a preconditioner P held fixed over the step (P = I for plain gradient descent),

```
V(phi, x)  = eta < grad F(phi), P E g_x(phi) > + O(eta^2)
grad F     = (1/|B|) sum_j p_j (1 - p_j) grad l_j          (l_j the logit of a right answer on bank item j)
K_P(x, j)  = < grad l_x, P grad l_j >                       (the logit kernel of panel B19 and B25; K = K_I)
```

V is one column of the learner's kernel, weighted by the bank. That column is the object the panel found under the response to one conflict case (S11); G reads the same column as a choice of experience.

## 2. The assumptions

Each assumption is listed with what fails with it.

A1. First order. One step is small, so V is the inner product above. If steps are large or the selected items interact within a batch, the value of a batch is not the sum of item values, and every ranking below can fail (result R8).

A2. The expected step is the gradient of the pass rate. For a score-function (REINFORCE) step with a baseline that depends only on the prompt, E g_x = grad p_x. A group mean that includes the scored answer changes the expected step by the constant (G - 1)/G for a group of G answers. A division by the group SD changes it by a factor that depends on the prompt's pass rate (R1). Blueprint section 3 lists four differences of arm G from the critics under the words "three things" (blueprint.md:55). Under a prompt-dependent factor the rankings of R1 change by that factor.

A3. One factor along training. The logit gradients share one direction, grad l_x = alpha_x v, so K = alpha alpha^T |v|^2. Here alpha_x is the learner's own discrimination along the direction training moves it, with alpha_x > 0 on the bank. If training moves several directions (format and content, panel S14), the value of an item depends on its alignment with the bank, not on one number, and R1 and R3 fail.

A4. The population axis is the training axis. A 2PL fit across many models gives a_x, the change of l_x per unit of the population ability. a_x predicts alpha_x only if training moves the learner along the population ability axis. If training changes the scale of the logits (panel S1.4 finds a scale term on the GRPO states that a decoding temperature reproduces) or moves format, a_x carries no information about value (R3, R4).

A5. Training moves location, not precision, on a binary logistic item curve. The logit of item x changes by a shift of theta. If training changes precision (p_x = sigmoid(beta delta_x), beta rises), the gain curve moves away from p = 0.5 (R4). A four-option item with a guessing floor moves both curves (R4).

A6. The pool is fixed and the target is disjoint. Gains are read on bank B, never on the selected items. If the target overlaps the selected items, the selection effect and regression to the mean of a noisy p_hat enter the reading.

A7. Exogenous variation (tier 3 only). Which item a student meets is independent of the student's unmeasured state, given the measured history. If a tutor or teacher chooses by state, the item and the later gain share a cause and no comparison of schedules identifies V.

A8. The optimizer. R1 to R5 hold for a plain gradient step and for a preconditioner P fixed over the window read, with K replaced by K_P. An adaptive optimizer such as Adam sets P from recent gradients, so the drawn items change P, and the derivations below are not made for it. Panel S1.3 states that no gain-shape claim stands without the optimizer named (panel-2026-10-19.md:1429). The blueprint names no optimizer for the Princeton recipe; the tier 1 predictions are read only after it is read from that code.

## 3. What can be derived

R1. Four rankings under one factor (derivation). Under A1 to A3 and A8, with a plain score-function step,

```
V(phi, x) = eta c_B alpha_x p_x (1 - p_x),      c_B = (|v|^2/|B|) sum_j alpha_j p_j (1 - p_j) > 0   (alpha_j > 0)
```

so items rank by alpha_x p_x (1 - p_x). The rules in use rank by other products.

| rule | ranks by | source |
|---|---|---|
| value, plain score-function step | alpha p (1 - p) | this derivation |
| value, step divided by the binomial SD at the true p | alpha sqrt(p (1 - p)) | this derivation |
| value, GRPO, group of G, group mean and group SD | alpha E[sqrt(q (1 - q))], q = k/G, k ~ Binomial(G, p) | this derivation |
| maximum Fisher information (Fluid, CAT) | a^2 p (1 - p) | Hofmann et al. 2025, Eq. 4 and 5 |
| learnability (LILO), balanced filtering | p (1 - p); a uniform draw within a band symmetric about 0.5 | Foster et al. 2025 abstract; Bae et al. 2025 Section 3.4 |

The GRPO row follows from E[s | r = 1] = grad p / p and E[s | r = 0] = -grad p / (1 - p), with s the score. Given k right answers of G, the summed step is G sqrt(q (1 - q)) grad p / (p (1 - p)), and grad p = p (1 - p) alpha v. At G = 5 its shape relative to p = 0.5 is 0.40, 0.68 and 0.86 at p = 0.1, 0.2 and 0.3, against 0.36, 0.64 and 0.84 for p (1 - p) and 0.60, 0.80 and 0.92 for sqrt(p (1 - p)) (src/surrogates/theory_g/derived_numbers.py). So GRPO at G = 5 ranks close to p (1 - p), not to sqrt(p (1 - p)). A sample SD with ddof 1 changes the step by a constant and not its shape.

Even when a = alpha, maximum information is not maximum value. Value is information divided by a, so information favors high-discrimination items more than their value warrants. Information is the value of an answer for measuring theta; V is the value of a step for changing F. The two are different objects that share a factor.

R2. Where the rules cannot be told apart (derivation). If alpha is constant (the learner is Rasch along training) and a is constant, value, information and learnability rank by p (1 - p), and a band symmetric about 0.5 draws uniformly from the items of highest p (1 - p). The rules then choose from the same items, and any outcome difference between them comes from noise in the estimate of p, not from item response theory. This restates blueprint section 1 for training. They come apart in three cases. Alpha varies across items within the band. Alpha is constant and a varies, so information ranks by a, which carries no value for this learner. Or a comes from other respondents at no sampling cost to this learner.

R3. When the population item side helps (derivation). Within a band, choosing items with weight w_x instead of uniformly raises the summed first-order value if and only if

```
Cov_band( w_x , alpha_x^eff p_x (1 - p_x) ) > 0,     alpha_x^eff = sum_j p_j (1 - p_j) K(x, j) / (|B| |grad F|)
```

alpha^eff is the learner's own alignment of item x with the bank. With w = a^2 from a population fit, IRT helps training only if population discrimination predicts this learner's kernel alignment. Under A3 and A4 it does; under a multi-factor kernel or a training direction away from the population axis it need not. A weight that carries no information gives zero gain at first order in expectation over pools. The calibration finds no loss from the fewer distinct items such a weight draws (N1 -0.00002, SE 0.00003; N2 -0.00002, SE 0.00005).

R4. Shift against sharpen (derivation, checked numerically). If training changes location, a step on item x changes F by eta c_B alpha_x p_x (1 - p_x) (R1), largest at p_x = 0.5. If training changes precision, l_j = beta delta_j on every item, a score-function step on x moves beta by eta delta_x p_x (1 - p_x) in expectation, and

```
V_x  = eta c'_B delta_x p_x (1 - p_x),        c'_B = dF/dbeta = (1/|B|) sum_j delta_j p_j (1 - p_j)
item factor (l_x / beta) p_x (1 - p_x),  extreme where l tanh(l/2) = 1,  l* = +-1.5434,  p* = 0.8240 or 0.1760
```

(src/surrogates/theory_g/calibrate_tier1.py, precision_optimum). The item factor is odd in l and zero at p = 0.5. The bank factor c'_B has the sign of the p (1 - p)-weighted mean logit of the bank. So a sharpening learner's gain is zero at 0.5, peaks at 0.824 if c'_B > 0 and at 0.176 if c'_B < 0, and is negative on the other side of 0.5. With bank logits N(mu, 1.5^2) at beta = 1, c'_B is -0.018 at bank accuracy 0.465 and +0.019 at 0.535 (derived_numbers.py). Wilson et al. 2019 meet only the positive case, since every item there has error at most 0.5. They derive error 0.1587 for Gaussian noise and 0.1839 for Laplace noise; the logistic error 0.1760 lies between them.

The curve assumes a binary logistic item centered at 0.5. A four-option item with guessing floor g = 0.25, p = g + (1 - g) sigmoid(l), has its location peak and its sharpening zero at p = 0.625, and its sharpening peaks at 0.868 and 0.382 (derived_numbers.py), if training leaves the floor unchanged.

The two explanations are told apart by the bank gain as a function of the base pass rate of the items trained on. A location learner peaks at the midpoint of the item curve. A sharpening learner is zero there and peaks on the side of the bank's weighted mean logit. The sign of c'_B comes from the base model's bank read (at beta = 1, c'_B = mean_j l_j p_j (1 - p_j), l_j the logit of the base pass rate), so it is known before training. This is not the logistic peak of panel S1.2. That peak is the gain on an item under a constant shift; R4 is the gain on a fixed bank from training on an item.

R5. The critic changes the spread of the step, not its expectation (derivation). With one answer per prompt and a baseline b that does not depend on the answer, E[(r - b) s] = grad p_x for every b, with s = grad log pi(y). The second moment of the advantage is p (1 - p) + (p - b)^2, smallest at b = p. The gradient variance is E[(r - b)^2 |s|^2] - |grad p_x|^2, and the constant b that minimizes it is E[r |s|^2] / E[|s|^2]. This equals p only if |s|^2 is uncorrelated with r. On a one-bit learner (two answers, p = sigmoid(l), |s|^2 = (1 - p)^2 or p^2) it is 1 - p, 0.700 at p = 0.3. Blueprint arms C and I0 differ only in b (blueprint.md:55). Their comparison asks which estimator of p is closer only if |s|^2 and r are uncorrelated, and in no case asks which items carry signal. A GRPO group of G answers gives zero advantage with probability p^G + (1 - p)^G, which is 0.328 at p = 0.2 or 0.8 and 0.591 at p = 0.1 or 0.9 for G = 5.

R6. Withdrawn; kept as the record of blueprint v2.1. Attempt 0 derived that the v2.1 advantage of arm I1, divided by sqrt(p_hat (1 - p_hat)), gives the weight a_j^2 sqrt(p_j (1 - p_j)) and not the stated information. The blueprint had been rewritten before G.md closed (blueprint.md:29 to 34; blueprint.md saved 11:08:17, G.md 11:14:26, 19 October 2026), after an external review found the same weight. Under the current blueprint A_j = 2 a_j (r - p_hat_j), with no divisor and no clip. The expected step of I1 is 2 a_j alpha_j p_j (1 - p_j) v, the information a_j^2 p_j (1 - p_j) when a = alpha, as blueprint.md:32 states. No claim of G rests on R6.

R7. A fixed pool makes a fixed weight a treatment (checked numerically). With one pool, any fixed weighting draws a particular subset of items, so its effect given the pool is not zero even when the weight carries no information; only its average over pools is zero. A test across seeds of one pool asks about the pool-conditional effect (section 4 gives its size). One permuted arm does not remove it, since arm I keeps its own fixed subset. The null that matches it is a set of K fixed weightings, each the weights of arm I permuted across the band, each trained as arm I is. With Y the bank read of a weighting,

```
p = (1 + #{k : Y_k >= Y_I}) / (K + 1),     declare "I helps" if p <= 0.05
```

If the weight carries no information about the learner's items, the K + 1 weightings are exchangeable given the pool, so the size is at most 0.05 by construction. The smallest p is 1/(K + 1), so K = 19 is the smallest K that can reach 0.05; arm I must then have the highest read of 20. If a_pop correlates with pass rate inside the band, a permutation across the band also changes the pass-rate mix, and a permutation within bins of p_hat is the stricter null (not calibrated here).

R8. Interference beyond first order (idea, from a source). The value of a batch S is sum_x V_x plus a second-order term in eta^2. Through the change of p_y of a selected item, K(x, y) enters with the factor (1 - 2 p_y); through the bank, a batch gives products K(x, j) K(y, j) with the curvature of F. The term is not sign-definite, so a batch can be worth less or more than the sum of its items (the form is stated, not derived here). Huang et al. 2026 (Theorem 3) show that a capacity-limited learner learns features in order of utility, frequency times spectral weight, and in Section 3.2 that rare complex tasks are lost by interference. So selecting rare high-value items on a small model can give gains that later steps overwrite. First-order value then overstates what lasts.

R9. What one learner's answers identify about an item (derivation, checked numerically). Take l_jt = a_j theta_t - c_j along training, with k answers per item at each of C checkpoints spaced evenly over a range Delta of theta. With theta known, the Fisher information about a_j, with c_j a nuisance, is sum_t k p_jt (1 - p_jt) (theta_t - theta_bar)^2. At p (1 - p) = 0.2 it is k 0.2 Delta^2 C (C + 1) / (12 (C - 1)).

| k | C | Delta (logit) | information | SE of a_j |
|---|---|---|---|---|
| 16 | 10 | 1.0 | 3.259 | 0.554 |
| 16 | 10 | 0.5 | 0.815 | 1.108 |
| 64 | 20 | 1.0 | 23.579 | 0.206 |

(derived_numbers.py). The logit velocity of item j is a_j d theta / dt. With theta known it is a_j times a known factor, so its relative SE equals that of a_hat_j. Velocity and online discrimination are then one estimate, not two predictors. Without theta known, the logit change between two checkpoints has SE sqrt(2 / (k p (1 - p))) = 0.79 logit at k = 16. At the counts of the table neither is identified for discriminations near 1. At first order the velocity is row j of the kernel contracted with the training distribution q,

```
d l_j = eta sum_x q_x p_x (1 - p_x) K_P(j, x)
```

so it reads the kernel only through q. The noise floor follows. Let the logit change over a window have SD s across items and reliability rho = s^2 / (s^2 + SE^2). With independent noise, the Pearson correlation of two estimates in adjacent windows is the true correlation times rho. Reliability 0.8 at C = 10 and p (1 - p) = 0.2 needs k = 79 answers per item per checkpoint if s = 0.5 logit, and k = 314 if s = 0.25 (derived_numbers.py). So tier 2 reads velocity on a probe set of items, not on the pool.

R10. Concept signal is the data's sensitivity (source, with one derivation). Park et al. 2024 define concept signal as sigma_i = abs(dG(z)/dz_i), the sensitivity of the data generating process to concept i, and show it sets the speed and order of concept learning. Item information a^2 p (1 - p) is the same kind of quantity for one answer about theta, the sensitivity of the answer distribution to the ability. The analogy fails at one point. Concept signal is a property of the data only; V also depends on the learner's state through p_x and alpha_x. So G's claim that the state tells which experience changes the learner is stronger than the concept-signal claim, and needs the state terms of R1 to matter.

## 4. The calibration of the tier 1 reading

The tier 1 reading. Arms I and B share one pass-rate band (base pass rate in [0.2, 0.8] from 8 base answers) and one cost (32 items per step, one answer each, 333 steps). Arm B draws uniformly within the band. Arm I draws within the band with probability proportional to a_pop^2. The reading is the difference in final bank accuracy, I minus B.

The learner with known truth. A 2PL learner with one ability theta, p_x = sigmoid(alpha_x (theta - b_x)), b ~ N(0, 1.5^2), a pool of 2,000 items and a bank of 300. Each step moves theta by eta times the mean of alpha_x (r - p_hat_x)(r - p_x), the score-function step with a stale critic p_hat from the 8 base answers, eta = 0.015, plain gradient ascent (A8). Its expectation is alpha_x p_x (1 - p_x) (A2 holds by construction). A second version adds a Gaussian move of theta per step, sd 0.15 before eta, as a stand-in for the training noise of a many-parameter learner.

| case | learner | a_pop | truth |
|---|---|---|---|
| N1 | alpha = 1 | lognormal(0, 0.5), independent | zero at first order |
| N2 | alpha lognormal(0, 0.5) | lognormal(0, 0.5), independent of alpha | zero at first order |
| P1 | alpha lognormal(0, 0.3) | alpha times lognormal(0, 0.3) | planted, size measured |
| P2 | alpha lognormal(0, 0.5) | alpha times lognormal(0, 0.3) | planted, size measured |

"Zero at first order", since arm I draws fewer distinct items and the step is nonlinear in theta. The planted size is the mean over all runs of the exact bank accuracy difference, computed from the true probabilities without sampling. The false positive rate is the share of experiments in N1 or N2 where a reading declares "I helps". The power is that share in P1 or P2. Each experiment draws a new pool and bank and trains 10 seeds per arm; 400 experiments per case.

Three readings are calibrated here, and a fourth below.
- BP, the peer blueprint's rule. I beats B by 2 paired SE over bank items in at least 2 of 3 seeds and on the seed mean, with the bank read at 16 answers per item.
- T3 and T10. A one-sided t test at 0.05 over the seed-level bank differences, 3 or 10 seeds.

| case | run noise | true effect, I minus B (bank accuracy) | run SD of the difference | item paired SE, one seed | SD over pools of the 10-seed mean | same, net of seed noise | BP rate | T3 rate | T10 rate |
|---|---|---|---|---|---|---|---|---|---|
| N1 | 0 | 0.0000 (SE 0.00003) | 0.0019 | 0.0081 | 0.0013 | 0.0013 | 0.0050 | 0.0625 | 0.0600 |
| N2 | 0 | 0.0000 (SE 0.00005) | 0.0029 | 0.0081 | 0.0024 | 0.0024 | 0.0025 | 0.0650 | 0.1050 |
| P1 | 0 | +0.0181 (SE 0.00004) | 0.0027 | 0.0081 | 0.0022 | 0.0022 | 0.6075 | 0.7350 | 0.9975 |
| P2 | 0 | +0.0381 (SE 0.00007) | 0.0041 | 0.0082 | 0.0038 | 0.0037 | 1.0000 | 0.9950 | 1.0000 |
| N2 | 0.15 | +0.0003 (SE 0.00014) | 0.0087 | 0.0081 | 0.0035 | 0.0024 | 0.0125 | 0.0400 | 0.0725 |
| P1 | 0.15 | +0.0181 (SE 0.00014) | 0.0085 | 0.0081 | 0.0036 | 0.0025 | 0.5600 | 0.5425 | 1.0000 |

Rates are shares of 400 experiments, so each has a binomial SE near 0.011 at a rate of 0.05. Rows N1 and N2 give false positive rates; rows P1 and P2 give power. Bank accuracy of arm B ends between 0.659 and 0.670 from a start near 0.5; the band holds 0.585 to 0.594 of the pool. The net column is sqrt(SD^2 - SE^2), with SE the within-pool SE of the 10-seed mean of the exact difference. The N2 row with run noise shows +0.0003 (SE 0.00014), 2.2 SE above zero. The first-order derivation does not cover it; it is 0.017 of the planted P1 effect.

Commands. `python -I src/surrogates/theory_g/calibrate_tier1.py > results/theory_g_calibration.json`, about 4 min on the PC. `python -I src/surrogates/theory_g/derived_numbers.py results/theory_g_calibration.json > results/theory_g_derived.json`, under a minute. Tiny synthetic records only.

Reading. The blueprint rule BP has a false positive rate of 0.0025 to 0.0125 on this learner, below 0.05, and power 0.56 to 0.61 against a planted gain of 0.018 in bank accuracy (I higher than B), rising to 1.0 at 0.038. The per-seed part binds. From the SEs of the table, each seed passes 2 paired SE with probability about 0.59 at +0.018 (two of three, 0.64), and the seed mean passes with probability about 0.97 (normal approximation, derived_numbers.py; observed joint rate 0.61). The t test over 10 seeds has power near 1.0 at 0.018 but a false positive rate of 0.105 in N2 without run noise, about 5 binomial SE above 0.05. That excess is R7. The t test sees the sampled bank read. The seed SE of its 10-seed mean is 0.0026, from the seed SD of the exact difference (0.0016) and the item paired SE (0.0081). The SD over pools of the pool-conditional effect, net of seed noise, is 0.0024, a ratio of 0.91. A one-sided t test whose mean carries an extra normal shift of that ratio has a false positive rate of 0.104 (derived_numbers.py, 400,000 draws), against 0.105 observed. The same computation gives 0.067 for N1 (observed 0.060) and 0.080 for N2 with run noise (observed 0.0725), where the run noise widens the seed spread.

The fourth reading, PT, the permutation test of R7 with K = 19. Per experiment, arm I and 19 permuted weightings are trained, each for 3 seeds, with one bank read at 16 answers per item. The statistic Y is a weighting's bank read from 1 seed (S1) or the mean of 3 (S3). 200 experiments per case.

| case | run noise | true effect, I minus the mean of the permuted weightings | PT rate, 1 seed | PT rate, 3 seeds |
|---|---|---|---|---|
| N2 | 0 | 0.0000 (SE 0.0002) | 0.045 | 0.060 |
| P1 | 0 | +0.0180 (SE 0.0002) | 0.795 | 0.995 |
| N2 | 0.15 | -0.0001 (SE 0.0003) | 0.040 | 0.040 |
| P1 | 0.15 | +0.0182 (SE 0.0003) | 0.665 | 0.965 |

Rates are shares of 200 experiments, binomial SE near 0.015 at 0.05. The false positive rate is 0.040 to 0.060, within one SE of 0.05, as exchangeability predicts. Power against a planted 0.018 is 0.665 to 0.795 with 1 seed per weighting and 0.965 to 0.995 with 3. PT removes the excess that took T10 to 0.105 in N2, at the cost of 20 trained weightings per experiment. Command `python -I src/surrogates/theory_g/calibrate_permutation.py > results/theory_g_permutation.json`, 4 min 50 s on the PC.

What this calibration does not show. The run SD of the difference here is 0.0019 to 0.0087, at most 1.07 times the item SE. The run SD of a 1B model under GRPO is not measured. BP compares each seed with 2 item SE and ignores the seed spread, so its false positive rate is expected to rise once the run SD exceeds the item SE; that is a heuristic argument, and no row here shows it. PT is calibrated on a null where a_pop is independent of difficulty. The learner meets A2, A3 and A8 by construction, so every reading is calibrated only where the theory holds.

## 5. One prediction per tier, written before any data

Tier 1, on a model. Prediction. On Llama-3.2-1B-Instruct trained by the Princeton GRPO recipe on MMLU prompts, with its optimizer read from the code (A8), drawing within a [0.2, 0.8] band with weight a_pop^2 from the Fluid 2PL does not give the highest final bank mean@16 among arm I and 19 weightings with the weights permuted within the band (PT, R7). Reason, a doubt and not a derivation. a_pop is discrimination along the axis that separates pretrained models, and nothing shows that GRPO moves this model along that axis (A4, R3). Panel S1.4 finds that the scale term of the GRPO states is a decoding temperature, a change of scale rather than a move along the population axis. Refuted if arm I has the highest bank read of the 20 (p = 0.05). On the calibration learner this rule had a false positive rate of 0.040 to 0.060 and power 0.665 to 0.795 at a gain of 0.018 with 1 seed per weighting, 0.965 to 0.995 with 3. What each outcome changes. A positive tier 1 says the population axis is a training axis for this learner, and tier 2 asks whether the learner's own kernel row predicts as well without a population. A null sends tier 2 to the learner's own kernel row and drops the population item side from G.

Tier 1, second prediction, the shape of the gain (R4). The bank gain is measured per run, not per item, so this needs its own arms. Four narrow band arms at equal cost each draw uniformly from one bin of base pass rate from 8 answers, {1/8, 2/8}, {4/8}, {5/8} and {7/8}, 3 seeds each. The reading is the final bank gain per arm against the bin's mean base pass rate, compared by 2 SE over seeds. Before training, the base bank read gives the sign of c'_B. Prediction, for a sharpening learner on the binary curve. If c'_B > 0, the gain in {7/8} exceeds the gain in {4/8}. If c'_B < 0, the gain in {1/8, 2/8} exceeds it. Refuted if {4/8} has the largest gain by 2 SE. Under a guessing floor of 0.25 the midpoint moves to 0.625, so {5/8} takes the place of {4/8} (R4); both readings are reported. What each outcome changes. A peak away from the midpoint, on the side of c'_B, says training changes precision for this learner and that the band of every classical arm is centered wrongly; this finding stands without IRT. A peak at the midpoint supports the location learner of R1. This reading is not calibrated, since the calibration learner has location only (choice 3).

Tier 2, the item side online. Prediction. On a probe set of items read at k answers per checkpoint, with k set so that the reliability of the window logit change is at least 0.8 (R9, k near 79 at C = 10 if the spread is 0.5 logit), the pass-rate change of an item over window w + 1 is predicted better by p_j (1 - p_j) v_hat_j(w) than by p_j (1 - p_j) alone, by more than a permutation null over items. Here v_hat_j(w) is the item's logit change over window w, the kernel row contracted with the training distribution. Refuted if the velocity predictor does no better. Two explanations of a refutation are then told apart by the correlation of v_hat across adjacent windows divided by its reliability (R9). A value near 1 with a small spread of v_hat means one factor with constant alpha (R2). A value well below 1 means the kernel row turns over between windows. Below reliability 0.8 no failure is attributed to turnover. What each outcome changes. A persistent kernel row makes guided training possible with no population; a row that turns over is the reordering that surrogate M reads, and G tier 3 loses its model counterpart.

Tier 3, on students. Prediction. In a log with practice difficulty assigned independently of the student's state, read the delayed gain on unpracticed sibling items per practice item, as a function of the predicted pass rate of the practice item. The siblings' weighted mean logit, the c'_B of the siblings, is computed before the gain is read. The prediction is the precision learner. The gain is near zero at 0.5 and peaks between 0.7 and 0.9 if that mean logit is positive, and between 0.1 and 0.3 if negative. Refuted if the peak sits at 0.5 within its SE. What each outcome changes. A peak at 0.5 supports the location learner for students and the information rule as a teaching rule; a peak away from 0.5 on the side of the siblings' mean logit supports precision learning (Wilson et al. 2019) and makes item information a measurement rule only. The item of maximum Fisher information differs from the item of maximum gain under both learners whenever alpha varies (R1), so that comparison does not separate them. Without exogenous variation (A7) no outcome is read.

## 6. The literature

Every source below has a row in sources/reading.md with its rung.

| source | what G takes from it | rung |
|---|---|---|
| Park et al. 2024 | concept signal as data sensitivity, setting speed and order (R10) | source, full text Sections 3 and 4 |
| Wilson et al. 2019 | the precision learner and its optimal error rates (R4) | source, full text |
| Mindermann et al. 2022 | the value of a point as the gain in holdout likelihood; learnable, worth learning, not yet learnt | source, full text Section 3 |
| Bae et al. 2025 | the classical band symmetric about 0.5, online, 16 rollouts; its bound is on KL, not on improvement | source, full text |
| Foster et al. 2025 | sampling by p (1 - p) | abstract only |
| Huang et al. 2026 | utility order (Theorem 3) and interference (Section 3.2) in capacity-limited learners (R8); the setup to copy for known truth | source, full text Section 3 |
| Hofmann et al. 2025 | the 2PL, the Fisher information and the selection rule of the population item side | source, Sections 3 and 4.1 |
| Weinshall et al. 2018 | two difficulties, against the optimal and against the current hypothesis, with opposite signs | source, full text Section 2 |
| Clement et al. 2015 | learning progress as the signal of a tutor; a classroom design with four groups | source, Sections 4.2 and 6 |
| Wang et al. 2020 | the Eedi selection task scores measurement, not learning | abstract only |

Foster et al. 2025 is arXiv 2502.12272 (Foster, Sims, Forkel, Fellows, Foerster); the NeurIPS 2025 version, LILO, lists Foster, Sims, Forkel and Foerster.

Three points from the full texts change how the field is read here. Bae et al. state in the abstract that policy improvement is bounded below by the variance of success; the body proves a lower bound on the reverse KL between the initial and the optimal policy, p (1 - p) / (2 beta^2) (Proposition 3.1), which is a different statement. Weinshall et al. state in the abstract that the convergence rate increases with difficulty; Theorem 1 states that it decreases with difficulty measured against the optimal hypothesis, and Theorem 2 that it increases with the current loss. The current loss is the state term of G, the optimal-hypothesis loss is the item term. Clement et al. pool the pre and post tests over the three software groups, so their study does not compare the selection rules on learning.

Setups to copy, in order of use.
1. Known truth before any model. Huang et al. 2026 Section 3, the mixture of K = 32 linear regression tasks with power-law frequencies and spectra, trained with Adam on a width-N encoder. Tasks are items; utility is known; a selection rule is a reweighting of task frequencies; capacity is a knob.
2. The classical arm. Bae et al. 2025 Section 3.4, balanced filtering with a [0.2, 0.8] band and online resampling.
3. The model run. The peer's Princeton recipe with G's own arms, arm I and 19 permuted weightings within the band (PT, section 4). The peer's current Q1 arms are critics over all prompts and do not supply these arms.
4. Students. The four groups per class of Clement et al. 2015, with assignment randomized and stated, and the outcome read per group on unpracticed items after a delay.

How the peer's arms relate to the tiers. The archived v2.0 arm I against arm M was the nearest form of tier 1 (archive/blueprint-v2.0-2026-10-19.md:28 to 33); the current blueprint replaced the selection arms (blueprint.md:25). The current Q1 arms C against I0 test the critic, which R5 shows changes the spread of the step and not its expectation, so they are not a tier of G; they test an estimator of p as a baseline. Arm I1 against I0 weights each item's step by a_j, which at first order equals drawing items with probability proportional to a_j at higher variance. Its expected step is the information when a = alpha (blueprint.md:32). It is a soft form of tier 1 with no permutation null, so its reading carries the pool-conditional effect of R7. Arm G against the critics is a classical result. Q2 belongs to surrogate M.

## 7. What the theory cannot show

It shows no gain on any learner; every number in section 4 is on a one-ability synthetic learner built to meet A2, A3 and A8. The run noise of a 1B model is unknown here, so the false positive rate of the blueprint rule on that model is not computed, only its form. PT holds its size by exchangeability on any learner, but its power on a model is not known. R1 to R5 hold at first order, for one factor and a fixed preconditioner; a deep network's kernel changes during training, and the size of that change is the question of tier 2, not an assumption the theory can grant. R4 separates location from precision for one logistic item curve with a known floor; a learner that does both gives a gain curve between the two shapes, and the mixture weight is not identified from the peak alone. R9 shows that one learner's answers identify item velocity only on a probe set at high answer counts. R10 is an analogy at the level of definitions; Park et al. do not test item selection. No source here shows that practice at maximum information helps students more than other practice. The theory cannot say what the selected experience installed; that is R and E.

## 8. Choices for the author and proposed edits to the research program

Choice 1. Whether the peer's Q1 serves as G tier 1 (the research program, section 9).
- Option A, Q1 as it stands. Mechanism, the critic arms C, I0, I1. Cost, none beyond the peer's runs. Shows whether a population baseline changes the outcome (C against I0) and whether a weight a_j changes it (I1 against I0). It does not test item choice at the learner's state (R5), and I1 against I0 carries the pool-conditional effect (R7).
- Option B, G tier 1 on the peer's recipe now. Mechanism, arm I draws within a base pass-rate band with weight a_pop^2, and 19 arms draw with the weights permuted within the band, one fixed permutation each (PT). Cost, 20 runs with 1 seed each at 3.6 to 3.8 h per run on 4 GPUs (blueprint.md:105), 72 to 76 h of runs; 3 seeds give 60 runs, 216 to 228 h. Shows tier 1 with a calibrated null, power 0.665 to 0.795 (1 seed) or 0.965 to 0.995 (3 seeds) at a gain of 0.018 on the calibration learner.
- Option C, G tier 1 as its own study after tier 1a below, with the seed count set by a power computation. The run SD of the recipe measured on the Q1 arms enters that computation as a first estimate. Those arms differ from G's (160 prompts by 1 answer over all prompts, blueprint.md:43), so their seed spread is not a measurement of G's arms. Cost, as B, later.
- Pick, C. Tier 1a tests the reading on a learner whose features move, and the Q1 runs give a first run SD for the recipe before G spends 72 h or more.

Choice 2. A tier 1a before the model tier.
- Option A, none. Cost, none. Shows nothing about the reading on a learner with features.
- Option B, copy Huang et al. 2026 Section 3. Mechanism, tasks as items, reweighting by known utility against uniform and against permuted utilities, small and large width. Cost, CPU hours, a small build. Shows whether PT recovers a planted selection gain on a learner whose features move, and whether interference erases it at small width (R8).
- Pick, B.

Choice 3. The second reading of tier 1, the shape of the gain (R4).
- Option A, the four narrow band arms of section 5, 3 seeds each. Cost, 12 runs, 43 to 46 h. Shows location against precision for this learner, given the sign of c'_B.
- Option B, regress the per-checkpoint bank change on the pass-rate mix of the items drawn since the last checkpoint, within the tier 1 runs. Cost, logging only. Within a band arm the mix varies little between checkpoints, so the power is low (heuristic argument, not calibrated).
- Option C, first on the tier 1a learner with location and precision both planted, then on the model only if the four-arm design recovers both shapes. Cost, CPU hours, then as A.
- Pick, C.

Choice 4. The decision rule of tier 1.
- Option A, the blueprint rule over bank items. Cost, 3 seeds per arm. False positive rate 0.0025 to 0.0125 and power 0.56 to 0.61 at a gain of 0.018 on the calibration learner. On a model its false positive rate depends on the seed spread, which the item SE omits.
- Option B, PT with K = 19, permuted across the band. Cost, 20 or 60 runs. False positive rate 0.040 to 0.060; power 0.665 to 0.795 with 1 seed per weighting and 0.965 to 0.995 with 3, at 0.018 (section 4).
- Option C, PT with the permutation inside bins of base p_hat. Holds the pass-rate mix fixed, so an a_pop that tracks difficulty inside the band cannot pass as discrimination. Same cost; not calibrated.
- Pick, B, and C if the Fluid a correlates with the base pass rate inside the band.

Proposed edits to the research program, section 4 G.
1. Tier 1 reads "item choice by population discrimination within a pass-rate band, against 19 fixed permutations of the weights within the band, a permutation test at 0.05, at equal training cost"; the current wording "information at the learner's state against a pass rate band" mixes R2's indistinguishable case with the case that can differ.
2. Add tier 1a, the copied setup of Huang et al. 2026 Section 3, as the calibration on a learner with features.
3. Tier 2 reads "the learner's own logit change over a window, read on a probe set at an answer count with reliability at least 0.8, predicts which items move next; a failure counts as turnover only above that noise floor", and names R9 as the reason.
4. Tier 3 replaces "practice on high information items raises later learning more" with "the delayed gain on unpracticed siblings, by the predicted pass rate of the practice item, peaks at 0.5 for a location learner and away from 0.5, on the side of the siblings' weighted mean logit, for a precision learner", and keeps the requirement of difficulty assigned independently of the state.
5. Section 3 idea 2 adds the distinction of R1, the value of an answer for measurement against its value for change.
6. Section 9 records that the current Q1 critic arms test the spread of the step, not item choice (R5), and that I1 against I0 carries the pool-conditional effect (R7), whatever the author decides on Q1.

## Fix log of attempt 1

| item | what changed | line |
|---|---|---|
| C1 | R4 adds the bank factor c'_B; the sharpening peak is 0.824 if c'_B > 0 and 0.176 if c'_B < 0; the guessing floor is stated; the second tier 1 and the tier 3 predictions are conditional on the sign of c'_B | 86 to 97, 182, 186 |
| C2 | R9 states that velocity and a_hat are one estimate, adds the noise floor and the answer counts for reliability 0.8; velocity is the kernel row contracted with q; tier 2 is rewritten above that floor | 113 to 127, 184 |
| C3 | R6 withdrawn and kept as the record of v2.1; the R1 table drops "blueprint critic arms"; section 6 states the current I1 weight | 101, 64 to 70, 215 |
| M1 | The 0.105 reading compares the net pool SD 0.0024 with the read SE 0.0026 (ratio 0.91, predicted rate 0.104) | 163 |
| M2 | The table gives the SD over pools with and without seed noise | 150 to 157, 159 |
| M3 | K = 19 is the smallest K for 0.05; costs are 20 or 60 runs, 72 to 76 h or 216 to 228 h at the current 3.6 to 3.8 h per run | 103 to 109, 225, 242 |
| M4 | PT is calibrated (calibrate_permutation.py), false positive rate 0.040 to 0.060, power 0.665 to 0.995; the tier 1 refutation cites it | 165 to 174, 180 |
| M5 | The R1 table separates the step divided by the SD at the true p from GRPO at G = 5, with its shape | 64 to 72 |
| M6 | R5 states the second moment, the gradient variance and the baseline 1 - p of the one-bit learner | 99 |
| M7 | Assumption A8 names the optimizer condition; section 1 carries the preconditioner P | 24 to 30, 52 |
| M8 | The second tier 1 reading has its own design, four narrow band arms with bins that include 7/8 | 182, 234 to 238 |
| M9 | Choice 1 and setup 3 use G's own arms; the Q1 seed spread enters only as a first estimate | 212, 223 to 227 |
| m1 | A2 gives the constant (G - 1)/G for the leave-in mean and the four differences under "three things" | 40 |
| m2 | alpha_j > 0 stated in A3 and R1 | 42, 59 |
| m3 | R2 names the band as a uniform draw and adds constant alpha with varying a | 76 |
| m4 | R3 drops the unproven loss and cites the calibration | 84 |
| m5 | R8 states the term is not sign-definite and cites Huang Section 3.2 for interference | 111 |
| m6 | R9 table uses evenly spaced checkpoints, SE 0.554, 1.108, 0.206 | 115 to 119 |
| m7 | "zero at first order"; the N2 run-noise +0.0003 is stated | 144, 159 |
| m8 | The per-seed part binds, 0.59 per seed against 0.97 for the seed mean | 163 |
| m9 | The BP rise past the item SE is marked as a heuristic argument | 176 |
| m10 | The script docstring gives about 4 min | calibrate_tier1.py:26 |
| m11 | Tier 3 states that the information clause does not separate the learners | 186 |
| m12 | "mostly sharpens" replaced by the decoding-temperature statement of S1.4 | 44, 180 |
| m13 | Foster et al. 2025 with both author lists; reading.md row updated | 70, 205 |
| m14 | reading.md cites Mindermann et al. Eq. (3) from the selection function Eq. (2) | sources/reading.md |
| m15 | Proposed edit 4 replaces the program's tier 3 claim | 250 |
