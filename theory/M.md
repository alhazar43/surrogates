# Theory of surrogate M, measurement under learning

Unit TM, 19 October 2026, fix attempt 1 after the review theory/M-review.md. The theory end of surrogate M of the research program (C:/Users/YuanW/PLMS/surrogates/program/research-program.md, section 4). Rungs are idea, definition, derivation, checked numerically (with the script), and source (with the row in sources/reading.md).

## 1. The object

Notation. The bank is a fixed set of items j = 1, ..., J. A learner state s gives each item a probability of a right answer p_s(j) in (0, 1), and the log odds eta_s(j) = logit p_s(j). A change is a route from the base state s0 to the after state s1. The observation is either the exact p_s (option probabilities of a model) or y_s(j) ~ Binomial(k, p_s(j)), the number of right answers in k sampled answers.

A calibration made before the change is a set of after states that the base calibration still reads. Five nested sets, each a function of the base log odds eta0 = eta_{s0}.

```
C_0     = { eta0 }                                         no change
C_shift = { eta0 + alpha 1 : alpha real }                  one common shift
C_aff   = { alpha 1 + c eta0 : alpha real, c > 0 }         shift and scale
C_mono  = { h(eta0) : h strictly increasing, item by item } any strictly increasing map
C_all   = R^J                                              any change
C_0 within C_shift within C_aff within C_mono within C_all
```

Beside the chain sits one set that uses item parameters from a population of other respondents, with slopes a in R^J (or a J by K matrix A):

```
C_pop(A) = { h(eta0) + A delta : h strictly increasing, delta in R^K }
```

The reading of one change is the map

```
M : (y_{s0}, y_{s1}) -> (p value of the order test against C_mono,
                         reorder size sigma_hat, read against its calibrated quantiles,
                         fitted map h_hat, residual r = eta1 - h_hat(eta0))
```

The order test and sigma_hat are defined and calibrated in section 4. The sets C_shift and C_aff have no calibrated test yet. Their test is the tier 2 deviance of section 5, and until it is calibrated M places a change only against C_mono.

C_shift is the base calibration read with the person parameter moved, which is what "the calibration still reads the learner" means under the Rasch model with the learner's own difficulties b_j = -eta0(j). C_aff and C_mono are the base calibration repaired by a relink of two parameters or of one strictly increasing function common to all items. A change outside C_mono reorders at least one pair of items for this learner.

Which part of the question it answers. What a change leaves fixed in the learner's ranking of problems, and how much of the change one function of the base log odds absorbs. It leaves open why the change happened, what was acquired, and whether the change reaches problems outside the bank.

## 2. The assumptions

| | assumption | what fails with it |
|---|---|---|
| A1 | The bank is fixed and the items are the same objects before and after (same text, same option orders) | a change of the prompt or option order enters as a change of the learner |
| A2 | For a fixed state, the k sampled answers of one item are independent draws given p, and the items are read independently | the binomial noise model of section 4 |
| A3 | The reading temperature and the prompt are the same before and after | a scale term becomes a property of the reading, not of the learner (panel D9) |
| A4 | The answer is scored right or wrong; the option chosen among wrong answers is not used | the multi option results of P3 move into the residual unseen |
| A5 | The fallback of a learner that does not use its capability gives the same success probability on every item | the map of P2(b) becomes item specific, and P3b reorders items by their fallback |
| A6 | For C_pop, the population slopes a hold for this learner up to the map h | the split between a move along the population axis and an item specific change |
| A7 | For tiers 2 and 3, a set of anchor items or a reference group is unchanged, as a convention | the split of a common change into person and items (P1) |
| A8 | The option orders read cover every position of the key, so that a letter bias averages out | the control of P3b; with two orders it is partial |
| A9 | The prior of the null's posterior draws fits the distribution of p over the bank | the false positive rate and the bias of sigma_hat (section 4) |

## 3. What can be derived

P1. One learner does not identify person change against item change. Under the Rasch model eta_s(j) = theta_s - b_s(j), the pairs (theta1, b1) and (theta1 + t, b1 + t 1) give the same eta1 for every t. Under a 2PL with fixed slopes, eta1 = eta0 + a delta + d and (delta, d) is equivalent to (delta + t, d - t a). Identified are the contrasts eta1(j) - eta1(i) - (eta0(j) - eta0(i)) and, for the 2PL, the part of the change orthogonal to a in any fixed inner product. A split into person and items needs a convention, anchor items with d = 0 or a reference respondent known to be unchanged. Rung, derivation. Fischer and Formann (1982, p. 406 to 407) state the same for response sequences, a change in the person "can also be postulated" as a change of item difficulty. Strobl et al. (2021) state that the anchor assumptions, balanced DIF or a DIF free majority, cannot be tested a priori (source).

Reading. The research program's "shift" is not a person change or an item change. It is the common part of the change, and it is the part a single learner cannot attribute.

P2. Every change in C_mono keeps the order of the items for the learner. Spearman rho(eta0, eta1) = 1 and Kendall tau = 1 without noise, and rho is invariant to any strictly increasing map of either vector. A weakly increasing h makes ties and rho below 1. Members of C_mono, each with its condition.
- (a) KL regularized reward with a right or wrong reward, at its optimum, on the prompts it trains. The optimum is pi_r = pi_ref exp(r / beta) / Z (Rafailov et al. 2023, Eq. 4, source). Summing over the right answers gives logit p1 = logit p0 + 1 / beta for every trained prompt, a member of C_shift (derivation). The condition is the optimum per prompt. It says nothing about untrained bank items, which move through the learner's similarity between items (panel B3, B17).
- (b) A change of use as a mixture. If p_s(j) = u_s q(j) + (1 - u_s) g, with q the capability and u the weight on using it, then p1 = g + (u1 / u0)(p0 - g), strictly increasing in p0 (derivation). In log odds this map is curved at the asymptotes, so it lies in C_mono and outside C_aff. Condition A5.
- (c) A decoding temperature on a right or wrong outcome, eta1 = eta0 / tau, a member of C_aff (panel D9, derivation).
- (d) Item blind geometry. If each bank item's gradient is a function of its own log odds, the same function for every item, the gradient flow keeps the order of the bank items along the whole path (panel B3, Proposition 2, panel:368; derivation). For training with a finite step, the step p to p + eta f(p) keeps the order only while eta f'(p) > -1 at every item (panel B4 point 4, panel:442), the same condition as g' > -1 in P4(a).

P3. Two changes without a change of capability that reorder items. Each breaks the reading "a reorder means capability".
- (a) A decoding temperature on four options. It changes the reading, not the learner's use or capability, and it breaks A3. Take two items with the same correct logit z. Item A has three wrong options at logit w; item B has one wrong option at w + log 3 and two far below. At tau = 1 the wrong mass is 3 e^w for both, so p_A = p_B. At temperature tau the wrong mass is 3 e^{w / tau} for A and 3^{1 / tau} e^{w / tau} for B, so p_B < p_A for every tau < 1 (derivation). On a bank of 200 items, each of kind A or B by a Bernoulli(0.5) draw, tau = 0.5 gives median noise free Spearman rho 0.986 over 200 banks (checked numerically, section 4). Condition for telling it apart, the option logits are observed, so the family z_s(j) = z0(j) / tau can be tested directly, or the base is read at the matched temperature (panel D9, the cooled base).
- (b) A change of use with an item specific fallback. If g = g_j varies, for instance with the position of the key under a letter bias, then p1 = (u1 / u0) p0 + (1 - u1 / u0) g_j depends on g_j, and a change of u reorders items (derivation). This is the one derived case in which a change of use reorders, and it needs A5 to fail. Condition for telling it apart, each item is read under every option order and the probabilities are averaged (A8).

P4. Two changes of capability that keep the order. Each breaks the reading "no reorder means no change of capability".
- (a) Any change Delta = g(eta0) with g' > -1 lies in C_mono, whatever its cause (derivation). A rise of a general capability that every item uses in proportion to its base log odds is of this kind. Under the Rasch model a general capability rise is a common shift, the same observation as the KL optimum of P2(a). The use change of P2(b) is a curved map, not a common shift.
- (b) The Rasch kernel. If the similarity between bank items and trained items is constant, the outcome part of every route gives a pure person change, a common shift on the bank (panel B3, Proposition 3, panel:372 to 376). The teacher route of SFT adds the import term J_B G, the projection of the teacher's paths on the bank, and that term "is item specific in general" (panel:376, 382). So under a Rasch kernel the policy gradient keeps the order exactly, and SFT on teacher text reorders by its import term (derivation, from the panel). This is the derived reason for an SFT reorder in section 5.

The bound. C_mono contains every change g(eta0) with g' > -1, whatever its cause. A failure to reject C_mono is therefore not evidence that the capability stayed the same.

P5. A 2PL person change can reorder. With eta0 = a (theta0 - b) and eta1 = eta0 + a delta, the pair i, j with eta0(i) > eta0(j) crosses exactly when delta (a_j - a_i) > eta0(i) - eta0(j) (panel B3, panel:378; derivation). For delta > 0 the move reorders iff delta > min over pairs with eta0(i) > eta0(j) and a_j > a_i of (eta0(i) - eta0(j)) / (a_j - a_i); for delta < 0 the same with the pairs a_j < a_i and |delta|. So the move is told apart from C_mono if and only if it crosses at least one pair. When a = A(eta0) is a function of eta0, the move is g(eta0) = delta A(eta0) and it keeps the order iff delta A'(eta0) > -1 on the bank, by P4(a). Monotonicity of a in eta0 alone settles nothing. A decreasing a = exp(-0.1 eta0) with delta = 5 keeps the order (rho 1.000), and an increasing a = exp(0.3 eta0) with delta = -5 reorders (rho -0.845) (review, chk/checks.py). With a ~ LogNormal(0, 0.35), independent of b, and delta = 1 logit, the median noise free Spearman rho is 0.958 over 200 banks (checked numerically, section 4). The move along the population axis is told apart from an item specific change only through C_pop(a). Condition A6.

P6. What a reorder is, once P3 to P5 are removed. The residual r = eta1 - h_hat(eta0), with h_hat the strictly increasing map, the temperature and the population axis fitted, is the item specific part of the change (definition). Its structure is the evidence. A residual concentrated on one subject of the bank is a change specific to that subject, which is the thesis gate's "practice on A raises a set clearly and the rest little". A residual spread over all items with no structure is either noise beyond the model of section 4 or a change of the learner's similarity between items (idea).

P7. Relinking with m anchor items (tier 2). Let eta1 = alpha + c eta0 + r, fitted on m anchors drawn at random from the bank. If r = 0 on the anchors and eta0 is known, the maximum likelihood relink has sampling error of order 1 / sqrt(m k1) in (alpha, c), and the predicted bank mean has the same order (derivation, standard). With an estimated base eta0_hat, its error attenuates c, as for any regression with an error in the regressor, and that bias does not shrink with m; the synthetic RMSE falls from m = 10 to m = 50 by a ratio of 2.47 against sqrt 5 = 2.24 (review row 26). If eta1 is in C_mono and not in C_aff, the relink of single items has a bias that does not shrink with m (derivation); for the bank mean the bias is small when the curve is mild (checked numerically, section 4, the use weight row). If r is item specific with mean zero, the bank mean stays nearly unbiased, since random anchors average r, while the prediction of single items fails (derivation). Stocking and Lord (1983) reject a regression of one calibration on another for two reasons, a regression is not symmetric in the two calibrations, and the regressor carries error (source). Here the asymmetry is intended, since the after state is predicted from the base and the base is read with more draws (k0 = 64 against k1 = 16). The error in the regressor remains and is the attenuation above. Few anchors suffice for the bank mean when the change is in C_aff and the base is read with more draws than the after state; they do not suffice for single items when r is not zero.

P8. Between cohorts of students (tier 3). Each cohort gives its own calibration of the items, and the cohort's mean ability and a common drift of the items are not identified apart, by P1 applied to groups. Identified are the contrasts of item difficulties between cohorts, once an anchor convention is chosen. So scale stability on a student log is testable only as the absence of a reorder beyond resampling, and its shift part rests on an anchor assumption that the log cannot test (Strobl et al. 2021, source; derivation). The LLRA of Fischer and Formann (1982) is the same structure inside one person, item specific person parameters theta_ai at the first occasion and theta_ai + delta_a at the second. Its test of whether the effects generalize over ability domains is a test of one common effect against effects that differ by domain (source). That is the student form of the subject residual of P6, a block structure in r, not the order test against C_mono.

Where two explanations cannot be told apart, and the condition that separates them.

| explanation 1 | explanation 2 | separated when |
|---|---|---|
| person change | common item drift | never for one learner; an anchor convention defines the split (P1) |
| use change (any strictly increasing map) | capability change g(eta0) with g' > -1 | never from the bank alone (P4); a bank built so that capability and base log odds come apart, or a second reading such as revival speed (surrogate E) |
| capability reorder | temperature on multi option items | option logits observed, or the base read at the matched temperature (P3a) |
| capability reorder | change of use with an item specific fallback | every option order read and averaged (P3b, A8) |
| item specific change | move along the population axis | population slopes known (A6), and the move crosses at least one pair (P5) |
| item specific change | binomial noise | the calibrated test of section 4 |

## 4. The calibration of the tier 1 reading

The reading. Input, the numbers of right answers y0 and y1 of one learner on J items, k0 and k1 draws per item. Statistic, Spearman rho(y0, y1) over items, invariant to any strictly increasing map, so its value under C_mono without noise is 1. Null distribution, by the co-monotone rearrangement.
1. Fit a prior for each state, in two forms. The logit normal prior, eta ~ N(mu, sd^2), with (mu, sd) chosen by marginal maximum likelihood of the binomial counts over the bank, on a grid of 181 log odds from -9 to 9. The Beta prior of attempt 0 by the method of moments, Var(p) = (Var(y/k) - m (1 - m) / k) / (1 - 1/k) with m the mean of y/k.
2. In each of B = 199 draws, draw p0* and p1* from the item posteriors. Sort p1* and give it the order of p0*, so the null state is in C_mono and keeps the spread of the after state.
3. Draw new numbers of right answers from p0* and the rearranged p1*, and compute rho*.
4. The p value is (1 + #{rho* <= rho_obs}) / (B + 1), one per prior.
5. The reading rejects C_mono when both p values are at or below 0.05 (chosen level). Each prior alone exceeds the level in some cell of the tier 1 design (table below), and the rule of step 5 cannot exceed the smaller of the two rates.

The reorder size. For each sigma on the grid {0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1} logit, add sigma times a standard normal draw per item to the log odds of the rearranged p1*, redraw the counts, and record the median of rho* over 49 draws, under the logit normal prior. sigma_hat is the sigma at which the median equals rho_obs, by linear interpolation. It is the item specific logit SD that, added to a co-monotone version of the after state, gives the observed rho (definition). The script also inverts the test into a 90 percent interval; that interval covers the true SD in 0.19 and 0.21 of learners at a true SD of 0.25 and 0.5 logit, so it is not used, and sigma_hat is read against its calibrated quantiles instead.

The prior. Script C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_m/prior_check.py, output results/theory_m/prior_check.json, 16 learners per cell on the tier 1 design. Mean sigma_hat at a true item specific SD of 0.1 logit is 0.161 under the Beta prior, 0.039 under a grid prior by nonparametric maximum likelihood (its atoms tie the draws), 0.159 under the same grid prior smoothed, and 0.111 under the logit normal prior. At a true SD of 0.25 the four give 0.304, 0.230, 0.304 and 0.280. The logit normal prior has the smallest bias on this synthetic bank, whose log odds are close to normal by construction, so the choice rests on A9.

The learner with known truth. Slopes a ~ LogNormal(0, 0.35), difficulties b ~ N(0, 1.2^2), theta0 = 0.3, eta0 = a (theta0 - b). The conditions:
- No effect. eta1 = eta0, read again; and the same with eta0 multiplied by 3, a bank with a U shaped distribution of p, as a check of A9.
- Order preserving changes. A common shift of 1 logit; a stretch eta1 = c eta0 + 0.3 with c = 1.378 (the scale that the panel quotes for GRPO checkpoints, F34, panel:449) and c = 3, which makes the after state U shaped; a change of use as a mixture, u from 0.6 to 0.95 with fallback 0.25.
- The confounds of P3a and P5. Four option items as in P3a at tau = 0.9 and tau = 1 / 1.378 = 0.726; a move of 0.1, 0.3 or 1 logit along the population slopes.
- Inserted reorders of known size. A common shift of 1 plus an item specific change with SD 0.05, 0.1, 0.25 or 0.5 logit; one subject of J / 10 items raised by a further 0.5 or 1 logit on top of a common shift of 0.5.

The false positive rate is the rejection rate over the learners of a condition with no reorder. The power is the rejection rate under a reorder. Each rate carries its binomial standard error sqrt(rate (1 - rate) / R).

Script C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_m/calibrate.py, run with python -I and 4 worker processes, output C:/Users/YuanW/PLMS/surrogates/results/theory_m/calibration.json (attempt 0 output kept as calibration_attempt0.json).

The tier 1 design, J = 1,500 items. Rejection rate of the order test at level 0.05 under each prior and under the rule of step 5, and the median of sigma_hat over learners with its 5 and 95 percent quantiles. Spearman rho noise free is the median rho between the true p0 and p1.

| condition | R | rho noise free | rejection rate, logit normal / Beta / both | sigma_hat median (5 to 95 percent) |
|---|---|---|---|---|
| split half, k = 8, no effect | 200 | 1.000 | 0.105 / 0.085 / 0.085 (SE 0.020) | 0.155 (0.000 to 0.354) |
| split half, k = 8, no effect, U shaped p | 200 | 1.000 | 0.005 / 0.175 / 0.005 (SE 0.005) | 0.000 (0.000 to 0.328) |
| k = 32, no effect | 200 | 1.000 | 0.115 / 0.065 / 0.065 (SE 0.017) | 0.084 (0.000 to 0.169) |
| k = 32, no effect, U shaped p | 200 | 1.000 | 0.000 / 0.230 / 0.000 | 0.000 (0.000 to 0.032) |
| k = 32, common shift of 1 logit | 100 | 1.000 | 0.140 / 0.330 / 0.140 (SE 0.035) | 0.091 (0.000 to 0.192) |
| k = 32, stretch, c = 1.378 | 100 | 1.000 | 0.120 / 0.080 / 0.080 (SE 0.027) | 0.038 (0.000 to 0.211) |
| k = 32, stretch, c = 3 | 100 | 1.000 | 0.030 / 0.100 / 0.030 (SE 0.017) | 0.082 (0.000 to 0.344) |
| k = 32, use weight 0.6 to 0.95 | 100 | 1.000 | 0.010 / 0.010 / 0.010 (SE 0.010) | 0.000 (0.000 to 0.178) |
| k = 32, move of 0.1 logit along 2PL slopes (P5) | 100 | 1.000 | 0.130 / 0.050 / 0.050 (SE 0.022) | 0.090 (0.000 to 0.160) |
| k = 32, move of 0.3 logit along 2PL slopes (P5) | 100 | 0.996 | 0.340 / 0.270 / 0.270 (SE 0.044) | 0.132 (0.000 to 0.188) |
| k = 32, move of 1 logit along 2PL slopes (P5) | 100 | 0.959 | 1.000 / 1.000 / 1.000 | 0.359 (0.305 to 0.423) |
| k = 32, temperature 0.9, four options (P3a) | 100 | 0.999 | 0.050 / 0.530 / 0.050 (SE 0.022) | 0.039 (0.000 to 0.187) |
| k = 32, temperature 0.726, four options (P3a) | 100 | 0.996 | 0.370 / 0.890 / 0.370 (SE 0.048) | 0.204 (0.000 to 0.319) |
| k = 32, item specific, SD 0.05 logit | 100 | 0.999 | 0.060 / 0.170 / 0.060 (SE 0.024) | 0.041 (0.000 to 0.169) |
| k = 32, item specific, SD 0.1 logit | 100 | 0.996 | 0.260 / 0.580 / 0.260 (SE 0.044) | 0.136 (0.000 to 0.220) |
| k = 32, item specific, SD 0.25 logit | 100 | 0.977 | 1.000 / 1.000 / 1.000 | 0.283 (0.221 to 0.354) |
| k = 32, item specific, SD 0.5 logit | 100 | 0.918 | 1.000 / 1.000 / 1.000 | 0.584 (0.523 to 0.633) |
| k = 32, one subject of 150 items, +0.5 logit | 100 | 0.992 | 0.700 / 0.700 / 0.670 (SE 0.047) | 0.176 (0.084 to 0.245) |
| k = 32, one subject of 150 items, +1 logit | 100 | 0.969 | 1.000 / 1.000 / 1.000 | 0.321 (0.266 to 0.375) |

Reading of the tier 1 design.
- The order test does not hold its level at J = 1,500. Under the rule of step 5 the false positive rate is 0.065 (SE 0.017) with no effect at k = 32, 0.140 (SE 0.035) under a common shift, 0.080 (SE 0.027) under the stretch of 1.378, and 0.085 (SE 0.020) in the split half at k = 8. Each prior alone does worse, the logit normal prior 0.115 with no effect and 0.120 under the stretch of 1.378, the Beta prior 0.330 under the shift and 0.230 on the U shaped bank. With 1,500 items rho has a small standard error, so a small misfit of the null's prior (A9) shows as a rejection. The order test is therefore reported at tier 1 and does not decide.
- The decision uses sigma_hat with a threshold of 0.22 logit (chosen). In five of the six order preserving cells at k = 32 the 95 percent quantile of sigma_hat is at most 0.211, so the false positive rate of sigma_hat > 0.22 is at most about 0.05 there. The exception is the stretch of 3, an after state with a U shaped p, where that quantile is 0.344. In the cells with an item specific SD of 0.25 or 0.5, a move of 1 logit along the slopes, and one subject raised by 1 logit, the 5 percent quantile is at least 0.221, so the power of sigma_hat > 0.22 is at least about 0.95.
- sigma_hat is biased upward by 0.03 to 0.09 logit at true SDs of 0.1 to 0.5 (medians 0.136, 0.283, 0.584; 0.041 at a true 0.05), and its median is 0 to 0.091 in the order preserving cells. Below about 0.15 logit it does not separate an item specific change from no reorder.
- The confounds sit near the threshold. Temperature 0.726 on four options gives median 0.204 (5 to 95 percent, 0 to 0.319); a move of 0.3 logit along the slopes gives 0.132; one subject raised by 0.5 logit gives 0.176.
- The split half at k = 8 is too noisy for sigma_hat. Its 95 percent quantile is 0.354 with no effect. At k = 8 only the order test is used, with its false positive rate of 0.085 (SE 0.020), and 0.005 on the U shaped bank.
- These rates hold for independent items with binomial noise and the synthetic bank (A2, A9).

The first design, J = 200 items at k = 16 and k = 64, with the same reading.

| condition | rho noise free | rejection rate, both priors, k = 16 | k = 64 |
|---|---|---|---|
| no effect | 1.000 | 0.015 (SE 0.009) | 0.035 (SE 0.013) |
| common shift of 1 logit | 1.000 | 0.065 (SE 0.017) | 0.065 (SE 0.017) |
| stretch, c = 1.5 | 1.000 | 0.015 (SE 0.009) | 0.020 (SE 0.010) |
| use weight 0.6 to 0.95 | 1.000 | 0.020 (SE 0.010) | 0.020 (SE 0.010) |
| move of 1 logit along 2PL slopes (P5) | 0.958 | 0.275 (SE 0.032) | 0.995 (SE 0.005) |
| temperature 0.5, four options (P3a) | 0.986 | 0.140 (SE 0.025) | 0.675 (SE 0.033) |
| item specific, SD 0.25 logit | 0.976 | 0.190 (SE 0.028) | 0.750 (SE 0.031) |
| item specific, SD 0.5 logit | 0.918 | 0.800 (SE 0.028) | 1.000 |
| item specific, SD 1 logit | 0.763 | 1.000 | 1.000 |
| one subject of 20 items, +1 logit | 0.968 | 0.285 (SE 0.032) | 0.950 (SE 0.015) |
| one subject of 20 items, +2 logits | 0.907 | 0.900 (SE 0.021) | 1.000 |

On 200 items the rule of step 5 holds its level, 0.015 to 0.065 over the four order preserving conditions at k = 16 and k = 64, so the failure at J = 1,500 comes with the bank size. Its power at k = 16 is 0.800 for an item specific SD of 0.5 logit and 0.900 for one subject raised by 2 logits. The confounds are rejected at 0.275 (P5) and 0.140 (P3a) at k = 16, so a rejection says that the order changed and not why. Attempt 0 reported this table under the Beta prior alone with the inexact variance; its numbers are in calibration_attempt0.json.

The relink of tier 2, P7, on the J = 200 learners. The base is read with k0 = 64 draws per item, the after state with k1 = 16 on m anchors drawn at random. The target is the after state's true bank mean. RMSE is over 200 learners, in units of proportion correct. The relink does not use the prior, and this table is unchanged from attempt 0 (same seed).

| condition | m = 10, relink / classical | m = 25, relink / classical | m = 50, relink / classical |
|---|---|---|---|
| common shift of 1 logit | 0.037 / 0.067 | 0.023 / 0.047 | 0.015 / 0.026 |
| stretch, c = 1.5 | 0.039 / 0.095 | 0.024 / 0.059 | 0.016 / 0.040 |
| use weight 0.6 to 0.95 | 0.046 / 0.076 | 0.029 / 0.047 | 0.020 / 0.032 |
| item specific, SD 0.5 logit | 0.046 / 0.071 | 0.029 / 0.046 | 0.017 / 0.030 |
| one subject of 20 items, +2 logits | 0.052 / 0.084 | 0.028 / 0.046 | 0.021 / 0.034 |

The relink has the smaller RMSE in all 15 cells, and its mean bias lies within plus or minus 0.007. It wins under the reorders too, as P7 says for the bank mean with random anchors. So the bank mean does not test whether the map is affine, and the tier 2 prediction below is stated on single items. Attempt 0 reports a first fit by Nelder-Mead with a bias of -0.013 under the pure shift; no artifact of that run was kept, so that history is unchecked.

## 5. One prediction per tier, written before any data

Tier 1. The material is the peer's S1 states (irt4llm notes/blueprint.md section 2; notes/state.md, row S1), Llama-3.2-1B-Instruct at the base, three SFT seeds and three GRPO seeds, on the 1,500 item MMLU bank. Each item has 16 sampled answers at temperature 1 per option order and two option orders, so the counts pooled over the orders give k = 32 per item and state. The reading pools the two orders (A8 holds in part). Before any end state is read, the base is split into two halves of 8 draws and the reading runs on the two halves, the no effect case on the real bank.

- T1.0, the reading on the real bank. The order test of the split half of the base does not reject at 0.05 under the rule of step 5. Calibrated false positive rate 0.085 (SE 0.020) with no effect and 0.005 on a U shaped bank, so a false refutation has a chance of about 0.085. Refuted if it rejects. Then A2 or A9 fails on the real bank, and no tier 1 reading of the end states is used until the prior is refitted.
- The order test of each end state against the base at k = 32 is reported with no prediction, since it does not hold its level at J = 1,500 (section 4).
- T1.1, SFT reorders. Each SFT state has sigma_hat above 0.22 logit (chosen threshold). The reason is the import term of P4(b), which is item specific in general. Rung, derivation for the existence of the term, idea for its size reaching the threshold. Calibrated power at least about 0.95 for an item specific SD of 0.25 logit or more; false positive rate at most about 0.05 in five of six order preserving cells. Refuted if any SFT state has sigma_hat at or below 0.22.
- T1.2, GRPO within a tolerance. Each GRPO state has sigma_hat at or below 0.35 logit (chosen tolerance). The assumption is that the GRPO change is a sharpening of the option logits plus a kernel part that is item blind on the bank and meets the step condition (P2(d)). Under that assumption the synthetic counterparts stay below the tolerance, temperature 0.726 on four options with median 0.204 and 95 percent quantile 0.319, and the U shaped stretch with 95 percent quantile 0.344. Refuted if any GRPO state has sigma_hat above 0.35. The tolerance rules out an item specific SD of 0.5 logit (5 percent quantile 0.523). It does not rule out an item specific SD of 0.25 (median 0.283) or a move of 1 logit along the population slopes (median 0.359, 5 to 95 percent 0.305 to 0.423). Rung, idea, since the size of P3a under the real option logits is not derived.
- T1.3, SFT above GRPO. The median sigma_hat of the three SFT states exceeds that of the three GRPO states by more than 0.1 logit (chosen margin, about half the 5 to 95 percent width of sigma_hat at k = 32 in the item specific cells). Refuted if the GRPO median exceeds the SFT median; otherwise undecided, and the cooled base decides. Rung, idea for the direction, derivation for the SFT import term.
- What was seen before this was written. Stability 0.269 at the base against 0.797 to 0.836 for GRPO (state.md, row S1), and person fit lz of +2.0 to +2.5 for GRPO under the population 2PL (results/Q2a/personfit.csv). Row S1 also holds a reading of the change of the modal answer against a split half floor at k = 8. Its symmetric null gives an excess over the floor of 0.130 to 0.137 for SFT and 0.112 to 0.145 for GRPO, so the peer withdrew a reorder split by route on that statistic. That statistic counts mode changes, not the order of p, and the order of p under each route was not seen. F34's scale of 1.378 is a number of the signature project, on hold, with three candidate mechanisms (panel:751).
- If T1.1 to T1.3 hold, reward on this model changes the bank by a sharpening plus one increasing map within 0.35 logit per item, SFT reorders, tier 2 relinks GRPO states by a curved map with the option temperature, and the research program's row "M tier 1 feeds G tier 2" holds with one map per GRPO state. If T1.2 fails, the residual is read by subject (P6) and along the population slopes (P5), tier 2 needs anchors per subject, and "reward changes use, not order" is withdrawn on this model. If T1.1 fails, the SFT import term is smaller than 0.22 logit per item on this bank, and SFT and GRPO are not told apart by order at k = 32. If T1.3 is undecided, the cooled base (decision 3) reads GRPO against the base at tau = 0.726 and removes P3a.

Tier 2. For states with sigma_hat at or below 0.22, the affine relink fitted on m = 25 anchors predicts the right answers on each of the other items within the noise. The statistic is the binomial deviance of the relinked p on the items outside the anchors, against its parametric bootstrap distribution under the fitted C_aff, with eta0_hat redrawn from its posterior in each bootstrap draw. Not yet calibrated; its calibration repeats section 4 with this statistic.
- Refuted if the deviance exceeds the 0.95 quantile (chosen) of that distribution on two or more of the states read. An exceedance has three causes, a curved map (C_mono and not C_aff), an item specific residual below the tier 1 tolerance, and error in eta0_hat if the bootstrap holds it fixed. The third is removed by the redraw. The first two are told apart by refitting with an increasing step function on more anchors; if the deviance then falls within its band, the map is curved.
- The bank mean is not the test. In the synthetic table the relink beats the classical estimate under every condition, reorders included.
- Q2b of the peer. Its linked method enters a shift delta of all b in the prediction step only, fitted on the base (irt4llm blueprint, Rewrite for Q2b). Under the 2PL a common shift s of all b gives a(theta - b - s) = a((theta - s) - b), so it is absorbed into theta, and lz at the fitted theta is unchanged up to the prior (derivation). On the six S1 states lz is therefore already seen, +2.03, +2.18 and +2.52 for GRPO (results/Q2a/personfit.csv), and is not a prediction. The prediction is for the 12 Q1 final states, unseen. If GRPO sharpens, lz of each Q1 state stays above +1.645. Refuted if any falls at or below +1.645. A value below -1.645 is a misfit of the other kind, an item specific change that the 2PL does not hold, and sends those states to the residual reading of P6.

Tier 3. On a student log with cohorts, and once an anchor convention is fixed (the Gini anchor point of Strobl et al. 2021), the contrasts of item difficulty between cohorts stay within their resampling spread for items whose position in the curriculum is the same in both cohorts. The reorder sits on items whose position moved.
- The test. An item is flagged when its contrast lies outside its 95 percent bootstrap interval over learners (chosen). The flag rates of moved and unmoved items are compared by a one sided Fisher exact test at 0.05 (chosen). Refuted if the test does not reject and the flag rate of moved items is within 0.05 (chosen tolerance) of that of unmoved items. Not calibrated; its false positive rate and power follow the rule of section 4 on resampled cohorts before use.
- It needs cohort and curriculum position fields, which the audit of surrogate S, tier 1, decides.
- If it holds, scale stability holds for unmoved items and drift is an effect of the order of learning, a thesis result. If it fails, drift is item level (wording, exposure), and a tracker must estimate the item side online, which is surrogate G, tier 2.

## 6. The literature

| source | what this theory takes from it | setup to reproduce |
|---|---|---|
| Fischer and Formann 1982 (LLTM, LLRA) | person change written as item change (P1); change as item specific base plus a common delta, with a test of whether effects generalize over ability domains (P8) | the LLRA two occasion design, one learner's base log odds as the item specific baseline and delta tested for equality over subjects |
| Stocking and Lord 1983 | linear linking of two calibrations, mean and sigma, robust weights, characteristic curves; regression of one calibration on the other is wrong, being asymmetric and having error in the regressor (P7) | the robust mean and sigma relink of the Appendix for C_aff, characteristic curve matching for the bank mean |
| Strobl et al. 2021 | anchors are a convention; balanced DIF and a DIF free majority are untestable a priori (P1, P8) | the Gini index anchor point for the shift of P8, with its criterion plot to show clusters of items |
| Rafailov et al. 2023 | the optimum of KL regularized reward (P2a) | none, a closed form |
| Jain et al. (arXiv 2023, ICLR 2024) | the wrapper, its localization by pruning, revival by reverse fine-tuning. When the pretraining prior of the fine-tuning operand is low (a weakly relevant capability), the fine-tuned model exploits the spurious attribute of the fine-tuning data; when that prior is raised, it performs well on the test data "regardless of the presence of the spurious attribute" (section 5.1). That the model learns a wrapper over the weakly relevant capability is their hypothesis, supported by pruning (Fig. 7) and probing (Fig. 8) | their Tracr counter task with a spurious attribute at a low sampling prior, read on a bank that mixes the pretraining task and the fine-tuning task, as a learner with known route for tier 1 |
| Hofmann et al. 2025 (Fluid) | population 2PL held fixed across checkpoints; the peer's slopes a | the checkpoint evaluation of OLMo, read with the order test instead of the fixed 2PL |
| Maia Polo et al. 2024 (tinyBenchmarks) | IRT anchors to estimate accuracy; a test set of 40 specialized models that the authors expect to have correctness patterns that "might be different" by category; the paper reports only the estimation error on them (Fig. 5) | their 40 specialized models as real learners for the tier 2 relink; whether they reorder by category is untested |

The panel (C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md) supplies P2(d), P4(b) and P5 (B3, B4), the temperature term (D9), and the warning that the split between person and items is a choice of response model and anchor set (S1 point 6). Not opened, and not relied on: Bechger and Maris (2015) on item pair DIF, cited inside Strobl et al. as making the same point as P1; Embretson (1991) and Andersen (1985) on models of change; Meredith (1993) on measurement invariance; the equating texts on nonlinear linking. Their primary texts were behind a paywall, so no claim of this file rests on them, including any name for the C_mono relink.

## 7. What the theory cannot show

- Why a change happened. C_mono holds the result of many mechanisms (P2), and a reorder has several causes (P3, P5, P6).
- That the capability stayed the same when the order holds (P4).
- Person change against item change for one learner, or cohort change against item drift for students, without an anchor convention (P1, P8).
- Anything about problems outside the bank. The bank's items are fixed and the reading is of their order.
- Anything about the computation inside the learner. Two learners with the same eta1 are the same object for this reading.
- The calibration of section 4 holds for the synthetic bank. For a fixed model the sampled answers are independent given p (A2), so the untested risk on the real bank is a misfit of the logit normal prior to the real distribution of p (A9). The split half of T1.0 measures it on the base. On the synthetic bank the four priors give mean sigma_hat from 0.039 to 0.161 at a true SD of 0.1 logit (section 4), so sizes below about 0.15 logit are not read, and on 1,500 items the order test does not hold its level.

## 8. Choices for the author and proposed edits to the research program

Decision for the author 1. Which calibration is "the calibration made before".
| option | mechanism | cost | what it shows |
|---|---|---|---|
| the learner's own base | difficulties b_j = -eta0(j) read on the base state, the chain C_0 to C_mono | none beyond the base reading | whether the learner's own ranking of items survives the change |
| the population 2PL (Fluid) | slopes and difficulties from leaderboard models, C_pop(a) | none, the peer holds the parameters | whether the change moves the learner along the population axis or off it (P5) |
| both | the order test on the own base, then C_pop(a) on the residual | both readings, CPU only | the full decomposition of P6 |
Pick, both, with the own base first. The population slopes alone mix a stretch into the person step (panel D4, D9).

Decision for the author 2. The reading of a model state.
| option | mechanism | cost | what it shows |
|---|---|---|---|
| sampled answers at temperature 1 (the peer's S1) | y ~ Binomial(32, p) per item, two option orders pooled | exists for 7 states | sigma_hat as calibrated in section 4; the order test only at J = 200 or for the split half |
| option probabilities of a forced letter | exact p and the option logits z | one forward pass per item and order | P3a tested directly, no binomial noise |
Pick, option probabilities where the format allows, with sampled answers beside them. Only the option logits separate a temperature from a reorder without a new run.

Decision for the author 3. The cooled base.
| option | mechanism | cost | what it shows |
|---|---|---|---|
| no cooled base | the stretch is read as part of h | none | the order test only |
| the base read at tau in {1, 0.726, 0.5} | the counterfactual of panel D9 | about 12 min 27 s per temperature on one L40, the peer's S1 base job for 48,000 answers (irt4llm notes/state.md, row S1) | whether the GRPO state equals the base made more certain plus a shift |
Pick, the cooled base. It is the one control a student cannot give, it removes P3a from the account of a GRPO reorder, and it decides T1.3 when that rule is undecided.

Decision for the author 4. Whether the peer's Q2 serves as M's tiers (the research program, section 9).
| option | mechanism | cost | what it shows |
|---|---|---|---|
| peer data as tier 1 and tier 2 material | the S1 states and the Q2b linked method read with the order test, read only | CPU on the cluster, no change to the peer | M's tier 1 on real states with known routes, and the C_shift relink that Q2b uses |
| a separate project | M builds its own learner (Jain et al. counter task) | a build and GPU time, not measured | a route known to install a wrapper over a weakly relevant capability |
Pick, peer data first. The Jain setup follows only if the peer states give no reorder at all, since then the reading has not yet been shown to see one on a real model.

Decision for the author 5. Four option orders. The two orders of S1 control P3b only in part (A8). Reading the bank under all four cyclic orders doubles the S1 sampling cost (about 12 to 32 min per state and order pair, row S1) and closes P3b. Pick, only if T1.3 is undecided after the cooled base.

Proposed edits to the research program.
1. Section 4, M, Theory. Replace "A change of use predicts a shift that a small relinking repairs. A change of capability predicts a reordering of items." with "A change that is a strictly increasing function of each item's base log odds keeps the order of items, and a relink of one function repairs it. A change that depends on more than the base log odds can reorder items, and does so when it crosses at least one pair. A decoding temperature on multi option items reorders without any change of use or capability, a change of use reorders when the fallback differs by item, and a capability gain that tracks the base log odds keeps the order (theory/M.md, P3 to P5)."
2. Section 3, idea 1, and M's Theory line. Add, as an idea and not a prediction, that when the capability is weakly relevant Jain et al. find that the fine-tuned model exploits the spurious attribute of the fine-tuning data, so on a bank that mixes tasks the wrapper may reorder items by task; with a strongly relevant capability they find no dependence on the attribute.
3. Section 4, M, Tier 1. Read "the way a base calibration fails" as one of three outcomes, an increasing map, a move along the population axis, or an item specific residual, with the cooled base and the option orders as the two controls, and with the size sigma_hat and a stated tolerance beside the order test.
4. Section 4, M, Tier 3. Add that between cohorts the common shift is not identified apart from cohort ability, so scale stability is tested as the absence of a reorder, under a stated anchor convention.
5. Section 6, the row "M tier 1 shows calibrations fail by shift". Replace "by shift" with "by an increasing map".

## Fix log of attempt 1

| item | what changed | line |
|---|---|---|
| C1 | P4(b) restated, the outcome part gives a pure person change and the teacher route adds the item specific import term; T1.1 rests on that term | 72, 186 |
| C2 | Recalibrated at J = 1,500, k = 32 and the split half at k = 8 (new script and output). The order test exceeds its level there (0.065 to 0.140), so the decision moved to sigma_hat with a threshold of 0.22 and a GRPO tolerance of 0.35, each with its assumption and calibrated rates; P3b left open by two orders is A8 | 104 to 166, 184 to 188 |
| C3 | Jain et al. claim stated with its condition, low pretraining prior, and the wrapper as their hypothesis; proposed edit 2 is an idea; sources/reading.md row narrowed | 210, 260 |
| M1 | P5 condition rewritten as the crossing rule with the sign of delta, and delta A'(eta0) > -1 when a is a function of eta0, with the two counterexamples | 76, 92 |
| M2 | The bound restated, C_mono contains every g(eta0) with g' > -1 | 74 |
| M3 | P2(d) holds for the flow; a finite step needs eta f'(p) > -1 | 64 |
| M4 | P3a called a change of the reading that breaks A3; P3b named as the one derived reorder by a change of use | 66 to 68 |
| M5 | Proposed edit 1 reads "can reorder", "when it crosses at least one pair" | 259 |
| M6 | The c > 1 part removed from tier 1; each of T1.0 to T1.3 has a refutation rule that matches its wording | 184 to 188 |
| M7 | Q2b lz restated as seen on the six S1 states (absorbed shift); prediction moved to the 12 Q1 states, with the case below -1.645 | 195 |
| M8 | Tier 2 lists the three causes of an exceedance; the bootstrap redraws eta0_hat; curvature is told apart by a step function refit | 192, 193 |
| M9 | M outputs the order test and sigma_hat only; C_shift and C_aff wait for the tier 2 deviance | 29 to 34 |
| M10 | P5 number 0.958, from calibration.json (0.9578 at k = 16, 0.9587 at k = 64) | 76 |
| M11 | Maia Polo et al. row says "might be different" and that only the estimation error is shown; reading.md row narrowed | 212 |
| M12 | k = 32 per item, 16 per order and two orders pooled | 182 |
| m1 | P3a number 0.986, as in the table | 67 |
| m2 | h strictly increasing in C_mono, P2 and C_pop | 15, 60 |
| m3 | P4(a) compares the Rasch shift with the KL optimum of P2(a), not with the curved use change | 71 |
| m4 | P6 residual marked definition, its reading idea | 78 |
| m5 | P7 names the attenuation bias that does not shrink with m, with the ratio 2.47 against 2.24 | 80 |
| m6 | P7 gives both reasons of Stocking and Lord and why the asymmetry is intended here | 80 |
| m7 | P8 calls the LLRA test a test of one common effect against domain effects, the block structure of P6 | 82 |
| m8 | Beta variance divided by 1 - 1/k in the code; stated in step 1 | 98 |
| m9 | "Holds its level" replaced by the measured rates per design; the failure at J = 1,500 stated | 143, 166 |
| m10 | Section 7 names the prior misfit (A9) as the risk; A2 restated | 45, 223 |
| m11 | "each of kind A or B by a Bernoulli(0.5) draw" | 67 |
| m12 | Jain et al. as arXiv 2023, ICLR 2024 | 210 |
| m13 | The name of the C_mono relink removed; unopened texts carry no claim | 214 |
| m14 | Tier 3 test is a one sided Fisher exact test with a chosen flag rule and tolerance, not calibrated | 198 |
| m15 | The peer's mode change split half reading and its withdrawn route split named under what was seen | 189 |
| m16 | Nelder-Mead history marked unchecked, no artifact | 178 |
| m17 | Separation table row 5 follows the crossing rule | 92 |
| m18 | Proposed edit 2 is an idea with the condition | 260 |
