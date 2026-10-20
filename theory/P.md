# Theory of surrogate P, the process of solving

Written 19 October 2026 for unit TP, against section 4 of the research program (C:/Users/YuanW/PLMS/surrogates/program/research-program.md). Revised the same day in the fix attempt after the review C:/Users/YuanW/PLMS/surrogates/theory/P-review.md; the fix log is at the end. Rungs are idea, derivation, checked numerically, and source. A source rung names the opened text and its section. Panel sections (D17, D33, B17 and others) are in C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md. The numbers of section 4 come from C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_p/calibration.py, output in C:/Users/YuanW/PLMS/surrogates/results/theory_p_calibration.txt. The numbers of the per-token departure in D5 come from C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_p/grpo_departure.py, output in C:/Users/YuanW/PLMS/surrogates/results/theory_p_grpo_departure.txt.

## 1. The object

```
x                     a problem with a declared step structure v_1, ..., v_k and key a*(x) = v_k
Ans(x)                the declared answer set (four letters, or the residues mod m for a numeric key)
pi_w(y | x)           the model; y = (y_1, ..., y_n) a sampled chain that ends in an answer A(y) in Ans(x)
b_0 = 0 < b_1 < ... < b_k     token positions where the chain has written v_1, ..., v_k (cuts)
R(y) = 1[A(y) = a*(x)]        the outcome of a run

F_j(x, y)   = softmax over Ans(x) of pi_w( . | x, y_{<=b_j}, s )              forced answer after cut j
              s is a fixed forced-answer suffix; F_0 is the forced answer with no chain
o_j(x, y)(a) = P_{y' ~ pi_w( . | x, y_{<=b_j})}[ A(y') = a ]                 outcome distribution per prefix
tau_text(y) = min{ j : the string of A(y) occurs in y_{<=b_j} }               first appearance in the text
tau_F(y)    = min{ j : argmax F_{j'} = A(y) for every j' >= j }               first commitment of the forced answer
```

F_j is what the model answers in one pass if the chain stops at cut j. o_j is where the runs from that prefix end. Both are defined per item and per sampled chain. F_j is exact given the prefix. o_j has to be estimated from S continuations. tau_text and tau_F are recorded per chain as descriptions, the point where the answer first appears that the program asks for; they enter no test until a statistic on them is calibrated (section 8, choice 5).

Training maps a base pi_0 to pi_1. The surrogate's question becomes a hypothesis on the runs, for items with 0 < p_0(x) < 1, where both conditionals exist.

```
H_freq :  pi_1(y | x, R) = pi_0(y | x, R)   for R in {0, 1} and every item x with 0 < p_0(x) < 1
          (training changed only p(x) = P(R = 1 | x))
H_proc :  H_freq fails for some x; the distribution of runs inside an outcome changed
```

For a declared run property phi(y) in {0, 1} (a check step, a step count above a bound, a repaired error), H_freq gives a prediction per item (panel D33).

```
a_R(x)                 = E_{pi_0}[ phi(y) | x, R ]          base rate inside outcome R
rate_1(x) predicted    = p_1(x) a_1(x) + (1 - p_1(x)) a_0(x)
```

Tier 1 asks whether the chain does work at all on the item set, since H_proc is empty where the chain is idle. Tier 2 tests H_freq against H_proc. Tier 3 asks the same of an attention probe belief along the chain.

The object answers one part of "what is learning". It says whether a change of accuracy came with a change of how runs are made, as seen in the text a model writes and in its answers at cuts. It leaves what happens inside one forward pass, whether the written chain is the whole computation, reach and generality across problems (surrogate R), the response to new evidence (surrogate E), and every student question.

## 2. The assumptions

| | assumption | it fails when | what fails with it |
|---|---|---|---|
| A1 | F_j is the model's answer from that prefix, not an effect of a forced suffix the model rarely sees mid chain | forced answering falls below probe accuracy for distilled 1.5B and 7B models, likely from off-policy prompting, and drops in the second half of reasoning for 14B and 32B (Boppana et al. 2026, appendix F) | a low F_j at early cuts is read as "the chain is needed" |
| A2 | the change of F_j with prefix length that does not come from content (format drift) adds a constant on a known scale, the same on items that need the chain and items that do not | drift adds on another scale, or differs between item kinds | the difference in differences of section 4 is biased; its false positive rate is 0.37 to 0.98 under the wrong scale and 0.70 to 1.00 under unequal drift (section 4, B) |
| A3 | the path tokens and the continuations come from one sampling kernel (same temperature and truncation) | greedy base paths, as in Forking Fast, or continuations drawn with a kernel other than the one that wrote the path | o_j is no longer a martingale along the path; drift along a path is not a fork and not noise |
| A4 | continuations from one prefix are independent, so answer counts are multinomial | shared seeds or batch effects in the sampler | the noise law and the fork test lose their stated false positive rate |
| A5 | the built problem has no route to the key except its steps | the key is guessable from the wording or from a bias of the generator | one-pass accuracy above chance at large k; "needed" items are not needed |
| A6 | the reward depends only on the outcome, the KL penalty is on the whole sequence, the loss has no length weight, and training reaches the fixed point, with each item free to reach its own | finite training, plain gradient narrowing, shared weights at finite capacity, format or length terms in the reward, a per-token KL term in the loss, a 1/\|y\| weight (GRPO as published, D5) | the D33 prediction of H_freq is not the null of the training that ran |
| A7 | phi is declared from the text by a rule fixed before training and applied unchanged after | training changes the wording, so the rule catches more or fewer runs with the same content | a change of the rule's coverage reads as a change of process |
| A8 | rates are computed per item, then averaged | rates are pooled across items inside an outcome | pooled rates move under H_freq when p(x) rises unevenly across items (section 3, D7) |

## 3. What can be derived

D1. A chain adds serial computation. Source, Merrill and Sabharwal 2024, introduction, Theorems 1 and 2 (lower bounds) and Theorem 4 (the SPACE upper bound). With t(n) decoding steps, TIME(t(n)) is contained in CoT(t(n)), which is contained in SPACE(t(n) + log n). With no steps, log-precision transformers lie in uniform TC0. With O(log n) steps they lie in L. With O(n) steps they recognize every regular language. With polynomial steps they recognize exactly P. The separations assume TC0, NC1, NL and P do not collapse, and the lower bounds assume strict causal saturated attention and projected pre-norm. Panel D16 states the same in plain terms (derivation from the architecture). Condition. These are statements about input length growing. At one fixed problem size a fixed-depth network whose width grows with the map can store any finite map, so no theorem says a chain is needed on a finite built item set. Need at finite size is a property of a model and an item set together, and it is measured, never assumed (A5).

D2. The outcome distribution is a martingale along chains sampled with the kernel of the continuations (A3). Derivation. Let o_t be the outcome distribution after t tokens, with y_{t+1} drawn from pi_w( . | x, y_{<=t}).

```
E[ o_{t+1} | y_{<=t} ] = o_t                                     (tower rule)
E[ sum_t || o_{t+1} - o_t ||^2 ] = E[ || e_{A} - o_0 ||^2 ] = 1 - sum_a o_0(a)^2
```

The total squared movement of o along a sampled chain is fixed by the outcome uncertainty at the start. Checked numerically (section 4, A1). On a toy chain with two outcomes and o_0(1) = 0.739, the summed squared increments in the vector norm average 0.3842 against the predicted 1 - sum_a o_0(a)^2 = 0.3855. The one-coordinate form gives 0.1921 against o_0(1)(1 - o_0(1)) = 0.1927, half of the vector form. The mean increment is -0.0003. Along the greedy path of the same chain o rises by 0.261, so a greedy base path drifts by construction (A3). Three learners follow.
- An idle chain, where the outcome is fixed before writing, has o_0 near one point and no movement.
- A coin at the first token puts all movement at t = 1.
- A working chain spreads the movement over the cuts where intermediate values are written.
The first two differ from the third by where the movement sits, not by its total. Forking Fast departs from the sampled path twice. It uses greedy base paths, and it resamples only alternative tokens with probability at least 0.05 at temperature 1 (its section 2). Its o_t is therefore an outcome distribution from x_{<t} with a truncated token at t, which is not on a sampled path, and its drift between forks need not average to zero.

D3. What the cut tests. Derivation from the definitions.
- Necessity. Accuracy of F_0 below the accuracy of the normal completion means this item set needs the chain for its accuracy.
- Dependence. F_j changing with j means the one-pass answer depends on the prefix.
- Use. A changed intermediate value v_j' placed at cut j, with the run continued, gives an answer that follows v_j' when the model uses the tape. The answer after the change is known by construction, h(v_j') on the declared steps.
A flat F_j shows the chain is not needed for the answer. It does not show the chain is not used, since a model can compute the answer in one pass and recompute it in the text (panel D20). Only the use test separates the two. Lanham et al. 2023 measure necessity and dependence by early answering and use by adding mistakes. Their adding mistakes is the use test without a known answer after the change.

D4. Two explanations that the cut cannot tell apart, and the observation that can. Derivation from D2 and D3; the filler and paraphrase rows are source.

| explanation pair | same observation | observation that separates them |
|---|---|---|
| chain idle; answer computed twice (one pass and text) | flat F_j | use test (D3) |
| chain needed; forced suffix fails mid chain (A1) | low F_j at early cuts | items with the answer given in the question, read with the same suffix at the same cut positions |
| computation in the chain; format drift (A2) | F_j rises with prefix length | a filler prefix of equal length (Lanham et al. 2023 section 2.5), and the difference against given items on the scale where drift adds |
| content of the steps; information hidden in the wording | answer depends on the prefix | a paraphrased prefix (Lanham et al. 2023 section 2.6) |
| a fork; sampling noise | a jump of the estimated o | a test with a stated false positive rate at the S used (section 4, A2) |
| a working chain; a coin at the first token | o moves | where the movement sits relative to the cuts (D2); no calibrated statistic yet for the first W tokens (section 8, choice 5) |

Filler tokens carry computation when a model is trained with dense parallelizable demonstrations (Pfau, Merrill and Bowman 2024, section 4.3, 3SUM with a 4-layer 34M Llama). A model trained on instance-adaptive chains did not use filler tokens on 3SUM (their section 4.3). So a null filler result rules out computation carried by filler tokens on this item set. It does not rule out computation hidden in the representations of the chain tokens themselves.

D5. The sequence-level tilt keeps the runs inside each outcome, and GRPO as published is outside its class. Derivation, extending panel D33 and B17 point 1. Let the reward be any function of the outcome, scaled per prompt, and the objective E[c(x) R(y)] - beta KL(pi( . | x) || pi_0( . | x)), with the KL on whole sequences.

```
pi*(y | x)      = pi_0(y | x) exp( c(x) R(y) / beta ) / Z(x)
pi*(y | x, R)   = pi_0(y | x, R)                   for R in {0, 1}
```

The factor is the same for every run with the same outcome, so it cancels inside the outcome. This covers plain reward with a sequence-level KL penalty, a group-normalized advantage held fixed in the step with c(x) = 1 / sd of R, and the limit beta to 0, which conditions pi_0 on R = 1. Only p_1(x) differs between these members. Checked numerically (section 4, C). Over 2000 random items the exact tilt changes the within-right rate by at most 1e-4.

GRPO (Shao et al. 2024, eq. 3, eq. 4 and section 4.1.2) departs in two terms. It adds a per-token KL estimate to the loss, not to the reward, and it weights each token by 1 / |o_i|. With the gradient not passing through the prefix distribution, the stationary rule at prefix h with exact per-token KL is pi(a | h) = pi_ref(a | h) exp(c q(h, a) / beta) / Z(h), where q(h, a) is the success probability after token a under the trained policy (the rule is from the review, re-derived). Summing its logarithm along a run, with q(h_D) = R(y), gives

```
log pi(y | x) / pi_ref(y | x) = (c / beta) ( R(y) - q(x) ) + sum_{t=0}^{D-1} KL( pi( . | h_t) || pi_ref( . | h_t) )
pi(y | x, R = 1) proportional to pi_ref(y | x, R = 1) exp( sum_t KL_{h_t} )
```

Derivation, checked numerically (largest error 3.2e-11 over 200 random token trees of depth 3 with 3 tokens per step). Inside the right outcome, a run gains mass by the summed per-token KL of the prefixes it passes. A prefix after a wrong step is a fork between repair and failure, where the trained policy moves most from the reference, so repaired right runs gain (idea; the tree numbers check the direction). On the same trees, with equal run lengths and c = 1, the exact per-token form at beta = 0.5 raises the within-right rate of repair (the run passes a prefix whose base success probability is below half that of the item, and ends right) by 0.0228 in mean (median 0.0158, range -0.0070 to 0.1179), positive in 0.950 of 179 trees. It moves a random run property by 0.0219 in mean absolute value with signed mean -0.0021. The k3 estimator of GRPO, defined only where beta > c, raises the repair rate at beta = 2 by 0.0050 in mean (range 0.0001 to 0.0168), positive in all 179 trees. The sequence-level tilt moves both by 0.0000 at nearly equal accuracy (0.864 against 0.854 at beta = 0.5). The length weight acts as a reward c (R(y) - p(x)) / |y| with the advantage held fixed. Under a sequence-level KL its fixed point is pi_0(y | x) exp(c (R(y) - p(x)) / (beta |y|)) / Z(x), so shorter right runs and longer wrong runs gain (derivation, not checked numerically; the joint effect of both terms is not derived).

So the D33 null holds per item for the D5 class and not for GRPO as published. Conditions of the D5 class. It fails when a reward term depends on the run beyond its outcome (a format reward, a length penalty), when the KL enters per token in the loss, and when token losses carry a 1/|y| weight. The tree computation ignores clipping and the old-policy ratio, and it is a table of probabilities, not a network.

D6. Plain gradient narrows inside the right outcome. Derivation, panel B17 point 2. With one logit per run and the plain policy gradient, two right runs y and y' obey d(l_y - l_y')/dt = (1 - p)(pi_y - pi_y'). The more probable right run gains faster. So finite training by the plain gradient moves the within-right rate of phi toward the value of phi on the base's most probable right runs. That is a change of the runs inside an outcome with no new method, and H_freq fails for it. Checked numerically (section 4, C). With 6 methods per item and accuracy raised by half the gap to 1, a property held only by the modal right method rises inside right runs by 0.070 in mean (median 0.066, smallest 0.000) over 2000 items. A property drawn at random per method moves by 0.035 in mean absolute value, with signed mean 0.0009.

D7. Pooling across items breaks the null. Derivation. Take two item types, p_0 = 0.2 with a_1 = 0.8, and p_0 = 0.8 with a_1 = 0.2. Raise each p by half its gap to 1 under the exact tilt. No item changes inside its outcome. The pooled within-right rate moves from 0.32 to 0.44 (section 4, C). So every tier 2 rate is per item first (A8).

D8. Changes inside an outcome that tier 2 must separate. Derivation from D5 and D6, with the last row an idea.

| change of training | within-outcome rate of phi |
|---|---|
| outcome-only reward in the D5 class at its fixed point (H_freq) | equal to the base rate, per item |
| plain gradient narrowing | moves toward phi on the base's most probable right runs |
| GRPO as published at its stationary point | repair rate rises inside right runs (per-token KL); short right runs gain (length weight) |
| A6 fails by another route (finite training through shared weights, a reward term beyond the outcome, a new method) | moves toward values of phi that are rare among base right runs of high base probability |

No judge that reads only the outcome changes within-outcome rates at the D5 fixed point, whatever it reads (D5; panel D17 point 1). So a change of the last row shows that A6 failed. It does not name the route. The rate alone cannot separate the second row from the last when phi is common on the base's most probable right runs. They separate when the direction of the change differs from what those runs carry. In practice that means the change of phi conditioned on the base log probability of the run (idea).

D9. The published readings, with their conditions. Source.
- Lanham et al. 2023, sections 2.3 to 2.6 and 3. One 175B RLHF model, 8 multiple choice tasks, 100 chains per problem. Early answering area over the curve ranges from 0.02 (ARC Easy) to 0.44 (AQuA), with MMLU at 0.12. Filler text gave no accuracy gain, and paraphrase kept accuracy. On 7 of 8 tasks the 13B model changed its answer after the chain more often than the 175B model. Models below 13B gained little accuracy from chains (section 3.1, appendix C).
- Boppana et al. 2026, sections 3 to 6 and appendices A, C, F, I. Performativity rate (difference in slope of forced answer accuracy and monitor accuracy) for DeepSeek-R1 is 0.505 on MMLU-Redux and 0.010 on GPQA-Diamond. By whole responses on MMLU, inflections per step are 0.020 in responses whose probe starts at 0.9 or above and never falls below it (High Confidence, 215 of 522) against 0.045 in the rest for DeepSeek-R1 (Table 2), and 0.053 against 0.096 for GPT-OSS 120B (Table 7, 124 of 514). Backtracking is the rarest type, 0.001 against 0.003 per step for R1. No such table exists for GPQA-Diamond. By single steps, with a probe shift of 0.2 between consecutive steps and a window of 10 steps, reconsiderations follow a probe shift about twice as often as in windows without one on MMLU (R1, section 6.2; GPT-OSS 59 against 35 percent, appendix I). On GPQA-Diamond R1 shows no significant trend and GPT-OSS the reverse. Section 6.2 and appendix I disagree on the direction for GPT-OSS, and every step-level result changes with window size and threshold. The abstract's "almost exclusively in responses where probes show large belief shifts" claims more than the ratio near 2 in Tables 2 and 7. Linear probes stayed near chance where attention probes reached 88 percent (appendix C).
- Bigelow et al. 2026 (Forking Fast), sections 2, 3 and appendices A, D, F. Answer counts per position fit a multinomial within 2 percent of the independent null through S = 200. The log-log slope of replicate distance against S is -0.49. Smoothing by change points and kernel pooling raised the effective sample size 3.3 times at S = 30 and 5 times at S = 5 (4-token spacing, Llama-3-8B-Instruct, tinyMMLU). The advantage near forks shrank at a fork threshold of 0.15 and reversed at 0.20. Its credible intervals covered 48 to 64 percent at nominal 90.

## 4. The calibration of the tier 1 reading

Two readings are calibrated, the forced answer at a cut (part B of the script) and fork detection in o (part A). The tier 2 test is calibrated in part C. Each case below has its truth set by construction.

B. The forced answer at the last cut before the key is written. Reference learners on problems with k = 8 steps and a one-pass capacity of c = 2 steps.

```
cut read:       j* = k - 1 = 7, the last cut before the key is written (k - j* <= c)
given item i:   z_ij = logit(g_i) + d_given * j/k + sigma * eps_ij,     g_i ~ Beta(9, 1)
needed item i:  z_ij = logit(q_i) + d_needed * j/k + sigma * eps_ij,    q_i ~ Beta(2, 8)
                with probability w the forced answer reads the prefix: z_ij = logit(0.9) + sigma * eps_ij when k - j <= c
key probability at cut j = sigmoid(z_ij);  sigma = 0.5;  d_given = d_needed = 0.5;  eps standard normal
statistic:  T = mean_needed( z_ij* - z_i0 ) - mean_given( z_ij* - z_i0 ),  one-sided Welch t at alpha = 0.05
```

T reads the cut before the key, so the forced answer must take the last step from the written intermediates; at cut k it would copy the written key. T is a dependence statistic (D3). The case with no effect is w = 0, a learner whose forced answer does not depend on the prefix beyond drift. The case with a planted effect is w > 0. Its size is the rise of the key probability at cut j*, averaged over needed items. The false positive rate is the rejection rate at w = 0 over 2000 replications. The power is the rejection rate at w > 0.

| items per kind | w = 0 (false positive rate) | w = 0.05 (rise 0.03) | w = 0.10 (rise 0.06) | w = 0.20 (rise 0.12) | w = 0.40 (rise 0.25) |
|---|---|---|---|---|---|
| 25 | 0.048 | 0.128 | 0.274 | 0.608 | 0.968 |
| 50 | 0.038 | 0.223 | 0.489 | 0.898 | 1.000 |
| 100 | 0.051 | 0.373 | 0.781 | 0.994 | 1.000 |
| 200 | 0.047 | 0.608 | 0.959 | 1.000 | 1.000 |

The rises in the header are means over replications; across rows they lie within 0.030 to 0.031, 0.061 to 0.062, 0.121 to 0.123 and 0.245 to 0.246. With 100 items per kind the test holds its level (0.051) and detects a rise of 0.06 in key probability with power 0.78. Two cases break it, both with w = 0.
- Taken on the probability scale while drift adds on the logit scale, the false positive rate is 0.372, 0.590, 0.826 and 0.983 at 25, 50, 100 and 200 items. Equal logit drift moves a key probability near 0.2 more than one near 0.9.
- With drift 1.0 on needed items and 0.5 on given items, the false positive rate is 0.699, 0.930, 0.997 and 1.000.
So the given items correct for drift only on the scale where drift adds and only when it is equal across kinds. The filler prefix of equal length on both kinds is the check of A2 and is run before the test.

A. Fork detection in o. T = 100 positions, 5 answers, base distribution (0.6, 0.2, 0.1, 0.05, 0.05). The case with no effect is a constant o. The planted effect is a jump at position 50 that moves total variation delta from the first answer to the second. Every rate is over 400 replications. Each position gets one multinomial of size S; Forking Fast draws S continuations per alternative token, so its sample per position is larger and this simulation understates it.

| rule | S = 30 | S = 100 | S = 200 |
|---|---|---|---|
| raw rule, TVD of adjacent estimates > 0.10, false forks per pair | 0.738 | 0.353 | 0.105 |
| adjacent chi-square, Bonferroni over 99 pairs, any false fork per chain | 0.003 | 0.028 | 0.022 |
| same, power at delta = 0.2 / 0.3 | 0.018 / 0.062 | 0.207 / 0.762 | 0.725 / 0.993 |
| pooled windows of 10 positions each side, any false fork per chain | 0.022 | 0.013 | 0.040 |
| same, power at delta = 0.1 / 0.2 | 0.138 / 0.938 | 0.890 / 1.000 | 1.000 / 1.000 |

In this simulation a threshold rule in the form of Forking Fast's appendix D marks a fork at 10 percent of pairs even at S = 200 with no fork present, so it describes a high-S curve and is not a test. A test between adjacent positions holds its level and needs S near 200 for a jump of 0.2. Pooling 10 positions on each side, which assumes o is flat between forks as Forking Fast observed, detects a jump of 0.2 at S = 30 with power 0.94. Its locating precision is the window, 10 positions, and it cannot test the first or last 9 positions. It is a test in the spirit of their pooling, not their estimator, which is PELT change point detection with an exact multinomial cost and kernel pooling within segments.

C. The tier 2 test, per item. N = 100 items, 32 runs before and 32 after per item, 6 methods per item, accuracy raised by half the gap to 1. The statistic is the per-item change of the within-right rate, t-test of its mean over items at alpha = 0.05, 500 replications. In the planted case a share s of right runs after training is set to phi = 1; about half already had it, so the planted change of the within-right rate is s (1 - a_1(x)), averaged over the items drawn.

| case (truth by construction) | change of the within-right rate | rejection rate |
|---|---|---|
| exact tilt, no change inside outcomes | 0 | 0.052 |
| planted, share s = 0.05 / 0.10 / 0.20 / 0.40 | 0.025 / 0.050 / 0.098 / 0.197 | 0.516 / 0.974 / 1.000 / 1.000 |
| plain gradient narrowing, phi drawn at random per method | 0.035 in mean absolute value, signed mean near 0 | 0.060 |
| plain gradient narrowing, phi held by the modal right method | 0.070 in mean | 0.996 |

The test holds its level under the tilt and detects a planted change of 0.05 with power 0.97, and of 0.025 with power 0.52. The random-phi narrowing row is not a level, since H_freq is false there (D6). Its rate of 0.060 shows that the t-test of the mean change detects only changes with a common direction across items, and changes of mixed sign pass it. It flags narrowing with rate 0.996 when phi goes with the modal right method, so a rejection alone does not show a new method (D8).

What this calibration does not show. The reference learners are equations, not networks. They calibrate the statistics and their noise, not the claim that a real model's chain is used. A trained network with known truth is the next calibration, and section 8 proposes the setup to take from Prystawski, Li and Goodman 2023. Their Theorem 3.1 shows that a risk minimizer's direct estimate of a pair never seen together is biased toward the uniform distribution, by |p(y_i | y_j) - 1/|X||, and that the scaffolded estimate through intermediate variables has a smaller bias (section 3, appendix A). The pull of trained transformers toward the marginal is their empirical finding, which their theory does not explain (Limitations). A pair seen together needs no intermediate variables (section 5.3). The reasoning gap on the trained network is measured, depends on the amount of training, and vanishes with fully observed data (section 5.2). So need is known by construction for the data, and for the trained network it is measured (32M parameters, 10 layers, and a 4.5M variant in their appendix).

## 5. One prediction per tier

Tier 1. On the built problems and a learner from Choice 1, there is a step count k* above which F_0 sits at chance while the normal completion stays above it. For k > k*, T at cut k - 1 (section 4, B) is positive at the calibrated level with 100 items per kind (level 0.051, power 0.78 for a rise of 0.06 in key probability). On given items F_j stays flat within the filler control. On sampled chains of needed items, the pooled window test of section 4, A (S = 30, W = 10, any false fork per chain 0.022, power 0.94 at a jump of 0.2) finds forks within W positions after the cuts where intermediates are written, and on given items it finds none. A changed intermediate v_j' is followed, meaning the answer equals h(v_j'), more often than the answer of the unchanged run equals h(v_j'), by a one-sided two-proportion test over items.
- Refuted if T at cut k - 1 is not positive for any k while the normal completion beats F_0. Then the written intermediates do not carry the one-pass answer one step before the key. Either the suffix fails mid chain (A1, read on given items at the same cuts) or the gain passes through something other than the written content, and the filler and paraphrase tests decide between hidden computation and format.
- Refuted in its fork part if forks on needed items lie no nearer to the cuts than forks on given items. The rule that counts forks within W positions after a cut has no calibration of its own yet, and neither has the follow-rate test; both wait for choice 5 before use.
- If tier 1 holds, tier 2 runs on these items. If it fails, tier 2 cannot read a process change on them, and the research program's feed table ("P tier 1 shows chains do work on built problems") stops at this row.
- A side prediction from Lanham et al. 2023 section 3 and Boppana et al. 2026 section 5.2. For two learners that both gain accuracy from chains on the items, on items within the one-pass capacity of the larger, the smaller shows the larger T. Lanham et al. found little gain from chains below 13B, and Boppana et al. attribute weak forced answers in small distilled models to off-policy prompting, so a single 0.5B to 1.5B learner under option 1a does not test it.

Tier 2. Two predictions, one per training class, from the same base, read per item by the test of section 4, C (N = 100 items, 32 runs before and after, level 0.052, power 0.52 at a change of 0.025 and 0.97 at 0.050). The properties are step count, check steps, the follow rate of a changed intermediate, and the repair rate of a planted wrong step.
- Under the D5 class (outcome-only reward, sequence-level KL penalty, no length weight), the within-outcome rates of the declared properties match the tilt prediction per item, except a shift toward the base's most probable right runs (D6).
- Under GRPO as published, the repair rate inside right runs rises and the step count inside right runs falls, with no change of method (D5, D8). The size of the rise on a network is not derived; on the trees of D5 it is 0.0228 in mean at beta = 0.5 with the exact per-token KL, a change the test detects with power near 0.5.
- Refuted, for the D5 class, if a property inside right runs changes in a direction the base's most probable right runs do not carry. That shows A6 failed by a route that is not narrowing (D8). It does not name the route. A rise of the repair rate under GRPO as published refutes nothing, since D5 predicts it.
- If refuted, tier 3 looks for the change in the attention probe belief at repaired steps. If it holds, tier 3 asks whether the probe belief changes when the text does not.
- The positive control is SFT on chains where a known share f carries a check step. It should move the within-right check rate toward f, and the test must detect it.
- The decision is per training run. The number of seeds per class is section 8, choice 4.

Tier 3. An attention probe for the final answer (Boppana et al. 2026, section 3.2) trained on the small model's chains shows a probe shift, a change of 0.2 in the top answer probability between consecutive steps (their section 6.2), within W steps after the cut where v_{k-1} is written, on more needed items with k > k* than given items at the same cuts. Inflection steps occur at a higher rate per step in responses that are not High Confidence (their section 6.1 rule), as in their Tables 2 and 7 on MMLU.
- Refuted in its first part if the probe is confident (0.9 or above) on the key at the first step on needed items with k > k*. Then the key is decodable before the chain writes the intermediates, and tier 1's reading of "needed" holds for the forced answer only. This would go beyond Ye et al. 2024, whose Result 4 is that a probe reads which parameters are necessary by the end of the problem statement while the values are written step by step. A probe confident at the first token matches Ye's finding only if it reads which intermediates are needed, not the key.
- Refuted in its second part if the inflection rate per step in High Confidence responses equals or exceeds the rate in the rest.
- The probe-shift rule has no calibration on a case with no effect and a planted one; it waits for choice 5 before use.
- A large model run through the DeepSeek route gives the same cut only if the route accepts a prefilled reasoning prefix and returns top-k log probabilities of the next token, as Boppana et al. used through OpenRouter (appendix A.2). Without these only o, by sampling, can be read there.

## 6. The literature

Rows for each source are in C:/Users/YuanW/PLMS/surrogates/sources/reading.md.

| source | what this file uses | setup to take |
|---|---|---|
| Bigelow et al. 2026, Forking Fast, arXiv 2608.19611 | o_t, the multinomial noise law, change point plus kernel pooling, limits near large forks | the S and N grid and the leave-replicate-out reference for any o reading |
| Boppana et al. 2026, Reasoning Theater, arXiv 2603.05488 | forced answering prompt and settings, performativity rate, inflection rates by response and by step, attention probes, the off-policy failure of forced answers in distilled models | forced answering after `</think>` with a one-token answer at temperature 0, and attention probes per layer for tier 3 |
| Lanham et al. 2023, arXiv 2307.13702 | early answering, adding mistakes, filler tokens, paraphrasing, model size | the four tests as the baseline checks of tier 1; the synthetic addition tasks of section 3.2 as a built set with set difficulty |
| Merrill and Sabharwal 2024, arXiv 2310.07923 | what a chain adds as computation, by number of steps | none (theory) |
| Prystawski, Li and Goodman 2023, arXiv 2304.03843 | a built task where the data fix which pairs need intermediate steps, with a theorem on the bias toward uniform and a trained transformer | Bayes nets with local observation, direct against scaffolded against free generation, negative scaffold as control |
| Pfau, Merrill and Bowman 2024, arXiv 2404.15758 | filler tokens carry computation only with dense parallelizable supervision; no transfer from instance-adaptive chains on 3SUM | the reading of the filler control |
| Ye, Xu, Li and Allen-Zhu 2024, iGSM, arXiv 2407.20311 | built grade-school problems with declared dependency graphs, mod 23 values, probes for necessary parameters (Result 4), depth matters | the generator's declared intermediate values for the use test; too costly to pretrain here (GPT2-12-12 from scratch) |
| Shao et al. 2024, DeepSeekMath, arXiv 2402.03300 | the GRPO objective (eq. 3), its k3 KL estimator (eq. 4), the outcome advantage (section 4.1.2) | the training rule whose departure from D5 tier 2 predicts |

Not opened, and so not relied on. Bigelow et al. 2024 (Forking Paths), Bogdan et al. 2025 (Thought Anchors), Macar et al. 2026 (Thought Branches), Kantamneni et al. 2025 (attention probes), and any paper on the length weight of the GRPO loss.

## 7. What the theory cannot show

- Whether the written chain is the whole computation. A followed changed intermediate shows the tape is used, not that nothing else is.
- That a chain is needed at a finite size from complexity theory alone (D1).
- That an attention probe's belief is used by the model. A probe finds information that can be recovered from activations.
- What finite training does between the tilt and narrowing, beyond the per-method flow of D6. Shared weights move unseen items through the learner's kernel (panel B17), and nothing here bounds that.
- The size of the GRPO departure on a network, with clipping, the old-policy ratio and the length weight together (D5).
- Whether a property declared from text means the same act before and after training (A7).
- Anything about students. Their written work would be the only record, and the research program leaves it out.
- Planning inside chains on recall items, which the research program leaves out (section 8 there).

## 8. Choices for the author and proposed edits to the research program

Choice 1. The learner of tier 1.

| option | mechanism | cost | what it shows |
|---|---|---|---|
| a. small instruction-tuned model (for example a 0.5B to 1.5B Qwen or the Llama-3.2-1B of the earlier week) on built chain problems with fresh names | cuts, filler, paraphrase and use tests on a pretrained model | 200 items per kind, 2 kinds, 9 cuts, 1 chain each, so 3,600 forced passes; o at S = 30 on 100 positions per chain, so 400 x 100 x 30 = 1,200,000 continuations; GPU time not measured | whether chains do work on this model and item set |
| b. a transformer trained from scratch on the Bayes net task of Prystawski et al. | need fixed by the data, unseen pairs against pairs seen together; the trained network's need measured as the reasoning gap | their 32M model took about 20 hours on a Titan Xp for 300,000 steps; the 4.5M variant less, not measured here | calibration of the cut reading on a real network whose data fix the need |
| c. GPT2-12-12 pretrained on iGSM | declared intermediate values, probes of necessary parameters | pretraining from scratch on V100 and A100 GPUs (Ye et al., appendix); the cost is not stated in the parts I opened | the cut and the use test where a probe of planning is documented |

My pick is b first, as the calibration on a network, then a for the tier 1 reading. Option c is out of reach.

Choice 2. The statistic of tier 1. Options are the logit difference in differences at cut k - 1 of section 4, Lanham's area over the curve, and Boppana's performativity rate. The area over the curve has no drift correction, and the performativity rate needs a second model as monitor. My pick is the logit difference in differences, with the area over the curve reported beside it for comparison with Lanham.

Choice 3. The test for forks in o. Options are the raw threshold rule, the adjacent test, and pooled windows. My pick is pooled windows of 10 positions at S = 30, with the adjacent test at S = 200 only on items where a fork's exact position matters.

Choice 4. The training arms of tier 2.

| option | mechanism | cost | what it shows |
|---|---|---|---|
| a. D5 class only (group baseline, sequence-level KL in the reward, no length weight) | the training rule whose fixed point is the tilt | one RL run per seed | a test of H_freq with a derived null |
| b. GRPO as published only | the common rule, outside D5 | one RL run per seed | the D5 departures (repair up, short right runs up) mixed with any process change; no derived null |
| c. both from one base | the contrast of b against a reads the per-token and length departures | two RL runs per seed | H_freq under a, and the size of the GRPO departure on a network |

My pick is c with 3 seeds per arm, the test of section 4, C run per seed, and the claim made only when every seed of an arm agrees. The seed rule has no calibration yet.

Choice 5. Five readings have no calibration yet. They are the early-decision statistic for the first W tokens, the count of forks near cuts, the follow-rate test, the tier 3 probe-shift rule, and a statistic on tau_text and tau_F. Option a calibrates all five on reference learners in one synthetic unit before tier 1, a few CPU hours. Option b calibrates each when its tier starts. My pick is a, since tier 1 uses three of them.

Proposed edits to the research program, section 4, surrogate P.
1. Theory line. Replace "Belief shifts at backtracking steps on hard items" with "By whole responses on MMLU, inflections (reconsideration, realization, backtrack) occur at about half the rate per step when the attention probe is confident from the first step (DeepSeek-R1 0.020 against 0.045, GPT-OSS 120B 0.053 against 0.096). By single steps on MMLU, reconsiderations follow a probe shift of 0.2 within 10 steps about twice as often as in windows without one (R1; GPT-OSS 59 against 35 percent). On GPQA-Diamond R1 shows no significant trend and GPT-OSS the reverse, and the step-level results change with window and threshold". The source does not show belief shifts at backtracking steps on hard items. Backtracking is its rarest inflection type (0.001 to 0.003 per step for R1), and the abstract's "almost exclusively" claims more than its tables show.
2. Tier 1. Add the four baseline checks of Lanham et al. 2023 (early answering, a changed intermediate with a known answer, filler of equal length, paraphrase). Read the forced answer at cut k - 1, before the key is written. State that the noise model applies to o, the outcome distribution per prefix, and not to the forced answer, which is exact per prefix.
3. Tier 1 calibration. Name the setup to take for a trained network whose data fix the need, Prystawski et al. 2023. Name the scale assumption of the drift correction (A2).
4. Tier 2. State the prediction of ideal reward training for the class D5 covers (outcome-only reward, sequence-level KL, no length weight), and state that GRPO as published is outside it, with its predicted departures (repair rate up and short runs up inside right runs). Add plain gradient narrowing as the derived alternative (D6), the per-item rule (D7), and SFT on chains with a known share of a property as the positive control.
5. Tier 3. State the requirement for the DeepSeek route, a prefilled reasoning prefix and top-k log probabilities of the next token.

## Fix log of attempt 1

Line numbers refer to this file after the fix. Items marked "none" in the review need no change and are not listed one by one; R2, R5, R9 to R12, R15, R16, R18, R20 to R22, R25, R26, R28 to R33, R35 to R37, R39 to R41, R44 to R48, R51, R54, R56, R64, R67 to R73, R75 to R77 hold, and their numbers were recomputed where the rerun changed them (R31 at line 111, R44 to R47 at lines 148 to 157, R51 at lines 174 to 179).

| item | what changed | line |
|---|---|---|
| R27 (critical) | D5 restricted to sequence-level KL and no length weight; GRPO shown outside it by the per-token identity, a tree check and the length-weight fixed point; tier 2 split into a D5-class prediction and a GRPO prediction; edit 4 rewritten | 51, 91 to 109, 193 to 199, 267 |
| R52 (critical) | planted size given as the change of the within-right rate (0.025, 0.050, 0.098, 0.197), computed by the script; share 0.40 added | 172 to 181 |
| R55 (critical) | Theorem 3.1 stated as bias toward uniform; pull toward the marginal stated as empirical and unexplained by their theory | 183 |
| R65 (critical) | tier 3 rewritten on Boppana's probe-shift definition and Tables 2 and 7; the "not within a window" claim removed; refutation added for the second part | 201 to 205 |
| R66 (critical) | Ye's Result 4 stated as a probe of necessary parameters, values written step by step | 202 |
| R8 (major) | A3 restated as one sampling kernel for path and continuations | 48 |
| R17 (major) | script prints the vector and one-coordinate forms; D2 compares 0.3842 with 0.3855 | 66 |
| R23 (major) | filler control limited to computation carried by filler tokens | 89 |
| R34 (major) | D8 last row and tier 2 refutation name a failure of A6, not the judge | 115 to 124, 196 |
| R38 (major) | D9 gives both levels for both models, the GPQA results and the paper's internal disagreement | 128 |
| R53 (major) | random-phi narrowing relabeled as power against mixed-sign change; the common-direction limit stated | 181 |
| R58 (major) | T read at cut k - 1 in the script, section 4 and tier 1; refutation rewritten | 135 to 146, 187 to 188 |
| R59 (major) | the share-of-squared-movement threshold withdrawn; tier 1 uses the calibrated pooled window test | 187, 189 |
| R60 (major) | Choice 1a budgets o on 100 positions per chain, 1,200,000 continuations | 241 |
| R62 (major) | side prediction restricted to two learners that both gain from chains | 191 |
| R63 (major) | tier 2 decision by the test of part C per run; seed rule moved to choice 4 and marked uncalibrated | 193, 199, 251 to 259 |
| R74 (major) | edit 1 includes the MMLU step-level result and the GPT-OSS replication | 264 |
| R1 (minor) | declared answer set Ans(x) added | 9, 10, 14 |
| R3 (minor) | tau_text and tau_F marked descriptive until calibrated; choice 5 | 21 |
| R4 (minor) | H_freq restricted to 0 < p_0(x) < 1 | 23, 26 |
| R6 (minor) | A1 restates appendix F with "likely" and the 14B and 32B drop | 46 |
| R7 (minor) | A2 cites both failure ranges from the rerun | 47 |
| R13 (minor) | Theorem 4 named for the SPACE bound | 57 |
| R14 (minor) | width growing with the map added | 57 |
| R19 (minor) | the 0.05 truncation of Forking Fast added | 70 |
| R24 (minor) | Pfau cited to section 4.3, reading.md row updated | 89 |
| R42 (minor) | k and c enter the statistic through j* = k - 1 and k - j* <= c | 138 |
| R43 (minor) | planted learner described as dependence, not use | 146 |
| R49 (minor) | per-position sample of Forking Fast noted as larger | 160 |
| R50 (minor) | pooled windows described as a test in the spirit of their pooling, not their estimator | 170 |
| R57 (minor) | need known for the data, measured for the trained network | 183 |
| R61 (minor) | follow rate compared with the coincidence rate of the unchanged run | 187 |
| R78 (minor) | docstring gives the runtime per part | calibration.py:14 to 16 |
