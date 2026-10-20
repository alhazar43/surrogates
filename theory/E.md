# Theory of surrogate E, response to new evidence

Theory end of surrogate E of C:/Users/YuanW/PLMS/surrogates/program/research-program.md, section 4. Written for unit TE (C:/Users/YuanW/PLMS/surrogates/notes/units/TE.md), 19 October 2026, and revised in its fix attempt 1 after the review C:/Users/YuanW/PLMS/surrogates/theory/E-review.md. Rungs are idea, derivation, checked numerically, source. A panel section (B25, D27 and so on) refers to C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md and is a project file, not a source. Every panel result this file uses was derived again here before use. Where a panel result and this file differ, the difference is stated where it occurs. Scripts are in C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_e/, each with its output file beside it.

## 1. The object

Symbols, one line each.
- x is a problem, y(x) in {-1, +1} its key, coded as key against every other option.
- theta in R^P is the learner's state. z_theta(x) in R is the log odds of the key on x.
- e = (x_e, y_e) is a conflict case, a problem on which the declared cue and the declared rule give different answers, shown with the rule's answer.
- B is a set of test problems, split by content into four groups. Pr, practiced kind (cue and rule agree). CS, conflict siblings (same rule pattern as e, the cue disagrees). Cu, cue siblings (same cue as e, another structure). N, unrelated (shares neither).
- d is the delivery. grad is one plain gradient step of size eta on the loss l(y_e, z_theta(x_e)). ctx is the case placed in the context with theta fixed.
- g_e = -dl/dz at x_e is the signed scalar the delivery puts on the case. Cross entropy on the shown key gives g_e = y_e (1 - p_e), with p_e the probability of the key.

The response, for the gradient delivery.

    R(b; e, theta) = y_b [ z_{theta'}(b) - z_theta(b) ],   theta' = theta - eta grad_theta l(y_e, z_theta(x_e))
    E : (theta, d, e, B) -> R^{|B|},   b -> R(b; e, theta)

The kernel column estimate divides each case by its own delivery scalar.

    k(b; e, theta) = y_e R(b; e, theta) / (eta g_e)

The response ratio averages k over cases e_1..e_m per learner and compares groups, with the unrelated group as the reference.

    rho(theta) = mean_j [ kbar_Pr(e_j) - kbar_N(e_j) ] / mean_j [ kbar_CS(e_j) - kbar_N(e_j) ]

For the context delivery, R_ctx(b; c, theta) = y_b [ z_theta(b | context c) - z_theta(b | empty) ], with c holding N conflict cases and their keys, and rho_ctx is formed from R_ctx in the same way with no division. For a held capability, the revival time T_rev(theta; tau) is the number of optimizer steps of the revival protocol on that capability's task until its test accuracy reaches tau (tau chosen, 0.95 unless stated). The steps are plain gradient steps in tier 1 and the AdamW steps of the copied reverse fine-tuning in tier 2.

Reading. rho is minus one for a learner whose kernel follows the cue only, zero for one whose kernel weighs cue and rule equally, plus one for one that follows the rule only (B25 table), and it is undefined for a learner that moves on no test problem. The group Cu enters no derivation below. It is kept because it separates a learner that copies the case's surface, which moves on Cu toward the case's answer and not on CS (panel D28 table, second row). E11 does not list that learner.

Part answered. Whether two learners, or one learner before and after a change, differ in what one disconfirming case does to their later answers, when their present answers can be equal. Parts left. What content was acquired. Why the change happened. What happens inside a student (the research program, section 4). Whether the moved answers follow the intended rule, which needs the declared structure of surrogate R.

## 2. The assumptions, each with what fails with it

| assumption | what fails with it |
|---|---|
| A1. eta is small enough that the first order term dominates R. | rho then depends on eta and on the second order term. Checked by halving eta from 0.01 to 0.005 on the same learners, cases and test problems. The largest change of one learner's read rho is 0.0015 over the dense and diagonal learners (calibrate_diagonal_output.txt, block 1), below the seed spread of section 4. |
| A2. The loss of the test step depends on theta only through z_theta(x_e). True for binary key against rest, and for softmax cross entropy on the key, since -log p_key = log(1 + exp(-z)) with z the log odds of the key against all other options. | A loss that is not a function of z alone, for example one that also scores the split among the wrong options. One step then moves several log odds, R is a sum of kernel columns, and E1 needs one column per scored option. |
| A3. The groups Pr, CS, Cu, N are declared from the content of the problems, by a rule a reader can check (C15, 7A3). | If the learner's own categories cut across the declared ones, rho near zero reads "holds both" for a learner that holds a third rule. A null on rho is decisive only if the base learner's kernel shows a positive structural term on some declared structure (B19, the paragraph on the gauge of the question). |
| A4. The test step is plain gradient descent in the learner's own parameterization, even when training used Adam. | Under a preconditioned step the kernel is reweighted by the optimizer state, and rho of two learners trained with different optimizers is not on one scale (E6). |
| A5. The parameterization is fixed by the architecture, and every compared learner uses the same one. | rho magnitudes are not invariant to rescaling of the weights (E6). Comparisons across architectures are void, and one architecture is necessary but not sufficient. |
| A6. The unit of inference is the learner (a seed). Cases and test problems are nested in learners. | A test that uses cases as replicates rejects a true null in 0.690 of replicates on null N1, where a learner's random units shift all its cases together. On null N0 the same test rejects in 0.050, since the two linear learners share one kernel (section 4). |
| A7. For the context delivery, the learner uses context as a Bayesian mixture over a few rules it represents (D26). Bigelow et al. 2025 fit the discounted form of this on the behavior of 7B to 70B models. | If the context reaches test problems by similarity of internal states (B27 point 2) and not by likelihood, rho_ctx is a ratio of attention similarities and E8 does not hold. |
| A8. For the revival link, the revival task gradient at the start is the one-step response summed over that task's problems. | This fixes the initial slope only. Revival time also depends on where the start lies on the path of the flow. In the toy of E10 this enters as a bounded gap between held and absent at equal kernel. |

## 3. What can be derived

### E1. The response is one kernel column times the delivery scalar

Proposition. Under A1 and A2, R(b; e, theta) = eta g_e y_b K(b, e) + O(eta^2), with K(b, e) = grad z(b) . grad z(e). Therefore k(b; e, theta) = y_b y_e K(b, e) + O(eta).
Rung. Derivation (chain rule, the same result as B25 Proposition 6). Checked numerically. On five hand-built learners, rho from one step of size 0.01 matches rho from the weights within 0.026 (calibrate_output.txt, block 1), and on three diagonal learners within 0.013 (calibrate_diagonal_output.txt, block 1).

### E2. What rho is invariant to

Proposition. To first order rho does not depend on eta, on the size or sign of g_e, or on which loss of z_e made the step. A step on the learner's own answer (no feedback) and a step on the shown key give the same rho.
Rung. Derivation from E1. Checked numerically. With and without feedback, rho differs by at most 0.044 on the five learners (block 1). The raw move on conflict siblings changes sign between the branches, +5.87 with feedback and -0.82 without, for the dense two-layer learner with the rule absent (block 1, log odds per unit step size). So the feedback branch changes the move and not rho.

Condition found here. The division by g_e must be done per case. A ratio of raw means over cases weights each case by |g_e|, which depends on the case's position relative to the learner's decision boundary, and therefore on the same noise that enters K(b, e). On two linear learners with the same kernel, that version rejects a true null in 0.120 of 400 replicates (95 percent interval 0.090 to 0.156) at a chosen level of 0.05 (block 3b). Per case division brings the rate to 0.050 (block 2). This is B31 point 6 in numbers. One case gives one column, and its weight in an average must not depend on the learner's state. So a reading needs several cases, each divided by its own scalar.

Not invariant. The rescaling of weights (E6), the optimizer's preconditioner (A4), the declared groups (A3), and the set of cases.

### E3. A linear learner carries no state in its response

Proposition. For z_theta(x) = theta . phi(x) with fixed features phi, K(b, e) = phi(b) . phi(e) does not depend on theta. Therefore any two states of a linear learner, whatever their histories and answers, have the same gradient response and the same rho.
Rung. Derivation. Checked numerically. A linear learner with cue weights (log odds +2.01 on practiced, -1.99 on conflict) and one with mixed weights (+2.01 on practiced, +0.03 on conflict) both give rho within 0.026 of 0 (block 1), and the seed-level test separates them in 0.050 of replicates (block 2). This is the linear pair the research program asks the reading to report as indistinguishable. It is indistinguishable by construction, while its answers on conflict problems differ.

### E4. In a balanced diagonal learner the response is a function of the answers

Proposition. Let z(x) = sum_k a_k s_k phi_k(x), with fixed features phi_k, last-layer weight a_k, feature strength s_k, and P_k = a_k s_k. Then K(b, e) = sum_k (a_k^2 + s_k^2) phi_k(b) phi_k(e). If every pair is balanced, a_k^2 = s_k^2, then a_k^2 + s_k^2 = 2|P_k| and K is a function of the coefficients P alone. With a cue feature u and a rule feature v in {-1, +1},

    rho = (|P_v| - |P_u|) / (|P_v| + |P_u|),

and when P_u, P_v >= 0 this equals z(CS) / z(Pr), the ratio of the present log odds on a conflict sibling and on a practiced problem.
Rung. Derivation. Reading. In a balanced diagonal learner rho adds nothing to the present answers on conflict siblings and practiced problems. The panel's statement that learners with the same practiced answers can have any rho (B25) holds, and in the balanced case those learners also differ in their conflict answers. This does not carry to the dense learner z = a . (W x) of calibrate.py, whose kernel (W x_b).(W x_e) + norm(a)^2 x_b.x_e has a use part shared by all inputs.

### E5. A held, unused rule needs imbalance, and weight decay removes it

Proposition. Under gradient flow on any loss of z with no weight decay, delta_k = a_k^2 - s_k^2 is constant in time for each pair. With the penalty (lambda/2) norm(theta)^2, d delta_k / dt = -2 lambda delta_k. With the penalty lambda norm(theta)^2 the rate is -4 lambda. This file uses the first convention, with the loss (1/2)(residual)^2, as E7 and calibrate.py block 4 do.
Consequences.
1. A held, unused rule (a_v = 0, s_v > 0) has delta_v < 0. Gradient flow from a balanced start cannot reach it with one output head, since a_v -> 0 forces s_v -> 0.
2. It is reached when the feature has a second consumer. With heads h, the constant is s_v^2 - sum_h a_{h,v}^2, so a head that keeps training on another task keeps s_v large while this task's head is zero. This is not the structure of Jain et al. 2023. Their fine-tuning trains one task and operand pair (Appendix C.2) through one unembedding, so the pretraining task sends no gradient during fine-tuning. There the capability persists through finite fine-tuning at small rates under AdamW, and at the large rate an inhibitor is learned (Appendix H.1). Neither route is derived here.
3. With weight decay and a fitted training set (residual near zero), a_v receives no gradient and s_v(t) = s_v(0) exp(-lambda t). The held part of the kernel then decays at rate 2 lambda. The step needs only that s_v gets no loss gradient. Under AdamW with decoupled decay the loss part of the update also vanishes for a zero gradient, since the ratio of Adam's moments shrinks by beta_1 / sqrt(beta_2) < 1 per step, and s_v shrinks by the factor (1 - eta lambda) per step. The conservation law itself does not hold under Adam.
Rung. Derivation for 1 and 3. Idea for 2 as a description of any transformer. Reading. What rho adds over the present answers is history stored as imbalance. Weight decay erases it on the clock of training steps.

### E6. rho magnitudes are not invariant to rescaling of the weights

Proposition. The map (a_k, s_k) -> (c a_k, s_k / c), c > 0, leaves z unchanged on every input and changes the k-th kernel term from a_k^2 + s_k^2 to c^2 a_k^2 + s_k^2 / c^2. At a_v = 0 the held strength s_v takes any positive value along this orbit. The rank of the Jacobian of z along phi_v, zero when a_v = s_v = 0 and one otherwise, is invariant under every smooth invertible change of parameters.
Rung. Derivation. Reading. "Held against absent" is a property of the tangent space at the learner's state and survives a change of parameters. "How strongly held" is a property of the parameterization and the step's metric. The orbit lies inside one architecture, so two learners of one architecture that compute one function can have different rho. One architecture under the plain test step (A4, A5) is necessary for comparing rho magnitudes and not sufficient. A natural gradient test step would remove the magnitude and keep the rank.

### E7. Entrenchment, reproduced and solved

Proposition. In the learner of E4 with balanced pairs, trained by gradient flow on practiced problems only, where the cue enters with input scale r_u and the rule with r_v, dP_k/dt = 2 r_k (residual) P_k. Therefore P_u / P_u(0) = (P_v / P_v(0))^(r_u / r_v). The stronger feature grows by a power of the weaker, and the weaker is never erased. With weight decay lambda, the end point minimizes (y - r_u P_u - r_v P_v)^2 + 2 lambda (|P_u| + |P_v|), since the least a^2 + s^2 at a s = P is 2|P|. This is a lasso. On practiced data the two features are collinear, and the lasso puts all weight on the feature with the larger scale. Then rho = -1.
Rung. Derivation. Checked numerically (block 4). With r_u = 1.5, r_v = 1 and no weight decay, the exponent ln(P_u/P_u(0)) / ln(P_v/P_v(0)) is 1.500, and rho falls from -0.105 to -0.655 while the practiced log odds reaches 1.000. With weight decay 0.05, P_v falls to 0.0002 and rho falls from -0.105 to -0.999 at practiced log odds 0.967. B26 point 3 starts at -0.385 and ends at -0.793 with no penalty and at -1.0 with penalty 0.05, at practiced log odds 0.966 and 0.967. The penalty end points agree. The no-penalty end depends on the start through the power law, and the two starts differ, so those ends are not compared. Reading. Entrenchment is a fall of rho at equal practiced answers. It needs a scale asymmetry between cue and rule, and weight decay drives it to the end.

### E8. The context delivery under the mixture premise

Proposition. Under A7 with two rules r (the rule) and s (the cue), slip rate epsilon spread over K - 1 wrong options, and a key that follows r, one case with its key changes the log odds of r against s by log[(1 - epsilon)(K - 1) / epsilon], whatever the prior. Practiced problems do not move, since both rules give the key there. Therefore rho_ctx = 0 for every prior weight in (0, 1). A learner whose support lacks r does not move at all, and rho_ctx is undefined. Bigelow et al. 2025 add a sublinear discount, log o(N) = b + gamma N^(1 - alpha) (their Eq. 9 without steering), so the number of conflict cases at which the rule takes over is N* = (-b / gamma)^(1/(1 - alpha)).
Rung. Derivation for the mixture. Source for the discounted form (Bigelow et al. 2025, Sections 4.1 and 4.3). Checked numerically for the mixture (block 5). At epsilon 0.1 and K = 4 one case multiplies the odds by 27 and moves the rule weight from 0.01 to 0.214, from 0.1 to 0.750, and from 0.3 to 0.920, as C23 and C24 state.
Reading. In this delivery entrenchment is a more negative prior log odds b, and N* grows faster than linearly in |b| when alpha > 0. The behavioral move on a conflict sibling has a ceiling, P(key) <= 1 - epsilon, which is log odds log 9 = 2.20 at epsilon 0.1 and K = 4. A gradient learner has no such ceiling. The fit form of Bigelow et al. has no ceiling either, so where the measured log odds saturates, the fitted gamma and alpha absorb the saturation and gamma is not the per case evidence weight. A fit reads gamma only on the N below saturation.

### E9. Reward at its fixed point leaves the context response unchanged

Proposition. If training ends at pi*(y | x) proportional to pi_0(y | x) exp(r(x, y) / beta) with r the indicator of the key, and the same tilt holds for prompts that carry cases, then z*(x | c) = z_0(x | c) + 1/beta for every prompt and context. Therefore R_ctx is unchanged, and in the form of E8 only b moves, upward by 1/beta, while gamma and alpha stay.
Rung. Derivation (agrees with D27). For GRPO with group size G the end point shift is w_G(p*)/beta, with w_G(p) = s_G(p) / (p (1 - p)) and s_G(p) = E[sqrt(rbar (1 - rbar))] over the group's mean reward rbar (B11 point 2, derived again here with the group's population standard deviation). At G = 5, w_5 is smallest at p = 1/2, 1.7247, rises monotonically toward 2 as p goes to 0 or 1, and so the context response differs from the base's by less than (2 - 1.7247)/beta = 0.275/beta. B31 point 1 states 0.274/beta, with 1.999 as the upper value. Checked numerically (revival_toy_output.txt, block 4).

### E10. Revival time and the one-step response

Proposition. In the learner of E4, train one pair (a_v, s_v) on a revival task with target log odds t on phi_v, squared loss (1/2)(t - P)^2, unit input, gradient flow, P = a_v s_v. Then delta = s_v^2 - a_v^2 is conserved (E5), and

    dP/dt = sqrt(delta^2 + 4 P^2) (t - P),   K0 = a_v(0)^2 + s_v(0)^2,

with K0 the initial kernel on phi_v, the first order term. Held (a_v = 0, s_v = kappa) gives K0 = kappa^2 and P(0) = 0. Absent (a_v = s_v = epsilon) gives K0 = 2 epsilon^2 and P(0) = K0 / 2. The time to P = t/2 is

    T_held   = integral from 0 to t/2 of dP / [ sqrt(K0^2 + 4 P^2) (t - P) ],  -> (1/(2t)) ln(4t / K0) when K0 << t,  -> ln 2 / K0 when K0 >> t
    T_absent = (1/(2t)) ln(2t / K0 - 1),  exact for K0 < t.

Both decrease in K0. At equal K0 << t the held rule is slower by (1/(2t)) ln 2. A held rule with kernel K_h revives slower than an absent rule with kernel K_a when K_a <= K_h < c K_a and faster when K_h > c K_a, with the crossing c between 1.9 and 2.0 for K_a << t (the asymptotic forms give c = 2). The initial slope of the revival loss under a plain step of size eta is -eta g^T K g, with g the residual vector over the revival task's problems, the summed one-step response. An absent pair whose product starts with the wrong sign does not revive under exact balance, since P stays on its side of zero, and is outside the comparison.
Rung. Derivation in the toy. Checked numerically (revival_toy_output.txt). Block 1, the ODE and the quadrature agree to four decimals. At t = 1 and K0 = 0.1, 0.02, 0.002 the held times are 1.820, 2.644, 3.800 and the absent times 1.472, 2.298, 3.453, with the gap at 0.347 = (ln 2)/2 from K0 = 0.02 down. At K0 = 8 the held time is 0.086 against ln 2 / K0 = 0.087. Block 2, the held rule is slower at K_h / K_a = 1.9 and faster at 2.0, at K_a = 0.02 and 0.002. Source for the observed pattern. In the PCFG Counter task of Jain et al. 2023, a strongly relevant held capability revives in about 0.1K to 1K iterations, a weakly relevant one in about 3K, and the Scr.+FT baseline (pretrained to count the fine-tuning target O_FT, then fine-tuned to count O_PT) reaches perfect accuracy only at about 4.5K iterations and only at the larger rate eta_M (their Figure 9, Appendix E.3 and Figure 26).
Reading. In the toy the one-step response predicts the rank of revival time for held and absent rules alike, except for a held rule whose kernel lies within about a factor of 2 above an absent rule's. At equal kernel the held and absent rules differ by their start, P(0) = 0 against P(0) = K0/2, and the gap is bounded. The first version of this file said the response predicts revival time only for a held capability. That compared the kappa^(-2) law, valid only for K0 >> t, with the logistic law, and it is withdrawn.

### E11. Which reference learners can be told apart, and by what

The reference learners of tier 1. L1 memorizer, a table over seen problems. L2 cue user, the diagonal learner of E4 with the rule absent. L3 rule user. L4 Bayesian mixture over cue and rule (E8). L5 cue user that stores the corrected case as an exception, z = P_u u(x) plus a table entry at x_e. L6 linear learner (E3) and L7 diagonal two-layer learner (E4, E5), each with two histories. Assumption for L5 in rows 1 and 2. The test step writes the case to the table only, and P_u receives no gradient (the cue weight is frozen). Row 2b drops it.

| pair | present answers on fresh test problems | gradient rho | other observation that separates |
|---|---|---|---|
| 1. L1 against L5 | differ (L1 at log odds 0 on fresh practiced problems, L5 at the cue's value) | equal, both undefined (no move off the case) | none needed, the answers separate |
| 2. L2 against L5, P_u frozen | equal | differ (L2 at -1, L5 undefined) | the response is the only separating observation |
| 2b. L2 against L5, P_u trained | equal | equal, both -1 (with a balanced cue pair, L5's kernel on fresh problems is 2 abs(P_u) u(b) u(e)) | none by any gradient response off the case |
| 3. L2 against L3 | differ on conflict siblings | differ (-1, +1) | answers suffice |
| 4. L4 against L7 using both features equally | can be equal | L4 has no gradient rho; rho_ctx(L4) = 0 and gradient rho(L7) = 0 | no feedback (L4 does not move, L7 moves toward its own answer); repeated cases (L4 saturates at log odds 2.20, L7 does not) |
| 5. L6 history 1 against history 2 | equal on practiced, may differ on conflict | equal, exactly (E3) | none by any gradient response |
| 6. L7 balanced, two histories with equal practiced answers | differ on conflict siblings | differ, by E4 a function of those answers | the answers already separate |
| 7. L7 rule absent against rule held unused | equal on every input | differ (true difference 0.379 in the diagonal learner at kappa = 1, 0.400 without random units; 0.166 in the dense learner of calibrate.py) | the response is the only separating observation |

Rung. Derivation for every row. Checked numerically for rows 5 and 7 (calibrate_output.txt blocks 1 and 2, calibrate_diagonal_output.txt blocks 1 to 3). Reading. The gradient rho carries information beyond present answers in two rows. Row 2 holds only with L5's cue weight frozen, so it separates a shared feature from an exception store that does not train its features. Row 7 is imbalance from history (E5). In row 4 the no-feedback branch and repeated cases separate learners with equal answers. Those are responses, but not the gradient rho.

## 4. The calibration of the tier 1 reading

The reading. For each learner, rho from m = 10 conflict cases, each with 20 fresh test problems per group, one plain step of size 0.01 with the key shown, k divided per case by g_e. For a pair of arms, the statistic is the difference of the arm means of rho over S = 8 learners per arm. The test is a two-sided permutation test on that difference, 2000 permutations, level 0.05 (chosen).

Known truth. The learners are built by hand in 16 input dimensions with 6 hidden units. Cue and rule are the orthonormal directions e1 and e2, and the unrelated group uses e3. Test problems and the case x_e carry Gaussian noise of standard deviation 0.3 per coordinate (calibrate.py:91). True rho is computed from the weights as K(Pr, e) / K(CS, e) on the noise-free problems.
- Linear learners. Weights w_u e1 + w_v e2 plus a random part of standard deviation 0.1 per coordinate (calibrate.py:72).
- Dense two-layer learners, z = a . (W x), all of W and a trained by the test step. Four hidden units are random per learner (rows of W of standard deviation 0.5 / sqrt(16), last-layer weights of standard deviation 0.3).
- Diagonal learners of E4, z = sum_k a_k s_k (f_k . x), only a and s trained. Built from the same random draws as the dense learner, with s_k f_k = W_k, so both compute the same function and differ only in which weights the test step moves (calibrate_diagonal.py).
- Null N0. Linear cue weights (2, 0) against linear mixed weights (1, 1), equal practiced log odds of 2. True difference 0 by E3.
- Nulls N1 (dense) and N1d (diagonal). Rule absent in both arms (a_u = s_u = sqrt 2, a_v = s_v = 0), different seeds. True difference 0.
- Planted effect. Rule absent against rule held unused (s_v = kappa, a_v = 0). The two arms give equal log odds on every input, up to the random units. True difference from the weights, the mean over 20000 learners per arm.

False positive rate is the fraction of 400 replicates of a null that reject. Power is the fraction of 400 replicates of a planted effect that reject. Intervals are Clopper-Pearson 95 percent.

| learner and condition | true difference in rho | mean measured difference | rejection rate [95 percent interval] |
|---|---|---|---|
| N0, linear pair | 0.000 | +0.003 | 0.050 [0.031, 0.076] |
| N1, dense, absent against absent | 0 | +0.006 | 0.058 [0.037, 0.085] |
| dense, planted kappa 0.25 | +0.012 | +0.015 | 0.058 [0.037, 0.085] |
| dense, planted kappa 0.5 | +0.046 | +0.043 | 0.185 [0.148, 0.227] |
| dense, planted kappa 0.75 | +0.099 | +0.098 | 0.593 [0.543, 0.641] |
| dense, planted kappa 1.0 | +0.166 | +0.167 | 0.958 [0.933, 0.975] |
| N1d, diagonal, absent against absent | 0 | +0.002 | 0.045 [0.027, 0.070] |
| diagonal, planted kappa 0.25 | +0.029 | +0.030 | 0.122 [0.092, 0.159] |
| diagonal, planted kappa 0.375 | +0.064 | +0.067 | 0.430 [0.381, 0.480] |
| diagonal, planted kappa 0.5 | +0.111 | +0.113 | 0.865 [0.828, 0.897] |
| diagonal, planted kappa 0.75 | +0.233 | +0.239 | 1.000 [0.991, 1.000] |
| diagonal, planted kappa 1.0 | +0.379 | +0.381 | 1.000 [0.991, 1.000] |
| N1, cases as replicates (paired t-test, one learner per arm) | 0 | not computed | 0.690 [0.642, 0.735] |
| N0, cases as replicates | 0 | not computed | 0.050 [0.031, 0.076] |
| N0 with a raw ratio of means (no per case division) | 0.000 | -0.033 | 0.120 [0.090, 0.156] |

Source. calibrate.py with calibrate_output.txt (3 min 41 s on the PC in the RevE rerun), and calibrate_diagonal.py with calibrate_diagonal_output.txt (4 min 50 s on the PC), both run with python -I. The true differences of the dense rows are recomputed in calibrate_diagonal_output.txt block 3 and replace the 50-learner values of the first version (0.007, 0.044, 0.099, 0.163).

Reading. The seed-level test holds its level on all three nulls. Power depends on the learner it was computed on, because the spread of rho between learners of one kind differs. In the dense learner power reaches 0.8 at a true difference in rho of about 0.14 (0.137), by linear interpolation between the kappa 0.75 and 1.0 rows. In the diagonal learner of E4 and E11 power reaches 0.8 at a true difference of about 0.10, by linear interpolation between the kappa 0.375 and 0.5 rows, and it is 1.000 at kappa 1.0, where the true difference is 0.379. So the power 0.958 at kappa 1.0 holds for the dense learner only. A test that treats cases as replicates rejects a true null in 0.690 of replicates on N1, since a learner's random units shift all its cases together. The calibration covers L2, L6 and L7 of E11, with L2 and L7 in both the dense and the diagonal form. It does not cover L1, L3, L4 and L5, whose responses are zero or exact by construction, and it does not cover the context delivery.

## 5. One prediction per tier, written before any data

### Tier 1, reference learners on CPU

Prediction. The full tier 1 run reproduces every row of the table in E11. The linear pair is rejected at a rate inside the binomial interval of 0.05. The absent against held pair at kappa = 1 is rejected at a rate of at least 0.8 in both the dense and the diagonal learner. The mixture learner L4 gives rho_ctx = 0 with a move that saturates at log odds 2.20.
Refuted by. A rejection rate on any null above 0.076 (the upper end of the interval at 400 replicates). Power below 0.8 at kappa = 1 in either learner. A row of E11 that the table calls separated and the run does not separate, or the reverse, by the seed-level test at level 0.05. For L4, rho_ctx different from 0 or a ceiling away from 2.20 means the mixture learner was coded wrong, since its values are exact by construction. That is a code test, not a refutation of the theory.
What each outcome changes above. Pass. Tier 2 uses the per case division and the seed-level test with at least 8 seeds per condition, and treats rho differences below the 0.8 power point of the learner nearest to it as untested at that size. Fail. Tier 2 does not start. The failed calibration is the result, and the reading is redesigned before any transformer is trained.

### Tier 2, the setup of Jain et al. 2023

Prediction 1. In the PCFG Counter task, after fine-tuning to count b, the summed one-step response on count-a test problems (one plain step on a count-a case, divided by g_e) is ordered across conditions as 1/T_rev is. The conditions are the pretraining prior of b, the fine-tuning target (0.001, 0.1, 0.3, which are the source's distributions with prior of a 0.999, 0.9, 0.5), the fine-tuning learning rate (1e-6, 1e-5, 1e-4) and the fraction of correlated fine-tuning data (0, 1), 18 in all. The absent condition is Scr. of Appendix E.4, a model trained from scratch to count a, which holds no counting capability. Scr.+FT is not the absent condition, since it holds a counting circuit for another operand. It is reported beside, outside the correlation. T_rev is counted in AdamW steps at the reverse fine-tuning rate, and a run that does not reach tau is censored at its length and takes the largest rank, ties averaged. Spearman r_s between the response and 1/T_rev over the 19 conditions is at least 0.6 (chosen). Under independent rankings r_s >= 0.6 occurs with probability 0.004. Under a bivariate normal pair with Pearson correlation 0.7, 0.8 or 0.9 it occurs with probability 0.691, 0.908 and 0.995 (revival_toy_output.txt, block 3).
Prediction 2, from E5.3. In an added arm that fine-tunes with AdamW and decoupled weight decay lambda = 0.01 (chosen; the source states no value), fine-tuning for 40K instead of 10K iterations at the rate 1e-6 lowers the one-step response for held conditions, and by E10 lengthens T_rev. At lambda = 0 E5.3 predicts nothing.
Refuted by. For prediction 1, r_s below 0.6. Or a held condition and Scr. whose one-step responses are within a factor of 2 (chosen) of each other and whose T_rev differ by more than a factor of 2 (chosen); in the toy with K << t, kernels within a factor of 2 give times within (ln 2)/t of each other. Either says revival is not ordered by the first order term in this setup. For prediction 2, the response after 40K fine-tuning iterations not below the response after 10K (chosen lengths) beyond the seed spread, in any held condition. Rung of prediction 2. Derivation in the toy, idea for the transformer.
What each outcome changes above. Pass. Tier 3 may use a single case reading on the weight channel. Fail. Tier 3 uses the context channel only, and revival is measured directly wherever it is needed.

### Tier 3, a post-trained language model

Prediction. Fit log o(N) = b + gamma N^(1 - alpha) (E8) to the key's log odds against the number of conflict cases in context, before and after reward training with a KL penalty, on one fixed set of prompts, using N up to the largest value whose mean log odds stays below the ceiling seen at the largest N. gamma and alpha change by less than their seed spread. b does not fall, and under E9 it rises by 1/beta where the reward marks the rule's answer as the key. Accuracy is reported beside the fit and is not used to select prompts, since prompts of equal accuracy are those on which the tilt of E9 did not act.
Refuted by. b falling beyond the seed spread, which is entrenchment through the prior (E8 reading). gamma falling beyond the seed spread, which answers the research program's tier 3 question yes (reward training lowered the weight the context gives each counterexample). gamma rising beyond the spread, which says reward training built a use of evidence the base lacked. alpha changing beyond the spread, which says the discount changed, against E9.
What each outcome changes above. Tier 3 is the top. Its outcome feeds surrogate R (whether reward entrenches the cue) and surrogate P (whether the chain uses counterevidence).

## 6. The literature

| source | what it gives this theory | rung | setup to reproduce |
|---|---|---|---|
| Jain, Kirk, Lubana, Dick, Tanaka, Grefenstette, Rocktäschel, Krueger 2023, Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks, arXiv 2311.12786 v2 | capability defined by a linear readout at some layer, wrapper as g composed with C (Section 3, Definitions 1 and 2); fine-tuning rarely alters capabilities (abstract); relevance set by the pretraining prior of the fine-tuning target O_FT (Figure 5, Appendix C.2); reverse fine-tuning revives a strongly relevant capability in about 0.1K to 1K iterations and a weakly relevant one in about 3K, while Scr.+FT (pretrained to count O_FT, then fine-tuned to count O_PT) reaches perfect accuracy only at about 4.5K iterations and only at eta_M (Figure 9, Appendix E.3, Figure 26); Scr., trained from scratch to count a, appears only as a baseline of Appendix E.4 (Figure 28); in PCFG a probe of the O_PT count drops only at eta_M with a weakly relevant capability (Figure 8), and in Tracr the attention pattern changes only with a large enough rate and no spurious correlation (Figure 10); task family tokens T_J1, T_J2, paired only with a in pretraining, recover counting a after fine-tuning when the pretraining prior of b was high (Appendix E.2, Figure 24); an inhibitor is learned with a strongly relevant capability at eta_L, mainly in the 200K-iteration pretrained model (Appendix H.1); no weight decay value is stated | source, full text opened (HTML v2, main text and appendices) | PCFG Counter. minGPT of about 3M parameters, 6 blocks, 6 heads, width 192. Pretraining with AdamW at 1e-3, cosine schedule with 20 percent warmup, 10K, 50K, 100K or 200K iterations, operand priors (a, b, c) of (0.999, 0.001, 0), (0.99, 0.01, 0), (0.9, 0.1, 0), (0.7, 0.2, 0.1), (0.5, 0.3, 0.2). Fine-tuning on one task and operand pair with AdamW, batch 96, 10K iterations, cosine with 20 percent warmup, rates eta_L 1e-4, eta_M 1e-5, eta_S 1e-6, correlated fraction C_Tr in {0, 0.5, 0.8, 1}. Reverse fine-tuning with the fine-tuning hyperparameters at a smaller rate (Appendix D). Baselines Scr.+FT (Section 5.1) and Scr. (Appendix E.4). |
| Bigelow, Wurgaft, Wang, Goodman, Ullman, Tanaka, Lubana 2025, Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering, arXiv 2511.00617 v2 | posterior odds of a concept against its complement, p = sigma(log o) (Eq. 3); log likelihood falls with the number of mismatched labels (Eq. 4); sublinear accumulation by a power law discount, log Bayes factor proportional to N^(1 - alpha) (Eq. 5, Appendix A.1); steering adds a m to the log odds (Eq. 7, 8); log o = a m + b + gamma N^(1 - alpha) and the crossover N* (Eq. 9, 10); cross-validated correlation r = 0.98 on Llama-3.1-8B and 0.97 to 0.98 on two other models; flipped labels need more shots (Appendix F) | source, full text opened (HTML v2, both parts) | Persona Yes/No questions with N up to 128 shots, and the flipped-label sentiment task (Appendices F, H). Four parameters fitted with L-BFGS-B and basin hopping, 10-fold cross-validation over steering magnitudes. |

Rows for both are in C:/Users/YuanW/PLMS/surrogates/sources/reading.md. The fetched text of Bigelow et al. writes the sign of the evidence term differently in Eq. 3 and Eq. 5. This file uses the form of Eq. 9 with gamma > 0 for evidence toward the concept, and the sign is to be checked against the PDF before any fit.

Panel results used, each derived again here. B11 point 2 (the GRPO factor), B19 (kernel on a crossed grid), B25 (Proposition 6 and the two-feature table), B26 points 1 to 3, B27 (context delivery), B31 points 1 and 6, B32 (split of the kernel), D26 and D27 (two deliveries and the tilt null), D28 (the group table), C23 and C24 (mixture learner numbers). The panel's citations of Gick and Holyoak, Schapiro and McClelland, Wei and others, and Koh and Liang are not used, since this unit did not open them.

## 7. What the theory cannot show

- Any empirical result. No transformer was trained, and no real log was opened.
- That the one-layer pictures of E4 to E7 and E10 describe a deep network. Every layer adds a kernel term (B32, last paragraph), and E4 to E7 and E10 are derived for one factorized layer over fixed features.
- That rho measures a rule in a learner whose categories differ from the declared ones (A3).
- That the context delivery of a transformer follows the mixture premise. Bigelow et al. fit the discounted form on behavior. They do not show the mechanism, and their tasks are personas and label flips, not conflict cases between a cue and a rule.
- That the rank link of E10 holds in a transformer. E10 has one pair on one fixed feature. In a transformer the revival task moves many features from unknown starts, and the inhibitor of Jain et al. Appendix H.1 is a second path that the toy lacks. In the toy itself the link inverts for a held rule whose kernel is within about a factor of 2 above an absent rule's.
- Anything about students. A student's two deliveries are mixed in one observation and separated only by delay (R8A3, C30), which this file does not treat.
- That a fall of rho is harm. A fall of rho with no fall of practiced accuracy is entrenchment in the sense of E7. Whether the cue was legitimate on the intended scope is outside the object.

## 8. Choices for the author and proposed edits to the research program

### Choice 1. What "equal present answers" means in tier 1

| option | mechanism | cost | what it shows |
|---|---|---|---|
| a. equal on practiced problems only | train both histories to the same practiced log odds | none extra | for balanced diagonal learners rho is then a function of the conflict answers (E4), so the reading repeats what the answers say |
| b. equal on every test problem | build or train pairs with one function, differing in imbalance (E5) | one extra pair per learner type | the only pairs where the response adds information beyond answers |
| c. both, reported apart | a and b | twice the pairs, still CPU minutes | separates "rho agrees with answers" from "rho adds to answers" |

Pick c.

### Choice 2. What "without feedback" means for a gradient learner

| option | mechanism | cost | what it shows |
|---|---|---|---|
| a. no update | the case is shown and nothing moves | none | a gradient learner and a mixture learner both stay still, so the branch separates nothing |
| b. a step on the learner's own answer | self-training; the delivery scalar is p_e toward the own answer | none | rho unchanged by E2 and the move reverses sign; separates L4 (no move) from L7 (moves) |
| c. a step that marks the own answer wrong | loss on giving the own answer | none | the outcome is shown and the key is not |

Pick b, with c as a second branch if the author wants the mark of wrong apart from the key.

### Choice 3. The context delivery in tier 2

| option | mechanism | cost | what it shows |
|---|---|---|---|
| a. the task family tokens of Jain et al. Appendix E.2 | tokens T_J1, T_J2 paired only with a in pretraining elicit counting a after fine-tuning on b | a new pretraining data mix with the tokens, one pretraining run per prior | context elicitation of a held capability; not R_ctx, since no conflict case and no key enters the context |
| b. in-context worked examples of counting a | prompts with N worked examples | a 3M model may not use context at all (C23 reports that small models keep their prior; that source was not opened here) | R_ctx, if the channel is open |
| c. the persona task of Bigelow et al. on a separate 1B model | the published setup | a second model, GPU hours not measured | the discounted curve, on a model whose weights tier 2 did not test |
| d. no context delivery in tier 2 | tier 2 reads the weight channel only; the context channel is read in tier 3 | none | E8 and E9 are tested only in tier 3 |

Pick d, with b as a first check of whether the channel is open if the author wants R_ctx at tier 2.

### Choice 4. What tier 2 measures for revival

| option | mechanism | cost | what it shows |
|---|---|---|---|
| a. revival time T_rev only, as the research program states | reverse fine-tuning to accuracy tau | the setup's own runs | held against absent, with no link to the one-step response |
| b. the one-step response and T_rev, with their rank correlation | add one plain step per condition before reverse fine-tuning, and train Scr. as the absent condition | one plain step per condition and one Scr. run | whether the tier 1 reading carries over to a transformer (E10, the tier 2 prediction) |

Pick b.

### Proposed edits to the research program, section 4, surrogate E

1. Theory line. Add "a linear learner's response does not depend on its state; in a balanced diagonal learner the response is a function of the answers; what the response adds over answers is history stored as imbalance, which weight decay erases (theory/E.md, E3 to E5)".
2. Tier 1. Replace "Two histories per adaptive learner with equal present answers" by "Two histories per adaptive learner, one pair equal on practiced problems and one pair equal on every test problem". Add "The unit of inference is the learner, at least 8 seeds per arm; each case is divided by its delivery scalar before averaging". Add the calibrated numbers with their learner. False positive rate 0.050 (linear pair), 0.058 (dense two-layer), 0.045 (diagonal two-layer). Power 0.8 at a true difference in rho of about 0.14 in the dense learner and about 0.10 in the diagonal learner, at 8 learners per arm. Power at kappa = 1 is 0.958 in the dense learner (difference 0.166) and 1.000 in the diagonal learner (difference 0.379). Power at a given difference depends on the spread of rho between learners of one kind, so it is computed again for each reference learner.
3. Tier 2. Replace "told apart by revival speed" by "told apart by revival time, beside the one-step response on the revived task and the rank correlation of the two, with Scr. of Jain et al. Appendix E.4 as the absent condition". Tier 2 has no context delivery (Choice 3).
4. Tier 3. Replace "lowers the response to counterevidence" by "lowers gamma, the per case evidence weight of the fitted curve log o(N) = b + gamma N^(1 - alpha), fitted below saturation on one fixed prompt set; the fixed point of KL-regularized reward predicts no change in gamma and alpha and a rise of b by 1/beta".
5. Section 2, condition 3. Add "The unit of inference is the learner, not the case or the test problem".
6. Section 3, idea 1. Add one sentence. Holding without using is a property of the tangent space at the learner's state, and its strength is not invariant to rescaling of the weights (theory/E.md, E6).
7. Section 3, the line on the weights. Replace "Read once, it shows a rule held and unused" by "Read over several cases, each divided by its delivery scalar, and over at least 8 learners per arm, it shows a rule held and unused (theory/E.md, E2 and section 4)".

## Fix log of attempt 1

Items are numbered as in theory/E-review.md. Lines refer to this file.

| item | what changed | line |
|---|---|---|
| 1, critical, E10 and section 7 bullet 5 | E10 derived again with the conserved delta. Held time (1/(2t)) ln(4t/K0) for K0 << t and ln 2 / K0 for K0 >> t; absent time (1/(2t)) ln(2t/K0 - 1). Both decrease in K0, gap (ln 2)/(2t) at equal kernel, inversion only for K_h between K_a and about 2 K_a. Checked in revival_toy.py. The old reading is withdrawn in the text. Section 7 bullet replaced, A8 and the tier 2 refutation reason revised | 45, 107 to 120, 187, 213 |
| 2, calibration coverage | calibrate_diagonal.py calibrates the diagonal learner of E4 from the same draws. False positive rate 0.045, power 0.8 at about 0.10, power 1.000 at kappa 1 (difference 0.379). Power is stated per learner in section 4, E11 row 7 and proposed edit 2 | 135, 161 to 166, 173, 262 |
| 3, E11 row 2 | Assumption on L5's cue weight stated (frozen). Row 2b added for a trained cue weight, both rho = -1. Reading revised | 124, 129, 130, 137 |
| 4, E5 consequence 2 | Relabeled idea. Says it is not the structure of Jain et al. (one task and operand pair, one unembedding) and names their routes (finite fine-tuning, inhibitor of H.1) as not derived | 82, 84 |
| 5, B31 point 1 | Derived again. w_5 from 1.7247 at p = 1/2 toward 2, gap below 0.275/beta (revival_toy_output.txt block 4). Line 3 now holds | 3, 105 |
| 6, Jain E.2 and Choice 3a | Tokens T_J1, T_J2 paired only with a, high prior of b, own pretraining data. Choice 3a costed and marked not R_ctx; option d added and picked; edit 3 and sources/reading.md row 43 corrected | 200, 243, 248, 263 |
| 7, Scr.+FT as absent | Absent condition is Scr. (Appendix E.4). Scr.+FT reported beside, outside the correlation. Censoring rule for runs that do not reach tau | 119, 185, 200 |
| 8, tier 2 second prediction | Restated from E5.3, which needs only a zero loss gradient on s_v and holds under AdamW with decoupled decay. lambda 0.01 chosen, lengths 10K and 40K chosen, refutation added, rung stated | 83, 186, 187 |
| 9, tier 3 prediction | b must not fall and rises by 1/beta under E9. Selection by equal accuracy removed. Refutation for alpha added. Fit below saturation | 192, 193, 264 |
| 10, A2 failure row | Softmax cross entropy on the key is a loss of z; failure row names a loss that scores the wrong options | 39 |
| 11, A3 citation | Cites the paragraph of B19 on the gauge of the question | 40 |
| 12, A6 | 0.690 stated for N1, 0.050 for N0; N0 case-level row added to the table | 43, 168 |
| 13, A1 check | Run. Halving eta changes one learner's rho by at most 0.0015 | 38 |
| 14, E4 reading | Restricted to the diagonal learner; dense kernel named | 68, 75 |
| 15, E5 convention | Penalty (lambda/2) norm(theta)^2 and loss (1/2)(residual)^2 stated; -4 lambda for the other convention | 79 |
| 16, E6 conclusion | One architecture necessary, not sufficient, in E6 and A5 | 42, 89 |
| 17, E7 against B26 | B26 start -0.385, no-penalty end -0.793 and penalty end -1.0 stated; penalty ends agree, no-penalty ends not compared | 94 |
| 18, E10 slope sign | -eta g^T K g | 118 |
| 19, E11 row 4 | Entry marks rho_ctx for L4; reading names row 4 as separated by responses other than gradient rho | 132, 137 |
| 20, E11 row 7 | 0.379 in the diagonal learner (0.400 without random units), 0.166 in the dense learner | 135 |
| 21, true differences | Recomputed with 20000 learners per arm (calibrate_diagonal_output.txt block 3): 0.012, 0.046, 0.099, 0.166; N1 truth 0 in the table | 156 to 160, 171 |
| 22, run time | 3 min 41 s from the RevE rerun; docstring of calibrate.py corrected | 171 |
| 23, section 4 design | Linear random part (sd 0.1) and noise on x_e stated; dense and diagonal constructions described | 143 to 146 |
| 24, Jain Section 5.1 | Replaced by the PCFG probe result (Figure 8) and the Tracr attention result (Figure 10), each with its condition | 200 |
| 25, tier 1 refutation | Every null, both learners and every row of E11 covered; L4 marked as a code test | 180 |
| 26, r_s threshold and T_rev | r_s >= 0.6 has false positive rate 0.004 and power 0.691, 0.908, 0.995 at Pearson 0.7, 0.8, 0.9; T_rev counted in the protocol's optimizer steps | 28, 185 |
| 27, E8 ceiling and fit form | Fit form has no ceiling; gamma read only below saturation; tier 3 fit restricted accordingly | 100, 192 |
| 28, read once | Proposed edit 7 revises the line; E2 states that a reading needs several cases | 59, 267 |
| 29, group Cu | Sentence states that Cu enters no derivation here and separates the surface copier of D28 | 30 |
