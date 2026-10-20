# R. Reach, shortcut against rule, the theory

Theory end of surrogate R of the research program (C:/Users/YuanW/PLMS/surrogates/program/research-program.md, section 4). Unit TR, 19 October 2026, fixed in attempt 1 after the review theory/R-review.md. Rungs are marked in brackets as idea, derivation, heuristic, checked numerically, or source. The scripts and their outputs are in C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_r/, calibrate.py with calibrate_output.txt and fix_checks.py with fix_checks_output.txt.

## 1. The object

```
problem      x = G(c, s, n)      c in C, the structure, declared from the content by a checkable rule
                                 s in S, the surface;  n, the nuisance
key          y = r(c, n) in {1..K}
cue answer   h(s) in {1..K}, the answer the surface points to
practice     D_T on cells T, a subset of {(c, s) : h(s) = r(c, n)}       (cue and rule agree)
learner      f_theta : X -> Delta({1..K});   change theta_0 -> theta_1 = U(theta_0, D_T), U the route
grid         Q = C x S, every structure in every surface; read on the cells of Q outside T
chance       pi(s) = P_{s'}(h(s') = h(s)),   rho(c,n) = P_{c'}(r(c',n) = r(c,n)),   s' uniform on S, c' uniform on C

Lambda_s(f) = avg over Q of [ P_f(h(s) | G(c,s,n)) - E_{s'} P_f(h(s) | G(c,s',n)) ] / (1 - pi(s))
Lambda_c(f) = avg over Q of [ P_f(r(c,n) | G(c,s,n)) - E_{c'} P_f(r(c,n) | G(c',s,n)) ] / (1 - rho(c,n))
estimand     bar Lambda_s(U, D_T) = E_seed Lambda_s(f_theta_1) - E_seed Lambda_s(f_theta_0),   bar Lambda_c likewise
```

Lambda_s is how much more often the learner gives the cue's answer when that surface is shown than when another surface is shown, scaled so that a pure cue learner reads 1. A change follows the surface when bar Lambda_s > 0 and the structure when bar Lambda_c > 0.

The estimand. Every tier 1 test reads bar Lambda, the mean over seeds, where a seed fixes the initialization, the data order and the random design draws of A2. It asks whether the route U on the practice D_T produces systematic use of the cue, a property of the training procedure and not of one network. The per learner value Lambda_s(f) is a second estimand. A seed whose answers depend on the surface with a sign drawn at random has Lambda_s(f) different from zero and adds only spread to bar Lambda_s. The written learner of section 4 with delta = 0.3 has E[T_s given delta] = 0.0500, equal to the planted lam = 0.05 [checked numerically, exact sum over the grid]. The item level tests of section 4 test the per learner estimand and are not used for tier 1 claims.

Reference values, exact whenever pi(s) < 1 and rho(c, n) < 1 [derivation]. A rule learner without slips has (Lambda_c, Lambda_s) = (1, 0). A cue learner has (0, 1). A learner that answers by the cue with probability lam and otherwise by any rule that ignores s has Lambda_s = lam (section 3, R2). A constant learner has (0, 0). The per cell divisor 1 - pi(s) makes these values hold without a balance condition. With h uniform over K options it is 1 - 1/K.

Lambda_s is invariance to one intervention measured on the outputs. Lubana et al. 2023 define a mechanism as the set of unit interventions on the latents of the data generating process to which a model is invariant (Definitions 2 to 4) [source]. Here the interventions are A_s, which resamples s and keeps c and n, and A_c, which resamples c and keeps s and n. Lambda_s = 0 holds for every learner invariant to A_s on Q. Shah et al. 2020 measure a related quantity, S randomized accuracy, the accuracy against the label after S is randomized (Definition 1) [source]. Lambda_s is the excess rate of the cue's answer, and the two agree only at the two pure learners. Hermann et al. 2023 define reliance in their Section 3 as a prefactor times the difference of the sums of y_hat sign(z_s) and of y_hat sign(z_c) over the probe grid [source, section from the review]. Lambda_s for K = 2 on a product grid equals the first sum up to that prefactor. Lubana et al. state invariance on the loss (Definition 3). The output form is used here because it gives the reference values above. The two forms agree on a pure cue learner and a pure rule learner.

Part of the question answered. Whether a change follows the surface of the practiced problems or their structure, the generality axis of the panel (panel B23). Part left. Reach, the base probability of what the learner now does. Features, whether the learner's similarity changed, except where tier 2 reads it inside the weights and tier 3 compares with the linearized base model. The response to new evidence (surrogate E). The process of solving (surrogate P). Students (surrogate S). Whether the declared structure is the one a teacher intended.

## 2. The assumptions

Each assumption is stated so that it can fail. The second column says what fails with it.

| assumption | what fails with it |
|---|---|
| A1. The structure is declared from the content, and the grid crosses structure with surface on the test cells. | If the declared c cuts across the learner's own categories, Lambda_c reads near zero for a learner that generalizes by those categories (panel B19, last paragraph). A third feature that agrees with c on every grid cell is read as the rule. |
| A2. The cue map is balanced and random. On the conflict cells of each structure, h(s) takes each wrong option equally often. The assignment of surfaces to structures in training, which fixes h, is drawn at random per seed, independent of the content of the surfaces. | The tests of section 4 need balance for their null. The K option item level test also needs every surface dependence the learner brings from content to be independent of h, which calibrate.py line 96 builds in by construction. If h is fixed across seeds or follows the content, a content preference is read as cue use by both the item and the seed level tests. Under balance the K option binomial null is conservative, 0.030 at alpha 0.05, since biased slips make the hit count Poisson binomial. |
| A3. The test cells are new to the learner. They lie outside T, and at tier 3 their surfaces are absent from the pretraining corpus. | A pretrained model may have practiced a "new" surface in its corpus (panel S7 point 8). Lambda_c is then inflated by practice the design cannot see. Fresh invented tokens are the only remedy. A3 fails for the binary design of Hermann et al., whose probe grid lies inside the support of the Gaussian training latents. There T_s reads invariance on the training support, not transfer to new cells. |
| A4. The unit of replication is the trained learner, one seed, and the estimand is the mean over seeds. | When the surface enters as a continuous input, one seed's random dependence on it is shared by all its test items. At bar Lambda_s = 0 the item level test then rejects in 0.305 of experiments, under a spread sigma_delta = 0.3 that was chosen, not measured on any trained network (section 4). |
| A5. The reading measures use, not hold. | A learner that holds the rule and does not use it gives Lambda_c = 0 (panel 7A2 point 1). A claim of absence does not follow from Lambda_c = 0. |
| A6. For the linear connectivity reading of tier 2, cue and rule differ in complexity. | Lubana et al. 2023 prove that mechanisms of different complexity give a loss barrier on the linear path only for a one hidden layer ReLU network with interpolating minimizers and perfectly predictive attributes (Appendix F, Theorem 1). The proof uses their Lemma 3, restated from Shah et al. without proof. With equal complexity, two mechanisms can be linearly connected (Section 8, the linear system example). A missing barrier then shows nothing. |
| A7. For the curvature reading of tier 2, the cue is a lookup over many surfaces and the rule is shared by all structures. | By the heuristic of R7, a single simple cue used by all the data, such as the offset of Jain et al. 2023, sits in high curvature beside the rule [heuristic]. The curvature split then reads narrow against shared, not cue against rule. |
| A8. For R5 and R5a, the features are fixed, training is full batch gradient descent from zero or its minimum norm limit, and the kernel and the base logits are invariant under the symmetry group of the design. For tier 3, the fine-tuned model's answers equal those of its linearization at the base model (R9). | Without the symmetry the answers under cross entropy depend on the number of steps and no closed form is derived (R5, consequence 3). With feature learning the answers depart from the linearized model. At tier 3 this departure is the quantity measured, not a defect. With AdamW the update is not the gradient, and the linearized prediction of R9 is not derived. |
| A9. For the item level test of the K option design, answers to distinct test problems are drawn independently given the learner, and each problem uses a distinct surface. | A learner with memory across problems, such as a model reading a shared context, makes answers dependent and the binomial null wrong. |

## 3. What can be derived

R1. Practiced cells identify nothing [derivation; source]. Let a cue learner f_s and a rule learner f_r give the same answer distribution on every cell of T. Then every statistic of answers drawn from D_T has the same distribution under f_s and f_r, at any sample size. Geirhos et al. 2020 state it for their toy task. They write "As long as tests are performed only on i.i.d. data, it is impossible to distinguish between these" (Section 3). Training loss, test accuracy on D_T and reward on D_T are all such statistics. So the information that separates the two lies on cells outside T where h(s) differs from r(c, n), the conflict cells. A statistic may also average agree cells, as T_s does over all 900 probes. Agree cells then add variance, not bias.

R2. The design null and the planted effect [derivation; checked numerically].

```
If P_f(a | G(c,s,n)) = q_{c,n}(a) for every s     (f invariant to A_s)
then  Lambda_s(f) = 0      exactly, on any grid.
If  P_f(a | x) = lam 1[a = h(s)] + (1 - lam) q_{c,n}(a)
then  Lambda_s(f) = lam    exactly, on any grid with pi(s) < 1.
Binary product grid (K = 2, z_s symmetric about zero):   E T_s = Lambda_s,   T_s = mean over Q of y_hat sign(z_s)
```

In the first case both terms equal q_{c,n}(h(s)). In the second the difference of the terms is lam (1 - pi(s)) on every cell. In the third, P(y_hat = sign(z_s) given x) = (1 + E[y_hat given x] sign(z_s)) / 2 and pi(s) = 1/2, and the subtracted term averages to zero because sign(z_s) sums to zero along each row of the grid. Balance is needed by the tests (A2), not by Lambda_s. Numerically, over 20,000 learners each, the mean of T_s is -0.0001 at lam = 0, 0.0499 at lam = 0.05 and 0.1001 at lam = 0.1 (calibrate_output.txt, Part A).

R3. Binary keys against K options [derivation; checked numerically]. With K = 2, every wrong answer on a conflict cell is the cue's answer, so a cue answer cannot be told from a slip on one cell. The product grid restores the null. If f ignores s and the values of z_s are symmetric about zero, the sum of y_hat sign(z_s) along a row of the grid at fixed z_c has expectation zero. With K > 2, a wrong answer that equals h(s) is evidence for the cue. A wrong answer that equals another option is not. Under the null a wrong answer equals h(s) with probability 1/(K - 1) by the balance of A2. One answer gives a likelihood ratio between cue and rule of about log(0.9 / 0.033) = 3.3 in log units at a slip rate of 0.1 over three wrong options (panel C15, computed there). The two designs differ in their unit of replication. In the binary design of Hermann et al., the surface is a continuous latent. A seed's random dependence on z_s has one sign for all items of that seed, so items are not independent. In the K option lookup design each test problem has its own surface, and items are independent given the learner when A9 holds. Section 4 gives the numbers.

R4. The data and the learner decide which is learned [source; derivation]. On cells where cue and rule agree, the loss is the same for both solutions, so the loss does not choose between them [derivation]. What chooses is the learner's bias together with the availability of each feature in the data, which Hermann et al. set by an amplification in the data (below). Panel B26 point 3 shows the same push toward the feature that is stronger on the data (panel C16, B26). The bias has published forms.
- Shah et al. 2020 prove that a one hidden layer ReLU network trained by gradient descent on the LSN data (one linear coordinate, one 3 slab coordinate, noise) relies only on the linear coordinate (Section 4, Theorem 1). On LMS-5 the measured S randomized AUC is 0.50 and the S^c randomized AUC is 1.00 (Section 4, Figure 3(b) and Table 4; Table 1 is the decision template) [source, location from the review].
- Lubana et al. 2023, Lemma 3. A minimizer found by gradient descent on data with several perfectly predictive attributes is invariant to all but the simplest attribute (Appendix F) [source; the lemma is restated from Shah et al. without proof].
- Hermann et al. 2023 separate predictivity, Pr(y = sign(z_i)), from availability, set by an amplification alpha_i and a nesting depth (Section 3). Their kernel analysis finds no availability bias in a linear network (Theorem 3) and a bias in a two layer ReLU network (Theorem 4). In experiments, deep ReLU networks prefer a feature with availability ratio 64 and predictivity 0.85 over a feature with predictivity 0.9 (Figure 3A), and depth raises the bias [source].
- Park et al. 2024 define concept signal as |dG(z)/dz_i| and find that it sets the speed and order of concept learning (Definition 3, Section 4.1) [source]. For a linear embedding it is the amplification of Hermann et al.

So the cue learner and the rule learner of tier 1 can be produced by design. Train on data in which the other feature is resampled, so that it carries no information. The truth is then known by construction.

R5. The threshold in surfaces per structure for a fixed feature learner [derivation; checked numerically]. Train on K structures, each in m surfaces, each surface used with one structure, so that h(s) = r(c) on every training cell. Let the learner's kernel on the grid be the split of panel B19.

```
k(x, x') = kappa_s 1[s = s'] + kappa_c 1[c = c'] + kappa_cs 1[s = s' and c = c']
targets  Y = one hot of r(c);   squared loss, minimum norm solution
conflict cell (c, s'), s' trained with c' != c:
   score of r(c)  = m kappa_c / lambda,   score of r(c') = kappa_s / lambda,   other options 0
   lambda = kappa_s + kappa_cs + m kappa_c
score ratio      = m kappa_c / (m kappa_c + kappa_s) = m / (m + m*),   m* = kappa_s / kappa_c
argmax answer    = r(c) if m > m*,  r(c') if m < m*,  tie at m = m*
new surface cell (c, s_new):  score of r(c) = m kappa_c / lambda > 0
```

The target lies in the subspace of vectors constant within a structure, an eigenspace of the training kernel matrix with eigenvalue lambda. Kernel regression and gradient descent from zero on the explicit features give score ratios 0.2000, 0.3333, 0.5000, 0.6667 and 0.8000 at m = 1, 2, 4, 8, 16 with kappa_s / kappa_c = 4, equal to the closed form to four decimals (calibrate_output.txt, Part C).

R5a. The answer does not depend on the loss in the symmetric design [derivation; checked numerically]. Let Gamma be the group that permutes the K structures together with their keys and permutes surfaces within a structure. Let the training kernel matrix G be invertible, let the loss be a sum over cells of a term invariant to a joint permutation of the classes (squared loss and cross entropy both are), and let full batch gradient descent start at W = 0. Then at every step t and on every cell x

```
f_t(x)_j = (a_t - b_t) S_j(x) + b_t T(x),    S_j(x) = sum over training cells i with y_i = j of k(x, x_i),   T(x) = sum_i k(x, x_i)
a_t - b_t > 0 for t >= 1,   so   argmax_j f_t(x) = argmax_j S_j(x)
```

Proof. The weights stay in the span of the training features, W_t = Phi^T A_t. One step maps A to A - eta grad L(G A), and this map commutes with Gamma because G and L are invariant. A_0 = 0 is invariant, so every A_t is. Gamma acts transitively on the training cells, and the stabilizer of a cell permutes the other K - 1 classes transitively, so A_t = a_t Y + b_t (1 - Y). The gap a_t - b_t starts at eta and grows, since G couples a cell only to cells of its own structure, where the gradient on the key class is below the gradient on every other class. Under cross entropy this holds at every step. Under squared loss it holds for a step size at most 1/lambda, with b_t = 0. On a conflict cell S_{r(c)} = m kappa_c, S_{r(c')} = kappa_s and the other S_j are 0, so the step at m* holds for both losses at every step. The answers of fix_checks.py, Part D, agree. In the symmetric design under cross entropy the rule is the argmax answer on no conflict cell at m = 3, on every cell at m = 5, and the scores tie on every cell at m = 4, after 200 and after 5,000 steps.

Four consequences.
1. On scores the ratio m / (m + m*) is a logistic function of log m with slope one. On argmax answers the rule share is a step at m*. A sharp transition in answers therefore occurs with fixed features. The reading of m* as the availability ratio of Hermann et al. in kernel units is an idea [idea].
2. Repeating every training cell the same number of times scales the loss. It leaves the minimum norm scores unchanged, and in the symmetric design it leaves the argmax answer unchanged at every step (R5a). Under cross entropy the softmax probability of the rule answer grows with training. At m = 8 it is 0.899 after 200 steps and 0.965 after 5,000 steps (Part D). Sampled answers therefore depend on the training length, and no closed form for their share is derived.
3. Without the symmetry no closed form for cross entropy is derived. With kappa_s drawn per surface as 4 times LogN(0, 0.5), the rule share of argmax answers at m = 3 is 0.500 under squared loss, and 0.333 after 200 and 0.417 after 5,000 steps of cross entropy (Part D). The answers of an asymmetric fixed feature learner depend on the loss and the number of steps. They are computed numerically from the kernel by running the same training on the linearized model (R9).
4. Any kappa_c > 0 raises the scores on new surface cells. That rise is spread through the base similarity, not the arrival of a rule (panel S6 point 2).

Raventos et al. 2023 report a sharp transition in a transformer trained on in-context linear regression. At D = 8, K = 16 examples and noise variance 0.25, a threshold lies between 2^14 and 2^15 pretraining tasks. Below it, more sequences per task move the model toward the Bayes estimator of the pretraining tasks. Above it, they move the model toward ridge regression on all tasks (Section 3.1, Figure 2). Weight decay lowers the threshold. Width has no effect on the threshold of the base model, and width and depth raise it only for the small model (Section 3.3, Figure 7) [source, the width and depth reading from the review]. Lubana et al. 2024 model the point of emergence as percolation on a bipartite graph of entities and properties, with threshold p_c of about 1/sqrt(|E||K|) (Section 7, Appendix B). The transition time of descriptive type checks scales with the square root of the number of properties. Unscrambling scales with |K|^{3/2}, and the authors give no account of that scaling (Section 7) [source]. By R5a and consequences 2 and 3, a step in answers and an effect of training length on softmax shares both occur with fixed features. Neither marks moved features by itself. What marks moved features is a departure of the trained model's answers from those of the linearized base model trained on the same cells with the same loss and steps (R9) [derivation]. The two sources are cited as observed transitions in deep networks, not as tests of this grid.

R6. Linear connectivity [source; idea]. Lubana et al. 2023 define mode connectivity on a path theta(t) as L(theta(t)) <= t L(theta_1) + (1 - t) L(theta_0) for all t (Definition 1). They conjecture that minimizers which are not linearly connected, up to permutation, are mechanistically dissimilar (Conjecture 1). They prove it in the setting of A6, and support it with ResNet-18 and VGG-13 on CIFAR-10 with box cues (Figures 4 and 5). Fine-tuning on data without the cue at a small or medium learning rate stays linearly connected to the pretrained model and keeps its reliance on the cue. A large learning rate, or a cue that predicted the label perfectly in pretraining, breaks both (Figure 5). Table 2 marks that linear connectivity of dissimilar mechanisms is possible in exceptional cases. So the reading runs one way. A barrier on the conflict cells is evidence of different mechanisms, and no barrier is evidence of nothing when A6 fails. Its contrapositive is the part a test can refute. Two models with the same mechanism should be linearly connected.

R7. Curvature [derivation, heuristic; source]. K-FAC approximates the Fisher F = E[g g^T] block by block (Merullo et al. 2025, Section 3). For a weight direction v used by a fraction phi of the data, with per example gradients along v of similar size, v^T F v is proportional to phi. A cue that is a lookup over M = K m surfaces uses each surface direction with mass about 1/(K m). A rule direction for one structure is used with mass about 1/K. The curvature ratio of rule to cue is then about m, and at m = 1 the two coincide [heuristic]. Merullo et al. report that in OLMo-2 7B at the layer 22 MLP input, the bottom 50 percent of K-FAC eigenvectors has 23.1 percent higher activation on memorized data than on clean data (Section 4). Removing the flat directions keeps 95 to 106 percent of logical reasoning, 66 to 74 percent of math and 3 to 16 percent of memorized recitation, against the unedited model (Section 6) [source]. The Fisher is taken on a data distribution, so narrow and shared are relative to it. A wrapper used by every fine-tuning example is shared under the fine-tuning distribution and narrow under the pretraining distribution [derivation].

R8. When a shortcut and a rule cannot be told apart, and when they can.

| case | told apart? | condition that would separate them |
|---|---|---|
| any test drawn from the practiced cells (R1) | no, at any size | conflict cells |
| a rule held and unused (A5) | no, by answers | a hint, or the hold part read from inside (panel B32) |
| linear path between mechanisms of equal complexity (A6) | no | a cue of lower complexity than the rule |
| curvature split with a single simple cue, or at m = 1 (R7) | no | a lookup cue over m > 1 surfaces per structure |
| binary design, item level test, seed level surface dependence (R3, A4) | false positives at 0.305 for bar Lambda_s = 0 | the seed as the unit, or the K option design with A2 |
| a third feature that agrees with c on the whole grid (A1) | no | a grid that crosses that feature too |
| fixed feature learner at m = m* in the symmetric design (R5a) | no, the scores of cue and rule tie | read the scores, or change m |

R9. The tangent kernel of LoRA at initialization [derivation]. LoRA writes an adapted matrix as W_l = W_l0 + B_l A_l, with B_l = 0 and A_l random at the start. At B = 0 the gradient of a logit with respect to A_l is zero, and with respect to B_l it is (dF_j/dW_l) A_l^T. The tangent kernel of the K answer logits on the grid is

```
Theta_jk(x, x') = sum_l tr[ (dF_j(x)/dW_l) A_l^T A_l (dF_k(x')/dW_l)^T ]        a K x K matrix per pair of cells
E_A Theta_jk(x, x') = r sigma_A^2 sum_l < dF_j(x)/dW_l , dF_k(x')/dW_l >_F       A_l entries i.i.d., variance sigma_A^2, rank r
linearized model   F_{t+1}(x) = F_t(x) - eta sum_{i in T} Theta(x, x_i) dL/dF(x_i) at F_t,   F_0 = the base logits
```

The prediction is the argmax of the linearized model run with the realized A_l, the same loss, the same cells and the same steps. It needs no reduction of Theta to one kappa. The summary (kappa_s, kappa_c, kappa_cs) is a least squares fit of (1/K) tr Theta(x, x') on the three indicators of R5 over pairs of grid cells, and its residual share is the rest of panel B19. The step at m* of R5a applies only when the residual share is small, the part of Theta that differs between classes is a common multiple of the identity, and F_0 is equal across the K answer options. The cost is one backward pass per cell and answer logit, K |Q| passes on the base model.

## 4. The calibration of the tier 1 reading

Learners with truth known by construction (calibrate.py, and fix_checks.py for Lambda_c and the probe grid of Hermann et al.).
- Binary design, Part A. A 30 x 30 probe grid over z_c and z_s in [-3, 3]. This is not the grid of Hermann et al., which spans plus and minus 3 mu_i with mu_c = 1.2816 (Section 4) [source, from the review]. The rule part answers +1 with probability sigmoid(3 z_c + delta z_s). delta is drawn per seed from N(0, sigma_delta^2), a seed level surface dependence with random sign, so bar Lambda_s = 0. With probability lam the answer is the cue, sign(z_s). The case with no effect is lam = 0 and the cases with planted effects are lam in {0.02, 0.05, 0.1, 0.2}.
- K option design, Part B. K = 4 options and 8 structures. Each structure has 5 conflict problems for each wrong option, 120 problems per learner, each with its own surface. The rule part is right with probability 1 - eps = 0.8. Its errors follow a structure specific bias drawn from Dirichlet(0.3) over the wrong options. With probability eta = 0.1 the answer is a surface specific option drawn independently of the cue map, a surface dependence unrelated to the cue, as A2 requires. With probability lam the answer is h(s).
- Lambda_c, fix_checks.py Part G. The same two designs with the roles of c and s exchanged. The base reads z_s, or answers h(s) with probability 0.8, with a seed level dependence on z_c or biased slips. With probability lam_c the answer is sign(z_c) or r(c).

Statistics and tests.
- Binary. T_s = mean over the grid of y_hat sign(z_s), with E T_s = Lambda_s (R2). The item level test is the exact sign test on the 450 pairs (z_c, +|z_s|) and (z_c, -|z_s|) of one learner. The seed level tests are a one sample t test of T_s over 10 seeds, and an exact sign test on the signs of T_s over the 10 seeds. T_c is the same with c and s exchanged.
- K option. w = the fraction of wrong answers on conflict problems that equal h(s). The item level test is the exact binomial test of one learner against 1/3. The seed level test is a one sample t test of w - 1/3 over 10 seeds. w_c is the fraction of answers other than h(s) that equal r(c), with the same tests.
- The false positive rate is the fraction of 2,000 experiments at lam = 0 in which the test rejects at alpha = 0.05, one sided. The power is the same fraction at a planted lam. One experiment is 10 learners.

What w estimates. w is not Lambda_s. For the learner of Part B, E Lambda_s = lam over the draws of the surface specific option, and by the ratio of expectations

```
E[w] ~ [ lam + (1 - lam) N0 ] / [ lam + (1 - lam) D0 ],    N0 = eta/K + (1 - eta) eps/(K - 1),   D0 = eta (K - 1)/K + (1 - eta) eps
```

At lam = 0 the ratio is 1/(K - 1) for every eps and eta, so the null of the test is free of the nuisance parameters. Its power is not. At lam = 0.1 and eta = 0.1, E[w] is 0.602, 0.536 and 0.469 at eps = 0.1, 0.2 and 0.4, the simulated means are 0.604, 0.536 and 0.470, and the item level power is 0.874, 0.789 and 0.660 (fix_checks_output.txt, Part F). An estimate of Lambda_s from w inverts the formula and needs N0 and D0, which agree cells give when the rule part slips at the same rate on agree and conflict cells.

Results for Lambda_s (calibrate_output.txt, 2,000 experiments per row).

| design | sigma_delta | lam | item level reject | seed t test reject | seed sign test reject |
|---|---|---|---|---|---|
| binary | 0 | 0 | 0.042 | 0.058 | 0.007 |
| binary | 0 | 0.02 | 0.292 | 0.963 | 0.627 |
| binary | 0 | 0.05 | 0.868 | 1.000 | 1.000 |
| binary | 0.3 | 0 | 0.305 | 0.053 | 0.011 |
| binary | 0.3 | 0.02 | 0.429 | 0.301 | 0.072 |
| binary | 0.3 | 0.05 | 0.633 | 0.893 | 0.482 |
| binary | 0.3 | 0.1 | 0.910 | 1.000 | 0.981 |
| K = 4 | none | 0 | 0.030 | 0.043 | |
| K = 4 | none | 0.05 | 0.334 | 0.991 | |
| K = 4 | none | 0.1 | 0.791 | 1.000 | |
| K = 4 | none | 0.2 | 0.999 | 1.000 | |

Results for Lambda_c (fix_checks_output.txt, Part G, 2,000 experiments per row).

| design | sigma_delta | lam_c | item level reject | seed t test reject |
|---|---|---|---|---|
| binary, T_c | 0 | 0 | 0.034 | 0.057 |
| binary, T_c | 0.3 | 0 | 0.303 | 0.056 |
| binary, T_c | 0.3 | 0.02 | 0.413 | 0.308 |
| binary, T_c | 0.3 | 0.05 | 0.636 | 0.886 |
| binary, T_c | 0.3 | 0.1 | 0.898 | 1.000 |
| K = 4, w_c | none | 0 | 0.032 | 0.051 |
| K = 4, w_c | none | 0.05 | 0.336 | 0.993 |
| K = 4, w_c | none | 0.1 | 0.788 | 1.000 |

Spread of the estimates. At sigma_delta = 0.3 and lam = 0, T_s over 20,000 learners has mean -0.0001, median 0.0000, standard deviation 0.0523 and 2.5 and 97.5 percent quantiles at -0.1022 and 0.1022. At sigma_delta = 0 the standard deviation is 0.0154. In the K option design at lam = 0, w has mean 0.3328, median 0.3333, standard deviation 0.0838, and quantiles at 0.1765 and 0.5000. T_c at sigma_delta = 0.3 and lam_c = 0 has mean 0.0010 and standard deviation 0.0523 (Part G).

Reading. The estimand is bar Lambda_s, the mean over seeds (section 1). Under a seed level surface dependence of standard deviation 0.3 in the logit slope, a chosen value, the item level test rejects in 0.305 of experiments at bar Lambda_s = 0, six times alpha. That learner does follow the surface by the per learner Lambda_s, so the 0.305 is a false positive rate for bar Lambda_s only. The seed level t test holds at 0.053, and its power is 0.893 at lam = 0.05 and 1.000 at lam = 0.1 with 10 seeds. The exact sign test over 10 seeds rejects at 0.007 and 0.011, below its largest attainable size at alpha = 0.05, 11/1024 = 0.0107, because T_s = 0 counts as not positive (calibrate.py line 64). In the K option design the item level test holds at 0.030 and has power 0.791 at lam = 0.1 from one learner, at eps = 0.2 and eta = 0.1. The seed level test has power 0.991 at lam = 0.05. The t test at sigma_delta = 0 rejects at 0.058, which is 1.6 standard errors above 0.05 at 2,000 experiments. Lambda_c has the same false positive rates and powers within simulation error, since each design is symmetric under the exchange of c and s. On the grid of Hermann et al., with mu = 1.2816 on both axes, the rates at sigma_delta = 0.3 are 0.316 (item level) and 0.051 (seed t test) at lam = 0, and the seed t test power is 0.905 at lam = 0.05 (Part H). No conclusion changes.

What this calibration does not test. The learners are written down, not trained. The size of the seed level surface dependence of a trained network, sigma_delta, is unknown. It sets the power of the seed level test and must be measured on trained rule learners, those trained with z_s resampled, before tier 1 reads anything. The power of the K option tests depends on eps and eta, which are measured on the trained learners too. The calibration does not test a learner whose cue use varies across test problems in a way correlated with the design, or a cue map that follows the content of the surfaces (A2).

## 5. One prediction per tier, written before any data

Tier 1. The setup of Hermann et al. 2023 as published (Choice 1a), 10 seeds per condition. The K option part holds only if the author picks Choice 1b.
- Prediction. (a) MLPs trained with z_s resampled have mean T_s equal to zero. (b) MLPs trained with z_c resampled have mean T_s above 0.9 (threshold chosen). (c) At core predictivity 0.9 and availability ratio 32, a ReLU MLP of depth 2 and width 128 has a larger mean T_s than a linear model, Hermann et al.'s Figure 3B read through Lambda_s. Under Choice 1b, the same three parts hold for w and w_c on the K option grid.
- What refutes it. (a) The two sided seed level t test rejects mean T_s = 0 at 0.05. Then trained learners with no information in the surface still prefer the cue's side, and the null of the reading fails for trained learners. (b) The upper 95 percent bound of mean T_s lies below 0.9. (c) The one sided Welch test of the ReLU against the linear mean T_s does not reject at 0.05. Hermann et al.'s reliance is computed on the same models. If reliance shows no difference either, the outcome is a failed reproduction of Figure 3B. If reliance differs and T_s does not, the two measures disagree on the same models, and the disagreement is traced before tier 2.
- What each outcome changes. If all parts hold, tier 2 uses Lambda_s with the measured seed spread as its null, and the seed count follows Choice 2 at that spread. If (a) fails, tier 2 waits until the source of the preference is found. If (b) fails, the cue learners are not pure and the planted scale of section 4 does not transfer. If (c) is a failed reproduction, the setup is fixed before any reading.

Tier 2. The PCFG Counter setup of Jain et al. 2023, reproduced, with the linear path protocol of Lubana et al. 2023.
- The reading. The cue answer is h = count(a) + 1, the answer of a wrapper over the pretrained count of a. The rule answer is r = count(b). Conflict strings are C_Te = 0 strings with count(b) different from count(a) + 1. A_s resamples the a tokens and keeps count(b). pi = P(count(a') + 1 = count(a) + 1) under that resampling is not 1/K and is computed from the string generator, so Lambda_s keeps the scale of section 1.
- Prediction [idea]. Jain et al. report low accuracy on C_Te = 0 strings and a probe that still reads count(a) (Figures 5 and 8). They do not report the answers. The prediction is that at the low prior P_L and learning rate S, fine-tuning with C_Tr = 1 gives a mean Lambda_s larger than fine-tuning with C_Tr = 0 at learning rate M, by the seed level Welch test. The linear path between a C_Tr = 1 model and a C_Tr = 0 model has a loss barrier on conflict strings. The path between two seeds of the same condition has none. The barrier threshold is chosen at the tier 2 build from the spread of the loss along paths between seeds.
- What refutes it. No difference in Lambda_s means the low accuracy is not the wrapper's answer count(a) + 1, and the cue answer is misdeclared. The distribution of answers is read before the reading is used. A barrier between two seeds of the same condition with equal Lambda_s refutes the joint use of Lambda_s and the barrier, since by Conjecture 1 the two would be dissimilar. No barrier between the C_Tr = 1 and C_Tr = 0 models refutes nothing, since A6 is not checked for count(a) + 1 against count(b). Both are counts of one token, and their order in complexity is not derived.
- The route comparison. Fine-tuning by a correctness reward on C_Tr = 1 data is compared with fine-tuning on the keys. No prediction is made. Panel B26 point 1 weights an example by p (1 - p) under reward and by 1 - p under keys. Point 3 shows, for training on keys with squared loss, that a feature stronger on the data entrenches (rho from -0.385 to -0.793). Nothing derived fixes the sign of the difference in Lambda_s between the two routes.
- What each outcome changes. A barrier that tracks Lambda_s, with none between seeds, makes linear connectivity a reading of mechanism inside the weights, usable at tier 3. The curvature reading is not predicted on this setup, since the offset is a single simple cue (A7, R7). It is tested on the lookup grid of section 8, choice 4, with the heuristic prediction that the curvature ratio of rule to cue directions rises with m and is near one at m = 1.

Tier 3. A pretrained language model below 1B parameters, with fresh problems built from invented tokens. K structures, each a declared transformation, are written in M surface templates, and the model is fine-tuned on m templates per structure, with the assignment of templates to structures random per seed (A2).
- Prediction. Under LoRA at a small learning rate, trained by full batch gradient descent or plain SGD on cross entropy, the argmax answer on each conflict cell equals that of the linearized base model of R9, run on the same cells with the same loss and steps. The agreement is at least 0.9 of conflict cells at every m (threshold chosen). If the fitted base kernel is symmetric (residual share below 0.1, chosen), the rule share of answers is a step at m* = kappa_s / kappa_c fitted before training (R5a). Under full fine-tuning at a large learning rate the agreement is lower than under LoRA.
- What refutes it. Under LoRA, an upper 95 percent bound of the mean agreement over 10 seeds below 0.9. Then the base kernel does not set the answers, and features moved even at a small learning rate.
- What each outcome changes. A match says the answers are set by the base model's similarity, which is spread and local in features. A departure says features moved, the change of the kernel that panel B19 point 2 calls arrival. That result goes to surrogate E as a learner whose response to evidence should differ.

## 6. The literature

Each source below was opened in full text in this unit unless the row says otherwise. Rows are in C:/Users/YuanW/PLMS/surrogates/sources/reading.md. In attempt 1 no source was reopened. Four locations and one reading are taken from the review and marked so in the text.

| source | what this theory takes from it | rung |
|---|---|---|
| Lubana, Bigelow, Dick, Krueger, Tanaka 2023, Mechanistic Mode Connectivity, ICML 2023, arXiv 2211.08422 | mechanism as invariance (Definitions 2 to 5), Conjecture 1, Theorem 1 and its conditions (Appendix F), Lemma 3 restated without proof, the fine-tuning results (Figure 5, Table 1), the slab data of Appendix F | full text |
| Jain, Kirk, Lubana, Dick, Tanaka, Grefenstette, Rocktaschel, Krueger 2023, Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks, arXiv 2311.12786, ICLR 2024 (reading.md line 7 lists it under 2024) | the PCFG Counter setup (Section 4, Appendices B.2, C.2), the cue count(b) = count(a) + 1, C_Tr and C_Te, low accuracy on C_Te = 0 and the probe of count(a) (Figures 5 and 8), pruning and probing protocols (Appendix D), reverse fine-tuning revives in 0.1K to 1K iterations against 4.5K for the baseline (Figure 9) | full text |
| Hermann, Mobahi, Fel, Mozer 2023, On the Foundations of Shortcut Learning, arXiv 2310.16228, ICLR 2024 | predictivity and availability, the generative framework and the reliance measure (Section 3), the 30 x 30 probe grid over plus and minus 3 mu_i (Section 4, Appendix C.1), Theorems 3 and 4 | full text |
| Shah, Tamuly, Raghunathan, Jain, Netrapalli 2020, The Pitfalls of Simplicity Bias in Neural Networks, NeurIPS 2020, arXiv 2006.07710 | S randomized accuracy and AUC (Definitions 1 and 2), LMS-k and LSN data, extreme simplicity bias (Section 4, Theorem 1, Figure 3(b), Table 4) | full text |
| Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, Wichmann 2020, Shortcut learning in deep neural networks, Nature Machine Intelligence, arXiv 2004.07780 | the indistinguishability statement on i.i.d. tests (Section 3), conditions for o.o.d. tests (Section 6.2) | full text |
| Raventos, Paul, Chen, Ganguli 2023, Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression, NeurIPS 2023, arXiv 2306.15063 | the task diversity threshold between 2^14 and 2^15 tasks, the reversal of the effect of more sequences per task at the threshold, the effect of weight decay, width and depth (Sections 3.1 to 3.3, Figure 7) | full text |
| Lubana, Kawaguchi, Dick, Tanaka 2024, A Percolation Model of Emergence, arXiv 2408.12578 | percolation threshold about 1/sqrt(|E||K|), square root scaling of the transition in the number of properties, the unexplained |K|^{3/2} scaling (Section 7, Appendix B) | full text |
| Merullo, Vatsavaya, Bushnaq, Lewis 2025, From Memorization to Reasoning in the Spectrum of Loss Curvature, arXiv 2510.24256 | K-FAC eigenbasis of MLP weights, flat directions carry memorized and narrow content, the edit and its retention by task (Sections 3 to 6) | full text |
| Park, Okawa, Lee, Tanaka, Lubana 2024, Emergence of Hidden Capabilities, arXiv 2406.19370 | the definition of concept signal (Definition 3) and its effect on learning speed (Section 4.1) | full text, definition and Section 4.1 only |

The setups to reproduce, named. Tier 1 reproduces the generative framework and probe grid of Hermann et al. 2023, Section 4, with d = 100, 3,200 training, 1,000 validation and 900 probe instances, rho_sc = 0.6, predictivity of the core feature 0.9, availability set by amplification. Tier 2 reproduces the PCFG Counter of Jain et al. 2023, a minGPT with about 3 million parameters, 6 blocks, 6 heads, embedding 192. Pretraining uses AdamW at 10^-3 with sampling priors P_L = (0.999, 0.001, 0), P_M = (0.9, 0.1, 0) and P_H = (0.5, 0.3, 0.2). Fine-tuning runs for 10K iterations at batch size 96 with learning rates M = 10^-5 and S = 10^-6 and C_Tr in {0, 0.5, 0.8, 1} (Appendix C.2). Tier 2 adds the linear path evaluation of Lubana et al. 2023, Section 5. Tier 3 reproduces the sweep design of Raventos et al. 2023, with diversity on a powers of two scale, and adds the linearized base model of R9 at each m.

Not opened in this unit and not relied on. Huang et al. 2026, the second candidate of the research program's shared ground. Results of panel members (C15, C16, B19, B23, B25, B26, B32) are cited by section and keep the rungs the members gave them.

## 7. What the theory cannot show

- Whether the declared structure is the one a teacher intended, or the one the learner computes. Lambda_c is relative to the declared c (A1).
- That a rule is absent. Answers bound use, not hold (A5). A rule held and unused needs surrogate E or the hold part read from inside.
- That linear connectivity implies the same mechanism. It runs one way, and only under A6.
- That the curvature split separates cue from rule in general. It separates narrow from shared directions relative to the data the Fisher is taken on (R7). The ratio of about m is a heuristic.
- A closed form for the answers of a learner trained by cross entropy outside the symmetric design, or for the share of sampled answers in any design. R5a gives the argmax answers in the symmetric design for squared loss and cross entropy alike. Elsewhere the fixed feature prediction is a numerical run of the linearized model (R9), and it holds only under full batch gradient descent or plain SGD, not AdamW.
- That a deep network outside the kernel regime follows R5 or R9. The linearized model is the reference against which tier 3 measures a departure.
- That the percolation model or the task diversity threshold applies to the problems built here. Both are cited as observed transitions, not as models of this grid.
- Anything about students. The features axis has no student counterpart (panel B23 point 3).
- Anything about real data. Nothing is trained or fitted here, and the calibration uses written down learners.

## 8. Choices for the author and proposed edits to the research program

Choice 1. The tier 1 setup.
- Option a. Hermann et al. 2023 as published, binary keys. Lambda_s is estimated by T_s, and the seed level test is the reading. The cost is CPU, minutes per MLP. It shows cue use with power 0.893 at lam = 0.05 when the seed spread matches sigma_delta = 0.3. A slip cannot be told from a cue answer on one cell, and the probe grid is inside the training support (A3).
- Option b. Option a first, then a K option extension. Each structure maps to K answers, and the surface feature becomes a lookup over many surfaces with the cue naming one wrong option, the assignment random per seed (A2). This departs from the published setup. The reason is a stated failure of the binary form. Every wrong answer is a cue answer (R3), and the item level test fails under seed level dependence (A4). The cost is one generator change and CPU. It shows cue use from one learner with power 0.789 to 0.791 at lam = 0.1 when eps = 0.2 and eta = 0.1, falling to 0.660 at eps = 0.4. It supplies the lookup grid for choice 4 and for tier 3.
- Option c. Jain et al. 2023 PCFG Counter as tier 1. The cost is pretraining a 3 million parameter transformer on the cluster GPU. It merges tiers 1 and 2 and keeps the shared ground of program section 5.
- My pick is b.

Choice 2. Seeds per condition. Powers of the one sided seed level t test at alpha 0.05 by the noncentral t distribution, at the T_s spread of sigma_delta = 0.3 (fix_checks_output.txt, Part E). The formula gives 0.891 at 10 seeds and lam = 0.05, against 0.893 simulated.

| seeds | power at lam = 0.02 | power at lam = 0.05 | power at lam = 0.1 |
|---|---|---|---|
| 5 | 0.177 | 0.571 | 0.978 |
| 10 | 0.299 | 0.891 | 1.000 |
| 20 | 0.500 | 0.995 | 1.000 |
| 40 | 0.766 | 1.000 | 1.000 |

- 5 seeds is half the cost of 10 and detects lam = 0.1 only.
- 10 seeds detects lam = 0.05 with power 0.891.
- 20 seeds is double the cost and still detects lam = 0.02 with power 0.500 only. Power 0.8 at lam = 0.02 needs 44 seeds.
- My pick is 10. The power is recomputed once sigma_delta is measured on trained rule learners.

Choice 3. The route comparison at tier 2.
- Option a. Answers only, as in Jain et al. Reward and keys are compared with no prediction (section 5). The cost is none beyond tier 2. It measures the difference that panel B26 points 1 and 3 leave open.
- Option b. Add a written intermediate format in which the teacher writes the running count of the operand, so that a teacher's text can carry the rule's steps (panel B26 point 4). The cost is a change of the data format and new pretraining. It separates an import of steps from an outcome signal.
- My pick is a, then b if the author wants the teacher route read at tier 2.

Choice 4. Where the curvature reading is tested.
- Option a. On the PCFG Counter. No prediction exists, since the cue is a single relation (A7).
- Option b. On a small transformer trained on the K option lookup grid of choice 1b, with m varied. It predicts a curvature ratio of about m by the heuristic of R7. The cost is a small GPU job per m. It uses a generator built here, not a published one.
- My pick is b.

Choice 5. The tier 3 regime.
- Option a. LoRA only. Low rank updates at a small learning rate keep the model near its linearization, so this tests the prediction of R9. The cost is one sweep over m with 10 seeds per m, plus K |Q| backward passes for Theta.
- Option b. Full fine-tuning only, at a large learning rate. Features can move, but without the LoRA match a departure from the linearized model cannot be told from a poor linearization. The cost is the same sweep with more memory per run.
- Option c. Both on one model below 1B parameters. The cost is about double that of either.
- My pick is c, since the difference in agreement between the two is the measurement.

Choice 6. The tier 3 optimizer.
- Option a. Plain SGD or full batch gradient descent. R9 gives the linearized prediction. The cost is slower training than AdamW.
- Option b. AdamW, the usual optimizer for LoRA. No linearized prediction is derived here, so tier 3 reads only the LoRA against full fine-tuning difference.
- My pick is a.

Proposed edits to the research program, section 4, surrogate R.
1. Tier 1. Add "The reading is bar Lambda_s and bar Lambda_c, the means over seeds on the crossed grid (theory/R.md section 1), each calibrated in section 4. The unit of replication is the trained seed. The setup is Hermann et al. 2023, then a K option extension."
2. Tier 2. Add "Linear connectivity needs a cue of lower complexity than the rule, and its refutable part is a barrier between two models of the same mechanism. The curvature split needs a lookup cue over several surfaces per structure. Reward and a teacher's answers are compared on agree data with no prediction."
3. Tier 3. Replace "the threshold in the number of surfaces per structure at which the rule appears" with "the argmax answers on conflict cells against those of the linearized base model trained on the same cells, a step at m* = kappa_s / kappa_c when the base kernel is symmetric. A departure marks moved features."
4. Sections 5 and 6, shared ground. Program section 5 has R and E share tier 1 and tier 2 ground on Jain et al. 2023, and the section 6 table has E tier 1 feed R tier 1. Choice 1b puts R tier 1 on Hermann et al. 2023 and keeps the shared ground at tier 2 only. Then the section 6 row "E tier 1 passes, E tier 2 and R tier 1 share its reading" holds only if E's reading applies on the crossed grid of Hermann et al. Choice 1c keeps the program as written. The choice is the author's, with my pick b.
5. Section 6. Add the row "R tier 3 departs from the linearized base model | E tier 2 receives a learner whose features moved".

## Fix log of attempt 1

One row per row of theory/R-review.md. Lines refer to this file after the fix. Scripts rerun with python -I on synthetic records, fix_checks.py in 21 seconds on the PC. calibrate.py was not changed and not rerun.

| item | what changed | line |
|---|---|---|
| 1 minor, reference values | per cell divisor 1 - pi(s) and 1 - rho(c, n) in the definitions; values exact without a balance condition | 16 to 18, 26 |
| 2 minor, mixture gives lam | holds exactly per cell with the new divisor; derivation restated | 57 to 61 |
| 3 minor, null needs balance | Lambda_s = 0 exactly on any grid; balance moved to the tests | 55 to 61 |
| 4 none | no change | 28 |
| 5 none | no change | 28 |
| 6 minor, Shah "the same thing" | now "a related quantity", accuracy against the label, agreeing only at the pure learners | 28 |
| 7 minor, Hermann reliance | defined in Section 3 with a prefactor; Lambda_s equals the first sum up to it | 28, 214 |
| 8 major, estimand | estimand bar Lambda, the mean over seeds, defined; per learner Lambda_s named as a second estimand; E[T_s given delta = 0.3] = 0.0500 rechecked | 19, 24, 183 |
| 9 none | no change | 38 |
| 10 minor, binomial null exact | stated as conservative, 0.030 | 39 |
| 11 major, h fixed is enough | A2 now requires the assignment random per seed and independent of surface content; failure under a fixed or content driven h stated | 39, 136 |
| 12 minor, A3 in the binary design | A3 stated to fail for the probe grid inside the training support | 40, 241 |
| 13 minor, 0.305 at a chosen spread | sigma_delta = 0.3 marked chosen and unmeasured | 41, 183 |
| 14 none | no change | 42 |
| 15 minor, Lemma 3 unproved | stated in A6 and in the source row | 43, 212 |
| 16 minor, A7 as fact | marked heuristic | 44 |
| 17 none | A8 rewritten for R5a and R9 (item 59) | 45 |
| 18 none | no change | 46 |
| 19 none | no change | 50 |
| 20 minor, conflict cells only | information lies on conflict cells; agree cells add variance, not bias | 50 |
| 21 minor, zero sum | now expectation zero along a row | 64 |
| 22 none | no change | 64 |
| 23 none | no change | 64 |
| 24 minor, no push from data | the loss does not choose; bias and availability in the data choose | 66 |
| 25 none | no change | 67 |
| 26 minor, Shah Table 1 | measured AUCs placed in Section 4, Figure 3(b), Table 4, from the review | 67, 215 |
| 27 minor, Lemma 3 proved | restated without proof | 68 |
| 28 none | no change | 69 |
| 29 none | no change | 70 |
| 30 none | no change; argmax line added | 76 to 87 |
| 31 major, no sharp transition | score ratio logistic, answers a step at m*; availability reading marked idea | 83, 99 |
| 32 minor, repetitions | squared loss scores and symmetric argmax unchanged; cross entropy softmax 0.899 to 0.965 at m = 8 (Part D) | 100 |
| 33 none | no change | 102 |
| 34 none | no change | 104 |
| 35 minor, Raventos width | width no effect on the base model; width and depth raise it for the small model only, from the review | 104 |
| 36 none | no change | 104 |
| 37 critical, sharper rise marks moved features | withdrawn; R5a derived and checked (step at m* for squared loss and cross entropy at every step); moved features now marked by departure from the linearized model | 89 to 104 |
| 38 none | contrapositive of Conjecture 1 added as the refutable part | 106 |
| 39 none | no change | 108 |
| 40 none | no change | 108 |
| 41 minor, mix at m* | now a tie of scores | 120 |
| 42 none | no change | 114 to 119 |
| 43 none | Part B text names eps and eta; Lambda_c design added | 134 to 137 |
| 44 none | no change | 152 to 166 |
| 45 none | T_c spread added | 181 |
| 46 minor, sign test size | 0.007 below 0.0107 explained by T_s = 0 counted as not positive | 183 |
| 47 minor, probe grid | grid of the script stated as [-3, 3], not Hermann's; rerun on plus and minus 3 mu (Part H) gives 0.316, 0.051, power 0.905 | 135, 183 |
| 48 major, w is Lambda_s | w separated from Lambda_s; E[w] formula, null free of nuisance, power 0.874, 0.789, 0.660 at eps 0.1, 0.2, 0.4 (Part F) | 144 to 150, 242 |
| 49 major, Lambda_c uncalibrated | Lambda_c calibrated in both designs (Part G), false positive rates 0.056 and 0.051, powers given | 137, 168 to 179, 183 |
| 50 none | eps and eta and content driven h added | 185 |
| 51 none | no change | 190 |
| 52 major, tier 1 refutation | three refutable parts, a two sided null test, an upper bound below 0.9, a Welch test | 189 to 192 |
| 53 minor, failed reproduction | reliance on the same models separates a failed reproduction from a disagreement | 191 |
| 54 none | no change | 194 |
| 55 major, answer count(a) + 1 and 0.9 | prediction made comparative and marked idea; pi computed from the generator; Jain's Figures 5 and 8 cited for what they show | 195, 196, 213 |
| 56 critical, no barrier refutes Conjecture 1 | withdrawn; refutation now a barrier between seeds of one condition; no barrier refutes nothing since A6 is unchecked for the two counts | 197 |
| 57 critical, reward equals keys | equality withdrawn; route comparison measured with no prediction; B26 points 1 and 3 reported as written | 198, 261 |
| 58 none | heuristic marked | 199 |
| 59 critical, share follows m / (m + m*) under LoRA | restated on argmax answers against the linearized model under cross entropy; closed form step only in the symmetric design; sampled shares withdrawn | 45, 202 to 204, 232 |
| 60 major, computing m* | R9 gives the tangent kernel at B = 0 through dF/dB = (dF/dW) A^T, uses the K x K kernel directly, and defines the kappa fit and its residual | 122 to 130 |
| 61 minor, Jain year | source row notes the 2024 listing in reading.md line 7 | 213 |
| 62 none | tier 3 setup adds the linearized model | 222 |
| 63 none | no change | 213 |
| 64 none | option b power qualified by eps and eta | 241, 242 |
| 65 critical, 20 seeds detect 0.02 | powers by noncentral t (Part E); 20 seeds give 0.500; 44 seeds for 0.8 | 246 to 258 |
| 66 minor, 5 seed power | 0.571 at lam 0.05 | 250, 255 |
| 67 minor, tier 1 on an open choice | tier 1 prediction on Choice 1a; K option part conditional on 1b | 189, 190 |
| 68 minor, edit 4 and shared ground | edit 4 rewritten against program sections 5 and 6 | 285 |
| 69 none | no change | 27, 281 to 286 |
| 70 minor, lazy prediction | section 7 now names the linearized model and its conditions | 232, 233 |
| 71 none | no change | none |
| 72 minor, run time | not fixed; calibrate.py not rerun, and no number in this file depends on its run time | none |
