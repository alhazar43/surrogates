# Surrogate S, student prediction. The theory

Unit TS, 19 October 2026, builder on Opus. Inputs: the research program (C:/Users/YuanW/PLMS/surrogates/program/research-program.md, section 4, surrogate S), the panel board (S1 to S15, member sections C4, C8, C12, C17, C21, C25, C30), goodfire-reading.md section 7, three-questions.md Part E, and the four primary texts of section 6. The Eedi data were not opened. Every statement about the log rests on the field list in C:/Users/YuanW/PLMS/signature/CLAUDE.md and on Wang et al. 2020, opened in full.

Rungs. idea; derivation; checked numerically (script and output named); source (primary text opened by this unit, section named).

## 1. The object

### 1.1 The log

```
L = { r = (u, q, o, k_q, y, g, w, s, d, kappa) }
u in U            student (UserId); 4,918 students in the train file of tasks 3 and 4
q in Q            question (QuestionId); |Q| = 948, each with an image
o in {1,2,3,4}    option chosen (AnswerValue);  k_q in {1,2,3,4} the key (CorrectAnswer)
y = 1[o = k_q]    right or wrong (IsCorrect)
g                 quiz (QuizId);  w  scheme of work (SchemeOfWorkId);  s  class (GroupId)
d                 DateAnswered, "to the nearest minute", used only to order a student's records
kappa in [0,100]  Confidence, 0 a guess, 100 total confidence, when present
S_q               the list of SubjectIds of q, nodes of a subject tree
t_u(r)            attempt rank of r among u's records, ordered by d; records of u in one minute share the mid rank
```

DateAnswered is recorded to the nearest minute (Wang et al. 2020, section 2.4, source), and the same source asks that a question take "less than 10 seconds" (section 3.3, source). So several records of one quiz share a minute, and the log gives no order among them. Every reading below uses only orders between minutes. A comparison that needs an order inside a minute drops the records that share the minute.

The release fixes two properties of L. "When a student has multiple answer records to the same question, we keep the latest answer record", so each pair (u, q) has one record (Wang et al. 2020, section 2.1, source). "For tasks 3 and 4, the UserIds are randomly split into 80%/10%/10% training/public test/private test sets" (same section, source); the cluster holds the train part only (signature CLAUDE.md, source). Time enters only as order; no quantity is measured in calendar units (PLMS requirement).

### 1.2 The declared content of the items

```
lambda : {(q, o) : o != k_q} -> M  union  {none}      the misconception an option answers
sigma  : Q -> Sigma                                   the structure of q (the method that solves it)
phi    : Q -> Phi                                     the surface of q (wording, format, numbers, context)
```

All three maps are declared from the item images by a written rule that a second reader can apply, and never from which options students choose together (panel C15, C25, derivation; the circularity is Proposition 4c). Each wrong option "is chosen to highlight a common misconception", and "these misconceptions are not labelled or linked between questions" (Wang et al. 2020, section 1.1, source). So lambda is not in the data and must be built. A crossed pair is two items with sigma equal and phi different, or phi equal and sigma different.

### 1.3 The learner the readings target

```
P(y = 1 | u, t, q)            = logistic( theta_u(t) - b_q - zeta * sum_m z_{u,m}(t) 1[m in M_q] )
P(o | y = 0, u, t, q)         = softmax_{o != k_q}( alpha_{q,o} + gamma * z_{u, lambda(q,o)}(t) + delta_{q,o} * theta_u(t) )
theta_u(t+1)                  = theta_u(t) + eta_u * n_u(t) + e_u(t),     e_u(t) ~ N(0, v)
z_{u,m}(t) in {0,1}           misconception m held at attempt t
```

theta_u(t) is the student's level in log odds; b_q the difficulty of q; M_q the misconceptions the options of q answer; zeta >= 0 the loss of a right answer that a held misconception causes; alpha_{q,o} the base preference among wrong options; gamma > 0 the pull of a held misconception toward its option; delta_{q,o} a dependence of wrong option choice on level with no misconception; n_u(t) = 1[S_{q(t)} meets K], where K is the declared set of SubjectIds of the skill a reading studies, so theta_u is the level on that skill and each reading carries its own K and its own law; eta_u the student's response to practice in log odds per practice attempt; v the variance of the random walk per attempt. This is the panel's mixture over rules (C17) joined to the scalar law of motion (C4, C8). It is a model of the answers, not of the inner process.

### 1.4 The reading of each tier

```
Tier 1  D(L) = (D1, ..., D9)                   design numbers, defined in section 3.9
        psi(j, k; m)                           local dependence of the options lambda^{-1}(m) of items j and k
                                               among students wrong on both, given the level
        T_MLS = sum_pairs U_eff / sqrt(sum U_eff^2)   pooled marginal-likelihood score statistic of psi = 0
                                               (Proposition 4f); log OR_MH on a rest score is reported beside
Tier 2  eta_1_hat = c4 / s_c, dAUC             the interaction (c4 of Proposition 7) of the posterior misconception probability with the
                                               amount of later practice, in log odds per attempt by the map s_c, and the AUC gain on test students
Tier 3  delta_delay = r_same - r_later         the change in the misconception's option rate on a sibling
                                               met in the same quiz, minus the change on a sibling first
                                               met in a later quiz, each against students without the case
```

### 1.5 The hint effect as held and unused

```
H_u in {0,1}  student u holds the rule        U_u in {0,1}  student u uses it on the transfer item
h in {0,1}    a hint that points to an earlier problem and names no rule
u_h = P(U = 1 | H = 1, h)                     P(U = 1 | H = 0, h) = 0  for h in {0,1}
```

This is the behavioral form of the split between what a learner holds and what it uses (panel B32, S11; research program section 3, idea 1).

What the object answers. Learning in students on behavior only. A change counts when theta, z or eta moves and the move is expressed in later options. What it leaves. The inner process, the features axis of B23, the arrival time of a rule (bounded from one side only, C15), and any equivalence between a student and a model.

## 2. The assumptions

| id | assumption | how it fails | what fails with it | checkable on Eedi |
|---|---|---|---|---|
| A1 | one b_q and one alpha_{q,.} for every student and occasion | item drift, a teacher's lesson that moves one item for a class | the scale, so theta is not comparable across students; every reading of tiers 2 and 3 | partly, through D4 (pace spread) |
| A2 | the law of motion of 1.3, a random walk plus eta per practice attempt | jumps, forgetting, a lesson outside the log | the path of theta; the floor of Proposition 2 is then stated under a wrong law | no; only the likelihood within one law chooses v |
| A3 | the next item depends only on the recorded past and on the class s as a whole | a teacher reassigns one student for an unrecorded reason | the interaction test of tier 2, which then needs the class term of Proposition 7 (c), and transfer (Proposition 9) | no; D6 says whether quizzes are class wide, not whether practice is independent of (theta, z) given the class |
| A4 | lambda is valid: an option answers the misconception the rule declares | a distractor chosen by guessing or by a misreading of the image | psi has no content; tier 2's state z_post is noise; tier 3 has no target | yes, through psi against unlinked pairs (prediction T1b) |
| A5 | local independence: given theta_u(t) and z_u(t), options on different items are independent | a shared reading habit, a test session effect, copying in a class | psi > 0 with no misconception; tier 3's sibling reading | no, only bounded by unlinked pairs of the same quiz |
| A6 | each record is the student's first encounter with q | a retake replaced the first record (release rule of 1.1) | the "before" of C21; memory of reused items; the reading of the first conflict item | no; D1 flags candidates and bounds the retake proportion from neither side (Proposition 3c) |
| A7 | the platform's feedback after an answer is the same for all students of a quiz | some students saw explanations and others did not | the route of a change; the response to a counterexample (C25) | no field records feedback |
| A8 | kappa is missing at random given the record | students enter confidence when sure | any use of kappa as a state | partly, through D8 |
| A9 | the hint names no rule, so it cannot install one | the hint teaches | the bound of Proposition 8 | not on Eedi; no hint field exists |

## 3. What can be derived

### 3.1 Proposition 1. Without a law of motion nothing about a path is identified
n items times T attempts of unknown logits face T answers. Each part a tracker reports is bought by an assumption, and most of the assumptions are design conditions (panel C4, derivation by counting, restated).

### 3.2 Proposition 2. The floor on the path of theta
Assume A2 with variance v per attempt and Fisher information I = a^2 p (1 - p) per binary answer, at most a^2 / 4. The scalar information recursion P = 1 / (1 / (P + v) + I) has the fixed point

```
P* = ( -I v + sqrt( I^2 v^2 + 4 I v ) ) / (2 I)   ~   sqrt(v / I)      for v I << 1
```

and the filter averages over about 1 / sqrt(v I) attempts. At v = 0.01 and I = 0.25, P* = 0.1951, the posterior standard deviation is 0.442 logit, and the window is 20 attempts (derivation; checked numerically, Part F of the script, closed form equal to the iterated recursion to five digits). The recursion is the posterior Cramér-Rao recursion of filtering, so the floor binds a causal tracker, one whose belief at t uses answers up to t. A smoother that uses the whole log has the steady-state variance Ps = (P* - J^2 (P* + v)) / (1 - J^2), J = P* / (P* + v), which is 0.100, a standard deviation of 0.316 logit, at the same v and I (derivation; checked numerically, Part F2 of calibrate_fix.py). A claim of trajectory recovery states its law and whether the tracker is causal, and is read against 0.442 or 0.316 (C8).

### 3.3 Proposition 3. What the release rule removes
Since each (u, q) keeps only its latest record (1.1, source):
a. Memory of reused items is not identified on this log under any model, since no student has two records of one item (derivation from the source). The program's design number "repeats and lags" is zero by construction.
b. A record is a first encounter only if no retake occurred. If a proportion r of records are retakes after instruction, a "before" rate p_before is estimated as (1 - r) p_before + r p_after, biased toward p_after (derivation).
c. A retake can leave a mark in order, since its kept record may sit out of block with the other records of its quiz. The mark is not specific. A student who interleaves two quizzes or resumes a quiz after another one gives out-of-block records with no retake, and a question reassigned in a later quiz keeps a record in block with that quiz. So D1 below counts candidates and bounds r from neither side (derivation). r is not identified on this log.

### 3.4 Proposition 4. The linked-option association
Fix a declared link: option a of item j and option b of item k answer the same m. Among students wrong on both j and k, let A = 1[o_j = a] and B = 1[o_k = b].
a. Under A5 with no misconception (gamma = 0) and delta = 0, A and B are independent, and the Cochran-Mantel-Haenszel test has its nominal level (derivation; checked numerically, Part A, rows "null, no ability dependence").
b. Under gamma = 0 and delta != 0, A and B are both functions of theta, so the crude odds ratio exceeds 1. Stratifying on a rest score from other items removes part of this and not all, since the rest score measures theta with error. The residual false positive rate grows with the number of students (derivation; checked numerically, Part A, rows "null, ability dependence 1.0").
c. A link chosen because the two options are chosen together has psi > 0 by selection, whatever the truth (derivation). So lambda is declared from content before psi is computed, and psi is a test of lambda, not a source of it.
d. Under gamma > 0 the association is positive. It grows with gamma, and it follows the variance q(1 - q) of z among students wrong on both, where q is the proportion of holders among them, so it rises and then falls in the prevalence pi. At gamma 1.4, delta 0 and the true theta in 50 strata, the population log OR_MH is 0.209, 0.402, 0.454, 0.374 and 0.177 at pi 0.1, 0.3, 0.5, 0.7 and 0.9, with q(1 - q) 0.115, 0.233, 0.244, 0.182 and 0.069 (derivation; checked numerically, Part Q of calibrate_fix.py).
e. Pooling brings the residue of (b) back. Let psi_0 > 0 be the mean of log OR_MH under gamma = 0 for the stratification used, and s1 its standard deviation for one pair. Pooled over K pairs of equal size, the MH statistic is close to N(psi_0 sqrt(K) / s1, 1), so its false positive rate is FPR(K) = Phi(-1.96 + psi_0 sqrt(K) / s1) + Phi(-1.96 - psi_0 sqrt(K) / s1), which tends to 1 in K (derivation). A finite rest score leaves psi_0 > 0 at any stratum width. A single pair on Eedi has no power (section 4.1), so every reading with power pools, and the stratified MH test is not calibrated at the pooled count.
f. A reading whose null holds at any pooled count. Model the level, not a stratum of it. For a pair (j, k), among students wrong on both,

```
P(A, B | theta) = p_a(theta)^A (1 - p_a)^(1-A) p_b(theta)^B (1 - p_b)^(1-B) exp(psi A B) / C(theta, psi)
p_a(theta) = logistic(a_j + d_j theta),  p_b(theta) = logistic(a_k + d_k theta)
w_u(theta) proportional to N(theta; 0, 1) prod_{i in R_u} P(y_ui | theta, b_i) (1 - P(y = 1 | theta, b_j)) (1 - P(y = 1 | theta, b_k))
U_u = A_u B_u - E_u[p_a p_b],   E_u over the posterior w_u(theta) P(A_u, B_u | theta, psi = 0)
```

with R_u the student's other answers under a Rasch model whose difficulties come from all train students, and (a_j, d_j, a_k, d_k) by maximum marginal likelihood. U_eff is U_u minus its projection on the four nuisance scores, and T_MLS = sum U_eff / sqrt(sum U_eff^2), summed over students and pairs, is the score test of psi = 0. If the level model and the option model hold, E[U_eff] = 0 at the true nuisance for every pair, so the mean of T_MLS stays 0 as K grows and the test keeps its level at any pooled count, any rest length and any slope d (derivation; checked numerically, Part L of calibrate_fix.py). The level error of a short rest score enters the posterior width and not a residue, so a student with 50 answers enters with a wider posterior.
g. The condition (f) cannot remove. If the linked items measure a level theta_L that the rest items measure only through a correlation rho < 1, then theta_L given the rest answers keeps a variance of at least 1 - rho^2. A and B both depend on theta_L, so they stay dependent given the rest answers, and the residue of (b) returns at a size that does not shrink with the rest length (derivation; checked numerically, Part L, rows "departure"). A continuous person trait of the linked skill and a misconception class are then two latent traits shared by j and k, and only a measure of theta_L from other items of that skill separates them (derivation; Part L2 of calibrate_fix.py).

### 3.5 Proposition 5. Two rules or one graded weight
Two discrete classes of students and one continuous propensity can both produce a positive psi. Let A_1, ..., A_J be the option indicators of J linked items, each with the class rates p_{1j}, p_{0j} and Delta_j = p_{1j} - p_{0j}.
a. Complete design, the same J items for every student. The two-class model has 1 + 2J parameters against 2^J - 1 free cells. At J = 2 it is not identified (5 against 3), at J = 3 it is just identified (7 against 7), and at J >= 4 it has testable restrictions (9 against 15) (derivation by counting; a necessary condition only).
b. Second moments do not separate the two. The two-class model gives cov(A_j, A_k) = pi (1 - pi) Delta_j Delta_k, and a one-factor propensity with loadings lambda_j gives lambda_j lambda_k, the same rank-one form (derivation).
c. Third central moments do. The two-class model gives E[(A_j - mu_j)(A_k - mu_k)(A_l - mu_l)] = pi (1 - pi) (1 - 2 pi) Delta_j Delta_k Delta_l, while a propensity that is symmetric about its mean gives a third moment near zero at small loadings (derivation; the propensity form is an assumption). So a test needs triples of linked items wrong for one student, and fails at pi near 0.5.
d. Incomplete design, students meeting different subsets of K linked items. If the class rates of an item do not depend on which other items a student met (an assumption that fails when teachers assign items by class, A3), the population model has 2K + 1 parameters, each observed triple gives 7 cells, and overlapping triples over-identify it (derivation by counting).
So the log cannot tell a rule switch from a graded shift unless enough students answer at least three linked items of one declared misconception wrong, in triples that overlap. This sharpens the warning C25 drew from Schapiro and McClelland into a count of triples.

### 3.6 Proposition 6. The gain on a raw score is predicted by any better estimate of the level
Let the true level be theta, the early score S = theta + e with error variance s_e^2, the later score Y = theta + eta n + e' with eta the same for all students, and the gain G = Y - S. Then cov(G, S) = -s_e^2 < 0, which is regression to the mean. Any state X that carries information on theta not in S, for example a filtered level or the option pattern, has a nonzero coefficient in the regression of G on (S, X), although no student responds to practice differently (derivation). The false positive rate of "X predicts the response to practice beyond the current score" then tends to 1 as the number of students grows, since a consistent test rejects a fixed nonzero coefficient with probability tending to 1 (derivation). The script shows the growth, 0.215 at 2,000 students and 0.440 at 4,918, not the limit (checked numerically, Part B, the naive column). So the comparison of tier 2 is never a state against a raw score.

### 3.7 Proposition 7. The interaction design identifies a difference in the response to practice
Model the answers of a test student after a cut tau as

```
logit P(y = 1) = c0 + c1 theta_hat_u(tau) + c2 z_hat_u(tau) + c3 n_u + c4 z_hat_u(tau) n_u - b_q
```

with theta_hat the filtered level and z_hat the tracked misconception state (z_post in the design of the fix, see below), both from records before tau, n_u the number of practice attempts on the skill between tau and the answer, and b_q estimated on train students only. If n_u is assigned independently of (theta, z, e) given the record before tau (A3 without a class), the level error enters c0 to c2 and not the slope in n. Under a linear link, which is collapsible, c4 = 0 then holds exactly when eta does not depend on z. Under the logistic link it holds when the residual variance of theta given (theta_hat, z_hat) does not vary with z_hat, and (b) gives the exception (derivation). Three conditions break it.
a. Practice that responds to later answers, a mastery rule, makes n a function of the outcome, and c4 then carries the outcome (derivation; Xiong et al. 2016, section 4.1, describe the ASSISTments skill builder in which students work "until they can correctly answer n right in a row", source).
b. The logistic link is not collapsible. Omitting the level error attenuates c3 by a factor that depends on the residual variance of theta given (theta_hat, z_hat). If that variance differs across z_hat, a spurious c4 appears even with A3 (derivation). The learner of section 4.2 has a similar residual variance across z_hat, so its calibration does not test this case.
c. A3 lets practice depend on the class s. If n_u depends on s and s correlates with level or with taught misconceptions, n_u is not independent of (theta, z) unless s enters the model. The model then adds a class intercept and a class slope in n, c0_s + c3_s n_u, so c4 is identified from the covariation of z_hat with the practice slope inside a class (derivation; not checked numerically). D6 does not decide whether this holds.
The unit of c4 is log odds per practice attempt per unit of the state in the product. If the state is the posterior probability z_post = P(z = 1 | linked options) from a two-class model fitted on train students, then E[z | z_post] = z_post under that model, and c4 estimates eta_1 = d eta / d z, the difference in log odds per practice attempt between a holder and a non-holder, up to the attenuation of the logistic link (derivation; the map is calibrated in section 4.2).

### 3.8 Proposition 8. The hint effect bounds the held and unused proportion
Under A9, the definitions of 1.5, and a hint assigned independently of H (a randomized hint), with pi_H = P(H = 1),

```
P(U = 1 | h = 1) - P(U = 1 | h = 0) = pi_H (u_1 - u_0)  <=  pi_H (1 - u_0)  =  P(H = 1, U = 0 | h = 0)
P(U = 1 | h = 1) <= pi_H
```

So the hint effect is a lower bound on the proportion of students who hold the rule and do not use it, and the hinted rate is a lower bound on the proportion who hold it (derivation, the bound of principal strata under monotonicity). If the hint teaches the rule with probability tau_h to non-holders, who then use it with probability u_t <= 1, the effect gains (1 - pi_H) tau_h u_t and the first bound fails. On Eedi the effect is not identified, since no field records a hint (field list, source). The model literature offers one analogue, the speed of revival. In Jain et al. 2023 a capability that fine-tuning wrapped returns in 0.1K to 1K iterations of reverse fine-tuning, while the baseline trained from scratch and then fine-tuned reaches perfect accuracy at 4.5K iterations, and only at the larger learning rate M (section 5.1, Figure 9 and its caption, source). In Tracr compiled models fine-tuned at the large learning rate 10^-1 the capability does not revive, which "hampers the pre-training capabilities" (appendix, Figures 39 and 43, source). In the PCFG models revival occurs after fine-tuning at their large learning rate 10^-4 (appendix, Figure 76, source). So the failure of revival is shown in one model family and at one learning rate. The student form is that a student who holds the rule and does not use it responds faster to practice (eta larger) than a student who does not hold it, at equal level. Tier 2 tests that form, with z_post as the mark of not holding the rule (idea; the mapping from weights to students is not derived).

### 3.9 The design numbers of tier 1
Each is computed before any model of change (panel C12, S4). A design number that enters a decision rule carries a calibration; a descriptive one enters no rule and is reported only.

| id | design number | definition on L | row of C12 it informs | role, calibration |
|---|---|---|---|---|
| D1 | retake candidates | proportion of records out of block with the other records of the same (u, g), by minute | A6, Proposition 3c | descriptive; bounds r from neither side |
| D2 | new subjects per window | leaf SubjectIds met for the first time by u, per 20 attempts (the window of Proposition 2 at v = 0.01, I = 0.25) | the person path against memory | descriptive |
| D3 | orders | for subject pairs met by at least 100 students (chosen), the proportion met in the minority order, within bins of equal attempt rank | transfer (Proposition 9) | enters T1a; binomial calibration in section 5.1 |
| D4 | pace spread | across students, the interquartile range of the attempt rank at which a quiz g is answered | item drift against experience, A1 | descriptive |
| D5 | option slope on level | for each declared linked option, the slope d_{q,o} of its logit given a wrong answer on the level, from the option model of Proposition 4f fitted to single items (no co-choice enters); C12 number 5 in effect size | A5, the null family of Proposition 4b | descriptive; reported beside T1b, since the reading of T1b estimates each slope |
| D6 | class schedule | proportion of quizzes answered by at least 90 percent (chosen) of the students of the class s that met them | A3 | descriptive; decides no row (Proposition 7c) |
| D7 | declared content | number of declared links lambda, number of crossed pairs in each direction, and number of students with three or more linked items of one misconception wrong | tiers 2 and 3, Proposition 5 | enters T1c; a count over thousands of students, sampling error not material |
| D8 | confidence present | proportion of records with kappa present, by quiz | A8 | descriptive |
| D9 | minute ties | proportion of records that share their minute with another record of the same student | 1.1, Proposition 10 | descriptive; sets how many records T3 drops |

### 3.10 Pairs of explanations the log cannot tell apart, and the condition that separates them

| explanation 1 | explanation 2 | separated when | holds on Eedi |
|---|---|---|---|
| memory of a reused item | growth of theta | reuse at varied lags | no, the release removes reuse (Proposition 3a) |
| a misconception held across items | wrong option choice driven by level | a level model and an option model under which the null holds at any pooled count (Proposition 4f) | only if the linked items measure the rest level; section 4.1 gives the rate under a departure |
| a discrete rule switch | a graded shift of one weight | third moments from overlapping triples of linked items wrong (Proposition 5) | D7 decides |
| a state that predicts the response to practice | a better estimate of the level | the interaction design of Proposition 7 with the class term | only if practice is independent of (theta, z) given the record and the class; D6 does not decide it |
| transfer from skill j to skill i | position in the scheme of work | both orders met by students of equal experience (panel C12) | D3 decides |
| use in the moment | lasting change | a sibling in the same quiz against one first met in a later quiz, and a comparison group (panel C30) | counts only; feedback unrecorded (A7) |
| a response to a counterexample | learning from one's own answer | a record that the outcome was shown (C25) | no |
| held and unused | not held | a hint that names no rule (Proposition 8) | no |

Proposition 9 (panel C12, derivation, restated). If every student meets skill j before skill i, the effect of practice on j on later answers to i is collinear with position in the scheme of work and is identified by no model.

Proposition 10 (panel C21 and C30, derivation, restated). The first conflict item of a structure is the only clean "before" for that student, so before and after are compared between students. The first conflict item is defined only when no other item of the structure shares its minute (1.1). On Eedi, with feedback unrecorded, delta_delay measures how long a change in the misconception's option rate persists after the first conflict item. It does not measure a response to a counterexample.

## 4. The calibration of the tier 1 and tier 2 readings

Scripts in C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_s/, run with python -I on synthetic records only. calibrate.py gives Parts F, A, B (calibrate_output.txt, seed 20261019) and A2, B2 (calibrate_output_extra.txt, seed 20261020). calibrate_fix.py gives Parts F2, P, P2, L, L2, Q, T and D of the fix (calibrate_fix_<part>.txt beside it, seeds in the script).

How the rates are computed. Each case draws R independent synthetic logs from the learner of 1.3, whose theta, z, gamma, delta and eta are known by construction. The false positive rate is the proportion of logs of a case with no effect (gamma = 0, or eta the same for all students) in which the test rejects at alpha = 0.05 (chosen). The power is the same proportion in a case with a planted effect of known size. The Monte Carlo standard error of a rate p is sqrt(p (1 - p) / R), 0.010 at p = 0.05 and R = 500, and 0.015 at R = 200. A test is calibrated when its false positive rate lies within two standard errors of 0.05 (chosen rule). Power is read only for a calibrated test, since rejection above alpha under the null inflates it.

### 4.1 The tier 1 reading, psi

The learner. theta ~ N(0, 1). Two linked items j and k with b = 0. A misconception z ~ Bernoulli(0.3) when planted, absent otherwise; it lowers the right answer by zeta = 0.5 log odds and pulls the wrong answer toward the linked option with gamma in log odds. Given a wrong answer, the linked option has logit -delta theta + gamma z against two other wrong options with logits 0 and -0.5. delta = 1.0 is the hard null, in which weaker students choose the linked option more with no misconception. The rest score counts right answers on 20 unlinked items (b ~ N(0, 1)) and is cut into 5 strata at its quantiles. The test is the Cochran-Mantel-Haenszel chi-square of A against B among students wrong on both items. The crude test is the same with one stratum.

| students | case | median students wrong on both | mean log OR_MH | rejection, MH | rejection, crude |
|---|---|---|---|---|---|
| 1000 | no effect, delta 0 | 294 | -0.022 | 0.058 | 0.058 |
| 1000 | no effect, delta 1.0 | 293 | 0.228 | 0.154 | 0.604 |
| 1000 | gamma 0.7, delta 1.0 | 325 | 0.300 | 0.232 | 0.732 |
| 1000 | gamma 1.4, delta 1.0 | 325 | 0.503 | 0.522 | 0.874 |
| 3000 | no effect, delta 0 | 879 | -0.003 | 0.054 | 0.058 |
| 3000 | no effect, delta 1.0 | 881 | 0.229 | 0.324 | 0.986 |
| 3000 | gamma 0.7, delta 1.0 | 980 | 0.288 | 0.516 | 0.990 |
| 3000 | gamma 1.4, delta 1.0 | 980 | 0.512 | 0.946 | 1.000 |

(Part A, 500 logs per row.) The MH test is calibrated when wrong option choice does not depend on level, 0.058 and 0.054 against 0.05. Under the hard null, delta 1.0, it is not calibrated. delta 1.0 is one point of a family whose residue grows with the product of the two slopes, and no Eedi number fixes it (Proposition 4e and Part P give delta 0.5 and 2.0). Its false positive rate is 0.154 at 1000 students and 0.324 at 3000, against 0.604 and 0.986 for the crude test. So stratifying on a 20 item rest score removes most of the level dependence and leaves a residue that grows with the number of students (Proposition 4b, checked numerically). The power figures of the rows with delta 1.0 are not read. This failed calibration is the first result of the surrogate.

Two remedies inside the MH design are a longer rest score and narrower strata (Part P, below). Eedi tasks 3 and 4 hold 1,382,727 / 4,918 = 281 answers per student on average (Wang et al. 2020, section 2.1).

| rest items | rejection, no effect, delta 1.0 | rejection, gamma 0.7, delta 1.0 |
|---|---|---|
| 20 | 0.257 | 0.497 |
| 100 | 0.087 | 0.220 |
| 250 | 0.070 | 0.197 |

(Part A2, 3000 students, 10 strata, 300 logs per row; standard error 0.013 at p = 0.05.) At 250 rest items the false positive rate is 0.070, inside the chosen rule (0.05 plus two standard errors is 0.075). The power at gamma = 0.7 is then 0.197. At 20 rest items the higher rejection under gamma = 0.7, 0.497, was mostly the false positive rate. These rates hold for one pair of 3000 students, about 880 wrong on both. They do not carry to Eedi.

The MH test on a 250 item rest score is not calibrated on Eedi once pairs are pooled. An item has on average 1,382,727 / 948 = 1,459 answers on Eedi, so one linked pair has at most a few hundred students wrong on both. At 1000 students per pair (292 wrong on both) one pair read by T_MLS has a rejection rate of 0.073 at gamma 0.7 against 0.060 with no effect (Part L), so it has no power, and any reading with power pools pairs. The MH test pooled over K pairs, 10 strata per pair cut at the quantiles of the students wrong on both, delta 1.0:

| K | pooled students wrong on both | MH false positive rate, simulated | logs | normal approximation (Proposition 4e) |
|---|---|---|---|---|
| 1 | 292 | 0.072 / 0.073 | 400 / 300 | 0.051 |
| 20 | 5,850 | 0.060 / 0.095 | 300 / 200 | 0.080 |
| 50 | 14,625 | 0.210 / 0.160 / 0.170 | 100 / 100 / 200 | 0.125 |
| 100 | 29,250 | not simulated | | 0.203 |
| 200 | 58,500 | not simulated | | 0.357 |

(Parts P, L and P2; one cell per independent run.) The approximation uses psi_0 = 0.0343, the mean pooled log OR_MH at K = 50 (Monte Carlo standard error 0.0028, median 0.0338, 5 to 95 percent -0.024 to 0.110, Part P2), and s1 = 0.304, the standard deviation for one pair (Part P). The population residue for the same rule, 400,000 students, is 0.0165 at 10 strata and 0.0151 at each score value, so the finite-sample pooled mean is twice the population value. This file does not trace the source of that gap, and the curve is read with it. A finer stratum lowers psi_0 by less than 10 percent and leaves the rise in K (Part P). The rise is steeper for steeper options. At delta 2.0 the population residue is 0.090, and at 50 rest items it is 0.094 (Part P). The false positive rate is therefore a function of the pooled count, the slopes and the rest length, not of the rest length alone.

The reading of the fix is the pooled marginal-likelihood score test T_MLS of Proposition 4f, run on the same draws as the MH test.

| case | K | logs | T_MLS rejection | MH rejection |
|---|---|---|---|---|
| no effect, delta 1.0, rest 250 | 1 / 20 / 50 | 300 / 200 / 100 | 0.060 / 0.060 / 0.060 | 0.073 / 0.095 / 0.160 |
| no effect, delta 1.0, rest 50 | 20 | 200 | 0.060 | 0.370 |
| no effect, delta 2.0, rest 50 | 20 | 200 | 0.030 | 1.000 |
| no effect, delta 0, rest 250 | 20 | 200 | 0.035 | 0.040 |
| gamma 0.7, delta 1.0, rest 250 | 1 / 20 | 300 / 200 | 0.073 / 0.530 | 0.087 / 0.705 |
| gamma 0.7, delta 1.0, rest 50 | 20 | 200 | 0.445 | 0.930 |
| departure rho 0.8, no effect, rest 250 | 1 / 20 | 300 / 200 | 0.210 / 0.990 | 0.243 / 0.990 |
| departure rho 0.95, no effect, rest 250 | 20 | 200 | 0.350 | 0.605 |
| rho 0.8, level from 10 items of the linked skill, no effect | 20 / 50 | 200 / 100 | 0.060 / 0.060 | 0.985 / 1.000 |
| rho 0.8, level from 10 items of the linked skill, gamma 0.7 | 20 | 200 | 0.345 | 1.000 |

(Parts L and L2; 1000 students per pair; the chosen rule admits 0.081 at 200 logs and 0.094 at 100.) T_MLS is calibrated at every pooled count tried, at 50 and 250 rest items, and at delta 0, 1.0 and 2.0, since it estimates each slope (mean fitted d_j -1.014 at true -1.0, -2.049 at -2.0). Its power at gamma 0.7 is 0.530 over 20 pairs with 250 rest items and 0.445 with 50. The MH power figures are not read, since its null rate is above the rule. Under the departure of Proposition 4g both tests fail, with 0.990 at rho 0.8 and 0.350 at rho 0.95 over 20 pairs. With the level taken from 10 items of the linked skill, T_MLS is calibrated again (0.060 at K 20 and 50) at power 0.345 over 20 pairs.

So tier 1 cannot be read on Eedi as first designed, with a level from all other answers in strata. It can be read by T_MLS with the level of the linked skill, if enough items of that skill precede or surround the linked pair for each student. The synthetic level of Part L2 is unidimensional inside the skill, and a skill defined by a subject leaf may hold too few items per student. The comparison of linked against unlinked pairs of the same subject leaf is not a remedy. A linked option is the distractor declared to answer a common misconception, which is the one weaker students choose most, so linked options can carry larger slopes than unlinked ones and the linked median can exceed the unlinked distribution with no shared misconception.

### 4.2 The tier 2 reading, c4, with the naive gain beside

The learner. theta ~ N(0, 1), z ~ Bernoulli(0.3). An early phase of 30 items (b ~ N(0, 1) per student), the first 10 linked, with zeta = 0.5 and gamma = 1.4. z_hat is the proportion of wrong answers on linked items that fall on the linked option, smoothed by (x + 0.5) / (n + 1). theta_hat is the posterior mean of theta under a N(0, 1) prior. Then n ~ Uniform{0, ..., 10} practice attempts, assigned at random, and 5 unlinked test items. The response to practice is eta = 0.05 + eta_1 z log odds per attempt. No effect is eta_1 = 0. The planted effect is eta_1 = -0.04, so a misconception holder gains 0.4 log odds less over 10 attempts. The naive test is the t test of z_hat in the least squares regression of the gain (test proportion right minus early proportion right) on the early proportion right and z_hat. The nested test is the cluster-robust Wald test of c4 in Proposition 7, with b_q known. dAUC is the AUC on a second, independent log of test students of the model with z_hat minus the model with theta_hat and n only.

| students | case | rejection, naive gain | rejection, nested c4 | mean c4 | dAUC mean | dAUC 5 to 95 percent |
|---|---|---|---|---|---|---|
| 2000 | no effect | 0.215 | 0.050 | 0.0023 | 0.0000 | -0.0004 to 0.0004 |
| 2000 | eta_1 = -0.04 | 0.070 | 0.305 | -0.0502 | 0.0000 | -0.0004 to 0.0004 |

| 4918 | no effect | 0.440 | 0.060 | -0.0002 | 0.0001 | -0.0001 to 0.0002 |
| 4918 | eta_1 = -0.04 | 0.130 | 0.610 | -0.0529 | 0.0001 | -0.0002 to 0.0003 |

(Part B, 200 logs per row at 2000 students; Part B2, 100 logs per row at 4918 students, the train file's count, standard error 0.022 at p = 0.05.) The naive gain test rejects in 0.215 of logs with no effect at 2000 students and in 0.440 at 4918, so its false positive rate grows with the number of students as Proposition 6 says (checked numerically). It rejects less often under the planted effect (0.070) than under none, so it is not a test of the response to practice at all. The nested test is calibrated, 0.050 at 2000 students and 0.060 at 4918 against 0.05, so the noncollapsibility of Proposition 7b is below the Monte Carlo error on this learner. Its rejection rate under the planted effect is 0.305 at 2000 students and 0.610 at 4918, with every student in the c4 fit, 5 test items per student and a planted difference of 0.04 log odds per attempt. These are not the power of the design of Proposition 7, which fits c4 on test students only. The AUC gain on test students has the same distribution with and without the planted effect, within plus or minus 0.0004. So the AUC on test learners cannot decide tier 2 at this effect size. It is reported beside c4, and the decision rests on c4.

The design of the fix (Part T). The state is z_post, the posterior probability of z from a two-class binomial model of the linked option count, fitted by EM on train students. c4 is fitted on test students only, with b_q known. Two designs: one split with 20 percent of the 4,918 students as test students (983), and 5-fold cross-fitting by student, in which each fold is the test set once, the two-class model is fitted on the other four, and the five c4 are pooled by inverse variance.

| linked items | design | eta_1 | logs | rejection | mean c4 | median c4 | SD c4 | mean SE |
|---|---|---|---|---|---|---|---|---|
| 10 | 5-fold | 0 | 200 | 0.070 | -0.0003 | 0.0003 | 0.0178 | 0.0167 |
| 10 | 5-fold | -0.04 | 200 | 0.630 | -0.0380 | -0.0368 | 0.0171 | 0.0168 |
| 10 | 5-fold | -0.08 | 100 | 0.990 | -0.0741 | -0.0736 | 0.0176 | 0.0169 |
| 10 | one split | 0 | 150 | 0.047 | -0.0030 | 0.0002 | 0.0375 | 0.0374 |
| 10 | one split | -0.04 | 150 | 0.153 | -0.0324 | -0.0293 | 0.0385 | 0.0376 |
| 4 | 5-fold | 0 | 100 | 0.050 | -0.0034 | -0.0017 | 0.0240 | 0.0239 |
| 4 | 5-fold | -0.04 | 100 | 0.370 | -0.0400 | -0.0400 | 0.0218 | 0.0240 |

Both designs are calibrated (0.070 against the rule's 0.081 at 200 logs, 0.047 and 0.050). The map from c4 to eta_1 is s_c = mean c4 / eta_1, 0.950 at eta_1 -0.04 and 0.926 at -0.08, mean 0.938, and 1.00 with 4 linked items. On the old z_hat the slope of z on z_hat given theta_hat is 1.186, so c4 on z_hat is eta_1 times a factor that depends on how many linked wrong answers a student has, and on z_post it is 0.966 with 10 linked items and 0.983 with 4 (200,000 students). The power at eta_1 = -0.04 is 0.630 with 5-fold cross-fitting, 0.153 with one split, and 0.370 with 5-fold cross-fitting when each student has 4 linked items (2.11 linked wrong answers on average, against 5.28 with 10). The class term of Proposition 7c is not in this learner, so its calibration is still open.

## 5. One prediction per tier, written before any data

Each threshold below is chosen, with its value. Each prediction names what refutes it and what each outcome changes above it.

### 5.1 Tier 1, the audit
- T1a, design. D3 is small. Fewer than 20 percent (chosen) of subject pairs met by at least 100 students (chosen) have a minority order proportion of at least 0.1 (chosen). Reason. Quizzes are set per class inside a scheme of work (field list), so most students of a class meet subjects in one order. Refuted if 20 percent or more of the pairs pass. Calibration (binomial, 100 students per pair, Part D of calibrate_fix.py). A pair with true minority proportion 0.05 passes with probability 0.028 and one at 0.2 with 0.998, so a log of locked pairs at 0.05 gives an expected 2.8 percent passing, and a log with 30 percent of pairs at 0.2 gives 31.9 percent. Pairs near 0.08 to 0.1 pass with probability 0.28 to 0.55, so the rule is a statement about pairs at 0.05 against 0.2, not about the boundary. If it holds, transfer between subjects is not identified on Eedi by any model (Proposition 9), and the transfer criterion of the thesis needs another log. If refuted, a transfer row enters tier 2 as a second outcome.
- T1b, links. The decision rule is the calibrated one of section 4.1. For links declared from content, T_MLS pooled over all declared linked pairs rejects psi = 0 at alpha 0.05 (chosen, two-sided) with psi > 0. The level of each student is the posterior of a Rasch level from that student's other answers of the linked skill, with item difficulties from all train students, so students with 50 answers and with 900 enter with their own posterior width. Beside it, and deciding nothing, are the stratified MH test with its false positive rate at the pooled count (section 4.1), the linked against unlinked comparison with the direction of its bias, D5, and a check that the linked items load on the skill level and not on a second level (Proposition 4g). Calibration. False positive rate 0.060 over 1, 20 and 50 pairs; power 0.530 over 20 pairs at gamma 0.7 with 250 rest items and 0.345 with 10 items of the linked skill (section 4.1). Reason. Each wrong option "is chosen to highlight a common misconception" (Wang et al. 2020, section 1.1, source). The Ghosh et al. 2021 figure (among students who chose 69D, 51 percent chose 293C, the option an expert linked to the same error, and 34 percent the key 293B, section 5.1, source) gives no marginal rate of 293C, so it supports no direction, and its source reads it as students who "may not consistently demonstrate an error". Refuted if T_MLS does not reject, or rejects with psi < 0. If refuted, A4 fails for this rule of declaration, z_post of tier 2 has no content, tier 3 has no target, and the student counterpart of E tier 3 (research program section 6) is dropped. If it holds, tier 2 starts.
- T1c, rule against graded weight. Fewer than 10 percent (chosen) of students who meet a declared misconception answer three or more of its linked items wrong. If so, the third moments of Proposition 5c rest on too few students, the log cannot tell a rule switch from a graded shift, and every claim of tiers 2 and 3 is stated for the option rate, never for a rule. Refuted if 10 percent or more do; then the two-class model and a one-factor propensity are compared on the overlapping triples (Proposition 5d), with the assumption of 5d stated. The count is a property of the log, so this rule has no false positive rate; the comparison it opens needs its own calibration.

### 5.2 Tier 2, the tracked state against the level
- T2. With the design of Proposition 7 on z_post, the class term of 7c and 5-fold cross-fitting by student, eta_1_hat = c4 / s_c is negative, in log odds per practice attempt of a holder minus a non-holder, with s_c = 0.938 the calibrated map (section 4.2; s_c is recalibrated at the Eedi count of linked items per student). At equal filtered level, holders gain less per practice attempt on later items of the skill. The cluster-robust Wald test rejects eta_1 = 0 at alpha 0.05 (chosen). The AUC gain on test students of the model with z_post over the model with the level alone is reported beside it and enters no decision, since section 4.2 shows it does not separate the planted effect from none. Reason. Proposition 8's student form: a student who answers by a misconception does not hold the rule, and a student whose errors are scattered holds it and does not use it.
- Refuted if eta_1_hat is positive with its 95 percent interval above 0 (holders respond more, as targeted teaching on the platform would make them), or if the lower end of its 95 percent interval lies above -0.04 log odds per attempt (chosen size, 0.4 log odds over 10 attempts). Otherwise the outcome is inconclusive. At the synthetic design (4,918 students, 10 linked items, 5 test items each), the standard error of eta_1_hat is 0.0179, the power at eta_1 = -0.04 is 0.630, and the probability of refutation at eta_1 = 0 is Phi(0.04 / 0.0179 - 1.96) = 0.608. With 4 linked items they are 0.024, 0.370 and 0.385, and with one 20 percent split 0.040, 0.153 and 0.168. A bound of -0.01 would be reached at eta_1 = 0 with probability 0.081, so this design cannot show an effect below 0.01. A positive eta_1 withdraws the held and unused reading of z_post and leaves it as a mark of teachable error. A refutation by size means the option pattern adds less than 0.04 log odds per attempt to the response to practice beyond the level, and the contribution of S to the question is the identification table of section 3 and nothing more. An inconclusive outcome changes nothing above. In every case tier 3 runs only if T1b held.
- What this does not test. Whether z_post measures holding. A negative eta_1 is also what a lower engagement of misconception holders gives. The power figures hold for the synthetic design; Eedi students have more later answers than 5, which raises power, and fewer linked items, which lowers it.

### 5.3 Tier 3, the delay channel
- T3. For students whose first conflict item of a structure shows the misconception's option, the option rate on a sibling falls more when the sibling is in the same quiz, in a later minute than the first conflict item, than when it is first met in a later quiz, and the later fall is above zero against students who meet the same sibling at the same attempt rank without the first conflict item. Refuted if the two falls are equal (no part bound to the moment) or both are zero against the comparison group (no change at all). If refuted by equality, the program's first definition of learning (three-questions.md B1) has no moment-bound part on this log. The hint part of tier 3 has no prediction, since Eedi records no hint (Proposition 8). Siblings that share the minute of the first conflict item are dropped (D9 gives their number). The reading delta_delay has no calibration in this file.

## 6. The literature

Each source below was opened in full by this unit and has a row in C:/Users/YuanW/PLMS/surrogates/sources/reading.md.

| source | what this theory takes from it | setup to copy |
|---|---|---|
| Wang, Lamb, Saveliev et al. 2020, Instructions and Guide for Diagnostic Questions: The NeurIPS 2020 Education Challenge, arXiv 2007.12061v3 | the definition of a diagnostic question and the unlinked misconceptions (1.1); the fields (2.1 to 2.4); the latest record per (u, q) and the split by UserId (2.1); 948 questions, 4,918 students, 1,382,727 answers in tasks 3 and 4 | the split by UserId of tasks 3 and 4, applied inside the train file |
| Ghosh, Raspat, Lan 2021, Option Tracing: Beyond Correctness Analysis in Knowledge Tracing, AIED 2021, arXiv 2104.09043 | option prediction on Eedi; a domain expert labeled options of the BIDMAS subset with images into 14 errors, and clusters from embeddings agree with the labels only partly; the 69D and 293C figure (5.1); the CF setup holds out 20 percent of each student's time steps at random, the KT setup holds out all time steps of 20 percent of students (section 5, Training and Testing) | the expert labeling of options as the template of lambda; the KT setup as the student split of tier 2. Not the CF setup, since later answers of a student then enter training |
| Xiong, Zhao, Van Inwegen, Beck 2016, Going Deeper with Deep Knowledge Tracing, EDM 2016 | three sources of inflated AUC on ASSISTments 2009-2010: 23.6 percent duplicated rows, scaffolding problems mixed in, and response sequences repeated once per skill tag of a multiple skill problem (4.1); item difficulty from items met by one student leaks the answer (4.3); the mastery rule of skill builders (4.1) | the check list of leakage for tier 2: one row per (u, q) even when S_q lists several subjects, item difficulty from train students only, and no practice amount set by a mastery rule |
| Jain, Kirk, Lubana et al. 2023, Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks, ICLR 2024, arXiv 2311.12786 | wrapper and revival; revival in 0.1K to 1K iterations, against 4.5K for scratch and fine-tune at the larger learning rate M (5.1, Figure 9 caption); no revival in Tracr models fine-tuned at 10^-1 (appendix, Figures 39 and 43), revival in PCFG models fine-tuned at 10^-4 (Figure 76) | none for students; the revival speed is the model form of Proposition 8 |

Not relied on. Gick and Holyoak 1980, Analogical problem solving, Cognitive Psychology 12, 306 to 355: the repository returned HTTP 403 and the full text was not opened, so the transfer rates with and without a hint that the panel quotes (C15, C17) are not used here. Proposition 8 does not need them. Robins's g formula and the identification of incomplete designs are recalled by the panel (C4, C12) and not opened; no statement here depends on them.

Setups to copy, in order. The release's split by UserId and Ghosh et al.'s KT setup for tier 2. Ghosh et al.'s expert labeling for lambda. Xiong et al.'s list of leaks as the audit of the tier 2 pipeline. No new generator is invented for the student. The synthetic learner of section 4 is the response model of 1.3 and exists only to calibrate the readings.

## 7. What the theory cannot show

- The inner process of a student. Every reading is a statement about options under the response model of 1.3.
- Whether a student holds a rule. Without a hint (Proposition 8) and without overlapping triples of linked items wrong (Proposition 5), holding is not identified from options; z_hat and z_post mark a pattern of options.
- The route of a change. Feedback is not recorded (A7), so practice with feedback, instruction, and learning from one's own answers give the same records.
- Memory of reused items. The release removed reuse (Proposition 3a).
- Transfer between subjects, if T1a holds (Proposition 9).
- The size of any effect on Eedi. No number of the log is measured here, and the synthetic numbers of section 4 hold for the synthetic learner only.
- Whether the law of motion of 1.3 is true. A tracker's path and its floor are claims under that law (Proposition 2), 0.442 logit for a causal tracker and 0.316 for a smoother at v 0.01 and I 0.25.
- The retake proportion r. D1 bounds it from neither side (Proposition 3c).
- An order among records of one student within one minute (1.1).
- Whether the linked items measure the same level as the rest items. The tier 1 reading is calibrated under that condition only (section 4.1).
- Whether the student form of revival follows from the model form. Jain et al. measure weights under gradient steps; the student form is an idea.

## 8. Choices for the author and proposed edits to the research program

### 8.1 Choices for the author

| choice | options, with mechanism | cost | what each shows | pick |
|---|---|---|---|---|
| how lambda, sigma and phi are declared | (a) two readers label options from the images by a written rule, Cohen's kappa reported; (b) a vision language model labels, two readers check a random sample; (c) ask Ghosh et al. for their expert labels of the BIDMAS subset | (a) days of reading for 948 items, hours for one subject leaf; (b) one cluster GPU job and a sample check; (c) an email, unknown answer | (a) a link a reader can check, agreement known; (b) all items, agreement known only on the sample; (c) a published expert labeling, 14 errors on one subset | (a) on one subject leaf first, then (b) for the rest with (a) as its check |
| the tier 1 reading | (a) MH test on a rest score, pooled over pairs; (b) the pooled marginal-likelihood score test of Proposition 4f, level integrated under a Rasch model with item difficulties from all train students; (c) psi of linked pairs against unlinked pairs of the same subject leaf; (d) (b) with a second level for the subject of the linked items | (a) none; (b) one small fit per pair; (c) none; (d) a two-dimensional level model and a new calibration | (a) a false positive rate that rises with the pooled count (Proposition 4e); (b) the nominal rate at every pooled count while the linked items measure the rest level, and a rising rate when they do not (section 4.1); (c) biased toward a positive difference when linked options depend on level more than unlinked ones; (d) removes the departure of (b) if the second level is estimable from few items | (b) as the decision rule, (a) and (c) beside, (d) if D5 or a dimensionality check shows the departure |
| the level that tier 2 compares against | (a) raw proportion correct; (b) prior right and wrong counts per skill, in the form of performance factors analysis as Xiong et al. describe it; (c) the filtered level under the law of 1.3 | none beyond tier 2 | (a) leaks by Proposition 6 and gives a false positive; (b) a standard baseline with its own regression to the mean; (c) the comparison Proposition 7 needs | (c), with (b) beside and (a) only to show the leak |
| the split of tier 2 | (a) one split, 20 percent of students as test students as in Ghosh et al.'s KT setup, a cut by attempt rank inside each test student, item and two-class parameters from train students; (a') the same with 5-fold cross-fitting by student, so every student is a test student once and the five c4 are pooled by inverse variance; (b) Ghosh et al.'s CF setup; (c) a cut at one date for all students | (a') five fits instead of one | (a) power at the planted effect in section 4.2, single split; (a') the same claim on every student, each scored by parameters not fitted on that student; (b) later answers enter training; (c) a calendar cut, against the project requirement | (a') |
| the class in tier 2 | (a) no class term; (b) class intercept and class slope in n (Proposition 7c); (c) class random effects | (b) and (c) a larger fit and a new calibration with a class confound | (a) c4 confounded with class when practice is set per class; (b) c4 from within-class variation; (c) partial pooling, valid only if the class effects are independent of z | (b), calibrated before use |
| retake candidates (D1) | (a) exclude them; (b) keep them; (c) both, reported as two numbers | negligible | (a) removes part of the bias of "before" (Proposition 3b) together with records that are not retakes (Proposition 3c), and loses data; (b) keeps all data with the bias; (c) the size of the bias | (c) |
| the hint part of tier 3 | (a) drop it from Eedi; (b) search for a log that records hint requests, then define tier 3 there; (c) use kappa as a mark of held and unused | (b) a literature and data search, scope change; (c) none, but uncalibrated | (a) Proposition 8 stays theory; (b) a test of Proposition 8 on students; (c) a mark whose relation to holding is not derived | (a) now, (b) as a proposal for the author's scope |

### 8.2 Proposed edits to the research program, section 4, surrogate S
1. Theory line. Add "the release keeps one record per student and question, the latest, so reuse is absent and a record is a first encounter only if no retake occurred (Wang et al. 2020, section 2.1)".
2. Tier 1. Replace "repeats and lags" by "retake candidates (records out of block with their quiz, by minute), which bound retakes from neither side". Add the linked-option local dependence read by the pooled marginal-likelihood score test, its null family with level-dependent options, its false positive rate against the pooled count and its rate when the linked items measure a second level (section 4.1 of theory/S.md). Add the count of students with three or more linked items of one misconception wrong (Proposition 5).
3. Tier 2. Replace "better than the current score, on a held out later part of each log" by "better than the filtered level, by the interaction of the posterior misconception probability with later practice, in log odds per practice attempt, on the later part of each test student's log, with 5-fold cross-fitting by student and a class term; reported as the interaction and the AUC gain on test students". Add the condition that practice is not set by a mastery rule.
4. Tier 3. Replace "a hint analogue if the log carries one" by "the hint is not in the Eedi fields, so the held and unused proportion (Proposition 8) is not identified on Eedi; a log with hint records is a scope question for the author".
5. Never answers. Add "whether a student holds a rule, without a hint or overlapping triples of linked items wrong".

## Fix log of attempt 1

Items are those of theory/S-review.md. Lines refer to this file after the fix. New numbers come from calibrate_fix.py.

| item | what changed | line |
|---|---|---|
| 1 critical | MH false positive rate given against the pooled count, simulated at K 1, 20, 50 and approximated to 200 (0.051 to 0.357); stated that tier 1 cannot be read on Eedi as first designed; new reading T_MLS with its null kept at K 1, 20, 50 (0.060), its failure when the linked items measure a second level (0.990 at rho 0.8), and its recalibration with a level of the linked skill (0.060, power 0.345) | 116-127, 228-257 |
| 2 critical | T1b decides by T_MLS, the calibrated reading; MH and the unlinked comparison are beside it with the direction of the bias; 8.1 pick made the same | 293, 340 |
| 3 critical | T2 states its threshold on eta_1 = c4 / s_c in log odds per attempt of a holder minus a non-holder, state z_post, map s_c 0.938 calibrated (z_hat slope 1.186 shown beside); bound moved to -0.04 with the refutation probability; -0.01 shown unreachable (0.081) | 151, 285, 297-298 |
| 4 critical | power computed at test-student designs: 0.630 with 5-fold cross-fitting, 0.153 with one 20 percent split, 0.370 with 4 linked items; old 0.305 and 0.610 marked as all-student fits | 271, 273-285, 298 |
| 5 major | Proposition 2 limited to causal trackers; smoother floor 0.316 logit added (Part F2) | 102, 327 |
| 6 major | Proposition 3c and D1 state that out-of-block records bound retakes from neither side; A6 not checkable | 85, 108, 168 |
| 7 major | DateAnswered to the nearest minute stated; ties share a mid rank; D9 counts ties; T3 and Proposition 10 drop same-minute records | 18-24, 176, 193, 302 |
| 8 major | Proposition 7c adds class intercept and slope; A3 row and the table row no longer rest on D6; class choice added to 8.1; its calibration stated open | 82, 150, 185, 285, 343 |
| 9 major | Proposition 5 recounted (J 2 not identified, J 3 just identified), second moments shown not to separate, third moments and overlapping triples given; T1c uses three linked items | 129-136, 294 |
| 10 major | Ghosh figure no longer read as support for a direction; the source's own reading quoted | 293 |
| 11 major | T1b level is a Rasch posterior per student from that student's answers; T_MLS calibrated at 50 and 250 rest items | 117, 293 |
| 12 major | stratum width reported (population residue 0.0165 at 10 strata, 0.0151 at each value); finer strata do not stop the rise in K. The review's 0.046 used edges on all students; here edges are cut on students wrong on both | 228-238 |
| 13 major | D table gives each number its role; D5 redefined as the option slope on level (C12 number 5 in effect size); D3 rule calibrated by binomial (Part D); descriptive numbers enter no rule | 165-176, 292 |
| 14 major | MH stated at delta 0.5, 1.0, 2.0; T_MLS estimates each slope and holds at delta 0, 1.0, 2.0 | 216, 238, 242-255 |
| 15 minor | Proposition 4d rises and falls in pi, follows q(1 - q) (Part Q) | 115 |
| 16 minor | c4 = 0 stated exactly for a collapsible link, approximately for the logistic; learner does not stress 7b | 147-149 |
| 17 minor | limit 1 marked as derivation; numbers marked as growth only | 138 |
| 18 minor | randomized hint stated; added effect carries u_t <= 1 | 154, 161 |
| 19 minor | Jain et al. narrowed to Tracr at 10^-1 (Figures 39, 43), PCFG revival at 10^-4 (Figure 76), 4.5K at learning rate M | 161, 313 |
| 20 minor | n_u(t) defined by a declared SubjectId set K per reading | 47 |
| 21 minor | "D1 above zero" removed from T1a; AUC bound of T2 removed, AUC enters no decision | 292, 297 |
