# The research program of the surrogates project

Version 3, 19 October 2026. An editable statement, not a plan; a piece becomes work only at the author's go. Version 2 is in program/archive/research-program-v2.md.

## 1. The question

The author's question is "what is learning". Learning is a lasting change in a learner, caused by an experience, that changes what the learner does later. What a change installed is not observed directly. The project probes it.

## 2. Probes and surrogates

A probe is a question sent at a problem that cannot be observed directly. A surrogate is a measurable quantity that stands in for a probe's exact target, used because the exact target cannot be computed with the data or the model at hand. Each probe has several surrogates.

Every reading is calibrated before use on learners with no effect and with a planted effect, which gives its false positive rate and power. The unit of inference is the trained seed or the learner, never the test case; case level tests rejected a true null in 0.690 of replicates (theory/E.md) and 0.305 of experiments (theory/R.md).

A "Checked" line is a derivation that its theory file's review left standing. Effect sizes and budgets are proposals; a stop day counts from the go on that probe.

## 3. Probes on what practice moves

### A1. Transfer
- Exact target. The change of loss on task B after practice on task A only, minus the change on B that seed noise alone gives.
- Bet. Practice on A moves B beyond seed noise when A and B share a rule, and within seed noise when they share none.
- Surrogates.
  1. Inter-task affinity, the loss drop on B after one lookahead step on A (Fifty et al. 2021, Eq. 1). The source gives cases where higher affinity gives higher loss.
  2. TracInCP, gradient products of A and B summed over checkpoints (Pruthi et al. 2020, Eq. 1). Belongs to one run in one order.
  3. Task2Vec distance (Achille et al. 2019, section 3.1). Scores the pair of tasks, not what practice did to the weights.
- Checked. One step on A moves B by the step size times the learner's similarity of A and B, within 0.026 on five learners (theory/E.md, E1).
- Checked. A shifting learner gains most from items at pass rate 0.5, a sharpening learner at 0.824 or 0.176 (theory/G.md, R4).
- Checked. One item's change reaches reliability 0.8 at about 79 answers per checkpoint (theory/G.md, R9).
- Smallest effect that matters. A move on B of 0.1 times the gain on A from the same practice, above twice the seed SD on B.
- Learners. A two-layer ReLU network on a crossed vector task; the 3 million parameter transformer of Jain et al. 2023.
- Budget and stop date. 2 CPU days and 2 GPU days, 10 seeds per pair. Stop at day 21.
- Outcomes. Win, shared pairs move and unshared do not; A2 and C1 take those pairs. Loss, unshared pairs move as much, so transfer is not specific. Null, with power 0.8 at the smallest effect, the detection threshold bounds transfer.
- Lessons. Rows 4, 6, 57 (size against seed spread); row 10 (no-change learners as control).

### A2. Order
- Exact target. The distance between end states after two orders of the same practice items, against the distance between two seeds of one order.
- Bet. Blocked practice (all of A, then all of B) and interleaved practice end farther apart than two seeds of either order.
- Surrogates.
  1. Error barrier on the straight line between end states (Frankle et al. 2020, section 2). Zero shows connected error, not equal functions.
  2. Spread of the test score over data order seeds (Dodge et al. 2020, section 3.1). Shows a nonzero spread, not that two chosen orders differ.
  3. Backward transfer on the task by stage accuracy matrix (Lopez-Paz and Ranzato 2017, Eq. 2 to 4). Needs equal test sets and a known seed spread.
- Checked. In a balanced two-layer diagonal learner the response is a function of present answers (theory/E.md, E4), and history sits only in a weight imbalance that weight decay erases at rate 2 lambda (E5).
- Smallest effect that matters. An order difference of 0.1 times the gain from plain practice of the task, above the 95th percentile of seed pair distances.
- Learners. The learners of A1, with and without weight decay.
- Budget and stop date. 10 seeds per order, 2 orders, 2 learners, 3 GPU days. Stop at day 21.
- Outcomes. Win, the order design enters B2. Loss, orders sit inside seed spread. Null, the power bounds the order effect.
- Lessons. Row 34 (real unit of replication); row 45 (no verdict from one seed); row 52 (stop epoch).

## 4. Probes on what is learned

### B1. Cue or rule
- Exact target. The probability that the trained model uses cue s rather than rule t where they disagree, as a function of the cost of computing each.
- Bet. On conflict inputs the cheaper feature wins in at least 4 of 5 cost pairs, on both learners.
- Surrogates.
  1. Relative extractability, the ratio of probe code lengths (Lovering et al. 2021, section 2.2). The ratio fails on GloVe.
  2. Shortcut bias, cue reliance minus Bayes reliance on a probe grid (Hermann et al. 2023, sections 3 and 4). Some reliance is optimal when features correlate.
  3. Parameter count of the smallest network fitting the cue (Scimeca et al. 2022, Eq. 6). Depends on the tolerance and the network family.
- Checked. No test on practiced cells separates cue from rule at any sample size (theory/R.md, R1).
- Checked. On a crossed grid the mean cue rate is exactly zero without a cue and the planted share with one; over 20,000 learners -0.0001, 0.0499, 0.1001 (R2).
- Checked. A fixed feature learner switches from cue to rule at m* surfaces per structure (R5, R5a).
- Checked. The seed level t test over 10 seeds has false positive rate 0.053 and power 0.893 at cue share 0.05; the binary item level test rejects a true null in 0.305 (R3, R8).
- Smallest effect that matters. A cue share of 0.05 on conflict cells beyond that of the same network linearized at its start and trained on the same cells.
- Learners. A ReLU network on the vector task of Hermann et al. 2023 at depth 2 and depth 8; the transformer of Jain et al. 2023.
- Budget and stop date. 3 CPU days and 1 GPU day. Stop at day 21.
- Outcomes. Win, the crossed grid enters A1 and C2. Loss, the costlier feature wins, so this cost does not decide use. Null, the power bounds the cue share.
- Lessons. Rows 12, 16 (calibrate before the rule); row 26 (plant inside and outside); row 53 (sizes set by hand).

### B2. Gap
- Exact target. The interaction of the gap g between repeats of a rare task and the width N, in the loss on that task after its return.
- Bet. The loss after the return rises with g faster at the smallest width than at the largest.
- Surrogates. No source measures the interaction; each reads one margin.
  1. Forgetting k steps after last use, by membership inference or canary exposure (Jagielski et al. 2023, Definition 3.1). A canary is one example, not a task.
  2. Count of forgetting events per example (Toneva et al. 2019, section 3). A lower bound that mixes gap with sampling rate.
  3. Accuracy against the count of relevant documents, by model size (Kandpal et al. 2023, sections 3 and 4). Ignores when the documents were seen.
- Checked. With weight decay and a fitted training set an unused feature decays as exp(-lambda t), with no width term (theory/E.md, E5). A width effect must exceed this decay.
- Smallest effect that matters. An interaction of 0.1 times the loss drop that one return of the rare task gives.
- Learners. The linear regression mixture of Huang et al. 2026, section 3, and the transformer of Jain et al. 2023, each at three widths.
- Budget and stop date. 4 GPU days. Stop at day 28.
- Outcomes. Win, narrow models lose rare tasks faster. Loss, the gap effect does not depend on width. Null, the power bounds the interaction.
- Lessons. Row 18 (horizon of the cut); row 20 (state the detectable effect).

## 5. Probes on held capability

### C1. Revival
- Exact target. The fewest steps until accuracy on a capability returns to its level before fine-tuning, under training without its answers, against a model that never had it.
- Bet. At a small fine-tuning learning rate, revival takes under half the steps of the control trained from scratch.
- Surrogates.
  1. Reverse fine-tuning time against a control (Jain et al. 2023, section 5.1). Fails at a large learning rate.
  2. Recovery rate after retraining on independent facts (Deeb and Roger 2025, section 4.3). Recovers little when hiding is local to each fact.
  3. Linear probe accuracy on the hidden capability (Jain et al. 2023, Figure 8). Fails when the capability sits in weights, not activations (Deeb and Roger 2025).
- Checked. A held rule revives slower than an absent one when its initial similarity is within a factor of 1.9 to 2.0 above the absent one's and faster beyond, so revival time alone inverts there (theory/E.md, E10).
- Smallest effect that matters. Revival in at most 0.5 times the steps of plain training from scratch.
- Learners. The transformer of Jain et al. 2023; the two-layer diagonal learner of theory/E.md on CPU.
- Budget and stop date. 2 GPU days and 1 CPU day. Stop at day 21.
- Outcomes. Win, a held unused capability shows by speed. Loss, revival is as slow as from scratch. Null, the power bounds the step ratio.
- Lessons. Row 19 (room to move); row 27 (activations against behavior); row 59 (ablation keeps the trajectory).

### C2. One-step response
- Exact target. The relation, across examples z, between the response to one step on z and the later change of test loss.
- Bet. The one-step response predicts the later change better than the plain loss on z does.
- Surrogates.
  1. Idealized TracIn, the test loss drop at each step that uses z (Pruthi et al. 2020, section 3.1). Another data order gives another value.
  2. Influence function with the inverse Hessian (Koh and Liang 2017, Eq. 2). Approximates removal at convergence, not one step.
  3. LESS influence, a cosine of Adam steps (Xia et al. 2024, Definition 3.1). Discards the step size.
- Checked. The response is one column of the learner's similarity times a per case scalar, divided per case; a ratio of raw means rejects a true null in 0.120 against 0.050 (theory/E.md, E1, E2).
- Checked. A linear learner responds alike in every state (E3); in a two-layer learner only the response separates a held unused rule from an absent one (E11, row 7).
- Checked. The permutation test of a response difference has false positive rate 0.045 to 0.058 and power 0.8 at about 0.14 (dense) and 0.10 (diagonal) (theory/E.md, section 4).
- Smallest effect that matters. A gain of 0.1 in rank correlation with the later change over the plain loss on z.
- Learners. The reference learners of theory/E.md on CPU; the transformer of Jain et al. 2023.
- Budget and stop date. 1 CPU day and 1 GPU day. Stop at day 14.
- Outcomes. Win, the response becomes a reading for A1 and C1. Loss, the plain loss predicts as well. Null, the power bounds the gain.
- Lessons. Row 1 (floor from a free reference); row 58 (zero on no-learning data).

## 6. Probes on the process and the measurement

### D1. Chain use
- Exact target. On problems built to need k written steps, accuracy against the built truth when only the first fraction f of the model's own chain is kept, minus accuracy with the full chain.
- Bet. Accuracy falls as the chain is cut on problems that need it, and stays flat on problems that do not.
- Surrogates.
  1. Early answering area over the curve (Lanham et al. 2023, section 2.3). Scores agreement with the full chain, not accuracy.
  2. Forced answer accuracy by cut point (Boppana et al. 2026, section 3.3). The cut text is off the model's own distribution.
  3. Outcome distribution per prefix (Bigelow et al. 2026). Intervals cover 48 to 64 percent against 90 nominal.
- Checked. Along sampled chains the summed squared movement of the outcome distribution equals the start uncertainty, 0.3842 against 0.3855 (theory/P.md, D2).
- Checked. A flat curve shows the chain is not needed; only a changed intermediate value shows use (D3).
- Checked. The forced answer statistic has false positive rate 0.051 and power 0.78 at a rise of 0.06 in key probability, 100 items per kind (theory/P.md, section 4).
- Smallest effect that matters. A drop of 0.06 at f = 0 on needed problems beyond the drop on problems that do not need the chain.
- Learners. A transformer trained on the Bayes net task of Prystawski et al. 2023; a small instruction-tuned model.
- Budget and stop date. 3 GPU days. Stop at day 28.
- Outcomes. Win, the chain does work; the use test follows. Loss, accuracy stays flat on needed problems. Null, the power bounds the drop.
- Lessons. Row 30 (format is not content); row 42 (state what the response is).

### D2. Order of difficulty
- Exact target. Kendall tau between the error rates of a fixed item set under the base and the post-trained model, and whether 1 minus tau exceeds sampling noise.
- Bet. Supervised fine-tuning on teacher text reorders the items beyond noise; outcome reward with a KL penalty does not.
- Surrogates.
  1. Pass rate per item over G samples (Bae et al. 2025, section 3.1). At G = 16, ties at 0 and 1 bias tau.
  2. Two-parameter item response difficulty (Hofmann et al. 2025, section 3). Held fixed across checkpoints, so it cannot show a reorder.
  3. Item specific residual sigma_hat after the best increasing map (theory/M.md, section 4). Above 0.22 logit, false positive rate at most about 0.05 in five of six order preserving cells, power at least about 0.95 at an item specific SD of 0.25 logit.
- Checked. One learner does not separate a person change from a common item change (theory/M.md, P1); no reorder does not show an unchanged capability (P4).
- Checked. KL-regularized reward at its optimum adds 1/beta to every trained prompt's log odds and keeps the order (P2). A decoding temperature on four options reorders without a capability change, median rho 0.986 at 0.5 (P3).
- Smallest effect that matters. An item specific SD of 0.25 logit beyond the common shift of plain outcome reward.
- Learners. Two small pretrained language models of different families, each after supervised fine-tuning and after reward.
- Budget and stop date. 4 GPU days. Stop at day 28.
- Outcomes. Win, a change beyond a common shift; its residual by subject is read next. Loss, both trainings keep the order. Null, the power bounds the reorder.
- Lessons. Row 31 (test flattening and sharpening); row 25 (common factor); row 24 (reference population).

Derivations with no probe stay in their theory files, namely G R2 and R5, P D5 to D8, and theory/S.md.

## 7. Rules

1. A probe tests one fixed claim, its bet. A failed probe is closed, not reshaped.
2. A surrogate that disagrees with the others of its probe is reported, not dropped.
3. A reading enters a probe only with its false positive rate and power beside it.
4. Each outcome's action is written before the run.

## 8. How the probes feed each other, and their order

| step | probe | a win feeds | next after a win | next after a loss or a null |
|---|---|---|---|---|
| 1 | C2 and B1, on CPU | the response (C2) and the crossed grid (B1) to A1 | A1 | A1 with its own surrogates |
| 2 | A1 | its task pairs to A2 and C1 | C1 | C1 on the pairs of Jain et al. 2023 |
| 3 | C1 | read beside C2 | A2 | A2 |
| 4 | A2 | its order design to B2 | B2 with that design | B2 with blocked order |
| 5 | B2 | the author's next pick | the author | the author |
| beside, from step 1 | D1, then D2 | none | D2 | D2 |

## 9. Open for the author

- The student side stays an open question, not integrated into the probes; theory/S.md is its record.
- The agreed build ground is the setup of Jain et al. 2023 first and Huang et al. 2026 second, not opened now.
- The peer's work stays separate.

## 10. Change log

| date | change | by |
|---|---|---|
| 19 October 2026 | version 1 | the lead |
| 19 October 2026 | version 2. Four queued corrections (G tier 3, M theory, P theory, idea 1); the theory files' section 8 edits that correct an error; a calibration line per surrogate; unit of inference in condition 3; sections 5, 6 and 9 | builder of unit V2 |
| 2026-10-19T10:52Z | Section 5: the author decided the setup order, Jain et al. 2023 first, Huang et al. 2026 second. | |
| 19 October 2026 | version 3, eight probes with eight fields; old surrogate sections dropped, checked derivations moved into probes; rules, order, contingency | builder of unit V3 |
