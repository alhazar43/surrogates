# The research program of the surrogates project

Version 2, 19 October 2026. Written by the lead at the author's word, as an editable statement and not an executable plan: "your surrogate should have a tiered structure, what is doable and what is more ambitious and depending on results", "dont force each surrogate as one paper". Version 2 folds in the theory files theory/G.md, M.md, R.md, E.md, P.md and S.md. Version 1 is in program/archive/research-program-v1.md.

## 0. How this document works

This is a working statement of the question, the surrogates, their tiers and how they depend on each other. The author and the lead edit it. It holds no rows, jobs or dates. A piece of it becomes work only when the author gives a go on that piece, and then the lead writes a unit file under notes/units/. Every edit adds one line to the change log at the end. A superseded version goes to program/archive/ with its version in the name.

## 1. The question, and why it is answered in parts

The author's question is "what is learning". The working definition is broad on purpose. Learning is a lasting change in a learner, caused by an experience, that changes what the learner does later. Memorizing an arbitrary code counts, and so does acquiring a misconception.

No single study answers the question. The hard part is not the definition. It is to tell what a change installed, and to tell a true reading of it from noise. Two learners can give the same answers today and have learned different things. A reading that is too harsh finds nothing, and a reading that is too loose finds everything. Neither separates a true change from noise.

So the project answers the question through surrogates. A surrogate is a narrower question that answers one part. No surrogate answers all of it. Together they cover the state of a learner, the reach of a change, what a change does to later learning, the process of solving, and the student.

## 2. What counts as a surrogate

A narrower question counts if it meets four conditions.
1. It names the part of "what is learning" it answers, and the parts it leaves open.
2. Its answer would differ if learning were one thing rather than another.
3. Its reading is calibrated before use, on a case with no change and a case with a planted change of known size. The calibration gives the false positive rate and the power. The planted change comes from a learner whose truth is known by construction, not from a null invented for the reading. The unit of inference is the learner or the trained seed, not the case. Tests that treat cases as replicates rejected a true null in 0.690 of replicates (theory/E.md) and 0.305 of experiments (theory/R.md).
4. A positive result and a negative result each say something.

Condition 3 is the one the earlier work kept missing (the panel board, S7 and S9, and the author's remark on harsh and loose readings).

## 3. The theory the surrogates stand on

Three ideas recur across the literature read so far (C:/Users/YuanW/claude-thesis-docs/exploration/signature/goodfire-reading.md, section 7). Each is the theory of one or more surrogates.
1. Holding a capability is not using it. When the capability is weakly relevant, the fine-tuned model exploits a spurious attribute of the fine-tuning data, and Jain et al. 2023 hypothesize a thin wrapper over the capability; with a strongly relevant capability the model works without that attribute (section 5.1). Further training revives a hidden capability fast. Capabilities form before prompting shows them (Park et al. 2024). A small reportable workspace sits beside much larger automatic processing (Gurnee et al. 2026).
2. The data decide what is learned. A data property sets the speed and order of learning (Park et al. 2024). Frequency, complexity and capacity decide what survives interference (Huang et al. 2026). Narrow tasks improve abruptly once the general structure of the data is acquired (Lubana et al. 2024).
3. A mechanism is its set of invariances. Two models with different mechanisms are not linearly connected (Lubana et al. 2023). Memorized content and shared computation sit in different parts of the loss curvature (Merullo et al. 2025).

The panel's results that stand (C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md).
- "Local or beyond" splits into reach, change of features, and generality, and the three come apart (B23).
- A shortcut and a rule that agree on every practiced problem cannot be told apart by any test of the same kind (C15).
- On a model's weights, the response to one conflict case, divided by its delivery scalar, is one column of the learner's similarity. Read over several cases and at least 8 learners per arm, it shows a rule held and unused (7A2, B25, B32; theory/E.md, E2).
- On a student, the problems that identify a rule also teach it, so "before" is measured once, between students (C21). Delay separates use in the moment from lasting change (C30, R8A3).

Some of these sources were read at the abstract only. A theory file opens the full text before it relies on a detail, and sources/reading.md gives each rung.

## 4. The surrogates, each with tiers

Each surrogate has three tiers. Tier 1 is doable now with what exists or with a small build. Tier 2 is doable if tier 1 gives a usable result. Tier 3 is ambitious and depends on the results below it. A tier is entered only after the reading of the tier below passes its calibration. A failed calibration is itself a result and is written down. The line "Calibration" names what the theory file calibrated, on synthetic learners only, and what it did not.

### G. Guided learning
- Part it answers. Whether a learner's measured state tells which experience will change it most.
- Theory. Data signal and interference (section 3, idea 2).
- Tier 1. On a model, items drawn within a pass rate band with weight proportional to the squared population discrimination, against 19 weightings permuted within the band, by a permutation test at 0.05, at equal training cost. With equal discriminations, choice by information and a band symmetric about 0.5 pick the same items (theory/G.md, R2). The peer's current Q1 arms compare critics, which change the spread of the step and not its expectation, so they are not this tier (R5).
- Tier 2. The learner's own logit change per item over a window predicts which items move next. One learner's answers identify it only on a fixed item set read at reliability 0.8 or more, about 79 answers per item per checkpoint at a spread of 0.5 logit (R9). Turnover is claimed only above that noise floor.
- Tier 3. On students, the delayed gain on unpracticed siblings by the predicted pass rate of the practice item. It peaks at 0.5 for a learner that moves its location, and away from 0.5, on the side of the siblings' weighted mean logit, for a learner that sharpens its precision (R4). It needs practice difficulty assigned apart from the student's state.
- Never answers. What was acquired.
- Calibration. The tier 1 permutation test has false positive rate 0.040 to 0.060 and power 0.665 to 0.795 with 1 seed per weighting, 0.965 to 0.995 with 3, at a planted gain of 0.018 in bank accuracy. Not calibrated, the shape of the gain, the tier 2 predictor, tier 3, and any learner whose features move.

### M. Measurement under learning
- Part it answers. What learning changes and what it leaves fixed, read as whether a calibration made before a change still reads the learner after.
- Theory. A change that is a strictly increasing function of each item's base log odds keeps the order of items, and one relinked function repairs it. A change that depends on more than the base log odds can reorder items, and does so when it crosses at least one pair. A decoding temperature on multi option items reorders with no change of use or capability; a change of use reorders when the fallback differs by item; a capability gain that tracks the base log odds keeps the order (theory/M.md, P3 to P5). As an idea, a wrapper over a weakly relevant capability may reorder a mixed bank by task.
- Tier 1. On trained states with known routes, whether the change from the base calibration is an increasing map, a move along the population axis, or an item specific residual. The reorder size sigma_hat decides against a stated threshold, with a cooled base and four option orders as controls. The peer's Q2 and the population reading of the copied run hold material for it; the author's open call applies here too.
- Tier 2. A linking rule with few anchor items, and the condition under which it suffices.
- Tier 3. On a student log, item drift across cohorts or position in the curriculum, which is the thesis criterion of scale stability. A common shift is not identified apart from cohort ability, so stability is tested as the absence of a reorder under a stated anchor convention.
- Never answers. Why the change happened.
- Calibration. On 1,500 items at 32 answers each, sigma_hat above 0.22 logit has false positive rate at most about 0.05 in five of six order preserving cells and power at least about 0.95 at an item specific SD of 0.25 logit; the split half order test has false positive rate 0.085. The order test of an end state exceeds its level (0.065 to 0.140) and decides nothing. Not calibrated, the tier 2 deviance test and the tier 3 drift test.

### R. Reach, shortcut against rule
- Part it answers. Whether a change follows the surface of the practiced problems or their structure.
- Theory. A mechanism as its invariances (idea 3); the data decide (idea 2).
- Tier 1. On a copied synthetic setup with surface and structure crossed, the reading recovers a cue learner and a rule learner whose contents are known, with its false positive rate and power. The reading is the mean over trained seeds of the cue rate and the rule rate (theory/R.md, section 1). The setup is open (section 9).
- Tier 2. The same distinction inside the weights of small trained transformers. A loss barrier between two seeds of one condition refutes the joint use of the reading and linear connectivity; no barrier between conditions refutes nothing. The curvature split needs a cue that is a lookup over several surfaces per structure. Reward and a teacher's answers are compared with no prediction.
- Tier 3. On a pretrained small language model with freshly built problems, the answers on conflict cells against those of the linearized base model trained on the same cells. The linearized model alone gives a step in the rule share at a number of surfaces per structure (R5a), so a threshold does not mark a rule; a departure from it marks moved features.
- Never answers. Whether the rule is what a teacher intended, which needs a declared structure.
- Calibration. The seed level t test over 10 seeds has false positive rate 0.053 and power 0.893 at a planted cue share of 0.05, at a chosen seed spread of 0.3; the K option item test has 0.030 and power 0.791 at 0.1; the rule rate matches. The binary item level test rejects in 0.305 with no mean effect and is not used. Not calibrated, the seed spread of trained networks, the barrier threshold, the curvature ratio and tier 3.

### E. Response to new evidence
- Part it answers. Whether a change alters what later evidence does to the learner, the strongest sense of "beyond".
- Theory. Holding against using (idea 1); the kernel column result and its ratio (panel B25, B32); belief accumulation for the context channel (Bigelow et al. 2025). A linear learner's response does not depend on its state, and a balanced diagonal learner's response is a function of its answers; what the response adds is history stored as imbalance, which weight decay erases (theory/E.md, E3 to E5).
- Tier 1. Reference learners on CPU. A memorizer, a cue user, a rule user, a learner that weights two rules on evidence, a cue user that stores the corrected case as an exception, and a linear beside a two-layer gradient learner. Two histories per adaptive learner with equal present answers, each branched with and without feedback; only a pair equal on every test problem shows what the response adds (E4). Each case is divided by its delivery scalar, over at least 8 learners per arm. The reading recovers the predicted differences and reports the linear pair as indistinguishable.
- Tier 2. The copied setup of Jain et al. 2023. Held against absent capability told apart by revival time beside the one-step response, with a model trained from scratch (their Appendix E.4) as the absent condition. Revival time alone inverts when the held kernel is one to two times the absent one (E10). The context response is open (section 9).
- Tier 3. On a post-trained language model, whether reward training lowers gamma, the per case evidence weight of log o(N) = b + gamma N^(1 - alpha), fitted below saturation on one fixed prompt set. KL regularized reward at its fixed point predicts no change in gamma and alpha and a rise of b by 1/beta. Accuracy is reported beside and selects no prompts.
- Never answers. What happens inside a student.
- Calibration. The permutation test of a difference in rho has false positive rate 0.045 to 0.058 over three nulls and power 0.8 at a true difference of about 0.14 (dense learner) and 0.10 (diagonal learner). Not calibrated, the context delivery, the revival link in a transformer, and the tier 3 fit.

### P. The process of solving
- Part it answers. Whether training changes how a learner solves, or only how often it is right.
- Theory. On MMLU, a recall set, inflections occur at about half the rate per step when an attention probe is confident from the first step (DeepSeek-R1 0.020 against 0.045, GPT-OSS 120B 0.053 against 0.096), and chains are idle. On GPQA-Diamond the step level relation is absent for R1 and reversed for GPT-OSS (Boppana et al. 2026). Sampling noise in prefix analyses and its smoothing (Bigelow et al. 2026). The prediction of ideal reward training that properties inside right answers and inside wrong answers do not change (panel D33), for the class of tier 2.
- Tier 1. On a small model and built multi-step problems, the forced answer after each cut of the chain up to the cut before the key, with a forced answer from no chain, the normal completion, and the point where the answer first appears. The noise model applies to the sampled outcomes per prefix, not to the forced answer. A filler prefix of equal length checks equal drift on both kinds of problem. Calibrated on problems where the chain is needed against problems where it is not; need is known by construction for the data and measured for a trained network.
- Tier 2. The same before and after training, with properties inside right answers and inside wrong answers against the prediction of ideal reward training, and a changed intermediate result as a test. The prediction holds for outcome-only reward with a sequence-level KL penalty and no length weight. GRPO as published is outside it and predicts more repairs and shorter runs inside right answers (theory/P.md, D5).
- Tier 3. A probe of belief along the chain on small models, and DeepSeek through opencode as a check on a large model if the route allows the same cut.
- Never answers. Whether the written chain is the whole computation.
- Calibration. The forced answer statistic has false positive rate 0.051 and power 0.78 at a rise of 0.06 in key probability (100 items per kind), and fails under unequal drift; the pooled window fork test has 0.022 false forks per chain and power 0.94 at a jump of 0.2; the tier 2 per item test has 0.052 and power 0.97 at a change of 0.050. Not calibrated, the early decision statistic, forks near cuts, the follow rate, the tier 3 shift rule, the statistic on first appearance, and the seed rule of tier 2.

### S. Student prediction
- Part it answers. Learning in students, on behavior only.
- Theory. Held and unused as the hint effect (idea 1); the design conditions of a log (panel C12, C17, C21, C30). The release keeps the latest record per student and question, so reuse is absent (Wang et al. 2020, section 2.1).
- Tier 1. An audit of the Eedi log. Wrong options linked across questions by content, read by a pooled score test with the level of the linked skill; pairs where surface and structure cross; the design counts (retake candidates, which bound retakes from neither side, first encounters per window, orders of practice, pace).
- Tier 2. If the audit passes, whether a tracked misconception state predicts a student's response to later practice beyond the filtered level, as the interaction with practice in log odds per attempt, on the later part of each test student's log, with 5-fold cross-fitting by student and a class term. The raw score leaks (theory/S.md, Proposition 6), and the AUC gain on test students decides nothing. Practice must not be set by a mastery rule.
- Tier 3. The delay channel, a sibling in the same quiz against one in a later quiz. Eedi records no hint, so the held and unused proportion is not identified there.
- Never answers. The inner process of a student. Whether a student holds a rule, without a hint or overlapping triples of linked items answered wrong.
- Calibration. The pooled score test has false positive rate 0.060 at 1 to 50 pairs and power 0.345 over 20 pairs at a planted effect of 0.7 log odds; the tier 2 interaction has 0.070 and power 0.630 at 0.04 log odds per attempt. The stratified MH test rises with the pairs pooled (0.060 to 0.210, simulated) and the naive gain rejects in 0.215 to 0.440 with no effect; neither is used. Not calibrated, the class term, rule against graded weight, and the delay reading.

## 5. Shared ground

Decided by the author (2026-10-19T10:52Z): the shared ground copies the setup of Jain et al. 2023 first, and the setup of Huang et al. 2026 second. Nothing is built before the author's go.

Surrogates R and E share their tier 2 ground, the setup of Jain et al. 2023, copied from a published one before any generator is invented (the author's warning against self invention). R's tier 1 joins it only if the author picks that (section 9). G would copy Huang et al. 2026 Section 3 before its model tier. Surrogate P can share its built problems. Surrogate S stands apart. G and M touch the peer's work.

## 6. How the surrogates feed each other

| result | feeds |
|---|---|
| E tier 1 passes | E tier 2 uses its reading; R tier 1 shares it only if it applies on R's crossed grid |
| R tier 1 shows the reading separates cue from rule | R tier 2, and P tier 2 uses the same problems |
| P tier 1 shows chains do work on built problems | P tier 2 |
| S tier 1 finds linked options and crossed pairs | S tier 2, and E tier 3 gets a student counterpart |
| M tier 1 shows calibrations fail by an increasing map | G tier 2 can track the item side online |

## 7. Papers

A paper can carry one surrogate or several. The mapping is decided later, from results, not now.

## 8. What stays out

The change of features on students, since no record holds it. Dating the arrival of a rule in a student from answers alone. A claim that a student and a model learn by the same inner process. Planning inside a chain on recall items.

## 9. Open for the author

Each pick is the theory file's. Options and costs are in section 8 of each file.
- Whether the peer's Q1 and Q2 serve as tiers of G and M, or as a separate small project. Q1's critic arms test the spread of the step, not item choice (theory/G.md, R5). G picks its own study after tier 1a; M picks the peer's states first.
- Whether the shared ground copies Jain et al. 2023 or Huang et al. 2026.
- R tier 1 on Hermann et al. 2023, then an extension to K options, or on Jain et al. 2023, which merges R's tiers 1 and 2. R picks Hermann.
- A G tier 1a copying Huang et al. 2026 Section 3, on a learner whose features move. G picks yes.
- E tier 1 pairs equal on practiced problems, on every test problem, or both. E picks both.
- E tier 2 with no context delivery, or with in-context examples to check that a 3 million parameter model uses context. E picks none.
- P tier 1 on a transformer trained on the Bayes net task of Prystawski et al. 2023 first, then a small instruction-tuned model. P picks this order.
- S tier 3 hint dropped from Eedi, or a search for a log with hint records, a scope change. S picks dropping it, with the search as a proposal.

## 10. Change log

| date | change | by |
|---|---|---|
| 19 October 2026 | version 1 | the lead |
| 19 October 2026 | version 2. Four queued corrections (G tier 3, M theory, P theory, idea 1); the theory files' section 8 edits that correct an error; a calibration line per surrogate; unit of inference in condition 3; sections 5, 6 and 9 | builder of unit V2 |
| 2026-10-19T10:52Z | Section 5: the author decided the setup order, Jain et al. 2023 first, Huang et al. 2026 second. |
