# The research program of the surrogates project

Version 1, 19 October 2026. Written by the lead at the author's word, as an editable statement and not an executable plan: "your surrogate should have a tiered structure, what is doable and what is more ambitious and depending on results", "dont force each surrogate as one paper".

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
3. Its reading is calibrated before use, on a case with no change and a case with a planted change of known size. The calibration gives the false positive rate and the power. The planted change comes from a learner whose truth is known by construction, not from a null invented for the reading.
4. A positive result and a negative result each say something.

Condition 3 is the one the earlier work kept missing (the panel board, S7 and S9, and the author's remark on harsh and loose readings).

## 3. The theory the surrogates stand on

Three ideas recur across the literature read so far (C:/Users/YuanW/claude-thesis-docs/exploration/signature/goodfire-reading.md, section 7). Each is the theory of one or more surrogates.
1. Holding a capability is not using it. Fine-tuning mostly adds a thin wrapper over what the model has, and further training revives a hidden capability fast (Jain et al. 2023). Capabilities form before prompting shows them (Park et al. 2024). A small reportable workspace sits beside much larger automatic processing (Gurnee et al. 2026).
2. The data decide what is learned. A data property sets the speed and order of learning (Park et al. 2024). Frequency, complexity and capacity decide what survives interference (Huang et al. 2026). Narrow tasks improve abruptly once the general structure of the data is acquired (Lubana et al. 2024).
3. A mechanism is its set of invariances. Two models with different mechanisms are not linearly connected (Lubana et al. 2023). Memorized content and shared computation sit in different parts of the loss curvature (Merullo et al. 2025).

The panel's results that stand (C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md).
- "Local or beyond" splits into reach, change of features, and generality, and the three come apart (B23).
- A shortcut and a rule that agree on every practiced problem cannot be told apart by any test of the same kind (C15).
- On a model's weights, the response to one conflict case is one column of the learner's similarity. Read once, it shows a rule held and unused (7A2, B25, B32).
- On a student, the problems that identify a rule also teach it, so "before" is measured once, between students (C21). Delay separates use in the moment from lasting change (C30, R8A3).

These sources were read at the abstract for the arXiv papers. A surrogate's theory file opens the full text before it relies on a detail.

## 4. The surrogates, each with tiers

Each surrogate has three tiers. Tier 1 is doable now with what exists or with a small build. Tier 2 is doable if tier 1 gives a usable result. Tier 3 is ambitious and depends on the results below it. A tier is entered only after the reading of the tier below passes its calibration. A failed calibration is itself a result and is written down.

### G. Guided learning
- Part it answers. Whether a learner's measured state tells which experience will change it most.
- Theory. Data signal and interference (section 3, idea 2).
- Tier 1. On a model, item choice by information at the learner's state against a pass rate band, at equal training cost. The peer's Q1 runs a form of this; whether it counts as this tier is the author's open call.
- Tier 2. The item side, estimated online from the learner's own answers, predicts which items move next.
- Tier 3. On students, practice on high information items raises later learning more. It needs practice schedules that vary for reasons apart from the student's state.
- Never answers. What was acquired.

### M. Measurement under learning
- Part it answers. What learning changes and what it leaves fixed, read as whether a calibration made before a change still reads the learner after.
- Theory. The wrapper (idea 1). A change of use predicts a shift that a small relinking repairs. A change of capability predicts a reordering of items.
- Tier 1. On trained states with known routes, the way a base calibration fails, a shift against a reordering, against planted shifts and planted reorders. The peer's Q2 and the population reading of the copied run hold material for it; the author's open call applies here too.
- Tier 2. A linking rule with few anchor items, and the condition under which it suffices.
- Tier 3. On a student log, item drift across cohorts or position in the curriculum, which is the thesis criterion of scale stability.
- Never answers. Why the change happened.

### R. Reach, shortcut against rule
- Part it answers. Whether a change follows the surface of the practiced problems or their structure.
- Theory. A mechanism as its invariances (idea 3); the data decide (idea 2).
- Tier 1. On a copied synthetic setup with surface and structure crossed, the reading recovers a cue learner and a rule learner whose contents are known, with its false positive rate and power.
- Tier 2. The same distinction inside the weights of small trained transformers, by linear connectivity and by the curvature split, under reward and under a teacher's answers.
- Tier 3. On a pretrained small language model with freshly built problems, and the threshold in the number of surfaces per structure at which the rule appears.
- Never answers. Whether the rule is what a teacher intended, which needs a declared structure.

### E. Response to new evidence
- Part it answers. Whether a change alters what later evidence does to the learner, the strongest sense of "beyond".
- Theory. Holding against using (idea 1); the kernel column result and its ratio (panel B25, B32); belief accumulation for the context channel (Bigelow et al. 2025).
- Tier 1. Reference learners on CPU. A memorizer, a cue user, a rule user, a learner that weights two rules on evidence, a cue user that stores the corrected case as an exception, and a linear beside a two-layer gradient learner. Two histories per adaptive learner with equal present answers, each branched with and without feedback. The reading recovers the predicted differences and reports the linear pair as indistinguishable.
- Tier 2. The copied setup of Jain et al. 2023. Held against absent capability told apart by revival speed, and the context response against the weight response on the same small transformer.
- Tier 3. On a post-trained language model, whether reward training lowers the response to counterevidence at equal accuracy.
- Never answers. What happens inside a student.

### P. The process of solving
- Part it answers. Whether training changes how a learner solves, or only how often it is right.
- Theory. Belief shifts at backtracking steps on hard items, idle chains on recall items (Boppana et al. 2026); sampling noise in prefix analyses and its smoothing (Bigelow et al. 2026); the prediction of ideal reward training that properties inside right answers and inside wrong answers do not change (panel D33).
- Tier 1. On a small model and built multi-step problems, the forced answer after each cut of the chain, with a forced answer from no chain, the normal completion, and the point where the answer first appears. Calibrated on problems where the chain is needed against problems where it is not, known by construction, with the noise model applied.
- Tier 2. The same before and after training, with properties inside right answers and inside wrong answers against the prediction of ideal reward training, and a changed intermediate result as a test.
- Tier 3. A probe of belief along the chain on small models, and DeepSeek through opencode as a check on a large model if the route allows the same cut.
- Never answers. Whether the written chain is the whole computation.

### S. Student prediction
- Part it answers. Learning in students, on behavior only.
- Theory. Held and unused as the hint effect (idea 1); the design conditions of a log (panel C12, C17, C21, C30).
- Tier 1. An audit of the Eedi log. Wrong options linked across questions by content, pairs where surface and structure cross, and the design counts (repeats and lags, first encounters per window, orders of practice, pace).
- Tier 2. If the audit passes, whether a tracked state predicts a student's response to later practice better than the current score, on a held out later part of each log.
- Tier 3. The delay channel, a sibling in the same quiz against one in a later quiz, and a hint analogue if the log carries one.
- Never answers. The inner process of a student.

## 5. Shared ground

Surrogates R and E share their tier 1 and tier 2 ground. That ground is a setup with known truth, copied from a published one before any generator is invented (the author's warning against self invention). The candidate is Jain et al. 2023, with Huang et al. 2026 as the second. Surrogate P can share its built problems. Surrogate S stands apart. G and M touch the peer's work.

## 6. How the surrogates feed each other

| result | feeds |
|---|---|
| E tier 1 passes | E tier 2 and R tier 1 share its reading |
| R tier 1 shows the reading separates cue from rule | R tier 2, and P tier 2 uses the same problems |
| P tier 1 shows chains do work on built problems | P tier 2 |
| S tier 1 finds linked options and crossed pairs | S tier 2, and E tier 3 gets a student counterpart |
| M tier 1 shows calibrations fail by shift | G tier 2 can track the item side online |

## 7. Papers

A paper can carry one surrogate or several. The mapping is decided later, from results, not now.

## 8. What stays out

The change of features on students, since no record holds it. Dating the arrival of a rule in a student from answers alone. A claim that a student and a model learn by the same inner process. Planning inside a chain on recall items.

## 9. Open for the author

- Whether the peer's Q1 and Q2 serve as tiers of G and M, or as a separate small project.
- Whether the shared ground copies Jain et al. 2023 or Huang et al. 2026.

## 10. Change log

| date | change | by |
|---|---|---|
| 19 October 2026 | version 1 | the lead |
