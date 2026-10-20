# RevM Thorough read of theory/M.md

## Plan
- Target: Find every error in theory/M.md before it informs the research program, as the author asked for every finished document.
- Goal: Write theory/M-review.md, a checked list of every derivation, number, citation and script of theory/M.md.
- Establishes: Which statements of theory/M.md hold, which are contradicted, which are narrowed, and which are unchecked.
- Cannot establish: Whether the surrogate is worth pursuing.
- Inputs: C:/Users/YuanW/PLMS/surrogates/theory/M.md; the scripts and result files it names; C:/Users/YuanW/PLMS/surrogates/sources/reading.md; C:/Users/YuanW/PLMS/surrogates/program/research-program.md; the unit file C:/Users/YuanW/PLMS/surrogates/notes/units/TM.md, its Plan and Result only.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/M-review.md; the Audit table of notes/units/TM.md.
- Check: every derivation, number, citation and script of theory/M.md has a row with an outcome and evidence.
- Open choices: none
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 0
- If pass: the builder of TM gets the critical and major rows for one fix attempt.
- If fail: the unchecked rows are named.
- Resolution: none

### Specification
Read theory/M.md line by line. Do not edit it. For each statement that carries weight, write one row.
1. Derivations. Re-derive each one yourself. If it fails, give the corrected statement and where the error enters.
2. Numbers. Recompute each number. Run every named script with python -I and compare its output with the file. Scripts run on tiny synthetic records only.
3. Citations. Open the primary source of each citation. Confirm the claim is in it, and that title, authors, venue and year match one record. A claim not found is contradicted.
4. Rungs. Flag any claim stated above its rung.
5. Consistency. Flag contradictions inside the file, with the research program, and with the panel synthesis it cites.
6. Gaps. Flag an assumption that is used but not stated, and a prediction with no refutation condition.
The review file starts with a list of critical errors, those that change a prediction, a calibration number or a claim of a source. Then major, then minor. Then the full table: item, line, outcome (holds, contradicted, narrowed, unchecked), evidence, severity. Copy the table into the Audit section of notes/units/TM.md. Plain English. Do not open C:/Users/YuanW/PLMS/learner-state, learner-models or any docs/conversation folder. No commits.

## Result
theory/M-review.md exists with 72 rows, 3 critical, 12 major and 18 minor errors, and the table is copied into the Audit of notes/units/TM.md. Missing are a check of the Nelder-Mead history (no artifact) and of the four sources M.md names as unopened and not relied on.
Effect: 3 critical errors of 72 rows checked, ratio 0.042

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/M-review.md
- C:/Users/YuanW/PLMS/surrogates/notes/units/TM.md, Audit section
- Scratch, outside the project: C:/Users/YuanW/AppData/Local/Temp/revm/chk/checks.py, chk/power1500.py, rerun.json, src/*.txt (fetched source texts)

Requirements
- Met. theory/M.md not edited; no commits; learner-state, learner-models and docs/conversation not opened.
- Met. Every named script run with python -I on synthetic records (calibrate.py in 4 min 1 s, 159 of 159 values equal to calibration.json; power1500.py in 2 min 18 s at R = 40).
- Met. Every source of section 6 opened in its primary text (DPO PDF, UMN Digital Conservancy texts, Europe PMC XML, arXiv PDFs), title, authors, venue and year checked.

What was tried
- Critical. C1, P4(b) drops the import term of panel B3 Proposition 3, so the SFT reorder prediction has no derivation. C2, the tier 1 prediction is calibrated at J = 200, k = 16; at J = 1,500, k = 32 the test rejects 40 of 40 for an item specific SD of 0.25 logit and for 1 logit along 2PL slopes, so "GRPO keeps the order" needs a tolerance or the item blind assumption. C3, Jain et al. tie the wrapper to the spurious attribute only for a weakly relevant capability.
- Major, among others. P5's iff condition is false both ways (counterexamples computed); 0.963 is not reproduced (0.959); the Q2b lz prediction is already seen for the S1 states, since a common b shift is absorbed into theta.
- The numbers of the peer (stability, lz, cost, 48,000 rows) were read from irt4llm state.md, blueprint.md and personfit.csv, and agree.

Disagreement: none with the Plan. On the Disagreement of TM with the research program, M.md holds in substance at the rung of derivation. Capability does not imply a reorder (P4a, Rasch), and use does not imply a shift (P2b, P3b). Three narrowings. P3a is not a change of use. The Jain support holds only for a weakly relevant capability. Proposed edit 1 must read "can reorder" (M-review.md, M4, M5, C3).

Lesson: none

## Audit
| item | outcome | evidence |
|---|---|---|
