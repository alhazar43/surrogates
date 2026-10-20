# RevR Thorough read of theory/R.md

## Plan
- Target: Find every error in theory/R.md before it informs the research program, as the author asked for every finished document.
- Goal: Write theory/R-review.md, a checked list of every derivation, number, citation and script of theory/R.md.
- Establishes: Which statements of theory/R.md hold, which are contradicted, which are narrowed, and which are unchecked.
- Cannot establish: Whether the surrogate is worth pursuing.
- Inputs: C:/Users/YuanW/PLMS/surrogates/theory/R.md; the scripts and result files it names; C:/Users/YuanW/PLMS/surrogates/sources/reading.md; C:/Users/YuanW/PLMS/surrogates/program/research-program.md; the unit file C:/Users/YuanW/PLMS/surrogates/notes/units/TR.md, its Plan and Result only.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/R-review.md; the Audit table of notes/units/TR.md.
- Check: every derivation, number, citation and script of theory/R.md has a row with an outcome and evidence.
- Open choices: none
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 0
- If pass: the builder of TR gets the critical and major rows for one fix attempt.
- If fail: the unchecked rows are named.
- Resolution: none

### Specification
Read theory/R.md line by line. Do not edit it. For each statement that carries weight, write one row.
1. Derivations. Re-derive each one yourself. If it fails, give the corrected statement and where the error enters.
2. Numbers. Recompute each number. Run every named script with python -I and compare its output with the file. Scripts run on tiny synthetic records only.
3. Citations. Open the primary source of each citation. Confirm the claim is in it, and that title, authors, venue and year match one record. A claim not found is contradicted.
4. Rungs. Flag any claim stated above its rung.
5. Consistency. Flag contradictions inside the file, with the research program, and with the panel synthesis it cites.
6. Gaps. Flag an assumption that is used but not stated, and a prediction with no refutation condition.
The review file starts with a list of critical errors, those that change a prediction, a calibration number or a claim of a source. Then major, then minor. Then the full table: item, line, outcome (holds, contradicted, narrowed, unchecked), evidence, severity. Copy the table into the Audit section of notes/units/TR.md. Plain English. Do not open C:/Users/YuanW/PLMS/learner-state, learner-models or any docs/conversation folder. No commits.

## Result
theory/R-review.md exists with 72 rows covering every derivation, number, citation and script of theory/R.md. Missing is a check of the run time of calibrate.py (1 row unchecked).
Effect: 5 critical errors of 72 rows checked, ratio 0.069

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/R-review.md (critical, major and minor lists, then the full table)
- C:/Users/YuanW/PLMS/surrogates/notes/units/TR.md, Audit section, the same 72 rows in the three column audit form
- Outcomes: 35 holds, 29 narrowed, 7 contradicted, 1 unchecked. Severity: 5 critical, 8 major, 27 minor.

Requirements
- Met. theory/R.md not edited. No commits. Nothing opened under learner-state, learner-models or docs/conversation.
- Met. calibrate.py rerun with python -I, output identical to calibrate_output.txt. Extra checks ran on tiny synthetic records in a scratch folder outside the project. The scratch folder in notes/units was deleted.
- Met. All nine cited papers opened in full text (arXiv PDF through pdftotext), each cited item searched in the text.
- Met. The panel sections the file cites (B19, C15, C16, B23, B26, B32, S6, S7, 7A2) were read for consistency.

What was tried
- Critical 1 and 2 (rows 37, 59). R5 gives a ratio of squared loss scores. Argmax answers switch in one step at m*. Cross entropy on fixed features gives a sharper curve that steepens with training. So the tier 3 prediction and the claim "a sharper rise marks moved features" fail.
- Critical 3 (row 56). The tier 2 refutation reverses Conjecture 1 of Lubana et al. 2023.
- Critical 4 (row 57). Reward equal to keys is cited to panel B26 point 3, which says outcome signals push toward the stronger feature.
- Critical 5 (row 65). 20 seeds give power 0.50 at lam 0.02, not detection.
- A rerun of Part A on the true probe grid of Hermann et al. (plus and minus 3 mu) gave the same rates within Monte Carlo error. That row stays minor.

Disagreement: none.

Lesson: none.

## Audit
| item | outcome | evidence |
|---|---|---|
