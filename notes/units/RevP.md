# RevP Thorough read of theory/P.md

## Plan
- Target: Find every error in theory/P.md before it informs the research program, as the author asked for every finished document.
- Goal: Write theory/P-review.md, a checked list of every derivation, number, citation and script of theory/P.md.
- Establishes: Which statements of theory/P.md hold, which are contradicted, which are narrowed, and which are unchecked.
- Cannot establish: Whether the surrogate is worth pursuing.
- Inputs: C:/Users/YuanW/PLMS/surrogates/theory/P.md; the scripts and result files it names; C:/Users/YuanW/PLMS/surrogates/sources/reading.md; C:/Users/YuanW/PLMS/surrogates/program/research-program.md; the unit file C:/Users/YuanW/PLMS/surrogates/notes/units/TP.md, its Plan and Result only.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/P-review.md; the Audit table of notes/units/TP.md.
- Check: every derivation, number, citation and script of theory/P.md has a row with an outcome and evidence.
- Open choices: none
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 0
- If pass: the builder of TP gets the critical and major rows for one fix attempt.
- If fail: the unchecked rows are named.
- Resolution: none

### Specification
Read theory/P.md line by line. Do not edit it. For each statement that carries weight, write one row.
1. Derivations. Re-derive each one yourself. If it fails, give the corrected statement and where the error enters.
2. Numbers. Recompute each number. Run every named script with python -I and compare its output with the file. Scripts run on tiny synthetic records only.
3. Citations. Open the primary source of each citation. Confirm the claim is in it, and that title, authors, venue and year match one record. A claim not found is contradicted.
4. Rungs. Flag any claim stated above its rung.
5. Consistency. Flag contradictions inside the file, with the research program, and with the panel synthesis it cites.
6. Gaps. Flag an assumption that is used but not stated, and a prediction with no refutation condition.
The review file starts with a list of critical errors, those that change a prediction, a calibration number or a claim of a source. Then major, then minor. Then the full table: item, line, outcome (holds, contradicted, narrowed, unchecked), evidence, severity. Copy the table into the Audit section of notes/units/TP.md. Plain English. Do not open C:/Users/YuanW/PLMS/learner-state, learner-models or any docs/conversation folder. No commits.

## Result
theory/P-review.md exists with 78 rows and the Audit table of TP.md is filled; no proof of a cited theorem was re-derived, and the GRPO result is a tabular check, not a network run.
Effect: 5 critical errors of 78 rows checked, ratio 0.064

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/P-review.md (critical, major, minor lists, the Boppana answer by level, full table)
- C:/Users/YuanW/PLMS/surrogates/notes/units/TP.md, Audit section (78 rows)
- notes/units/revp_scratch/dl/ (arXiv full texts as .txt), tools/ (h2t.py, checks.py, grpo_fast.py), run/ (rerun.txt)

Requirements
- met: every derivation, number, citation and script of P.md has a row with outcome and evidence; unchecked rows 0.
- met: calibration.py rerun with python -I on synthetic records; output identical to results/theory_p_calibration.txt apart from line endings.
- met: all seven cited sources and DeepSeekMath (GRPO) opened as full text; P.md not edited; learner-state, learner-models and docs/conversation not opened; no commit.

What was tried
- Critical. R27, GRPO (Shao et al. 2024, eq. 3) puts a per-token KL in the loss and a length weight; its fixed point is not the tilt (within-right TVD 0.0585 mean, 0.121 max on 200 trees, against 0.0000 for the tilt). R52, the planted share of part C changes the within-right rate by delta (1 - a_1), so 0.025, 0.051, 0.101, half the label. R55, the Prystawski theorem gives a bias toward uniform; the marginal is empirical and unexplained by their theory. R65, Boppana found within-window co-occurrence on MMLU for both models. R66, Ye Result 4 is about which parameters are needed, not their values.
- Boppana by level. Whole responses, MMLU only: inflections 0.020 against 0.045 per step (R1) and 0.053 against 0.096 (GPT-OSS), High Confidence against the rest; the abstract's "almost exclusively" exceeds this. Single steps: R1 MMLU reconsiderations twice as often after probe shifts; GPQA-Diamond null for R1, reversed for GPT-OSS; sensitive to window and threshold. No backtracking-at-belief-shift result on hard items at either level. The writer's disagreement holds; its replacement line is narrowed.
- Major. 12 rows, among them an empty tier 1 refutation (R58), two uncalibrated decision statistics (R59, R63), and a refutation attributed to the D17 judge that D5 excludes at the fixed point (R34).

Disagreement: none

Lesson: none

## Audit
| item | outcome | evidence |
|---|---|---|
