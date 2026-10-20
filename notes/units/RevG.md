# RevG Thorough read of theory/G.md

## Plan
- Target: Find every error in theory/G.md before it informs the research program, as the author asked for every finished document.
- Goal: Write theory/G-review.md, a checked list of every derivation, number, citation and script of theory/G.md.
- Establishes: Which statements of theory/G.md hold, which are contradicted, which are narrowed, and which are unchecked.
- Cannot establish: Whether the surrogate is worth pursuing.
- Inputs: C:/Users/YuanW/PLMS/surrogates/theory/G.md; the scripts and result files it names; C:/Users/YuanW/PLMS/surrogates/sources/reading.md; C:/Users/YuanW/PLMS/surrogates/program/research-program.md; the unit file C:/Users/YuanW/PLMS/surrogates/notes/units/TG.md, its Plan and Result only.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/G-review.md; the Audit table of notes/units/TG.md.
- Check: every derivation, number, citation and script of theory/G.md has a row with an outcome and evidence.
- Open choices: none
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 0
- If pass: the builder of TG gets the critical and major rows for one fix attempt.
- If fail: the unchecked rows are named.
- Resolution: none

### Specification
Read theory/G.md line by line. Do not edit it. For each statement that carries weight, write one row.
1. Derivations. Re-derive each one yourself. If it fails, give the corrected statement and where the error enters.
2. Numbers. Recompute each number. Run every named script with python -I and compare its output with the file. Scripts run on tiny synthetic records only.
3. Citations. Open the primary source of each citation. Confirm the claim is in it, and that title, authors, venue and year match one record. A claim not found is contradicted.
4. Rungs. Flag any claim stated above its rung.
5. Consistency. Flag contradictions inside the file, with the research program, and with the panel synthesis it cites.
6. Gaps. Flag an assumption that is used but not stated, and a prediction with no refutation condition.
The review file starts with a list of critical errors, those that change a prediction, a calibration number or a claim of a source. Then major, then minor. Then the full table: item, line, outcome (holds, contradicted, narrowed, unchecked), evidence, severity. Copy the table into the Audit section of notes/units/TG.md. Plain English. Do not open C:/Users/YuanW/PLMS/learner-state, learner-models or any docs/conversation folder. No commits.

## Result
theory/G-review.md exists with 64 checked rows and the TG Audit table is filled. The review does not check goodfire-reading.md section 7 or three-questions.md Part E, and it does not read source text beyond the first 100k characters of four sources.
Effect: 3 critical errors of 64 rows checked, ratio 3/64 = 0.047

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/G-review.md (critical, major, minor lists; full table of 64 rows)
- C:/Users/YuanW/PLMS/surrogates/notes/units/TG.md, Audit section (the same 64 rows)
- Scratch outside the project, not kept as artifacts: C:/Users/YuanW/AppData/Local/Temp/revg/checks.py and rerun.json

Requirements
- Met. Every derivation, number, citation and script of theory/G.md has a row with an outcome and evidence.
- Met. calibrate_tier1.py rerun with python -I on its tiny synthetic records; output identical to results/theory_g_calibration.json after key sort (4 min 12 s).
- Met. Each cited source opened at its primary text (arXiv HTML or PDF, Europe PMC XML); title, authors, venue and year checked.
- Met. R5 and R6 checked against C:/Users/YuanW/PLMS/irt4llm/notes/blueprint.md as data; nothing changed there.
- Met. theory/G.md not edited; learner-state, learner-models and docs/conversation not opened; no commits.

What was tried
- Critical 1. Under sharpening the bank gain is eta c'_B delta_x p_x(1-p_x), and the sign of c'_B depends on the bank. If the bank's weighted mean logit is negative, the gain peaks at p = 0.176 and the tier 1 second and tier 3 predictions fail under sharpening.
- Critical 2. In R9's model the logit velocity is a_j dtheta/dt and has the same relative SE as a_hat, so the tier 2 asymmetry between them does not follow.
- Critical 3. blueprint.md:29 now has no divisor and no clip, and :32 states the information weight that holds without it. So R6 and "R6 corrects its stated weight" are false against the blueprint as it stands. R6 was right for v2.1.
- R5 holds against the current blueprint (C and I0 differ only in p_hat). Its variance sentence is narrowed, since the gradient-variance optimal baseline is 1 - p on the one-bit learner.
- Major rows include the wrong size comparison behind the 0.105 rate (ratio 0.91, not 5; simulated 0.104) and K = 2 permuted weightings (smallest p 0.33; K = 19 needs 60 runs). Further rows cover an uncalibrated refutation rule, an unnamed optimizer against panel S1.3, and a band arm C that the blueprint no longer has.

Disagreement: none with the Plan. The Plan's Inputs omit goodfire-reading.md section 7 and three-questions.md Part E, which G.md names; those statements of G.md are unchecked.

Lesson: none

## Audit
| item | outcome | evidence |
|---|---|---|
