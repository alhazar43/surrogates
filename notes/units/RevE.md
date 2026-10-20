# RevE Thorough read of theory/E.md

## Plan
- Target: Find every error in theory/E.md before it informs the research program, as the author asked for every finished document.
- Goal: Write theory/E-review.md, a checked list of every derivation, number, citation and script of theory/E.md.
- Establishes: Which statements of theory/E.md hold, which are contradicted, which are narrowed, and which are unchecked.
- Cannot establish: Whether the surrogate is worth pursuing.
- Inputs: C:/Users/YuanW/PLMS/surrogates/theory/E.md; the scripts and result files it names; C:/Users/YuanW/PLMS/surrogates/sources/reading.md; C:/Users/YuanW/PLMS/surrogates/program/research-program.md; the unit file C:/Users/YuanW/PLMS/surrogates/notes/units/TE.md, its Plan and Result only.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/E-review.md; the Audit table of notes/units/TE.md.
- Check: every derivation, number, citation and script of theory/E.md has a row with an outcome and evidence.
- Open choices: none
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 0
- If pass: the builder of TE gets the critical and major rows for one fix attempt.
- If fail: the unchecked rows are named.
- Resolution: none

### Specification
Read theory/E.md line by line. Do not edit it. For each statement that carries weight, write one row.
1. Derivations. Re-derive each one yourself. If it fails, give the corrected statement and where the error enters.
2. Numbers. Recompute each number. Run every named script with python -I and compare its output with the file. Scripts run on tiny synthetic records only.
3. Citations. Open the primary source of each citation. Confirm the claim is in it, and that title, authors, venue and year match one record. A claim not found is contradicted.
4. Rungs. Flag any claim stated above its rung.
5. Consistency. Flag contradictions inside the file, with the research program, and with the panel synthesis it cites.
6. Gaps. Flag an assumption that is used but not stated, and a prediction with no refutation condition.
The review file starts with a list of critical errors, those that change a prediction, a calibration number or a claim of a source. Then major, then minor. Then the full table: item, line, outcome (holds, contradicted, narrowed, unchecked), evidence, severity. Copy the table into the Audit section of notes/units/TE.md. Plain English. Do not open C:/Users/YuanW/PLMS/learner-state, learner-models or any docs/conversation folder. No commits.

## Result
theory/E-review.md exists with 74 checked rows and the Audit table of TE.md; one critical error stands unfixed (E10 reading and section 7 bullet 5), and two rows stay unchecked.
Effect: 1 critical errors of 74 rows checked, ratio 0.014

Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/E-review.md (1 critical, 8 major, 20 minor; full table of 74 rows).
- C:/Users/YuanW/PLMS/surrogates/notes/units/TE.md, Audit section (the same table).
- C:/Users/YuanW/PLMS/surrogates/notes/units/RevE-rerun.txt (rerun of calibrate.py, identical to calibrate_output.txt, 3 min 41 s).
- C:/Users/YuanW/PLMS/surrogates/notes/units/reve_tools/ (checks.py, truediff.py, h2t.py) and reve_src/ (arXiv HTML v2 and abstract pages of both sources).
- Documents read: theory/E.md, TE.md Plan and Result, research-program.md sections 2 to 4, sources/reading.md, panel B19, B25 to B27, B31, B32, C15, C23, C24, D26, D27, 7A3. None of them reviews E.md; the Plan names E-review.md as the artifact.

Requirements
- Met. Every derivation, number, citation and script of theory/E.md has a row with outcome and evidence.
- Met. calibrate.py and the check scripts ran with python -I on synthetic records only.
- Met. Both primary sources opened in full text; title, authors, arXiv number and year match the arXiv record.
- Met. theory/E.md not edited. Nothing opened from learner-state, learner-models or docs/conversation. No commits.

What was tried
- Critical. E10 says revival time follows the one-step response only for a held rule. Gradient flow on the toy gives, at equal initial kernel K, held and absent times that differ by about ln 2/(2t) and both grow as ln(1/K) (K = 0.002: 3.802 against 3.455). The kappa^(-2) law needs kappa^2 much larger than t.
- Kernel column result E1 and the ratio hold to first order. Their reading as -1, 0, +1 needs features of value plus or minus one and K(N, e) = 0. Per case division (E2) holds, and the sign of the pooled bias was derived.
- Distinguishable pairs. L2 against L5 holds only if L5's cue weight takes no gradient. L7 absent against held holds. The calibration numbers belong to the script's dense learner (true difference 0.166 at kappa = 1), not to E4's learner (0.400).
- Sources. Every Bigelow et al. 2025 claim holds. Jain et al. 2023: the revival numbers, the setup and Section 3 hold. Appendix E.2 needs a high prior of b and its own pretraining tokens. Scr.+FT starts from a model that counts another operand, so it is not an absent condition.
- Sources were read from arXiv HTML v2, not the PDF, as in TE.

Disagreement: none.
Lesson: none.

## Audit
| item | outcome | evidence |
|---|---|---|
