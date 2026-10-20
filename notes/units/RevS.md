# RevS Thorough read of theory/S.md

## Plan
- Target: Find every error in theory/S.md before it informs the research program, as the author asked for every finished document.
- Goal: Write theory/S-review.md, a checked list of every derivation, number, citation and script of theory/S.md.
- Establishes: Which statements of theory/S.md hold, which are contradicted, which are narrowed, and which are unchecked.
- Cannot establish: Whether the surrogate is worth pursuing.
- Inputs: C:/Users/YuanW/PLMS/surrogates/theory/S.md; the scripts and result files it names; C:/Users/YuanW/PLMS/surrogates/sources/reading.md; C:/Users/YuanW/PLMS/surrogates/program/research-program.md; the unit file C:/Users/YuanW/PLMS/surrogates/notes/units/TS.md, its Plan and Result only.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/S-review.md; the Audit table of notes/units/TS.md.
- Check: every derivation, number, citation and script of theory/S.md has a row with an outcome and evidence.
- Open choices: none
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 0
- If pass: the builder of TS gets the critical and major rows for one fix attempt.
- If fail: the unchecked rows are named.
- Resolution: none

### Specification
Read theory/S.md line by line. Do not edit it. For each statement that carries weight, write one row.
1. Derivations. Re-derive each one yourself. If it fails, give the corrected statement and where the error enters.
2. Numbers. Recompute each number. Run every named script with python -I and compare its output with the file. Scripts run on tiny synthetic records only.
3. Citations. Open the primary source of each citation. Confirm the claim is in it, and that title, authors, venue and year match one record. A claim not found is contradicted.
4. Rungs. Flag any claim stated above its rung.
5. Consistency. Flag contradictions inside the file, with the research program, and with the panel synthesis it cites.
6. Gaps. Flag an assumption that is used but not stated, and a prediction with no refutation condition.
The review file starts with a list of critical errors, those that change a prediction, a calibration number or a claim of a source. Then major, then minor. Then the full table: item, line, outcome (holds, contradicted, narrowed, unchecked), evidence, severity. Copy the table into the Audit section of notes/units/TS.md. Plain English. Do not open C:/Users/YuanW/PLMS/learner-state, learner-models or any docs/conversation folder. No commits.

## Result
theory/S-review.md exists with 77 checked rows and a copy in the Audit of TS.md; no row is unchecked, and no Eedi number exists to test the hard null size.
Effect: 4 critical errors of 77 rows checked, ratio 0.052

### Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/S-review.md (4 critical, 10 major, 7 minor errors, then the full table).
- The Audit section of C:/Users/YuanW/PLMS/surrogates/notes/units/TS.md (the same table).
- C:/Users/YuanW/PLMS/surrogates/notes/units/revs_scratch/: run/ (reruns of calibrate.py, byte identical to both output files), revs_check.py and revs_pi.py (own checks, synthetic only), dl/ (the four primary texts).

### Requirements
- Met. Every derivation, number, citation and script has a row with an outcome and evidence.
- Met. Scripts ran with python -I on synthetic records only. The two reruns ran detached on the PC for under 25 minutes.
- Met. theory/S.md not edited. Eedi data, learner-state, learner-models and docs/conversation not opened. No commits.

### What was tried
- Reran calibrate.py in both modes. Every number of sections 3.2, 4.1 and 4.2 reproduces exactly.
- Hard null. The failure is real. The population residue is a log OR_MH of 0.223 at 20 rest items and 5 strata, and 0.046 at 250 items and 10 strata. The null is the right kind but one size, delta 1.0.
- Residue split. At 250 items, 50 strata give 0.021. The true theta with 10 strata gives 0.027. Stratum width carries more than half.
- Eedi design. An item has 1,459 answers on average, so one pair has a few hundred students wrong on both. One pair has power 0.087 against a false positive rate of 0.100. Twenty pooled pairs give power 0.700 and a false positive rate of 0.133. A 250 item level does not calibrate psi on Eedi.
- Wang et al. 2020 in full. All quotes and counts hold. S.md omits two facts that matter, the filter of at least 50 answers and DateAnswered to the nearest minute.
- Tier 2. c4 is per unit of z_hat (slope 1.185), so the T2 threshold has the wrong unit. The power on about 984 test students is 0.174.
- Smoother floor SD 0.316 against the filter's 0.442. Proposition 4d is non-monotone in pi.

### Disagreement
none

### Lesson
none

## Audit
| item | outcome | evidence |
|---|---|---|
