# SweepB Surrogates used by others for two probes

## Plan
- Target: C:/Users/YuanW/PLMS/surrogates/sources/surrogates-B.md
- Goal: List the surrogates that published work uses for the exact target of each probe below.
- Establishes: Which measurable stand-ins exist for each probe, with their sources and failure modes.
- Cannot establish: Which surrogate is right for this project.
- Inputs: the two probes below; C:/Users/YuanW/PLMS/surrogates/sources/reading.md.
- Artifacts: sources/surrogates-B.md; new rows in sources/reading.md.
- Check: each probe has at least three surrogates, each with an opened primary source.
- Open choices: none
- Tier: sonnet, medium
- Budget: 40 turns
- Attempt: 0
- If pass: the lead writes version 3 of the research program.
- If fail: the probes with fewer than three sourced surrogates are named.
- Resolution: none

### Probes
1. Cue or rule: when cue and rule agree in training, the cheaper one wins
2. Gap: a longer gap between repeats of a rare task hurts narrow models more

### Specification
A probe asks a question about learning. A surrogate is a measurable quantity that stands in for the exact target of the probe. It is used because the exact target cannot be computed with the data or the model at hand.
1. For each probe, write its exact target in one sentence.
2. Find at least three surrogates that published work uses for that target. Search the literature. Do not invent one.
3. Open the primary source of each surrogate in full text. Note the section or equation that defines it.
4. For each surrogate, write one row with these columns:
   - name
   - what it computes, as a formula or one line
   - source, as authors, year, title, arXiv id, and section
   - the assumption under which it tracks the target
   - how it misleads when that assumption fails
   - cost for a small model
5. Add one row per opened source to sources/reading.md, with its rung.
6. Mark a source "abstract only" if the full text cannot be opened, and do not use it for a definition.
Plain English. At most 150 lines. Do not open C:/Users/YuanW/PLMS/learner-state, learner-models or any docs/conversation folder. No cluster jobs. No commits.

## Result
Exists: sources/surrogates-B.md with 5 sourced surrogates for probe 1 and 3 for probe 2, each from an opened full text. Missing: no source measures the gap by width interaction of probe 2, and Lovering et al. 2021 has no arXiv id.

Artifacts
- C:/Users/YuanW/PLMS/surrogates/sources/surrogates-B.md (35 lines)
- 6 new rows at the end of sources/reading.md (Lovering, Scimeca, Jagielski, Toneva, Kandpal, Hermann reopened)

Requirements
- Each probe has at least three surrogates with an opened primary source: met.
- Exact target in one sentence per probe: met.
- Six columns per row: met.
- One reading.md row per opened source, with rung: met.
- Hermann et al. text ended inside Appendix E (100k of 166k characters); the earlier row records the full text: met, noted.
- Toneva et al. venue (ICLR 2019) is not in the page text opened: stated in the file.
- No forbidden folder opened, no cluster job, no commit: met.

What was tried
- PDF fetches of arXiv failed (compressed bytes). ar5iv HTML and a text mirror worked.
- A heredoc write failed once; the file was written with the Write tool and reading.md rows were appended with printf.
- Probe 2 searched for a gap by width measure. None found; the three rows cover the two margins.
- Huang et al. 2026 was not reopened and is listed as not used.

Disagreement: none

Lesson: none

Effect: 8 sourced surrogates of 6 required surrogates (3 per probe), ratio 1.33

## Audit
