# SweepA Surrogates used by others for two probes

## Plan
- Target: C:/Users/YuanW/PLMS/surrogates/sources/surrogates-A.md
- Goal: List the surrogates that published work uses for the exact target of each probe below.
- Establishes: Which measurable stand-ins exist for each probe, with their sources and failure modes.
- Cannot establish: Which surrogate is right for this project.
- Inputs: the two probes below; C:/Users/YuanW/PLMS/surrogates/sources/reading.md.
- Artifacts: sources/surrogates-A.md; new rows in sources/reading.md.
- Check: each probe has at least three surrogates, each with an opened primary source.
- Open choices: none
- Tier: sonnet, medium
- Budget: 40 turns
- Attempt: 0
- If pass: the lead writes version 3 of the research program.
- If fail: the probes with fewer than three sourced surrogates are named.
- Resolution: none

### Probes
1. Transfer: practice on one task moves untouched tasks beyond seed noise
2. Order: the same practice in a different order gives a different end state

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
Exists: three sourced surrogates per probe, each with an opened full text. Missing: calibration of any surrogate on the exact targets, and a source that compares two designed orders against a seed null.

- Artifacts: sources/surrogates-A.md (28 lines); six new rows at the end of sources/reading.md.
- Requirements, not met first:
  - Not met: none.
  - Met: at least three surrogates per probe (Transfer: inter-task affinity, TracInCP, Task2Vec; Order: linear interpolation instability, data-order seed variance, backward transfer).
  - Met: each primary source opened in full text, section or equation noted. The PDFs were converted with pdftotext because WebFetch could not read PDF streams.
  - Met: reading.md rows with rung; no abstract-only source; plain English; under 150 lines.
  - Met: no learner-state, learner-models or docs/conversation folder opened; no cluster job; no commit.
- What was tried: WebFetch on six arXiv PDFs failed; the saved PDFs were converted locally to text and read for definitions, assumptions and stated limits. Cost columns are my counts from the definitions. Cells marked "derived" are my reading, not the source's claim. Frankle section 2 was read; experiments after it were not.
- Disagreement: none.
- Lesson: none.
- Effect: 6 sourced surrogates of 6 required (3 per probe, 2 probes), ratio 1.0

## Audit
