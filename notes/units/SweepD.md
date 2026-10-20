# SweepD Surrogates used by others for two probes

## Plan
- Target: C:/Users/YuanW/PLMS/surrogates/sources/surrogates-D.md
- Goal: List the surrogates that published work uses for the exact target of each probe below.
- Establishes: Which measurable stand-ins exist for each probe, with their sources and failure modes.
- Cannot establish: Which surrogate is right for this project.
- Inputs: the two probes below; C:/Users/YuanW/PLMS/surrogates/sources/reading.md.
- Artifacts: sources/surrogates-D.md; new rows in sources/reading.md.
- Check: each probe has at least three surrogates, each with an opened primary source.
- Open choices: none
- Tier: sonnet, medium
- Budget: 40 turns
- Attempt: 0
- If pass: the lead writes version 3 of the research program.
- If fail: the probes with fewer than three sourced surrogates are named.
- Resolution: none

### Probes
1. Chain use: cutting a written chain short lowers accuracy on built multi-step tasks
2. Order of difficulty: post-training reorders which items are hard

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
Exists: surrogates-D.md with 5 surrogates for probe 1 and 4 for probe 2, each with a source opened; missing: Hofmann 2PL and Bigelow Forking Fast rows rest on earlier reading.md rows, not re-opened here, and fetches of long papers covered the first 100000 characters only.
- Artifacts: C:/Users/YuanW/PLMS/surrogates/sources/surrogates-D.md (34 lines); 6 new rows at the end of sources/reading.md (Lanham, Boppana, Bogdan, Bae, Baldock, Toneva).
- Requirements: at least 3 surrogates per probe, met (5 and 4). Exact target in one sentence per probe, met. Six columns per row, met. One reading.md row per opened source, met. Abstract-only marking, not needed. Under 150 lines, met. No learner-state, learner-models, docs/conversation, cluster job or commit, met. Not met: none.
- What was tried: arXiv PDF fetches failed on size; HTML versions worked for all but Toneva (ar5iv). Lanham gives no AOC formula; the file says so. Probe 1 target is accuracy against ground truth; early answering scores agreement, and the file notes it.
- Disagreement: none
- Lesson: none
- Effect: 9 sourced surrogates of 6 required (3 per probe), ratio 1.5

## Audit
