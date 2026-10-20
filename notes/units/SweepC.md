# SweepC Surrogates used by others for two probes

## Plan
- Target: C:/Users/YuanW/PLMS/surrogates/sources/surrogates-C.md
- Goal: List the surrogates that published work uses for the exact target of each probe below.
- Establishes: Which measurable stand-ins exist for each probe, with their sources and failure modes.
- Cannot establish: Which surrogate is right for this project.
- Inputs: the two probes below; C:/Users/YuanW/PLMS/surrogates/sources/reading.md.
- Artifacts: sources/surrogates-C.md; new rows in sources/reading.md.
- Check: each probe has at least three surrogates, each with an opened primary source.
- Open choices: none
- Tier: sonnet, medium
- Budget: 40 turns
- Attempt: 0
- If pass: the lead writes version 3 of the research program.
- If fail: the probes with fewer than three sourced surrogates are named.
- Resolution: none

### Probes
1. Revival: a few steps of retraining bring back a capability that fine-tuning hid
2. One-step response: how a learner reacts to one example predicts its later learning

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
sources/surrogates-C.md exists with five sourced surrogates for Probe 1 and four for Probe 2. Nothing from the Check is missing. Hu, Lo and Koh were read in part (listed in the file).

Artifacts
- C:/Users/YuanW/PLMS/surrogates/sources/surrogates-C.md (80 lines)
- Six new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md (Pruthi, Koh, Xia, Deeb, Hu, Lo)

Requirements
- Met: at least three surrogates per probe, each with an opened full text source (Probe 1 S1 uses Jain, opened earlier).
- Met: a row per opened source in reading.md with its rung.
- Met: under 150 lines, plain English, no learner-state, learner-models or docs/conversation reads, no cluster job, no commit.
- Met: no row is abstract only.

What was tried
- Fetched six arXiv PDFs. The fetch tool returned raw bytes, so I converted the saved PDFs with pdftotext into a temporary folder and read the sections with grep.
- Probe 1 targets revival by retraining. Probe 2 targets one-step response. T3 (influence function) is a convergence quantity, not a one-step one. It is kept and marked as such.
- Not found: a source that tests Probe 2 against later learning directly. All four T rows are first-order proxies.

Disagreement: none

Lesson: none

Effect: 9 sourced surrogates of 6 required surrogates, ratio 1.5

## Audit
