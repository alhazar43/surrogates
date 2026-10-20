# TS The theory end of surrogate S, Student prediction

## Plan
- Target: Give surrogate S of C:/Users/YuanW/PLMS/surrogates/program/research-program.md section 4 its theory, so each tier has a defined object, a derived prediction and a calibration with known truth.
- Goal: Write theory/S.md, the theory end of surrogate S.
- Establishes: The object, its assumptions, what can be derived, the calibration of its tier 1 reading, and a prediction per tier.
- Cannot establish: Any empirical result. Nothing is trained or fitted on real data.
- Inputs: C:/Users/YuanW/PLMS/surrogates/program/research-program.md (all); C:/Users/YuanW/claude-thesis-docs/exploration/signature/goodfire-reading.md section 7; C:/Users/YuanW/claude-thesis-docs/exploration/signature/panel-2026-10-19.md (the synthesis sections S1 to S15 first, member sections as needed); C:/Users/YuanW/claude-thesis-docs/exploration/signature/three-questions.md Part E.
- Checklist groups: none
- Artifacts: C:/Users/YuanW/PLMS/surrogates/theory/S.md; new rows in C:/Users/YuanW/PLMS/surrogates/sources/reading.md; any tiny synthetic script under C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_s/.
- Check: theory/S.md holds the eight sections of the Specification, each derivation marks its rung, and every cited source has a row in sources/reading.md.
- Open choices: the formal object, the derivations and the calibration design are the builder's; the choices that change scope go to section 8 as options for the author.
- Tier: opus, high
- Budget: 60 turns, 3 hours
- Attempt: 1 (fix after RevS, theory/S-review.md)
- If pass: the lead reads section 8 and folds accepted edits into the research program.
- If fail: the missing sections are named and one attempt fixes them.
- Resolution: none

### Specification
The surrogate answers one part of "what is learning": learning in students, on behavior only. The theory file has these sections.
1. The object. Define it formally, as an equation or a signature, before any prose. Say which part of the question it answers and which parts it leaves.
2. The assumptions. List each so that it can fail, and say what fails with it.
3. What can be derived. Each result with its conditions and its rung (idea, derivation, checked numerically, source). Include the cases where two explanations cannot be told apart, and the conditions under which they can.
4. The calibration of the tier 1 reading. A case with no effect and a case with a planted effect of known size, on a learner whose truth is known by construction. Say how the false positive rate and the power are computed. If it is cheap, compute them on a tiny synthetic case and give the numbers with the script path.
5. One prediction per tier, written before any data. Say what refutes it and what each outcome changes in the tiers above.
6. The literature. Open the primary text of every source you rely on; for a detail, open the full text, not the abstract. Add one row per source to sources/reading.md with what it gives and its rung. Prefer setups that can be copied over inventing new ones, and name the setup to copy.
7. What the theory cannot show.
8. Choices for the author and proposed edits to the research program. Each choice gives options, mechanism, cost, what each shows, and your pick.
Rules. Plain English, short sentences, technical terms as their field defines them, no jargon of your own. The object comes first, then one sentence of reading. Do not edit the research program. Run no cluster jobs and no heavy computation; a local script runs on tiny synthetic records only, with python -I. Do not open C:/Users/YuanW/PLMS/learner-state, C:/Users/YuanW/PLMS/learner-models or any docs/conversation folder. No commits. At most 500 lines.

## Result
theory/S.md exists with the eight sections and a synthetic calibration of the tier 1 and tier 2 readings. It holds no Eedi number, and the tier 1 reading psi is calibrated only with a level from about 250 answers.
Effect: 8 of 8 sections written, ratio 1.0

### Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/S.md (279 lines).
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_s/calibrate.py; outputs calibrate_output.txt (seed 20261019) and calibrate_output_extra.txt (seed 20261020) beside it.
- Four rows appended to C:/Users/YuanW/PLMS/surrogates/sources/reading.md (Wang et al. 2020, Ghosh et al. 2021, Xiong et al. 2016, Jain et al. 2023, each opened in full).

### Requirements
- Not met, in part. Every reading calibrated before use. psi has a false positive rate of 0.154 (1000 students) and 0.324 (3000 students) with a 20 item rest score under level-dependent options. With 250 rest items it is 0.070, power 0.197 at gamma 0.7. The design numbers D1 to D8 are not calibrated.
- Not met, in part. Tiny records only. The script used about 25 minutes of one PC core on synthetic records. It used no real data and no cluster.
- Met. Eight sections, each derivation with its rung (theory/S.md sections 1 to 8).
- Met. Every cited source opened in full, with a row in reading.md. Gick and Holyoak 1980 returned HTTP 403, so the file does not cite it.
- Met. learner-state, learner-models, docs/conversation and the Eedi data were not opened. The research program was not edited. No commits.

### What was tried
- Wang et al. 2020 in full. The release keeps the latest record per student and question. Reuse is therefore absent, and a first encounter is not certified (Proposition 3).
- Derived the floor of the filter: posterior SD 0.442 logit, window 20 attempts at v 0.01 and I 0.25, checked numerically.
- Calibrated psi (Cochran-Mantel-Haenszel test on content-linked options) under no effect, a hard null and planted gamma.
- Calibrated tier 2. The naive gain test has a false positive rate of 0.215 at 2000 students and 0.440 at 4918. The nested interaction test has 0.050 and 0.060, with power 0.305 and 0.610 at eta_1 -0.04. The AUC gain on test students does not separate the effect from no effect (within 0.0004).
- Hint effect as held and unused: derived as a lower bound on the held and unused proportion. It is not identified on Eedi, since Eedi has no hint field.

### Disagreement
none

### Lesson
none

### Attempt 1
theory/S.md now fixes all 4 critical and 10 major review items, and the 7 minor ones. The class term of tier 2, the triple comparison of T1c and the tier 3 reading are not calibrated.
Effect: 14 of 14 critical and major items fixed, ratio 1.0

#### Artifacts
- C:/Users/YuanW/PLMS/surrogates/theory/S.md (380 lines), with "Fix log of attempt 1" at the end.
- C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_s/calibrate_fix.py (Parts F2, P, P2, L, L2, Q, T, D); outputs calibrate_fix_<part>.txt beside it. calibrate.py and its outputs are unchanged.
- No new source, so no new row in sources/reading.md. The Jain et al., Wang et al. and Ghosh et al. details were checked in the full texts of revs_scratch/dl/.

#### Requirements
- Not met, in part. Every reading calibrated before use. Calibrated: T_MLS (0.060 at 1, 20, 50 pairs; power 0.530 at 20 pairs), tier 2 on z_post with 5-fold cross-fitting (0.070; power 0.630), the T1a rule (binomial). Not calibrated: the class term of Proposition 7c, the triple comparison of T1c, delta_delay of tier 3.
- Not met, in part. Tiny records only. Synthetic records only, no Eedi data, no cluster. The scripts used about 52 minutes of one PC core, from file times.
- Met. File at most 500 lines (380).
- Met. learner-state, learner-models and docs/conversation were not opened. The research program and the review were not edited. No commits.

#### What was tried
- Pooled MH under the hard null. False positive rate 0.072 at 1 pair, 0.060 and 0.095 at 20, 0.160 to 0.210 at 50; approximation 0.203 at 100 and 0.357 at 200. The finite-sample pooled mean (0.034) is twice the population residue (0.0165); the gap is not traced.
- T_MLS, a marginal-likelihood score test with a Rasch level integrated per student. Calibrated at rest 50 and 250, delta 0, 1, 2. It fails when the linked items measure a second level (0.990 at rho 0.8, 0.350 at rho 0.95), and is calibrated again with a level from the linked skill (0.060, power 0.345).
- Tier 2. Map from c4 to eta_1 is 0.938 on z_post against 1.186 on z_hat. Power at eta_1 -0.04 is 0.630 with 5-fold cross-fitting, 0.153 with one split, 0.370 with 4 linked items.
- Smoother floor 0.316 logit (Part F2). Association against pi rises and falls (Part Q).

#### Disagreement
none. One note on item 12: the review's population residue 0.046 cuts strata on all students; this fix cuts them on students wrong on both (0.0165). The conclusion of the item holds.

#### Lesson
none

## Audit
Copied from C:/Users/YuanW/PLMS/surrogates/theory/S-review.md (unit RevS). Lines refer to theory/S.md; numbered items refer to the error lists of the review.

| item | line | outcome | evidence | severity |
|---|---|---|---|---|
| 4,918 students in the train file of tasks 3 and 4 | 13 | holds | Wang et al. 2020 sec 2.1, revs_scratch/dl/w.txt:169 | none |
| 948 questions, each with an image | 14 | holds | Wang sec 1.3 file list and sec 2.2, w.txt:122,174 | none |
| AnswerValue and CorrectAnswer in 1 to 4 | 15 | holds | Wang sec 2.1, w.txt:162-163 | none |
| QuizId, SchemeOfWorkId, GroupId fields | 17 | holds | Wang sec 2.4, w.txt:188-190; signature CLAUDE.md, Data on the cluster | none |
| DateAnswered used only to order records | 18, 21 | narrowed | Wang sec 2.4 "to the nearest minute", w.txt:186; ties within a minute | major (7) |
| Confidence 0 a guess, 100 total confidence | 19 | holds | Wang sec 2.4, w.txt:187 | none |
| SubjectIds as nodes of a tree | 20 | holds | Wang sec 2.2, w.txt:173 | none |
| latest record kept per (u, q), quoted | 24 | holds | Wang sec 2.1, w.txt:164 | none |
| tasks 3 and 4 split 80/10/10 by UserId, quoted | 24 | holds | Wang sec 2.1, w.txt:165 | none |
| cluster holds the train part only | 24 | holds | C:/Users/YuanW/PLMS/signature/CLAUDE.md, Data on the cluster | none |
| wrong options chosen for a misconception, not labelled or linked, quoted | 34 | holds | Wang sec 1.1, w.txt:92,98 | none |
| scalar theta with practice "on the skill" | 41, 45 | narrowed | definition gap, item 20 | minor (20) |
| A3 allows dependence on class s | 78 | holds as stated; consequence gap | item 8 | major (8) |
| A6 checkable through D1 | 81 | contradicted | D1 is not a bound, item 6 | major (6) |
| Proposition 1, counting | 89 | holds | panel C4 part 2 | none |
| Proposition 2 fixed point formula and P* 0.1951, SD 0.442, window 20 | 92-98 | holds | rerun Part F identical, run/rerun_main.txt; own algebra I P^2 + I v P - v = 0 | none |
| Proposition 2, no tracker beats the floor | 98 | narrowed | smoother SD 0.316 vs 0.442, revs_check.py part F2 | major (5) |
| Proposition 3a, reuse absent | 102 | holds | Wang sec 2.1, w.txt:164 | none |
| Proposition 3b, bias of "before" | 103 | holds | derivation re-done, mixture of two rates | none |
| Proposition 3c, D1 bounds r from below | 104 | contradicted | interleaving and reassigned questions, item 6 | major (6) |
| Proposition 4a, nominal level with delta 0 | 108 | holds | rerun Part A, 0.058 and 0.054 | none |
| Proposition 4b, residue grows with students | 109 | holds | rerun Part A 0.154 to 0.324; part H residue 0.223 at rest 20 | none |
| Proposition 4c, link chosen by co-choice is circular | 110 | holds | derivation; panel C15 | none |
| Proposition 4d, association grows with pi | 111 | contradicted | revs_pi.py, 0.211 to 0.456 to 0.158 | minor (15) |
| Proposition 5, J >= 4 per student needed | 114 | narrowed | J = 2 not identified; incomplete design over-identified, item 9 | major (9) |
| Proposition 6, cov(G, S) = -s_e^2 and nonzero coefficient of X | 117 | holds | derivation re-done, E[G given S, X] = const - S + E[theta given S, X] | none |
| Proposition 6, rate tends to 1 | 117 | narrowed | two points 0.215, 0.440 only | minor (17) |
| Proposition 7, c4 = 0 under A3 | 126 | narrowed | holds for a collapsible link; 7b states the opposite | minor (16) |
| Proposition 7a, mastery rule; Xiong quote "until they can correctly answer n right in a row" | 127 | holds | Xiong et al. 2016 sec 4.1, dl/x.txt:142 | none |
| Proposition 7b, noncollapsibility | 128 | holds | derivation; learner does not stress it | minor (16) |
| Proposition 7, no class term | 123 | gap | item 8 | major (8) |
| Proposition 8, two bounds | 134-138 | holds | derivation re-done under P(U = 1 given H = 0) = 0 and h independent of H | minor (18) |
| Proposition 8, no hint field on Eedi | 138 | holds | Wang sec 2.4 field list, w.txt:184-190 | none |
| Jain et al. revival 0.1K to 1K against 4.5K, sec 5.1, Fig 9 | 138 | holds | dl/j.txt:130,356 | none |
| Jain et al. no revival after a large learning rate, Fig 43 | 138 | narrowed | Tracr only, j.txt:1714; PCFG revives, j.txt:4524 | minor (19) |
| D1 retake candidates | 145 | contradicted | item 6, item 7 | major (6) |
| D2 window of 20 attempts from v 0.01, I 0.25 | 146 | holds as a chosen value | Part F; panel C8 says v must be estimated | none |
| D5 option dependence on level | 149 | narrowed | rejection count, not delta; differs from C12 number 5, panel line 1299 | major (13) |
| D1 to D8 calibrated | 141-152 | contradicted | no null or planted case; TS Result | major (13) |
| table row, state against level, "yes if A3 holds (D6)" | 161 | narrowed | item 8 | major (8) |
| Proposition 9, order lock | 167 | holds | panel C12, panel line 1288 | none |
| Proposition 10, first conflict item and delay | 169 | holds | panel C21 and C30 | none |
| Monte Carlo standard errors 0.010, 0.015, 0.013, 0.022 | 175, 202, 216 | holds | sqrt(0.05 x 0.95 / R) = 0.00975, 0.0154, 0.0126, 0.0218 | none |
| calibration rule 0.05 plus two standard errors, 0.075 | 202 | holds | 0.05 + 2 x 0.0126 = 0.0752 | none |
| section 4.1 learner matches the code | 179 | holds | calibrate.py:61-85 | none |
| Part A table, 8 rows | 181-190 | holds | rerun identical | none |
| 281 answers per student | 194 | holds as a mean; gap | 1,382,727 / 4,918 = 281.15; minimum 50, Wang sec 2.1 | major (11) |
| Part A2 table, 6 rows | 196-200 | holds | rerun identical | none |
| psi calibrated on Eedi with about 250 answers | 202 | contradicted | part H residue 0.046; part P 0.133 pooled over 20 pairs | critical (1) |
| remedy is a longer rest score | 202 | narrowed | part H and part S, stratum width | major (12) |
| hard null at delta 1.0 as the null | 179, 192 | narrowed | one point of a family, item 14 | major (14) |
| section 4.2 learner matches the code | 206 | holds | calibrate.py:146-175 | none |
| Part B and B2 tables, 4 rows | 208-214 | holds | rerun identical | none |
| naive gain rate grows, nested test calibrated, dAUC within 0.0004 | 216 | holds | rerun; 0.050 and 0.060 inside 0.05 plus two standard errors | none |
| power 0.305 and 0.610 apply to the Eedi design | 216, 228 | contradicted | test students only, power about 0.174, part C4 | critical (4) |
| T1a reason and refutation | 223 | holds for the D3 part | "D1 above zero" has no refutation | minor (21) |
| T1b decision rule calibrated | 224 | contradicted | line 202 says not covered; 8.1 pick differs | critical (2) |
| T1b Ghosh 51 and 34 percent, sec 5.1 | 224 | holds as a quote | Ghosh et al. 2021 sec 5.1, dl/g.txt:880 | none |
| T1b figure supports the direction | 224 | contradicted | no marginal rate; source reads it as inconsistency | major (10) |
| T1b level from about 281 answers | 224 | narrowed | item 11 | major (11) |
| T1c consequence from Proposition 5 | 225 | narrowed | item 9 | major (9) |
| T2 threshold -0.01 log odds per attempt | 229 | contradicted | c4 unit per z_hat; slope 1.185, part C4 | critical (3) |
| T2 AUC bound 0.005 | 228 | gap | no consequence stated | minor (21) |
| T3 same-quiz sibling after the first conflict item | 233 | gap | order inside a minute, item 7 | major (7) |
| Wang et al. record, title, authors, arXiv 2007.12061v3 | 241 | holds | w.txt:1-6 | none |
| Ghosh et al. record, AIED 2021, arXiv 2104.09043 | 242 | holds | g.txt:1-4; Crossref DOI 10.1007/978-3-030-78292-4_12 | none |
| Ghosh et al. 14 expert errors on BIDMAS, partial agreement | 242 | holds | g.txt:878-880 | none |
| Ghosh et al. CF and KT setups | 242 | holds | g.txt:795 | none |
| Xiong et al. record, EDM 2016 | 243 | holds | x.txt:1-12, 21 | none |
| Xiong et al. 23.6 percent duplicates, scaffolding, repeat per skill tag, sec 4.1 | 243 | holds | x.txt:148-152 | none |
| Xiong et al. item difficulty from one student leaks, sec 4.3 | 243 | holds | x.txt:289 | none |
| Jain et al. record, ICLR 2024, arXiv 2311.12786 | 244 | holds | j.txt header, "Published as a conference paper at ICLR 2024" | none |
| Gick and Holyoak not relied on | 246 | holds | no use in sections 3 to 5 | none |
| 8.1 pick for the level of psi | 268 | contradicted | differs from T1b, item 2 | critical (2) |
| 8.1 split of tier 2 | 270 | holds as a choice | its power consequence, item 4 | critical (4) |
| 8.2 edit 1, release rule | 275 | holds | Wang sec 2.1 | none |
| 8.2 edit 2, add psi false positive rate and power | 276 | narrowed | carries the error of item 1 | critical (1) |
| consistency with research program section 4 S | 274-279 | holds | edits are proposals; program unchanged | none |
