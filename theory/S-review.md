# Review of theory/S.md

Unit RevS, 19 October 2026, checker on Opus. theory/S.md was read line by line and not edited. Line numbers refer to theory/S.md as of this review (279 lines). The Eedi data were not opened. learner-state, learner-models and docs/conversation were not opened.

How it was checked.
- Scripts. calibrate.py was rerun with python -I in both modes. The outputs are in C:/Users/YuanW/PLMS/surrogates/notes/units/revs_scratch/run/ and are byte identical to calibrate_output.txt and calibrate_output_extra.txt (diff, no lines differ).
- Own checks. C:/Users/YuanW/PLMS/surrogates/notes/units/revs_scratch/revs_check.py (parts H, F2, C4, S, P) and revs_pi.py, synthetic records only. Each number below that is not from calibrate.py names its part.
- Sources. Wang et al. 2020 (arXiv 2007.12061v3), Ghosh et al. 2021 (arXiv 2104.09043), Xiong et al. 2016 (EDM 2016 proceedings PDF) and Jain et al. (arXiv 2311.12786) were downloaded to revs_scratch/dl/ and read in full text. The Ghosh et al. venue was checked against the Crossref record (LNCS, Artificial Intelligence in Education 2021, pages 137 to 149, DOI 10.1007/978-3-030-78292-4_12).
- Panel. Sections C4, C8, C12, C15, C17, C21, C25, C30, B23, B32, S4 and S11 to S15 of panel-2026-10-19.md, and the research program section 4.

## Critical errors

A critical error changes a prediction, a calibration number or a claim of a source.

1. The tier 1 calibration at 250 rest items holds only at the number of students it was run at (lines 202, 268, 276). The population residue of the hard null at 250 rest items and 10 strata is a log OR_MH of 0.046, not zero (part H, 400,000 students). A nonzero residue makes the false positive rate rise with the number of students wrong on both items. The 0.070 of line 202 is the rate at about 880 such students for one pair. On Eedi an item has on average 1,382,727 / 948 = 1,459 answers, so one linked pair has at most a few hundred students wrong on both. At 1,000 students (about 294 wrong on both) one pair gives a rejection rate of 0.100 under the hard null and 0.087 under gamma 0.7, so a single pair has no power (part P, 150 logs). Pooling 20 such pairs gives power 0.700 and a false positive rate of 0.133 (part P, 150 logs, standard error 0.018 at p 0.05). The normal approximation from the population residue predicts 0.13 at 20 pairs, 0.26 at 50 and 0.45 at 100. So psi is not calibrated on Eedi by a 250 item level. Any reading with power on Eedi pools pairs, and pooling brings back the false positive rate. The sentence "the false positive rate on Eedi is at least the 250 item figure" also fails as stated, since the rate depends on the pooled count and not on the rest score alone.

2. The decision rule of T1b is uncalibrated, and it differs from the pick of section 8.1 (lines 202, 224, 268). T1b decides by the linked median against the 95th percentile of unlinked pairs. Line 202 says the calibration "does not cover" that comparison. Section 8.1 picks the MH test against zero as the primary reading, with the unlinked comparison beside it. T1b reverses the two. The unlinked comparison also has a bias in a known direction. A linked option is the distractor declared to answer a common misconception, which is the distractor most likely to be chosen by weaker students. If linked options depend on level more than unlinked ones, the linked median exceeds the unlinked distribution with no shared misconception. The project requirement says every reading is calibrated before use. The tier 1 gate rests on the one reading that is not.

3. The refutation threshold of T2 is in the wrong unit (line 229). c4 multiplies z_hat times n, so its unit is log odds per attempt per unit of z_hat, not log odds per attempt for a holder. In the learner of section 4.2 the slope of z on z_hat at fixed theta_hat is 1.185, so eta_1 = -0.04 implies c4 of about -0.047 (part C4), and calibrate.py gives mean c4 of -0.0502 and -0.0529. The factor depends on how reliably z_hat marks z, which on Eedi depends on the number of linked wrong answers per student. So "excludes values below -0.01 log odds per attempt" has no fixed meaning as a difference between holders and non-holders. The threshold must be stated on eta_1, with the map from c4 to eta_1 calibrated, or stated on c4 with its unit.

4. The power of the tier 2 test does not apply to the design of Proposition 7 (lines 216, 228, 270). The power of 0.305 at 2,000 students and 0.610 at 4,918 uses every student of the log in the c4 fit, with 10 linked items each and 5.28 linked wrong answers on average (part C4). Proposition 7 and section 8.1 fit c4 on test students only. With the 20 percent student split of Ghosh et al. inside the train file, that is about 984 students, and the power scales to 0.174 (part C4, from the 0.305 at 2,000, which reproduces 0.623 against the simulated 0.610 at 4,918). T1c predicts that fewer than 10 percent of students answer four or more linked items of one misconception wrong. Then z_hat rests on fewer linked answers than in the calibration, and the power falls further. The figure 0.610 "at 4918 students, the train file's count" therefore overstates the power of the design that S.md proposes.

## Major errors

5. Proposition 2 holds for online trackers only (line 98). The recursion is the posterior Cramér-Rao recursion of filtering, which bounds an estimator that uses answers up to t. A smoother that also uses later answers has a steady-state posterior variance of 0.100 against 0.195, a standard deviation of 0.316 logit against 0.442 (part F2, Rauch-Tung-Striebel steady state at v 0.01, I 0.25). Trajectory recovery from a full log can use the smoother. So "No tracker beats this floor" holds for causal trackers, and the offline floor is 0.316.

6. D1 does not bound the retake proportion from below (lines 104, 145). A record is out of block with its quiz without any retake when a student interleaves two quizzes or resumes a quiz after another one. A question reassigned in a later quiz keeps the later record, which sits in block with the later quiz and is not flagged. So D1 can exceed r and can miss retakes of other kinds. It is neither a lower nor an upper bound.

7. DateAnswered is recorded "to the nearest minute" (Wang et al. 2020, section 2.4), and S.md does not state it (lines 18, 21). The golden rules of section 3.3 of the same source say a question should take less than 10 seconds. Several records of one quiz then share a minute, and the attempt rank t_u(r) is not defined among them. D1 (contiguity), D3 and D4 (bins of attempt rank), the cut tau of Proposition 7 and the order of the first conflict item and its same-quiz sibling in T3 all need an order inside a minute. The log does not give it.

8. Proposition 7 omits the class (lines 78, 123 to 126, 161). A3 allows the next item to depend on the class s as a whole. If the amount of practice n_u depends on the class, and the class correlates with level and with taught misconceptions, n_u is not independent of (theta, z) given the record unless the class enters the model. The model of line 123 has no class term. D6 measures whether quizzes are class wide. It does not decide whether n_u is independent of (theta, z), so "yes if A3 holds (D6)" in the table of line 161 is not established.

9. Proposition 5 counts a complete table of a fixed set of J items (lines 114, 225). At J = 2 the two-class model has 5 parameters against 3 cells, so it is not identified, not saturated. In an incomplete design, where students meet different subsets of a larger set of linked items, the population two-class model has 2K + 1 parameters for K items and each observed triple gives 7 cells, so overlapping triples over-identify it (derivation by counting). Fewer than four linked items per student is then not by itself a barrier. The consequence that T1c draws, that the log cannot tell a rule switch from a graded shift, is narrowed to a complete design. The count remains a necessary condition only, as the file says.

10. The Ghosh et al. figure does not support the direction of T1b (line 224). Among students who chose 69D, 51 percent chose 293C and 34 percent chose the key 293B (Ghosh et al. 2021, section 5.1, quoted correctly). The source gives no marginal rate of 293C, so the figure cannot show that 69D choosers pick 293C more than others. The source uses the figure for the opposite point, "students may not consistently demonstrate an error". The claim is above its rung.

11. The level of T1b is not the level of the calibration (lines 194, 202, 224). The 281 answers per student are a mean. The release removed students with fewer than 50 answers (Wang et al. 2020, section 2.1), so the count ranges from 50 upward. The synthetic rest score is a raw count over items that every student answered. Eedi students answer different items in different quizzes, so a raw count is not a level, and an item response model estimate is needed. Students with fewer answers carry a larger residue than the 250 item figure.

12. The remedy misses stratum width (line 202). At 250 rest items the population residue is 0.046 with 10 strata, 0.021 with 50 strata and 0.021 with each score value as a stratum. With the true theta it is 0.095 with 5 strata, 0.027 with 10 and near zero (-0.003) with 50 (part H). So at 250 items and 10 strata more than half the residue comes from the width of the strata. The false positive rate at 3,000 students is 0.067 with 10 strata and 0.057 with each score value (part S, 300 logs). A finer stratification is a second remedy that costs nothing.

13. The design numbers are not calibrated, and D5 is not the number the hard null needs (lines 149, 152). The project requirement asks for a null case and a planted case for every reading. D1 to D8 have neither (TS Result says so). D5 counts items that reject a chi-square test, which depends on the number of answers per item and not on the size of the level dependence. The false positive rate of psi depends on the slope delta of each linked option on level. D5 also differs from number 5 of panel C12 and S4, which asks whether wrong options are chosen at different rates by students of equal predicted ability.

14. The hard null is one point of a family (lines 179, 192). Every false positive rate of section 4.1 is at delta = 1.0 on both linked options with the same sign. The residue grows with the product of the two slopes. No statement says why 1.0 bounds the Eedi case, and no Eedi number fixes it. The null is the right kind (conditional independence given the true level, with options that depend on level), but its size is a choice, and the conclusion "calibrated" holds only at that size.

## Minor errors

15. Proposition 4d says the association grows with the prevalence pi (line 111). It rises and then falls. At gamma 1.4, delta 0 and the true theta in 50 strata, the population log OR_MH is 0.211, 0.397, 0.456, 0.387 and 0.158 at pi 0.1, 0.3, 0.5, 0.7 and 0.9 (revs_pi.py). The association of a binary mixture scales with the variance of z among students wrong on both.

16. Proposition 7 first states c4 = 0 as a derivation under A3 and then says a spurious c4 appears even with A3 (lines 126, 128). The first statement holds for a collapsible link. The learner of section 4.2 does not stress 7b, since the residual variance of theta is similar across z_hat there, so "below the Monte Carlo error" holds for this learner only.

17. Proposition 6 says the naive false positive rate tends to 1 (line 117). The script shows growth from 0.215 to 0.440. The limit is a derivation from the consistency of a test against a fixed nonzero coefficient, not a numerical check.

18. Proposition 8 needs the hint assigned independently of H, which is not stated (line 134). The added effect (1 - pi_H) tau_h assumes that a student taught by the hint uses the rule with probability 1.

19. The Jain et al. statements are narrower in the source (lines 138, 244). The failure of revival after a large learning rate is in Tracr compiled models with L = 10^-1 (Figures 39 and 43). In the PCFG models revival occurs after fine-tuning at a large learning rate of 10^-4 (Figure 76). The 4.5K iterations of the Scr. + FT baseline are reached only at the larger learning rate M (Figure 9 caption).

20. Section 1.3 gives a scalar theta and says n_u(t) marks practice "on the skill" (lines 41, 45). A scalar level has no skill, so the law of motion does not say which practice counts.

21. Two parts of the predictions have no refutation condition (lines 223, 228). "D1 is above zero" in T1a, and the AUC bound of 0.005 in T2, which has no stated consequence.

## Full table

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

Rows checked: 77. Seven table rows carry the 4 critical errors.
