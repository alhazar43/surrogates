# surrogates

The project of 19 October 2026 on the author's question "what is learning", answered through a set of surrogates, each answering part of the question and never all of it. Global rules are in ~/.claude/CLAUDE.md and ~/.claude/rules/. This file holds the paths, the environment and the requirements, and nothing else.

## Where things are
- program/research-program.md is the research program, the author's and the lead's editable working statement of the question, the surrogates, their tiers and their dependencies. It is not an executable plan. A superseded version goes to program/archive/ with its version in the name.
- theory/<id>.md holds the theory end of each surrogate, one file per surrogate.
- sources/reading.md indexes the opened sources with their rung.
- notes/units/ holds unit files. notes/decisions.md, notes/deferred.md, notes/mistakes.md and notes/session-note.md are the lead's logs.
- src/surrogates/ is the code, tests/ the tests, results/ the result summaries.
- The thinking record that led here is in C:/Users/YuanW/claude-thesis-docs/exploration/signature/ (panel-2026-10-19.md, three-questions.md, goodfire-reading.md section 7).

## Plan state
Plan state: none

## Environment
- Cluster hpc-head1.ewi.utwente.nl, QOS research, at most 100 submitted and 8 running jobs per user, shared with the other projects. Exclude ctit088, ctit092 and ctit093. Cancel jobs by explicit ID only.
- No heavy computation on the PC. Local scripts run on synthetic or tiny records only.

## Requirements
- Every reading is calibrated before use on a case with no effect and a case with a known planted effect, with its false positive rate and power written beside it.
- A source is cited only after its primary text is opened, with its rung.
- Nothing from C:/Users/YuanW/PLMS/learner-state or learner-models is opened, cited or copied. docs/conversation/ is never opened.
- No commits or pushes unless the author asks.
