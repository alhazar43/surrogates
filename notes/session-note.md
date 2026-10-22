# Session note (updated 2026-10-21)

**2026-10-21 CROSS-REPO HANDOFF (`item-params-llm` + `signature` + `surrogates`)**:
- The unified v4.0 synthesis (`Cannot`, `A helps B` Declarative–Procedural Double Dissociation $R^2 = 76.9\%$, `B helps A`, and `A <-> B` `Unit M1`) and the active cluster experiment specification are locked in:
  - `D:\surrogates\gemini\CLAUDE-HANDOFF.md` (mirror of `alhazar43/item-params-llm:notes/CLAUDE-HANDOFF.md`)
  - `D:\surrogates\gemini\MASTER-PLAN.md` (mirror of `alhazar43/item-params-llm:notes/MASTER-PLAN.md` v4.0)
  - `D:\surrogates\gemini\M1.md` (mirror of `alhazar43/item-params-llm:notes/units/M1.md`)
  - `D:\surrogates\gemini\CROSS-REPO-INDEX.md` (mirror of `alhazar43/item-params-llm:notes/CROSS-REPO-INDEX.md` v4.0)
- All active cluster runs (`Unit M1`: `M1a` token temperature $T=0.2$, `M1b` `t9_sft -> GRPO` 83 updates, `M1c` `t9_sft -> ZPD-GRPO` 83 updates) run in **`alhazar43/item-params-llm` (`plms-74`)**. Pull `origin/main` (`10e01cd`) in `item-params-llm` and read `notes/CLAUDE-HANDOFF.md` and `notes/units/M1.md` first.

---

## Earlier `surrogates` note (2026-10-19T08:54Z)

The research program is program/research-program.md, version 1. Six theory units run (notes/units/TG, TM, TR, TE, TP, TS), each writing theory/<id>.md. When they finish, the lead reads section 8 of each file and folds accepted edits into the research program as version 2, with version 1 archived. Open for the author: whether the peer's Q1 and Q2 serve as tiers of G and M; whether the shared ground copies Jain et al. 2023 or Huang et al. 2026. The peer project runs on its own. Nothing is built or run on the cluster from here.

### Corrections queued for version 2 of the research program
- Surrogate M, theory line: "a change of use predicts a shift, a change of capability predicts a reordering" is wrong as stated; theory/M.md's disagreement holds in substance with three narrowings (theory/M-review.md). Take the corrected line from M.md after its fix.
- Surrogate G, tier 3: the program says practice on high information items raises later learning more; G's derivation predicts otherwise (theory/G-review.md m15). Take the wording from G.md after its fix.
- Surrogate P, theory line on Reasoning Theater: wrong; the relation holds only on recall items (theory/P-review.md, "What Boppana et al. 2026 report, by level"). Take the replacement from P.md after its fix.
- Section 3, idea 1: the wrapper holds only when the capability is weakly relevant (Jain et al. section 5.1, theory/M-review.md C3).
