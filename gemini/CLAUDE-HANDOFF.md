# Cross-Repository Handoff Brief for Claude (`plms-74` / `item-params-llm` & `plms-c8` / `signature`)

**Date**: 2026-10-22 (Updated v4.1 — All 6 Pre-Launch Fixes of Claude Adopted in Full)  
**Author & Peer Session**: Author + Gemini Synthesis $\longrightarrow$ Handoff to Claude (`item-params-llm` Lead)  
**Primary Governing Files**:
1. [`notes/units/M1.md`](units/M1.md) (**Active Unit Contract v4.1 — Ready to Launch with All 6 Fixes Applied**)
2. [`notes/MASTER-PLAN.md`](MASTER-PLAN.md) (**v4.1 Final Locked Authoritative Plan**)
3. [`notes/CROSS-REPO-INDEX.md`](CROSS-REPO-INDEX.md) (**Cross-Repo Unit Status & Graveyard Ledger**)
4. [`src/irt4llm/verify_master_plan.py`](../src/irt4llm/verify_master_plan.py) (**One-Command Reproducibility & Verification Script**)

---

## 0. Resolution of Your 6 Pre-Launch Review Points (2026-10-22)

Thank you for the sharp pre-launch audit. **Every single one of your 6 picks is accepted and written into `notes/units/M1.md`, `notes/MASTER-PLAN.md`, and `notes/state.md`**:

1. **Fix 1 (`M1b` & `M1c` use `Q1`'s exact GRPO settings)**:
   - Adopted. Both `M1b` and `M1c` use `tools/q1_job.sh`'s exact `grpo` configuration: `lr = 5e-6`, `kl_beta = 0.05`, `temperature = 0.8`, `max_new_tokens = 3000`, `max_seq_length = 2000`, `per_device = 1`, `n_gen = 5`, `total = 32` prompts $\times 5$ answers $= 160$ answers per update. **Only `--model_name` and `--ref_model_name` change to `$SFT_PATH`**.
2. **Fix 2 (`333` updates with a save at update `83`, bank sampled at both)**:
   - Adopted. Both `M1b` and `M1c` run `UPDATES=333` with `SAVE_AT=83` (using `repo_s0`'s `--save_at_updates 83`), and sample the 1,500-item bank at **both** `u83` and `final` (`u333`) before deleting `checkpoint-u83`. This gives a clean 3-seed check of the update-83 hypothesis AND an exact 333-update comparison against `q1_grpo`.
3. **Fix 3 (`M1c` samples $k = 8$ answers per training prompt from each `t9_sft_<seed>_70` checkpoint first)**:
   - Adopted. Before `M1c` training starts, run `src/irt4llm/w1_sample.py` (`tools/w1_job.sh`, `~23 min` per seed on 1 GPU) on each `t9_sft_<seed>_70` checkpoint across the 10,682 training prompts (`sft_pass_rate_<seed>.csv`), and filter `M1c` to that SFT seed's own interior ZPD pass-rate band (`[0.2, 0.8]`, i.e., `2 to 6 of 8` as in `W1.md:29`, or `1 to 7 of 8` if declared before launch). *(Bonus: `sft_pass_rate_<seed>.csv` compared against `results/W1/pass_rate.csv` also measures SFT's exact gain on the 8,538 Seen-by-Both vs. 2,144 Seen-by-GRPO-Only training prompts!)*
4. **Fix 4 (`Rule 3`'s `zero_adv_share` branch dropped)**:
   - Adopted. A drop in `zero_adv_share` follows by construction from filtering out $0/8$ and $8/8$ prompts; it is now reported as description only. Rule 3 reads strictly on bank `mean@16` (`M1c` vs. `M1b`).
5. **Fix 5 (`Rule 1` adds the `0` null branch)**:
   - Adopted. Rule 1 now partitions outcomes into `+1` (beats both `t9_sft` and `q1_grpo` by $>2.92\text{ SE}_d$ in `3/3` seeds), `-1` (reduces `maj@16` or `pass@16` below `t9_sft` by $>2.92\text{ SE}_d$ in `3/3` seeds without beating `q1_grpo` on `mean@16`), and `0` (all other outcomes).
6. **Fix 6 (Verification of the two flagged claims)**:
   - **Mode reordering (`S1` & `M.md` Section M9 / Defect 5)**: You are right. In `results/review-2026-10-19/M.md` (`lines 132–149, 157`), `-4.4 to -5.4 SE` (`~50%` larger excess) was Reviewer M's *synthetic simulation* showing that `excess = obs - h0` is biased toward sharp states (`h0 = 0.341` GRPO vs. `0.477` SFT) at equal true mode-change share $\pi$, which is why `notes/state.md:32` withdrew the `S1 Rewrite 1` verdict (*"about the same share"*) and deferred `pi_hat = (obs - h0) / (h1 - h0)`. In committed files, the verified statistics are: (i) raw $k=16$ mode change `0.560–0.564` (SFT) vs. `0.398–0.436` (GRPO) (`results/S1/rewrite1_mode_change.csv`), and (ii) item log-odds correlation with Base $r(L_{\text{SFT}}, L_{\text{base}}) = 0.733$ vs. $r(L_{\text{GRPO}}, L_{\text{base}}) = 0.858$ in `S1` (`0.289` in free-text CoT `T9g/s3`, `0.822` vs. `0.908` in `T9r`). We have updated `MASTER-PLAN.md` and Red Line 2 below to state these exact verified numbers.
   - **GSM8K `-0.22`**: That came from an external Oct 5 pilot repo (`alhazar43/irt4llm`), not `item-params-llm`. We have removed it from `item-params-llm`'s claims; use only the committed in-repo correlations (`results/E1/difficulty.csv`: $r = -0.0130, \rho = 0.0889$; `signature F5`: $r(a_{\text{pop}}, a_{\text{traj}}) = 0.04$).

---

## 1. The Locked Scientific Synthesis (`Cannot`, `A helps B`, `B helps A`, `A <-> B`)

### Pillar 0 (`Cannot` — Admitting Failure Cleanly)
Static cross-sectional 102-LLM 2PL parameters $(a_j^{\text{pop}}, b_j^{\text{pop}})$ cannot be used as a plug-and-play 1D score replacement (`Q2a, Q2-A, Q2-B`) or 1D RL advantage multiplier (`Q1: pop0, pop1`) because:
1. **Molenaar's Non-Ergodicity**: Between-model differences across 102 heterogeneous LLMs are orthogonal to within-model post-training trajectories ($\operatorname{Corr}(a_j^{\text{pop}}, a_j^{\text{traj}}) = 0.04$ on MMLU; human-vs-LLM difficulty $r = -0.013$ in `E1`).
2. **Autoregressive Prior Scaling (`Table 1.1`)**: Within a model, post-training log-odds obey $\Delta L_{s,j} = (\tau_s - 1)L_{\text{base},j} + \delta_s + \gamma_s z_j^{\text{pop}} + \varepsilon_{s,j}$, where $L_{\text{base},j}$ explains **`48.7%–87.3%`** of shifts and $z_j^{\text{pop}}$ adds $\Delta R^2 \le 1.04\%$. By `Rule 4`, $\hat{\tau}_s = r(L_s, L_{\text{base}}) \cdot \frac{\sigma(L_s)}{\sigma(L_{\text{base}})}$ splits SFT ($r = 0.289$ in `T9g/s3`, $0.733$ in `S1`, $0.822$ in `T9r`) from RL ($r = 0.858$ in `S1`, $0.908$ in `T9r` + scale sharpening).

### Pillar 1 (`A helps B` — What Item Parameters CAN Tell Us: The Declarative vs. Procedural Double Dissociation, $R^2 = 76.9\%$)
Across the 15 MMLU subjects in `results/W0/items_bank.csv` ($r(\bar{a}_m^{\text{pop}}, \bar{b}_m^{\text{pop}}) = -0.8841$), cross-sectional $(a_m^{\text{pop}}, b_m^{\text{pop}})$ indexes a **Cognitive Construct Axis** (`verify_master_plan.py Part 4`, `MASTER-PLAN.md Table 1.2`):
- **High $\bar{a}_m^{\text{pop}}$ (`0.69–1.45`), Low $\bar{b}_m^{\text{pop}}$ (`-0.99 to +0.72`) = Declarative / Semantic Recall** (*Marketing, Sociology, Prehistory, Geography, World History, Biology, Philosophy, Clinical Knowledge*).
- **Low $\bar{a}_m^{\text{pop}}$ (`0.28–0.55`), High $\bar{b}_m^{\text{pop}}$ (`+0.25 to +2.30`) = Procedural / Multi-Step Quantitative Reasoning** (*High School Math, Physics, Statistics, Elementary Math, Econometrics, Macroeconomics*).
- **The Neuro-Cognitive Double Dissociation**:
  - **Outcome RL (`GRPO` + all 12 `Q1` runs) = Neural Gain Modulation ($\beta_s \uparrow$)**: $\overline{\Delta L}_{\text{GRPO},m}$ correlates **$r = +0.7464$ (`S1`) and $+0.6421\text{–}0.7229$ (`Q1`) with $\bar{a}_m^{\text{pop}}$**, beating SFT on **`8 of 8` Declarative domains** while stalling/regressing on Procedural domains (*Econometrics `-0.575`, Macroeconomics `-0.192`, Physics `-0.035`*).
  - **CoT SFT (70B Distillation) = Procedural Schema Acquisition ($\beta_{\text{SFT}} \approx 1$)**: Beats GRPO on **`5 of 6` Procedural Quantitative domains** (*Elementary Math `+1.491` logits, High School Math `+0.772`, Econometrics `+0.563`, Macroeconomics `+0.504`, Physics `+0.357`*).
  - Across the 15 domains:
    $$r\big(\bar{a}_m^{\text{pop}}, \, \overline{\Delta L}_{\text{SFT},m} - \overline{\Delta L}_{\text{GRPO},m}\big) = \mathbf{-0.8015} \;\; (R^2 = 64.2\%), \qquad r\big(\bar{b}_m^{\text{pop}}, \, \overline{\Delta L}_{\text{SFT},m} - \overline{\Delta L}_{\text{GRPO},m}\big) = \mathbf{+0.8771} \;\; (R^2 = \mathbf{76.9\%})!$$

### Pillar 2 (`B helps A` — What Counterfactual LLM Rollouts Teach Psychometric Theory)
1. **The $a_j^{\text{pop}}$-Weighting Attribution Reversal Theorem (`Corollary 4`)**: Because $\Delta\hat{\theta} \propto \overline{\Delta p} + \operatorname{Cov}(a_j^{\text{pop}}, \Delta p_j)/\bar{a}^{\text{pop}}$ and $a_j^{\text{pop}}$ weights declarative recall $5\times$ higher than procedural math, 2PL $\hat{\theta}$ cuts SFT's post-training credit from **`60%` (Accuracy) to `21%` ($\hat{\theta}$)** (`0/3` wins on `Base vs. SFT`) and inflates GRPO's credit from **`40%` to `79%`** (`9/9` wins on `SFT vs. GRPO`).
2. **The $l_z^{(1)} = +2.24 \to l_z^{(32)} = -482$ Person-Fit Sign-Flip Theorem (`Corollary 3`)** & **Option-Order Testlet Law (`Corollary 5`)**.

### Pillar 3 (`A <-> B` — The High-Leverage Bet: `Unit M1`, Ready to Launch)
All 15 `Base -> RL` runs (`3 S1 + 12 Q1`) hit a ceiling at **`Mean@16 = 0.4685–0.4790`** because RL is a concentration operator ($\Delta\beta_s \gg 0$) bounded by `t9_base`'s modal ceiling (`Maj@16 = 0.414 / 0.459*`). Meanwhile, `t9_sft_0..2_70` (`Base -> SFT`) *already* raised the modal ceiling to **`Maj@16 = 0.504` (`0.529` random tie / `0.555*` tie-correct)** and **`Pass@16 = 0.966`**, leaving a **`+0.065 to +0.116` unharvested Condorcet voting gap** (`H = 0.943`).
- **`Unit M1` (`notes/units/M1.md`)** tests whether we can harvest SFT's Condorcet gap and break the `0.479` wall via `M1a` ($T=0.2$), `M1b` (`t9_sft -> GRPO`, 333u with save at 83u, exact Q1 settings), and `M1c` (`t9_sft -> ZPD-GRPO` using SFT's own $k=8$ pass rates, 333u with save at 83u).

---

## 2. Red Lines (Updated With Your Fix 6)

1. **Do NOT claim $\hat{\tau}_{\text{SFT}} < 1$ means SFT "loses confidence" or "flattens its rollout distribution."** In `S1`, SFT rollout entropy matches Base (`0.943 vs. 0.963`), and in `T9g/s3`, SFT answer-token logit SD is $2.35\times$ Base (`7.44 vs. 3.17`). By `Rule 4`, $\hat{\tau}_{\text{SFT}} = r \cdot (\sigma_s/\sigma_b) = 0.289 \times 2.351 = 0.680 < 1$ (`0.430` at Token-0) is **regression attenuation from lower correlation with Base ($r = 0.289$ in `T9g/s3`, $0.733$ in `S1`)**.
2. **Do NOT cite the withdrawn `S1 Rewrite 1` verdict ("about the same share").** State the exact facts from `results/S1/rewrite1_mode_change.csv`, `results/S1/rewrite1_null.csv`, and `results/review-2026-10-19/M.md` (`lines 132–149, 157`): raw $k=16$ mode change is `0.560–0.564` (SFT) vs. `0.398–0.436` (GRPO); the symmetric-null `excess` (`0.130–0.137` SFT vs. `0.112–0.145` GRPO) scales with sharpness (`h0 = 0.477` SFT vs. `0.341` GRPO, giving a `4.4–5.4 SE` sharpness bias in `M.md`'s simulation at equal $\pi$), so the "same share" verdict was withdrawn in `notes/state.md:32` and `pi_hat = (obs - h0) / (h1 - h0)` was deferred.
3. **Do NOT run local patches on `q2_full.py` (`Q2-A/B`) or run larger-$n$ short-test CAT (`Q2a/Q2c`).**
4. **Do NOT cite small-$N$ NRM item axes from `P1–P5` ($N=61$) or `T1m–T1f` ($N=30$), or external unverified numbers.**
