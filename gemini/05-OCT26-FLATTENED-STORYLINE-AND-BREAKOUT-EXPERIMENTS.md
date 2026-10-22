# 05 — Unified Flattened Storyline & Breakout Plan (`plan-within-run` `R1–R7` + `Unit M1`)

> **Status (Updated 2026-10-22 after Claude's Combined Read on `alhazar43/surrogates#2`)**:
> This document integrates Gemini's audit (`00`–`04`) with Claude (`CC-bot`)'s combined read, theory notes (`notes/theory-kt-irt/math.md`, `cs.md`), and draft plan (`irt4llm/notes/plan-within-run.md`, posted in full at `https://github.com/alhazar43/surrogates/issues/2#issuecomment-6345174293`). All 5 file-level corrections noted by Claude have been verified on the CSVs and applied below.

---

## 1. Exact Empirical Alignment & File-Verified Corrections (`surrogates#2`)

1. **Same-Measurement Comparison of GRPO vs. SFT (`results/Q2g/s1/residuals.csv`, `results/T9r/items_wide.csv`)**:
   - Earlier drafts mixed token-0 reads (`r = 0.908` for RL in `T9r`) with greedy free-text reads (`r = 0.289` for SFT in `T9g`, where 23%–33% of items have greedy/mode path flips).
   - **On the exact same $K=32$ sampled answers (`results/Q2g/s1/residuals.csv`, `variant=all`, $J=1,472$ items)**:

     | Quantity on $K=32$ Sampled Answers (`Q2g/s1/residuals.csv`) | GRPO (`t9_rl_0..2_666`, 3 seeds) | CoT SFT (`t9_sft_0..2_70`, 3 seeds) |
     | :--- | :---: | :---: |
     | **$r(L_s, L_0)$ (uncorrected for Base sampling noise)** | **`0.731 to 0.746`** | **`0.592 to 0.611`** |
     | **$\operatorname{sd}(L_s) / \operatorname{sd}(L_0)$** | **`1.91 to 1.99`** | **`0.89 to 0.92`** |
     | **OLS slope $\tau_s = r(L_s, L_0)\cdot \operatorname{sd}(L_s)/\operatorname{sd}(L_0)$** | **`1.425 to 1.456`** | **`0.545 to 0.550`** |
     | **Reliability-corrected slope $\tau_s / \text{rel}_0$ ($\text{rel}_0 = 0.765$)** | **`1.86 to 1.90`** | **`0.71 to 0.72`** |
     | **Token-0 logit slope (`results/T9r/summary.md`)** | **`1.87 to 1.92`** | `0.34 to 0.58` *(format shift: letter mass `0.755–0.889`)* |
     | **$R^2$ of $L_s - L_0$ on $L_0$** | `0.095 to 0.101` | `0.269 to 0.292` |

   - **Key Non-Trivial Discovery (`plan-within-run.md §2.3–2.4`)**:
     - Once corrected for the Spearman-Brown split-half reliability of the 32-draw Base logit ($\text{rel}_0 = 0.765$ from its two 16-draw option-order halves, $r = 0.619$), **GRPO's $K=32$ CoT answer slope (`1.86 to 1.90`) equals its token-0 logit slope (`1.87 to 1.92`)!** One common **person slope (gain) $\tau_s \approx 1.88$** governs both measurements to first order (`Prediction P4` in `R1` tests this in the full latent-logit likelihood).
     - Furthermore, **GRPO's change is a respondent slope $\tau_s$, NOT a 1D ability shift $\Delta\theta_s$**: in `results/T9r/items_wide.csv` at step 666, the token-0 key probability **rises on `876 to 894` items and falls on `606 to 624` items per seed** (`876/624` seed 0, `891/609` seed 1, `894/606` seed 2). A 1D 2PL ability shift with fixed item parameters moves every item in the same direction; a respondent slope $\tau_s > 1$ moves items upward where the keyed margin is positive and downward where it is negative!

2. **Modal-Key Fraction & The Majority-Vote Bound Is OPEN (`results/S1/stability.csv`, `plan-within-run.md §2.6`)**:
   - Under pure person-slope sharpening (`Model N1`: $\pi_s(o\mid j) \propto \exp(\kappa_s \eta_{0jo})$ with $\kappa_s > 0$), every item's modal option is invariant ($\arg\max_o \pi_s(o\mid j) = \arg\max_o \eta_{0jo}$), so the fraction of items whose modal option is the key satisfies $A_\infty(s) = A_\infty(0)$, and accuracy approaches $A_\infty(0)$ as $\kappa_s \to \infty$ (while finite-$\kappa_s$ accuracy is not bounded by $A_\infty(0)$).
   - In `results/S1/stability.csv` (letters only, $K=16$ per option order):

     | Model State (`results/S1/stability.csv`, per option order) | Key Is the Unique Mode | Key Among Tied Modes |
     | :--- | :---: | :---: |
     | **Base (`t9_base`)** | `0.422 to 0.439` | **`0.476 to 0.497`** |
     | **GRPO (`t9_rl_0..2_666`, 3 seeds)** | `0.476 to 0.490` | **`0.482 to 0.498`** |
     | **CoT SFT (`t9_sft_0..2_70`, 3 seeds)** | **`0.486 to 0.525`** | **`0.535 to 0.576`** |

   - Because Base's tied-mode key fraction (`0.476 to 0.497`) is **at or above all 15 RL runs (`mean@16 = 0.4685 to 0.4790`)** and overlaps GRPO's modal-key fraction (`0.476 to 0.498`), **whether GRPO preserves Base's modal options ($A_\infty(s) = A_\infty(0)$) is OPEN and directly tested by `S2` in `Row R1`** (whereas SFT lies clearly above Base under both unique and tied rules).

3. **Why `Q2-A` Did Not Test Item-Level Change (`results/Q2g/s1/reproducibility.csv`, `plan-within-run.md §2.2`)**:
   - In `Q2-A`, top-10% reproducibility across 36 seed pairs was `0.524 to 0.789` for the IRT residual and `0.490 to 0.782` for raw pass-rate change (vs `0.10` random).
   - Both rules retained the common gain $(\tau_s - 1)\mu_{0j}$ AND shared the exact same Base sampling noise across seeds. **Testing whether genuine item-level change $\kappa_{sj}$ exists beyond location $\delta_s$ and person slope $\tau_s$ requires splitting Base into its two independent 16-draw option-order halves (one half per seed) and removing $\tau_s$ in the latent-logit likelihood (`Prediction P1` in `Row R1`)**.

4. **Exact Definitions of `pass@16` (`results/Q1/stats.csv`, `results/S1/stats.csv`) & Update-83 Trajectory (`results/Q1/trajectory.csv`)**:
   - The `pooled` `pass_at_16` values (`0.930` Base, `0.966` SFT, `0.656` S1 GRPO; in `Q1`: `0.686–0.709` [`0.694` mean] for `grpo`, `0.567–0.619` [`0.586` mean] for `ctt`, `0.524–0.525` for `pop0`, `0.583–0.590` [`0.587` mean] for `pop1`) are **`pass@16` averaged over the two option orders (`order 0` and `order 1`)**, not `Pass@32`.
   - In `results/Q1/trajectory.csv`, only **seed 0** of the 4 `Q1` arms (`4` runs) was sampled at intermediate updates `83, 166, 250`, where `grpo_0` reached **`83.7%`** of its final gain by update `83` (`83.2% to 107.1%` across the 4 seed-0 arms).

5. **Strict Repository Firewall for `learner-state`**:
   - Per `irt4llm/CLAUDE.md`, `learner-state` may **not** be opened, cited, or copied into `item-params-llm` plans, code, or talks. All references to `learner-state` stay strictly outside the LLM presentation and paper.

---

## 2. The Flattened End-to-End Storyline (Combining Pre-Registered `Q1/Q2/E1/S1` With `R1` & `M1`)

### **Central Question**:
> *What can an item response model give LLM evaluation and post-training that overall accuracy, `pass@k`, entropy, and per-prompt pass rates cannot?*

### **Act 1 — Population Item Parameters Between Respondents Fail on Post-Training (`E1`, `Q2-A`, `Q2-B`, `Q2a`, `Q1`)**
1. **Human-to-LLM (`E1`)**: 1D student difficulty $b_i$ (`4,918` students, `704` Eedi items) does not order items for `61` LLMs ($\text{Spearman} = 0.089$ vs $0.854$ for held-out students; `61/61` LLMs fail 1D person-fit). *(If discussing distractor geometry: option-centered distractor RDM `P2b` aligns at $\rho = 0.1477$, $29.8\%$ of student split-half ceiling.)*
2. **Adaptive Testing (`Q2c` vs. `Q2a`)**: 102-LLM 2PL adaptive selection cuts short-test error by **`35%` at $n=25$** (`6.4%` vs `9.8%` random) on `165` pretraining checkpoints (`Q2c`), but **reverses on post-trained states (`Q2a`: RMSE quotient `1.033` at $n=25$ and `1.163` at $n=100$; on SFT `11.4%` vs `7.3%` random at $n=25$)**.
3. **1D Ability $\hat{\theta}$ & Item Residuals (`Q2-B`, `Q2-A`)**:
   - On the 7 seen `S1` models, 1D 2PL $\hat{\theta}$ weights items by cross-model $a_j^{\text{pop}}$ (*Marketing* $a=1.45$ vs *High School Math* $a=0.28$), flipping post-training credit from `60% SFT / 40% GRPO` (accuracy) to `21% SFT / 79% GRPO` ($\hat{\theta}$), while on the 12 unseen `Q1` models 5 of 6 pairs do not differ in accuracy (`2/6` sharper).
   - Across the 15 MMLU subjects, $\bar{b}_m^{\text{pop}}$ correlates $-0.94$ with Base subject log-odds $\overline{L}_{\text{base},m}$, so the apparent 15-subject SFT vs. GRPO crossover ($r = +0.88$) reduces to **partial $r = +0.17$** once Base log-odds are controlled!
4. **Population 2PL Inside the RL Critic (`Q1`)**:
   - Across 12 RL runs (`grpo, ctt, pop0, pop1` $\times 3$ seeds), the 2PL critic does not beat the pass-rate band (`mean d = -0.0032, SE = 0.0039`), and $a_j^{\text{pop}}$-weighting (`pop1`) does not beat unweighted `pop0` (`mean d = +0.0059, SE = 0.0052`). Meanwhile, 1-draw critics reduce order-averaged `pass@16` from `0.686–0.709` (`grpo`) to `0.567–0.619` (`ctt`), `0.583–0.590` (`pop1`), and `0.524–0.525` (`pop0`).

---

### **Act 2 — What Item Response Modeling Inside a Single Run Reveals: The Person-Slope ($\kappa_s / \tau_s$) & Option-Level Decomposition (`Row R1`)**
Why did static population 2PL fail, and why did the 15-subject dissociation collapse when controlling for Base log-odds $L_{0j}$? **Because post-training acts primarily through a **person slope (common gain $\tau_s$)** on the Base model's option distribution $\eta_{0jo}$, NOT a 1D ability shift $\Delta\theta_s$ along the population axis!**

1. **GRPO Is a Person Slope $\tau_s \approx 1.88$ Across Both Token-0 and CoT Rollouts (`P4`)**:
   - At step 666 (`T9r`), GRPO's token-0 key probability **rises on `876–894` items and falls on `606–624` items per seed**—impossible under a 1D ability shift, but exact under a person slope $\tau_s > 1$ (which pushes items up when the **keyed margin** $\eta_{0jk} - \sum_{o\neq k}\frac{q_o}{1-q_k}\eta_{0jo} > 0$ and down when it is negative, `P9`).
   - Corrected for Base 32-draw noise ($\text{rel}_0 = 0.765$), GRPO's $K=32$ slope (`1.86–1.90`) matches its token-0 slope (`1.87–1.92`).
   - By the JEDM paper's affine identity (`main.tex:550-557`), a person slope $\tau_s$ and multiplying every item's discrimination by $\tau_s$ yield the exact same likelihood.
2. **SFT Flattens the Person Slope ($\tau_s / \text{rel}_0 = 0.71\text{–}0.72 < 1$) While Raising Modal-Key Fraction (`0.422–0.439 -> 0.486–0.525` unique, `0.476–0.497 -> 0.535–0.576` tied)**:
   - On unfamiliar teacher CoT traces $y^*$, the typicality push $c(y^*)$ along the logit scale is negative (`P12`), so SFT flattens the slope ($\tau_s < 1$) while shifting modal options toward the key on hard items.
3. **Why Two Person Parameters $(\kappa_s, \delta_s)$ Explain `Pass@k` Collapse (`P11`) & Test True Item-Level Change (`P1`, `S2`, `S4`, `EQ` in `Row R1`)**:
   - Under `Model N2` ($\pi_s(o\mid j) \propto \exp(\kappa_s \eta_{0jo} + \delta_s \mathbf{1}[o=k_j])$), two person parameters $(\kappa_s, \delta_s)$ predict the entire `pass@k` curve ($k=1, 4, 16$) from the Base distribution (`P11`), explaining why RL models with higher $\kappa_s$ have lower `pass@16` at large $k$ (`pop0` vs `grpo`).
   - **`Row R1` (CPU, 1–2 days, ready for Oct 26 if approved)** runs 8 win-win tests on existing `S1, Q1, T9r` data:
     - **`P1` (Item-level change beyond location & gain)**: Splits Base into independent option-order halves and removes $\tau_s$ in the latent-logit likelihood. Either **no item-level change repeats across seeds beyond $(\delta_s, \tau_s)$** (proving post-training on the bank is a 2-parameter person operator!) or **$\hat{\kappa}_{sj}$ isolates true item-level learning**.
     - **`S2` (Modal-option invariance)**: Tests whether $\Delta A_\infty$ for GRPO lies inside the Base order-to-order noise floor (settling the majority-vote bound).
     - **`S4` (Reliability-corrected rank agreement)**: Disattenuates $\rho(L_s, L_0)$ by $\sqrt{\text{rel}_s \cdot \text{rel}_0}$ to test whether SFT's lower correlation (`0.59–0.61` vs `0.73–0.75`) is true item reordering or high-entropy attenuation.
     - **`EQ` (Equal-accuracy test on the 4 `Q1` arms)**: Tests whether `grpo`, `ctt`, `pop0`, and `pop1` (all at `46.9%–47.5%` accuracy) differ only in $(\delta_s, \kappa_s)$ or also in item structure.

---

### **Act 3 — Composing `SFT -> GRPO` & Within-Model Pass-Rate Filtering (`Unit M1` + `Row R2` Pre-Registered Predictions)**
- **Start State of `Unit M1` (Verified on Cluster `~/irt4llm-runs/M1/results/sft_pass_rate_<s>.csv`, $K=8$ on 10,682 prompts)**:
  - `0/8` on `925–977` prompts, `1/8` on `1,322–1,355`, `7/8` on `963–1,094`, `8/8` on `899–1,006`.
  - Filtering to `2..6 of 8` (`M1c`) keeps **`6,286 to 6,529` prompts** (`58.8%–61.1%` of the pool).
- **Pre-Registered `M1` Rules (`notes/units/M1.md`) + `Row R2` Person-Slope & Modal-Answer Predictions (`surrogates#2`)**:
  1. **`P5` (`M1c` band raises $\tau_c \ge \tau_b$, reversing the naive binary intuition!)**:
     - Under the 4-option model, prompts with $p_{\text{key}} < 1/4$ (`0/8` and `1/8`) have negative keyed margins and push the gain $\tau$ *down*. Removing `0, 1, 7, 8 of 8` in `M1c` therefore **raises** the expected gain push: $\log \tau_c / \log \tau_b = \mathbf{1.18\text{ to }1.30}$ (`P5` rule: mean over seeds of $\log \tau_c - \log \tau_b \ge -2\,\text{SE}$).
  2. **`A1` (`M1b: SFT -> GRPO` modal-key test)**:
     - If GRPO after SFT acts as pure sharpening of SFT's distribution, $\Delta A_\infty(\text{M1b})$ relative to SFT stays within the order-split floor, while single-draw `mean@16` rises toward SFT's modal-key fraction (`0.486–0.525` unique / `0.535–0.576` tied). If $\Delta A_\infty(\text{M1b})$ drops below the floor, short-chain RL overwrote SFT's modal math answers (`-1` branch).
  3. **`A2` (`M1a: Token Temperature T = 0.2`)**:
     - Tests whether decoding at $T=0.2$ fits pure sharpening `N1` ($\Delta A_\infty$ inside the floor) or shifts modal answers by altering the CoT mixture.

---

## 3. Summary Table: Decisions for the Author (`plan-within-run.md §7`)

| Row / Unit | What It Delivers | Compute Cost | Target | Recommendation |
| :--- | :--- | :---: | :---: | :--- |
| **`Unit M1`** (`M1a, M1b, M1c`) | Pre-registered `SFT -> RL` & pass-rate band `[2..6/8]` vs. $T=0.2$ | Running on cluster (`ctit093` approved) | **Oct 26 & Nov 20** | **Keep running** |
| **`Row R1`** (`P1, P4, P9, P10, P11, S2, S4, EQ`) | Person-slope ($\kappa_s$) & option-level (`N0..N4`) analysis on `S1, Q1, T9r`: tests true item change beyond gain (`P1`), common gain across token-0 & CoT (`P4`), `pass@k` from 2 parameters (`P11`), modal bound (`S2`), SFT disattenuated rank correlation (`S4`), and equal-accuracy `Q1` test (`EQ`) | **CPU only**, 1–2 builder days | **Oct 26** (adds 1 high-depth slide) & **Nov 20** | **Approve `R1` now** |
| **`Row R2`** (`M1` through `R1`) | Evaluates `M1a, M1b, M1c` on $(\kappa_s, \delta_s, \Delta A_\infty)$ and tests `P5, P6, A1, A2, A3` alongside `M1`'s pre-registered rules | **CPU minutes** once `M1` samples finish | **Oct 26 / M1 result** | **Approve with `R1`** |
| **` Rows R3, R4, R6, R7`** | `R3`: Gain $\tau_s$ as final RMSNorm weight scale (`P2, P3`) & SFT trace typicality (`P12`).<br>`R4`: Within-run discrimination $\tilde{a}_j = J_j u_\perp$ (`P7, P8, C2, C3`).<br>`R6`: Offline KT-IRT person-state tracker $(\delta_t, \tau_t, \eta_t)$ on the 12 `Q1` reward logs.<br>`R7`: Primary-source literature verification. | `R3/R4`: GPU (after `R1`)<br>`R6`: CPU (~1 day) | **Nov 20 Workshop** | **Approve after `R1`** |
| **Gemini's earlier `Exp 2` (Gated 1-draw critic) & `Exp 3` (Uncorrected cross-repo rank plot)** | `Exp 2` is a 1PL-level heuristic with published neighbors (SPO, DAPO) requiring new training code; `Exp 3` mixed models/banks without reliability correction and is superseded by `R1` (`S4`). | — | — | **Drop `Exp 2` & `Exp 3` in favor of `R1 + R2`** |
