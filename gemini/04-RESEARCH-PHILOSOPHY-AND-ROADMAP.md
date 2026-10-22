# Research Philosophy, Operational Rules, & Three-Tier Roadmap (`04-RESEARCH-PHILOSOPHY-AND-ROADMAP.md`)

This document consolidates the user's intellectual North Star, strict anti-hype operational rules, and the concrete mapping of our verified cross-repository assets onto the three upcoming milestones.

---

## 1. Core Intellectual Identity & Research Philosophy

1. **Self-Explainable Models Through Latent Representations & Psychometric Measurement**:
   - **Background**: Applied Mathematics / Machine Learning + Psychometrics / Educational Measurement.
   - **Core Question**: When do a model's internal representations and external response distributions genuinely identify an interpretable latent measurement scale, and when do they alias format compliance, temperature/entropy scaling, or optimization shortcuts?
   - **True Interdisciplinary Leverage ($A \leftrightarrow B$)**:
     - Reject **ML leaderboard hounding** (chasing `0.005–0.01` accuracy gains with brute-force compute) and **old-school IRT pedantry** (treating 1980s binary 2PL curve-fitting on `0/1` matrices as an end in itself or using $\hat{\theta}$ as a trivial accuracy surrogate).
     - Show **A helps B** (psychometric decomposition & invariance diagnostics expose where standard ML accuracy and RL critics are blind), **B helps A** (open-weight LLMs provide repeatable, memoryless, counterfactual respondents that bypass the $\mathcal{O}(\Delta^2) \sim 10^{-4}$ nats passive-log identification wall in human education), or **$A \leftrightarrow B$**.
2. **Never Enter the `0.01` Noise Swamp**:
   - Any effect of size `~0.005–0.01` that fluctuates with random seeds (`MDE = 0.015–0.023` for 3 seeds in `R R9` / `M6`), batch ordering (`bfloat16` greedy flips in `T9r`), or covariate collinearity is **Status `0` (Noise)**. Never try to rescue a `0` with post-hoc math jargon.
3. **"A Good Research Is a Bet You'd Be Happy Losing" (True Failsafe)**:
   - A real failsafe is **not** inventing fancy names (`"Conservation Law"`, `"Neural Gain"`) to dress up a null result, nor is it getting trapped in a `hype -> audit -> withdraw -> new hype` loop.
   - A real failsafe is an experimental or theoretical question where **both `+1` and `-1` establish a clean, macro-scale (`>> 0.01`) structural mechanism**.
4. **"Show the Reader How I Learned This"**:
   - Papers and talks should not read like a defensive laundry list of caveats or a textbook of ungrounded definitions; they should take the audience through the genuine investigative arc: *natural hypothesis $\to$ surprising empirical anomaly / negative wall $\to$ white-box / mathematical diagnosis $\to$ constructive resolution*.

---

## 2. Mandatory Operational & Anti-Hype Rules

- **Rule 1 (Single Source of Truth in `D:\surrogates\gemini\`)**:
  - All synthesis, indexing, and slide/paper planning documents live exclusively in `D:\surrogates\gemini\`. Never scatter duplicate markdown files across other repos.
- **Rule 2 (Every Claim Carries Its Provenance, File Path, and Caveat Tag)**:
  - Before citing any number, verify it in [`02-VERIFIED-EMPIRICAL-ASSETS.md`](file:///D:/surrogates/gemini/02-VERIFIED-EMPIRICAL-ASSETS.md) or [`03-VERIFIED-THEORY-AND-MATH.md`](file:///D:/surrogates/gemini/03-VERIFIED-THEORY-AND-MATH.md) and check [`01-PROVENANCE-AND-SELF-AUDIT.md`](file:///D:/surrogates/gemini/01-PROVENANCE-AND-SELF-AUDIT.md) to ensure it is not on the blacklist or missing a mandatory caveat.
- **Rule 3 (Zero Coined Buzzwords)**:
  - Never invent grandiose labels (`"Declarative Factual"`, `"Procedural Schema"`, `"Condorcet Voting Gap"`, `"Theorem 1/2"`). Use standard, precise ML and psychometric terminology that both communities immediately understand.
- **Rule 4 (Regression Slope $\hat{\tau}$ vs. Correlation $r$ vs. Dispersion $\sigma_s/\sigma_b$)**:
  - In any linear regression $\Delta L_{s,j} = (\tau_s - 1)L_{\text{base},j} + \delta_s + \varepsilon_{s,j}$, always report the exact identity $\hat{\tau}_s = r(L_s, L_{\text{base}})\frac{\sigma(L_s)}{\sigma(L_{\text{base}})}$ so preference reordering ($r < 1$, regression attenuation) is never confused with logit flattening ($\sigma_s/\sigma_b < 1$).
- **Rule 5 (Visual Storytelling in Figures)**:
  - Figures exist to tell a clear visual story, not to dump textboxes, equations, or forced mechanical `CAN / CANNOT` banners onto a canvas. One clear visual contrast per figure, minimal text, immediately readable by both ML and IRT audiences.

---

## 3. Mapping Verified Assets to the Three-Tier Delivery Horizon

### Tier 1: Monday, Oct 26 Internal Group Presentation ("The Appetizer")
- **Audience**: Measurement, IRT/Psychometrics, and Neuro-Cognitive researchers.
- **Locked Title**: *"Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?"*
- **Role**: An honest, high-depth appetizer that frames the interdisciplinary bridge ($A \leftrightarrow B$) using our **verified `[HIDDEN GEMS]` and `[SOLID FACTS]`** across `irt4llm`, `signature`, `item-params-llm`, and `surrogates/theory`:
  1. **Motivation (Why Psychometricians Need LLMs & Why LLMs Need Psychometrics)**:
     - Open-weight LLMs provide **clonable counterfactual respondents** (`Base -> SFT` vs `Base -> GRPO`, $K=32$ memoryless draws/item across option orders), allowing us to separate population item parameters from within-run person-slope and item-level dynamics (`plan-within-run.md`).
  2. **Pillar 1 — Human vs. LLM Calibration (`E1` + `P4` + `P2b` + `U1a/m/n`)**:
     - **What 1st-Order 2PL Cannot Do**: Binary 2PL item difficulty $b_j$ does not transfer from $4,918$ students to $61$ LLMs ($\rho = 0.089$, `61/61` LLMs misfit at median $l_z = -11.08$, and diagram items have *lower* DIF `12.8%` than text items `20.7%`, `E1/P4`).
     - **What 2nd-Order NRM Distractor Geometry Can Do**: When we move from binary 2PL to **option-centered distractor co-selection (`P2b`)**, 61 LLMs and 4,918 students share **$\rho = 0.148$ ($p < 10^{-4}$, `29.8%` of student split-half ceiling)** across $245,390$ item pairs! And on humans (`U1a/m/n`), that 5D NRM distractor manifold tracks dynamic misconceptions with an **`8.24`-quiz half-life** (`40/40` history lengths positive).
  3. **Pillar 2 — Evaluating Pretraining vs. Post-Training (`Q2c/Q2a` + `K2/Theory E8` + `signature T1/T3/T5/T8-P3` + `Q2g`)**:
     - **Where Population 2PL Works (On-Scale Growth)**: During pretraining (`Q2c`, $n=25, 50$ cuts error by 35%) and rank-preserving LoRA (`T1/T3`, stays inside L18 PCA subspace `0.875` and preserves item rank), learning moves along a shared scale.
     - **Why White-Box Subspace Alone Is Blind, and How Item Rank + KL Diagnose Forgetting (`signature T1/T3/T5/T8-P3`)**: 5-shot ICL exits the L18 PCA subspace (`0.37–0.42`) *without* breaking item rank (`4/5` performance, reversible), whereas Full FT exits the subspace (`0.24–0.29`) AND breaks the item map (`4/5`), with trained-subject Forward KL ($\rho = 0.870$, `T8-P3`), Cosine-NTK distance ($\rho = 0.772$, `T6`), and initial loss ($\rho = 0.891$, `T5`) directly predicting OOD catastrophic forgetting!
     - **Why Static Population 2PL Fails on Post-Training (`Q2a/Q2g`)**: Because population $a_j^{\text{pop}}$ cannot be predicted from item text (`Surrogates P1` $R^2 = -0.15$) or single-model gradients (`Gate S2x` $+0.008$) and weights easy factual subjects $5.2\times$ higher than hard math, greedy Fisher selection biases CAT by `+0.04 to +0.12` (reversing at $n=100$, `Q2c/Q2a`), and on the 7 seen `S1` models flips SFT:GRPO credit from `60:40` (accuracy) to `21:79` ($\hat{\theta}$, `Q2g`).
  4. **Pillar 3 — From Single-Scalar Readouts to Multi-Draw Distributions & RL Training (`T9` + `S1` + `W1/Q1` + `Theory R/S` + `M1` Teaser)**:
     - **What Single-Draw Eval Misses (`T9` + `S1`)**: On greedy eval (`T9`), `100%` of SFT's `+5.01` pt MMLU gain is format repair (`gain_parsed = -0.55` pts vs GRPO `+2.51` pts). Under $K=32$ sampling (`S1`), GRPO wins `mean@16` (`0.479 vs 0.439`) by collapsing within-item entropy (`0.956 -> 0.287` nats, dropping `Pass@16` from `0.869 -> 0.673`), while SFT preserves diversity (`H = 0.938`, `Pass@16 = 0.901`) and overtakes GRPO on **`maj@16` (`0.504 vs 0.481` strict, `0.555 vs 0.490` tie=key)**.
     - **Why Static 2PL Critics Fail in RL (`W1` + `Q1` + `Theory R2`)**: In $G=5$ GRPO, **`35.0%`** (UMVUE) of prompt groups have zero reward variance (`W1`), and 83%–107% of gain finishes by step 83 (`Q1`). Replacing the group baseline with static $\hat{p}_j$ under $1/L(y)$ normalization introduces a non-zero length bias $-\hat{p}_j\nabla\mathbb{E}[1/L_j]$ (`R2/M1`), and even with $L_0$ fixed (`Q1`), static $G=1$ critics (`ctt, pop0, pop1`) tie `grpo` (`0.469–0.475`) while `pop0` worsens mode collapse (`stable_12` `+0.076`).
     - **Appetizer Hook & Breakout Results (`Row R1` + `Unit M1` + `Row R2`)**:
       - **`Row R1` (CPU, 1–2 days)**: Fits the key-contrast $(\tau_s, \delta_s, \kappa_{sj})$ and nested option-level models (`N0–N4`) on `S1`, `Q1`, and `T9r` (`74` states total: `7 S1 + 12 Q1 + 55 T9r`), testing `P1` (item-level change beyond location & gain), `P4` (one gain across token-0 and $K=32$ CoT), `P9` (keyed margin sign law), `P10` (order overdispersion), `P11` (`pass@k` from 2 person parameters), `S2` (modal-option invariance of GRPO), `S4` (reliability-disattenuated SFT rank correlation), and `EQ` (equal-accuracy `Q1` arm comparison).
       - **`Unit M1` + `Row R2` (Running on Slurm)**: Tests `M1a` (`t9_sft_0..2_70` at decoding $T=0.2$, `3` states), `M1b` (`t9_sft_0..2_70 -> GRPO` at `u83` and `u333`, `6` states), and `M1c` (`t9_sft_0..2_70 -> pass-rate [2..6/8] GRPO` on `6,286–6,529` prompts at `u83` and `u333`, `6` states), evaluated under both `M1` Rules 1–3 and `R2` Predictions `P5, P6, A1, A2, A3`.

---

### Tier 2: Nov 20, 2026 — 40th IRT Workshop ("The Main Course")
- **Locked Title**: *"Evaluating and Training Large Language Models: What Can Item Parameters Tell Us?"*
- **Deliverables**:
  1. Complete `Unit M1` (`M1a, M1b, M1c`, `15` new sampled states) + `Row R1` & `Row R2` person-slope and option-level decomposition.
  2. **`Rows R3, R4, R6, R7` (`plan-within-run.md`)**:
     - `R3`: Gain $\tau_s$ as final RMSNorm weight scale (`P2, P3`) and SFT trace typicality (`P12` on the `8,538` SFT traces).
     - `R4`: Within-run discrimination as trajectory derivative $\tilde{a}_j = J_j u_\perp$ (`P7, P8, C2, C3`).
     - `R6`: Offline KT-IRT person-state tracker $(\delta_t, \tau_t, \eta_t)$ trained on the 12 `Q1` reward logs (`predictions_rank=*.jsonl`).
     - `R7`: Primary-source literature verification (`Ferrando 2019`, `Cui et al. 2025`, `Yue et al. 2025`, `Huang et al. 2024`, `Duan 2026`).

---

### Tier 3: 14-Month PhD Thesis & Flagship Papers
- **Unified Thesis Spine**: *Identifiability and Representation Geometry in Neural Psychometrics: From Human Knowledge Tracing to Large Language Model Post-Training*.
  - **Chapter / Paper 1 (Human KT — `JEDM`)**: *Prediction Is Not Calibration* (Fisher slope-location asymmetry & affine logit absorption in sequential neural encoders).
  - **Chapter / Paper 2 (Dynamic Distractor Manifold — `signature U1a/e/m/n`)**: The `8.24`-quiz half-life of multidimensional human misconceptions (`U1m/n`).
  - **Chapter / Paper 3 (LLM Evaluation, Within-Run Person-Slope Models, & Post-Training Geometry — `irt4llm P2b` + `signature T1/T3/T5/T8/T9` + `item-params-llm S1/Q1/Q2/M1/R1–R7` + `surrogates/theory`)**: Cross-species NRM distractor geometry (`P2b`), white-box subspace vs. item-rank dissociation (`T1/T3/T5/T8`), and person-slope ($\kappa_s$) + within-run discrimination ($\tilde{a}_j = J_j u_\perp$) models of SFT and RL (`T9/S1/Q1/Q2/M1/R1–R6`).
