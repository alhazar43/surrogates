# Slide-by-Slide Presentation Plan (Monday, Oct 26, 2026)

## **Title**: *Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?*

- **Target Audience**: Both **LLM Evaluation / Post-Training Researchers** and **Psychometrics / IRT Researchers**.
- **Central Question of the Talk**:
  > When we evaluate or train an LLM, what **CAN** IRT item parameters ($\text{difficulty } b_i, \text{discrimination } a_i$) reveal or do that standard LLM metrics (`overall accuracy %`, uniform reward) miss — and what **CANNOT** they do?
- **Visual Design Rules Enforced Across All 5 Figures (`gemini/make_oct26_figures.py`)**:
  1. **Every figure shows both `CAN` and `CANNOT`**, directly contrasting what standard LLM evaluation/training sees against what IRT item parameters reveal.
  2. **Zero invented buzzwords**: All labels use standard LLM terms (`accuracy %`, `Mean@16`, `Maj@16`, `Pass@16`, `SFT`, `GRPO`) and standard IRT terms (`difficulty b`, `discrimination a`, `ability \theta`).
  3. **Zero statistical textboxes inside figures**: Axes are in `%`, `percentage points`, `percentile`, `number of questions`, or `IRT difficulty b`, with direct labels on the curves and individual seed dots.

---

## Executive Summary Table: What Item Parameters CAN vs. CANNOT Tell Us

| Dimension | What Standard LLM Eval / Training Sees | **What Item Parameters $(a_i, b_i)$ CAN Tell Us (`+`)** | **What Item Parameters $(a_i, b_i)$ CANNOT Tell Us (`-`)** | Figure & Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **0. Human vs. LLM Calibration** | Assumes human math difficulty transfers to LLMs | **Transfer across human students**: Student 2PL difficulty $b_i$ (`4,918` students) predicts held-out student difficulty (**`Spearman = 0.85`**). | **Cannot transfer to LLMs**: On the same `704` Eedi math questions, human $b_i$ is uncorrelated with `61` LLMs (**`Spearman = 0.09`**, `61/61` LLMs fail person-fit). | **`Fig01`** (`results/E1/`) |
| **1. Evaluating LLMs: Short Adaptive Testing (`CAT`)** | Random subset of $n$ questions (`RMSE` shrinks slowly as $1/\sqrt{n}$) | **Cut short-test error on pretraining checkpoints (`Q2c`)**: Using `102`-LLM 2PL parameters $(a_i, b_i)$, adaptive selection at $n=25$ cuts error by **`35%`** (`6.4%` vs. `9.8%` random; **`25` adaptive items beat `50` random items**). | **Cannot evaluate post-trained SFT models (`Q2a`)**: After SFT, 2PL adaptive selection **reverses and has higher error than random sampling at every $n \in \{25, 50, 100\}$** (`11.4%` vs. `7.3%` at $n=25$; on GRPO it wins at $n=25$ (`7.0%` vs. `8.0%`) before reversing at $n=50, 100$) because it greedily picks high-$a_i$ items where SFT models misfit. | **`Fig02`** (`results/Q2c/`, `results/Q2a/`) |
| **2. Evaluating LLMs: Diagnosing Post-Training (`SFT` vs. `GRPO`)** | Overall MMLU accuracy rises from `39.6%` (Base) to `45.3%` (SFT) and `49.3%` (GRPO) — making GRPO look like "more of SFT" | **Person-Slope Decomposition (`R1` + `S1` + `T9r`)**: GRPO's token-0 key prob rises on `876–894` items and falls on `606–624` (ruling out a 1D ability shift and consistent with a **person slope $\tau_s \approx 1.88$**: `1.86–1.90` on $K=32$ CoT after reliability correction $\text{rel}_0=0.765$, matching `1.87–1.92` on token-0; tested in `R1`), whereas SFT flattens ($\tau_s = 0.545\text{–}0.550$, corrected `0.71–0.72`) while raising modal-key fraction (`0.422–0.439 -> 0.486–0.525` unique, `0.476–0.497 -> 0.535–0.576` tied). | **Cannot collapse evaluation into 1D 2PL $\hat{\theta}$ (`Q2-B`) or raw 15-subject $\bar{b}_m^{\text{pop}}$**: Across the 15 subjects, $\bar{b}_m^{\text{pop}}$ is $-0.94$ collinear with Base log-odds $\overline{L}_{\text{base},m}$ (partial $r = 0.17$), and on the 7 seen `S1` models 1D $\hat{\theta}$ weights high-$a_i$ subjects $5\times$ over math, flipping credit from **`60% SFT / 40% GRPO` (Accuracy) to `21% SFT / 79% GRPO` (2PL $\hat{\theta}$)**. | **`Fig03`** (`results/W0/`, `results/Q2g/s1/`, `results/T9r/`) |
| **3. Training LLMs: Online IRT Critics in RL (`Unit Q1`)** | Standard GRPO samples `5` draws/question; `Mean@16` accuracy plateaus at `46.9%` | **1. Match accuracy with `0` pre-samples**: 1-draw online 2PL critics (`pop0: 46.9%`, `pop1: 47.5%`) match 5-draw GRPO (`46.9%`) and the pass-rate critic (`ctt: 47.2%`) **without needing `ctt`'s `85,456`-rollout pre-calibration pass (`W1`)**.<br>**2. Diagnose step-83 saturation (Seed 0)**: On `grpo_0`, `83.7%` of RL gain finishes by step `83` (`83.2%–107.1%` across the 4 seed-0 arms) as zero-variance groups rise from `33.2%` to `63.1%`. | **1. Cannot raise the `47%` RL accuracy level**: Weighting updates by static leaderboard $a_i$ (`pop1`) ties unweighted critics (`all 12 runs in 46.9%–47.5%`).<br>**2. Worsens answer diversity collapse**: Without GRPO's 5-draw zero-variance shutoff gate, 1-draw critics collapse order-averaged `Pass@16` (`results/Q1/stats.csv` `pooled`) from **`93.0%` (Base) and `69.4%` (`grpo`) down to `58.7%` (`pop1`), `58.6%` (`ctt`), and `52.4%` (`pop0`)**. | **`Fig04`** (`results/Q1/`, `results/W1/`) |
| **4. Combining Eval & Training: Within-Model Pass Rates (`S1` $\to$ `Unit M1`)** | Evaluates only single-draw `Mean@16` and trains RL uniformly from Base | **1. Reveal the `Mean@16` vs. `Maj@16` & Modal-Key Crossover (`S1`)**: On letters-only (`stability.csv`), Base's modal-key fraction is `42.2%–43.9%` unique (`47.6%–49.7%` with tied modes counted as key, matching GRPO's `47.6%–49.8%`), whereas **SFT raises modal-key fraction to `48.6%–52.5%` unique (`53.5%–57.6%` tied) and order-averaged `Pass@16` to `96.6%`**.<br>**2. Guide `Unit M1` & `Row R2`**: Filtering to SFT's `2..6 of 8` pass-rate band (`6,286–6,529` prompts) removes $p_{\text{key}} < 1/4$ prompts and is predicted (`P5`) to raise $\log \tau_c / \log \tau_b$ (`1.18–1.30`). | Static cross-model item parameters cannot identify which questions have mixed rewards or positive keyed margins for a specific model checkpoint; **within-model** pass rates and option distributions are required. | **`Fig05`** (`results/S1/`, `results/W1/`) |

---

## Slide-by-Slide Plan (10 Slides, ~25 Minutes)

### **Slide 1 — Title & The Central Question**
- **Title**: *Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?*
- **Core Premise**:
  - In psychometrics, **Item Response Theory (2PL IRT)** characterizes each test question $i$ by two parameters: **difficulty $b_i$** (where the 50% success point sits) and **discrimination $a_i$** (how steeply success rises with ability $\theta$):
    $$P(\text{correct}_{m,i}) = \frac{1}{1 + \exp\!\big(-a_i(\theta_m - b_i)\big)}$$
  - Recent LLM benchmarks (*Fluid Benchmarking*, *tinyBenchmarks*) and RL training papers (*SEELE*, *ZPD Detector*) plug static item parameters $(a_i, b_i)$ directly into LLM evaluation and RL training.
  - **Our Question**: Across **`4,918` human students + `61` LLMs (`E1`)**, **`102` leaderboard LLMs + `452` pretraining checkpoints (`Q2c`)**, and **`19` post-trained SFT & RL models (`1.48M` Chain-of-Thought rollouts in `S1` & `Q1`)**, what **can** item parameters tell us that standard LLM metrics miss — and where do they **fail**?

---

### **Slide 2 — Act 1 (`CAN` vs. `CANNOT`): Human Question Difficulty Transfers Across Students, Not to LLMs (`Unit E1`)**
- **Figure**: **`Fig01_human_vs_llm_difficulty.png`**
- **What the Figure Shows**:
  - **Left Panel (`CAN` — Student-to-Student Transfer)**: On `704` Eedi school math questions calibrated on `4,918` human students (`1.12M` responses), question difficulty percentiles transfer cleanly to held-out student groups (**`Spearman = 0.85`**).
  - **Right Panel (`CANNOT` — Student-to-LLM Breakdown)**: On the exact same `704` math questions, human student difficulty is flat and uncorrelated with `61` open-weight LLMs (`0.1B` to `70B+`, **`Spearman = 0.09`, `Pearson = -0.013`**, and **`61 / 61` LLMs fail person-fit** with median $Z_h = -6.12$).
- **Takeaway**: Human difficulty parameters cannot evaluate LLMs. To test whether item parameters help LLM evaluation and training, we calibrate 2PL parameters $(a_i, b_i)$ directly on **`102` Open LLM Leaderboard models** across **`1,472` MMLU questions (`15` subjects, `Unit W0`)**.

---

### **Slide 3 — Act 2A (`CAN` vs. `CANNOT` in Evaluation): Extending $n$ from `10` to All `1,472` Items Shows 2PL Adaptive Fails by Asymptotic Bias, Not Initialization (`Q2c` vs. `Q2a`)**
- **Figure**: **`Fig02_adaptive_testing_can_vs_cannot.png`** (`n = 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1472`)
- **Refuting the *"Is $n=100$ just initialization for an LLM?"* Critique (`Left Panel — Full n = 10..1,472 RMSE Curve`)**:
  - **At tiny $n \le 50$ (`Small-n / Prior Regime`), 2PL IRT actually *helps***: On the `165` pretraining checkpoints (`Q2c`), maximum-information 2PL selection cuts RMSE by **`35%` at $n = 25$ (`6.4%` vs. `9.8%`)** and still leads at $n = 50$ (`6.4%` vs. `7.0%`), because the prior and high-$a_j$ items compress $1/\sqrt{n}$ sampling variance.
  - **Past the Crossover ($n \ge 100$), IRT loses—and extending $n \to 1,472$ makes IRT lose *harder***:
    - Even on the `165` pretraining checkpoints (`Q2c`), 2PL adaptive already **loses to random sampling at $n = 100$ (`6.26%` vs. `4.97%`, ratio `1.26x`)**.
    - Across the `7` post-trained models (`Base`, `3 SFT`, `3 GRPO` on the `1,472`-item MMLU bank, `k=16`), unweighted random sampling drops along $1/\sqrt{n}$ (`10.9% -> 6.7% -> 4.7% -> 3.3% -> 2.2% -> 1.7% -> 1.3% -> 0.4%` at $n=1,472$), **beating BOTH `irt_adaptive` and `irt_random` on `7 / 7` (`100%`) of models for all $n \ge 300$**!
    - Meanwhile, `irt_adaptive` and `irt_random` stay stuck above a **`2.85%` asymptotic error floor even when all $n = 1,472$ items (`100%` of the bank) are administered** (`+2.38%` SFT, `+2.91%` Base, `+3.29%` GRPO).
- **Why Larger $n$ Cannot Rescue 2PL IRT (`Right Panel — Systematic |Bias| vs. Sampling SD`)**:
  - **Not an initialization issue**: `2PL Adaptive` sampling SD drops to **`< 0.4%` by $n = 100$** (`0.19%` at $n=1,472$)—the posterior $\hat{\theta}_n$ has completely converged.
  - **Two stacked bias mechanisms**:
    1. **Fisher Selection Bias (`+4%` to `+8%` for $n = 25\text{–}500$)**: Greedy Fisher selection ($a_j^2 p_j(1-p_j)$) picks high-$a_j$ easy factual subjects first (*Marketing*, *Sociology*), skewing $\hat{\theta}_n$ away from the full-bank mean.
    2. **2PL Parametric Misspecification Floor (`2.85%` at $n = 1,472$)**: Even when all `1,472` items are observed, mapping $\hat{\theta}_{1472}$ through the 1D cross-model curve $\frac{1}{J}\sum_{j=1}^J \sigma(a_j(\hat{\theta} - b_j))$ does not equal the respondent's true bank average.

---

### **Slide 4 — Act 2B (`CAN` in Evaluation): Plotting Gains Against IRT Difficulty $b_i$ Reveals Opposite SFT vs. GRPO Learning Profiles (`W0 + S1 + Q2g`)**
- **Figure**: **`Fig03_item_params_reveal_sft_vs_grpo.png` (Left Panel)**
- **What Standard LLM Eval Sees**:
  - On the `1,472`-item MMLU bank (`32` CoT draws/item at $T=1.0$), overall accuracy rises from **`39.6%` (Base)** to **`45.3%` (SFT, `+5.8%`)** and **`49.3%` (GRPO, `+9.7%`)**. (Note: `~35–40%` of SFT's `Mean@16` gain at $T=1.0$ comes from format/parse repair [`parse_rate: 90.3% -> 95.8%`], and on greedy `T9` decoding [`signature/T9`], `100%` of SFT's `+5.01%` raw gain is parse repair [`gain_parsed = -0.55%`] vs. GRPO's `+2.51%` parsed gain!)
- **What Item Parameters CAN Tell Us**:
  - When we plot each subject's accuracy gain against its **mean IRT difficulty $\bar{b}_m$ and discrimination $\bar{a}_m$** across the `15` MMLU subjects, SFT and GRPO form an **X-shaped crossover (`r = +0.85` in `%` gain, `r = +0.88 / R² = 76.9%` in log-odds)**:
    1. **GRPO (Outcome RL) wins on `8 / 8` low-difficulty, high-discrimination subjects ($\bar{b}_m < +0.75, \bar{a}_m \in [0.69, 1.45]$)**: *Sociology (`+28.9%` vs. `+7.1%` SFT), World History (`+25.8%` vs. `-0.6%`), Marketing (`+20.6%` vs. `+1.2%`), Biology (`+17.6%` vs. `-2.1%`)*.
    2. **SFT (70B CoT Distillation) wins on `5 / 6` high-difficulty, low-discrimination quantitative subjects ($\bar{b}_m > +1.0, \bar{a}_m \in [0.28, 0.55]$)**: *Elementary Math (`+38.3%` vs. `+3.8%` GRPO), High School Math (`+23.8%` vs. `+7.1%`), Macroeconomics (`+7.3%` vs. `-4.2%`), Physics (`+6.3%` vs. `-0.6%`), Econometrics (`-0.2%` vs. `-9.4%`)*.

---

### **Slide 5 — Act 2C (`CANNOT` in Evaluation): Why Collapsing 2PL Into a Single 1D Ability Score $\hat{\theta}$ Flips SFT vs. GRPO Credit (`Q2-B` & `Q2-A`)**
- **Figure**: **`Fig03_item_params_reveal_sft_vs_grpo.png` (Right Panel)** + **Table 1**
- **Why 1D 2PL Ability $\hat{\theta}$ Distorts Post-Training Credit**:
  - Across the 102 leaderboard LLMs, factual recall scales tightly with model size (**high discrimination $a_i$, e.g., *Marketing* $\bar{a} = 1.45$**), whereas multi-step math is noisy across base models (**low discrimination $a_i$, e.g., *High School Math* $\bar{a} = 0.28$**; across the 15 subjects, $\operatorname{Corr}(\bar{a}_m, \bar{b}_m) = -0.88$).
  - In the 1D 2PL maximum-likelihood score equation $\sum_i a_i(y_i - P_i(\theta)) = 0$, each question counts in proportion to $a_i$ — so **one *Marketing* question (`a = 1.45`) counts like five *High School Math* questions (`a = 0.28`)**!
  - Because SFT's gains sit on low-$a_i$ math subjects ($\rho = -0.17$) and GRPO's gains sit on high-$a_i$ subjects ($\rho = +0.14$), collapsing evaluation into 1D $\hat{\theta}$ flips the credit:

#### **Table 1: Classical Accuracy vs. 1D 2PL Ability $\hat{\theta}$ on the 1,472 Matched MMLU Questions (`results/Q2g/s1/personfit.csv`)**

| Model State (3-Seed Mean, 32 Draws/Item) | Unweighted Accuracy (%) | Accuracy Gain Over Previous Step | Share of Total Accuracy Gain | 1D 2PL Ability $\hat{\theta}$ | $\hat{\theta}$ Gain Over Previous Step | Share of Total $\hat{\theta}$ Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (`Llama-3.2-1B-Instruct`)** | `39.6%` | — | — | `-1.21` | — | — |
| **After CoT SFT (`t9_sft`, 3 seeds)** | `45.3%` | **`+5.8%`** | **`60%`** (`8.9–10.5 SE`) | `-1.05` | `+0.16` | **`21%`** (`2.9–5.1 SE`) |
| **After GRPO (`t9_rl`, 3 seeds)** | `49.3%` | **`+3.9%`** | **`40%`** (`3.8–5.9 SE`) | `-0.43` | **`+0.62`** | **`79%`** (`8.6–10.5 SE`) |

- **Additional `CANNOT` at the Single-Item Level (`Q2-A`)**: Individual 2PL item residuals $|c_i|$ do not repeat across training seeds any better than raw pass-rate changes (`65.4%` vs. `64.5%` top-10% overlap; `0/6` seed pairs in `S1`, `1/12` in `Q1`) because swapping multiple-choice option order (`Order 0` vs. `Order 1`) adds **$16\times$ binomial sampling variance**, swamping single-item drift.

---

### **Slide 6 — Act 3A (`CAN` vs. `CANNOT` in Training): Online IRT Critics in RL Training (`Unit Q1`)**
- **Figure**: **`Fig04_rl_training_can_vs_cannot.png`** + **Table 2**
- **Experimental Design (`Unit Q1`, `12` RL runs = `4` methods $\times$ `3` seeds, `333` updates of `160` answers/update on `10,682` MMLU training questions)**:
  1. **`grpo` (Reference, `5` draws $\times$ `32` questions)**: Group-normalized reward $(r_{ik} - \bar{r}_i)/\hat{\sigma}_i$.
  2. **`ctt` (Classical pass-rate critic, `1` draw $\times$ `160` questions)**: $r_i - \sigma(\operatorname{logit} p_{0,i} + \delta_t)$, using Base pass rates $p_{0,i}$ from **`85,456` pre-sampled rollouts (`Unit W1`)**.
  3. **`pop0` (Unweighted 2PL IRT critic, `1` draw $\times$ `160` questions)**: $r_i - \sigma(a_i(\theta_t - b_i))$, requiring **`0` pre-samples**.
  4. **`pop1` ($a_i$-weighted 2PL IRT critic, `1` draw $\times$ `160` questions)**: $w_i(a_i)[r_i - \sigma(a_i(\theta_t - b_i))]$, requiring **`0` pre-samples**.
- **What Item Parameters CAN and CANNOT Do in RL Training (`Fig04`)**:
  - **Left Panel (`CAN` Match Accuracy With `0` Pre-Samples & Diagnose Step-83 Saturation; `CANNOT` Beat the `47%` Level)**:
    - **`CAN`**: `pop0` (`46.9%`) and `pop1` (`47.5%`) match `ctt` (`47.2%`) and `grpo` (`46.9%`) **without needing the `85,456`-draw `W1` pre-calibration pass**, and on seed 0 (`grpo_0`), **`83.7%` of the final gain finishes by update `83`** (`83.2%–107.1%` across the 4 seed-0 arms) as zero-variance groups rise from **`33.2%` at step `0` to `63.1%` after step `83`**.
    - **`CANNOT`**: Weighting updates by static leaderboard discrimination $a_i$ (`pop1`) does not beat unweighted critics (`all 12 runs finish inside a 0.6% band: 46.9%–47.5%`).
  - **Right Panel (`CAN` Reveal Hidden Mode Collapse in 1-Draw Critics)**:
    - While single-draw accuracy `Mean@16` looks identical (`46.9%–47.5%`), **order-averaged `Pass@16` (`results/Q1/stats.csv` `pooled`) collapses from `93.0%` (Base) to `69.4%` (`grpo`), `58.7%` (`pop1`), `58.6%` (`ctt`), and `52.4%` (`pop0`)** (`stable_12` rises from `26.9%` to `75.7%` $\to$ `86.6%` $\to$ `88.0%` $\to$ `94.2%`, a **`+20 SE` difference**), because 5-draw `grpo` automatically shuts off updates ($A=0$) when all 5 draws agree, whereas 1-draw critics keep pushing $r_i - \hat{p}_i \neq 0$ on every step (`pop1`'s $a_i$-weighting downweights low-$a_i$ items and preserves `6.3%` more order-averaged `Pass@16` than unweighted `pop0`, `58.7%` vs. `52.4%`).

#### **Table 2: Summary of All 12 Pre-Registered RL Runs in `Unit Q1` (`results/Q1/stats.csv`, `contrasts.csv`)**

| RL Training Method (`3` Seeds Each, `333` Updates) | Draws per Question ($G$) | Pre-Sampling Required | Update-83 `Mean@16` (Seed 0) | Final `Mean@16` (`3`-Seed Mean) | Mode Lock-In `stable_12` ($\ge 12/16$ Same) | Order-Averaged `Pass@16` (`pooled`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`grpo` (Group-Normalized Reference)** | `5` (`shutoff gate`) | **`0` draws** | `45.5%` | **`46.9%`** | `75.7%` | **`69.4%`** (`68.6%–70.9%`) |
| **`ctt` (Classical Pass-Rate Critic)** | `1` (no shutoff) | `85,456` draws (`W1`) | `46.7%` | **`47.2%`** | `86.6%` | `58.6%` (`56.7%–61.9%`) |
| **`pop0` (Unweighted 2PL IRT Critic)** | `1` (no shutoff) | **`0` draws** | `47.4%` | **`46.9%`** | **`94.2%`** | **`52.4%`** (`52.4%–52.5%`) |
| **`pop1` ($a_i$-Weighted 2PL IRT Critic)** | `1` (no shutoff) | **`0` draws** | `47.2%` | **`47.5%`** | `88.0%` | `58.7%` (`58.3%–59.0%`) |

---

### **Slide 7 — Act 3B (Synthesis): Why All 15 `Base -> RL` Runs Plateau at `46.9%–47.9%`, While `SFT` Raises `Maj@16` & Modal-Key Fraction (`Unit S1`)**
- **Figure**: **`Fig05_sampling_crossover_and_m1.png` (Left Panel)** + **Table 3**
- **What Comparing `Mean@16` vs. `Maj@16`, Modal-Key Fraction, and Order-Averaged `Pass@16` Reveals (`Fig05 Left`)**:
  - Why did **all `15` `Base -> RL` runs (`3` in `S1` at 666 updates + `12` in `Q1` at 333 updates)** plateau at **`46.9%–47.9%`**?
  - Because outcome RL (`GRPO`) acts primarily via a person slope ($\tau_s / \text{rel}_0 = 1.86\text{–}1.90$) that collapses within-question answer diversity (`stable_12` jumps from `26.9%` to `82.0%`), sharpening the Base model's existing distribution (`Base` modal-key fraction on letters is `42.2%–43.9%` unique and `47.6%–49.7%` with tied modes counted as key, matching `GRPO`'s `47.6%–49.0%` unique and `48.2%–49.8%` tied):
    - For **GRPO**, single-draw accuracy (**`Mean@16 = 47.9%`**) is already equal to 16-draw majority vote (**`Maj@16 = 48.1%` strict / `49.0%*` tie-correct**), and order-averaged **`Pass@16` drops from `93.0%` to `65.6%`**.
    - For **SFT**, single-draw accuracy is lower (**`Mean@16 = 43.9%`**) because answer diversity stays wide (`stable_12 = 24.7%`, $\tau_s / \text{rel}_0 = 0.71\text{–}0.72$), but **majority voting over 16 draws jumps by `+6.5% to +11.6%` to `Maj@16 = 50.4%` (`52.9%` random tie / `55.5%*` tie-correct; letters-only modal-key fraction `48.6%–52.5%` unique / `53.5%–57.6%` tied) and order-averaged `Pass@16 = 96.6%` — overtaking GRPO!**

#### **Table 3: Single-Draw vs. Majority-Vote & Coverage Across `S1` (`1,500` Questions $\times$ `32` Draws, `results/S1/stats.csv`)**

| Model State (`3`-Seed Mean) | Mode Lock-In `stable_12` | Rollout Entropy (nats) | Single Draw `Mean@16` | Majority Vote `Maj@16` (`Strict` / `Random Tie` / `Tie-Correct`*) | Voting Boost (`Maj - Mean`) | Order-Averaged `Pass@16` (`pooled`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (`t9_base`)** | `26.9%` | `0.96` | `38.2%` | `41.4%` / `44.0%` / `45.9%*` | `+3.2%` (`+7.7%*`) | `93.0%` |
| **GRPO (`t9_rl_0..2`, 666 updates)** | **`82.0%`** | **`0.28`** | **`47.9%`** | `48.1%` / `48.5%` / `49.0%*` | **`+0.2%`** (`+1.1%*`) | `65.6%` |
| **CoT SFT (`t9_sft_0..2`, 70 steps)** | `24.7%` | `0.94` | `43.9%` | **`50.4%` / `52.9%` / `55.5%*`** | **`+6.5%`** (**`+11.6%*`**) | **`96.6%`** |

---

### **Slide 8 — Putting the Diagnosis Into Action (`Unit M1` & `Row R2`): Composing `SFT -> RL` & Targeting the Model's Own Pass-Rate Band**
- **Figure**: **`Fig05_sampling_crossover_and_m1.png` (Right Panel)** + **Table 4**
- **Two Direct Prescriptions From Our Findings (`Unit M1` & `Row R2`)**:
  1. **Start RL from `SFT` instead of `Base` (`M1b: SFT -> GRPO`)**:
     - `Base -> RL` plateaus at `47.9%` (matching Base's tied modal-key fraction `47.6%–49.7%`).
     - `SFT` already raised `Maj@16` to **`50.4%–55.5%`** (`48.6%–52.5%` unique / `53.5%–57.6%` tied modal-key fraction; order-averaged `Pass@16 = 96.6%`), leaving a **`+6.5% to +11.6%` gap** between single-draw accuracy (`43.9%`) and majority vote. Running RL starting from `t9_sft` tests whether RL sharpening converts SFT's `Maj@16` advantage into single-draw accuracy (`A1`).
  2. **Replace static leaderboard $(a_i, b_i)$ with the model's own pass-rate band (`M1c: SFT Pass Rate 2..6 of 8`, `6,286–6,529` prompts)**:
     - In 5-draw GRPO, a question produces a non-zero gradient only when the 5 draws have mixed rewards, which occurs with probability **$1 - p_i^5 - (1-p_i)^5$ (`Fig05 Right`)**.
     - On `W1` (`10,682` training questions), **`35.0%` (UMVUE) of $G=5$ groups at step 0** (`25.9%` all-wrong + `9.1%` all-right) — and **`63.1%` after step 83** — produce **zero gradient**.
     - Furthermore, under the 4-option model (`R2 / P5`), prompts with $p_{\text{key}} < 1/4$ (`0/8` and `1/8`) have negative keyed margins and push the person slope $\tau$ *down*; filtering to `2..6 of 8` (`M1c`) is therefore predicted to **raise** $\log \tau_c / \log \tau_b$ (`1.18 to 1.30`)!

#### **Table 4: Pre-Registered 3-Arm Design of `Unit M1` (`notes/units/M1.md`)**

| Arm in `Unit M1` (`3` Seeds Each on `t9_sft_0..2_70`) | Training / Inference Protocol | What It Tests (`+1` vs. `-1` Outcome) |
| :--- | :--- | :--- |
| **`M1a`: Inference Temperature Scaling (`T = 0.2`)** | Sample `t9_sft_0..2_70` at decoding temperature `T = 0.2` (`0` RL updates) | Tests whether lowering token temperature alone harvests SFT's majority-vote ceiling at **zero training cost** (`-1`), or whether trajectory-level RL updates are required (`+1`). |
| **`M1b`: Sequential Composition (`SFT -> GRPO`)** | Run `GRPO` (`333` updates, checkpoint at update `83`) starting from `t9_sft_0..2_70` on all `10,682` training questions | **`+1`**: `Mean@16` breaks the 15-run `47.9%` wall (`> 2.92 SE_d` over both `t9_sft` and `q1_grpo` in `3/3` seeds).<br>**`-1`**: Short-answer RL (`~113` tokens) overwrites SFT's `310`-token math CoT traces and drops `Maj@16`. |
| **`M1c`: Model-Own Pass-Rate Targeted RL (`SFT -> Targeted GRPO`)** | Filter training questions to each `t9_sft` checkpoint's own `20%–80%` pass-rate band (`2..6` of `8` correct); run `GRPO` (`333` updates, save at `83`) | Tests whether **within-model** pass-rate targeting beats unfiltered `M1b` (`+1`), turning `Q1`'s negative result on *cross-model* item parameters into a positive result for *within-model* item parameters. |

---

### **Slide 9 — Takeaways & Roadmap to the Nov 20 IRT Workshop**
- **Direct Answer to the Title**:
  1. **What Static Item Parameters CANNOT Do**:
     - Human student difficulty $b_i$ **cannot** transfer to LLMs (`Spearman = 0.09`).
     - Cross-model 102-LLM 2PL parameters $(a_i, b_i)$ **cannot** be used as a 1D score replacement (`2PL \theta` flips SFT/GRPO credit from `60%/40%` to `21%/79%`), **cannot** do short adaptive testing after SFT/GRPO (`Q2a`), and **cannot** raise the RL accuracy ceiling when used as static reward weights (`Q1: 46.9%–47.5%`).
  2. **What Item Parameters & Within-Model Pass Rates CAN Do**:
     - **Cut short-test error by `35%` ($n=25$)** on pretraining checkpoints (`Q2c`) from the calibration distribution.
     - **Decompose post-training in 2D $(b_i, a_i)$**: Reveal that GRPO amplifies low-difficulty, high-discrimination subjects (`+18% to +29%`) while SFT acquires high-difficulty, low-discrimination math/STEM subjects (`+24% to +38%`).
     - **Diagnose RL saturation & diversity collapse**: Explain why `>85%` of RL gain finishes by step `83`, why 1-draw critics collapse `Pass@16` (`93% -> 52%`), and why combining `SFT` (which raises `Maj@16` to `50.4%–55.5%`) with within-model pass-rate targeting (`Unit M1`) is the principled path past the `47.9%` RL ceiling.
