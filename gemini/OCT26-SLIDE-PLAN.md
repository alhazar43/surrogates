# Story-Driven Slide, Figure & Table Plan for Monday, Oct 26 Talk

- **Locked Title**: *Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?*
- **Design Principles**:
  1. **Storytelling First (4-Act Detective Arc)**: Every slide and visual answers the question raised by the previous slide and sets up the question for the next slide (**Puzzle $\rightarrow$ Clues & Mechanism $\rightarrow$ Multi-Draw Twist $\rightarrow$ Training Resolution**).
  2. **One Figure = One Visual Intuition (`Fig01`–`Fig11`)**: Eliminated all forced `1×2` unrelated subplots. Every figure is a single-panel visual with generous whitespace and a 3-second takeaway.
  3. **Numbers That Don't Build Visual Intuition Live in Clean Tables (`Table 1`–`Table 4`)**: Multi-modality slope decompositions, 19-state metrics, person-fit estimators ($l_z^{(1)}$ vs. $l_z^{(32)}$), $G=5$ vs. $G=1$ RL critic diagnostics, and the pre-registered `Unit M1` matrix are presented as clean, scannable Markdown tables rather than crammed into figure text boxes.
  4. **100% Audit-Proof (`review-oct26-slide-plan.md`)**: [`gemini/make_oct26_figures.py`](file:///D:/surrogates/gemini/make_oct26_figures.py) resolves `results/` portably (`Path(__file__).resolve().parent.parent / "results"`), contains zero hard-coded data arrays, obeys Rule 25 (OLS slope/intercept and identity line $y=x$ on agreement/scaling scatters), and states exact scopes (`J = 704`, `seen S1 states`, `Seed 0 u83`, `Unit M1 running`).

---

## Part 1: Visual & Table Inventory (`Fig01`–`Fig11` + `Table 1`–`Table 4`)

| Asset | Type | Single-Concept Visual / Table Focus | Committed Source Files (`results/`) | Slide |
| :---: | :---: | :--- | :--- | :---: |
| **`Fig01`** | **Single Plot** (`Fig01_human_vs_llm_difficulty.png`) | **Human Student vs. 61-LLM Difficulty Rank Percentiles on `J = 704` Eedi Math Items**: Uniform cloud around flat OLS line (Spearman $\rho = +0.089$, rank slope $+0.089$ vs. $y=x$; raw $b_j$ Pearson $r = -0.013$, OLS slope $-0.76$; `61/61` LLMs have $Z_h < -1.96$; per `closure.md:29`, raw LLM $b_j \in [-1574, +912]$ rests on the rejected 61-LLM fit so rank order reads). | `E1/difficulty.csv`<br>`E1/llm_scores.csv` | Slide 3 (Act I) |
| **`Fig02`** | **Single Plot** (`Fig02_attribution_puzzle.png`) | **The Central Attribution Puzzle (`Base -> SFT -> GRPO` on Seen `S1` States)**: Classical Accuracy attributes **`59.5%` to SFT (`+0.058`) / `40.5%` to GRPO (`+0.039`)**, whereas 102-LLM 2PL $\hat{\theta}$ flips credit to **`20.9%` SFT (`+0.164`) / `79.1%` GRPO (`+0.620`)** (`parsed`: `47/53` vs. `15/85`; unseen `Q1` vs. Base: $\hat{\theta}$ sharper in `3/12`). | `Q2g/s1/personfit.csv`<br>`Q2g/s1/precision.csv`<br>`Q2g/q1/precision.csv` | Slide 4 (Act I) |
| **`Fig03`** | **Single Plot** (`Fig03_mmlu_subject_irt_map.png`) | **Clue #1 — The 15 MMLU Subjects in the $(\bar{a}_m^{\text{pop}}, \bar{b}_m^{\text{pop}})$ Plane**: Across $n = 15$ subjects ($r = -0.8841, R^2 = 78.2\%$, OLS slope $-2.72$, intercept $+2.41$), declarative subjects have **`5.2x` higher discrimination $\bar{a}_m^{\text{pop}}$** (*Marketing `1.45`*) than quantitative math (*HS Math `0.28`*). | `W0/items_bank.csv` | Slide 5 (Act II) |
| **`Fig04`** | **Single Plot** (`Fig04_subject_sft_vs_grpo_gains.png`) | **Clue #2 — SFT vs. GRPO Subject Gains Aligned with `Fig03`**: Diverging horizontal bars of $\overline{\Delta L}_{\text{SFT},m} - \overline{\Delta L}_{\text{GRPO},m}$ sorted by $\bar{a}_m^{\text{pop}}$: **GRPO leads `8/8` high-$\bar{a}_m$ subjects** (`0.69–1.45`), while **SFT leads `5/6` low-$\bar{a}_m$ subjects** (`0.28–0.55` at $\pm 0.1$ logit; *HS Stats `-0.08`* tie, *Microecon `-0.08`* in middle; $R^2(\bar{b}_m^{\text{pop}}) = 76.9\%$, slope $+0.68$). | `W0/items_bank.csv`<br>`Q2g/s1/flags.csv`<br>`Q2g/s1/personfit.csv` | Slide 6 (Act II) |
| **`Fig05`** + **`Table 1`** | **Single Plot + Table** (`Fig05_within_model_prior_scaling.png`) | **Within-Model Prior Scaling ($L_{s,j}$ vs. $L_{\text{base},j}$) & Confound Check**: Plots $L_{s,j}$ vs. $L_{\text{base},j}$ against $y=x$ (`T9r` Seed 0, $\varepsilon = 10^{-4}$): GRPO stretches $L_{\text{base},j}$ ($\hat{\tau}_0 = 1.87 > 1$), while SFT compresses/lifts weak items ($\hat{\tau}_0 = 0.54 < 1$). **Table 1** gives all 3 modalities (`Token-0`, `K=32`, `Free-Text CoT`), $r(a_j^{\text{pop}}, \Delta L_j)$, and the $\bar{b}_m^{\text{pop}}$ vs. $\overline{L}_{\text{base},m}$ ($r = -0.94$, partial $r = 0.17$) confound check. | `T9r/items_wide.csv`<br>`Q2g/s1/residuals.csv`<br>`T9g/s3/matched.csv`<br>`W0/items_bank.csv` | Slide 7 (Act II) |
| **`Fig06`** | **Single Plot** (`Fig06_cat_sociology_bias.png`) | **Consequence for Short-Test CAT (`Q2c` & `Q2a`)**: In `Q2c` ($>0.45$ band), greedy max-info CAT (`irt_adaptive`) picks `40%–49%` Sociology (`K.md:47-51`), flatlining at `~0.063` (bias `+0.042 to +0.058`) and losing to $1/\sqrt{n}$ `classical` random sampling at $n = 100$ (`0.050 vs. 0.063`; `19/21` cells in `Q2a`). | `Q2c/errors.csv`<br>`Q2a/cells.csv` | Slide 8 (Act II) |
| **`Fig07`** | **Single Plot** (`Fig07_s1_readout_crossover.png`) | **Act III — The Multi-Draw Readout Crossover (`S1`)**: Across `Mean@16 -> Maj@16 -> Maj@16* -> Pass@16`, GRPO leads `Mean@16` (`0.479 vs. 0.439`) via low entropy ($H = 0.281$), but **SFT overtakes on `Maj@16`** (`0.504 vs. 0.481` in `2/3` seeds; `0.555* vs. 0.490*` tie-correct) and preserves `Pass@16` (`0.966`, `+0.036` over Base `0.930`; GRPO drops to `0.656`). | `S1/stats.csv`<br>`S1/rewrite1_parsed.csv` | Slide 9 (Act III) |
| **`Fig08`** + **`Table 2`** | **Single Plot + Table** (`Fig08_19states_entropy_vs_majority.png`) | **All 19 States in the $(H_s, \text{Maj@16})$ Plane + Table of Readouts & Person Fit $l_z$**: All 15 `Base -> RL` runs cluster at low entropy with `Maj@16` in `0.467–0.496`, while all 3 `SFT` runs sit at high entropy ($H_s = 0.943$) with `Maj@16 = 0.504 / 0.555*`. **Table 2** compares $l_z^{(1)}$ (`Q2a`) vs. $l_z^{(32)}$ (`Q2g`) across all families. | `S1/stats.csv`<br>`Q1/stats.csv`<br>`Q2a/personfit.csv`<br>`Q2g/s1/personfit.csv`<br>`Q2g/q1/personfit.csv` | Slide 10 (Act III) |
| **`Fig09`** | **Single Plot** (`Fig09_q1_rl_trajectories.png`) | **Act IV — `Unit Q1` RL Trajectories (Seed 0) & 12-Run Convergence Band**: In Seed 0, `83.7%–94.5%` of total gain finishes by update 83, and all 4 arms (`grpo, ctt, pop0, pop1`) land in a **`0.63 percentage point` band (`0.4685–0.4748`)** at update 333 (`12 Q1 runs: 0.459–0.480`). | `Q1/trajectory.csv`<br>`Q1/stats.csv`<br>`S1/stats.csv` | Slide 11 (Act IV) |
| **`Fig10`** + **`Table 3`** | **Single Plot + Table** (`Fig10_grpo_zero_advantage_starvation.png`) | **Why $G=5$ GRPO Stalls After Update 83 (`zero_adv_share = 63.1%`) + $G=5$ vs. $G=1$ Critic Table**: Plots `zero_adv_share` rising from `43.8%` (`W1` raw) to `63.1%` post-u83 (`84%–91%` peak). **Table 3** contrasts $G=5$ `grpo` against $G=1$ critics (`ctt, pop0, pop1`: nonzero $r - \hat{p}_j$, fixed $L_0 = 78.98$, `stable_12 = 0.866–0.942`). | `Q1/traces/grpo_0..2.csv`<br>`W1/summary.json`<br>`Q1/stats.csv` | Slide 12 (Act IV) |
| **`Fig11`** + **`Table 4`** | **Single Plot + Table** (`Fig11_w1_pass_rate_and_zpd.png`) | **Training-Prompt Pass-Rate Distribution & Pre-Registered `Unit M1` Design**: `W1` pass-rate histogram (`51.7%` interior `2..6 of 8` vs. `48.3%` exterior) overlaid with Bernoulli variance $p_j(1-p_j)$, motivating why `Unit M1` (`M1c`) filters `SFT -> RL` by SFT's **own** pass rate $p_{\text{SFT},j} \in [0.2, 0.8]$. **Table 4** gives the 4 committed baseline groups + pre-registered `M1a / M1b / M1c` rules. | `W1/pass_rate.csv`<br>`S1/stats.csv`<br>`Q1/stats.csv`<br>`notes/units/M1.md` | Slide 13 (Act IV) |

---

## Part 2: Slide-by-Slide Narrative Script & Full Tables (14 Slides)

### Act I: The Puzzle — What Happens When We Plug Population Item Parameters $(a_j^{\text{pop}}, b_j^{\text{pop}})$ into LLM Evaluation? (Slides 1–4)

- **Slide 1 — Title Slide**
  - **Title**: *Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?*
  - **Subtitle**: *An Empirical Audit Across 4,918 Human Students, 102 Leaderboard LLMs, and 1.488 Million Post-Training Rollouts*

- **Slide 2 — The Paradigm Under Test**
  - **Story**: In both LLM evaluation (*tinyBenchmarks*, *Fluid Benchmarking*, adaptive testing) and RL post-training (advantage weighting and curriculum filtering), we want **item parameters**: which questions are hard ($b_j$) and which questions are informative ($a_j$).
  - **The Standard Shortcut**: Calibrate static 2PL item parameters $(a_j^{\text{pop}}, b_j^{\text{pop}})$ across an external population of respondents (e.g., human students or 100+ leaderboard LLMs) and treat them as invariant properties of the items when evaluating or training a single target LLM.
  - **Hook to Slide 3**: *First, can item difficulty calibrated on human students transfer to LLMs at all?*

- **Slide 3 — Surprise #1 (`Unit E1`): What Is Hard for Human Students Is Orthogonal to What Is Hard for LLMs**
  - **Visual**: **[`Fig01_human_vs_llm_difficulty.png`](file:///D:/surrogates/gemini/figures/Fig01_human_vs_llm_difficulty.png)**
  - **3-Second Visual Takeaway**: Across **`J = 704` Eedi diagnostic math items**, plotting human 2PL difficulty percentile (`4,918` students) against 61-LLM difficulty percentile produces a uniform cloud with a flat regression line (**Spearman $\rho = +0.089$**, rank OLS slope $+0.089$ vs. the $y=x$ diagonal; raw $b_j$ Pearson $r = -0.013$, OLS slope $-0.76$). Every single LLM (**`61/61`**) falls below the human 2PL person-fit threshold ($Z_h < -1.96$, `E1/llm_scores.csv`). *(Per `closure.md:29`, the raw 61-LLM $b_j \in [-1574, +912]$ rests on the rejected 61-LLM calibration, so only the rank order reads.)*
  - **Hook to Slide 4**: *"Okay, humans $\neq$ LLMs. What if we calibrate $(a_j^{\text{pop}}, b_j^{\text{pop}})$ on **102 LLMs** (`W0`) and use them to evaluate our own 1B model across `Base -> SFT -> GRPO` (`S1`)?"*

- **Slide 4 — Surprise #2 (`Q2-B`): The Attribution Puzzle — Classical Accuracy and 102-LLM 2PL $\hat{\theta}$ Give Opposite Answers!**
  - **Visual**: **[`Fig02_attribution_puzzle.png`](file:///D:/surrogates/gemini/figures/Fig02_attribution_puzzle.png)**
  - **3-Second Visual Takeaway**: Evaluating the exact same `Base -> SFT -> GRPO` states (`S1`, seen states, `all` draws, `Q2g/s1/personfit.csv`):
    - **Classical Mean Accuracy ($\overline{\Delta p}$)** says **`59.5%` of the gain came from SFT (`+0.058`, $8.9\text{–}10.5\text{ SE}$)** and **`40.5%` from GRPO (`+0.039`, $3.8\text{–}5.9\text{ SE}$)**.
    - **102-LLM 2PL Ability ($\hat{\theta}$)** says the exact opposite: **`20.9%` came from SFT (`+0.164`, $2.9\text{–}5.1\text{ SE}$)** and **`79.1%` came from GRPO (`+0.620`, $8.6\text{–}10.5\text{ SE}$)**!
    - *(Scope note: `parsed` variant gives `47/53` vs. `15/85`; `report-q1q2.md:79` withdrew `S1` Rule 2 as a general finding because `S1` states were seen, and on the 12 unseen `Q1` finals vs. Base, $\hat{\theta}$ is sharper than Accuracy in `3/12` runs.)*
  - **Hook to Slide 5**: ***Why does 102-LLM $\hat{\theta}$ assign 79% of the credit to GRPO while Accuracy assigns 60% to SFT? What is $a_j^{\text{pop}}$ actually weighting?***

---

### Act II: Solving the Evaluation Mystery — What $(a_j^{\text{pop}}, b_j^{\text{pop}})$ Measure vs. How SFT and RL Change a Model (Slides 5–8)

- **Slide 5 — Clue #1 (`W0`): 102-LLM Discrimination $a_j^{\text{pop}}$ Weights Factual Recall $5.2\times$ More Heavily Than Math**
  - **Visual**: **[`Fig03_mmlu_subject_irt_map.png`](file:///D:/surrogates/gemini/figures/Fig03_mmlu_subject_irt_map.png)**
  - **3-Second Visual Takeaway**: In the 2PL score equation $\sum_j a_j^{\text{pop}}(y_j - P_j(\hat{\theta})) = 0$, each item's weight is $a_j^{\text{pop}}$. Plotting the $n = 15$ MMLU subjects in the $(\bar{a}_m^{\text{pop}}, \bar{b}_m^{\text{pop}})$ plane ($r = -0.8841, R^2 = 78.2\%$, OLS slope $-2.72$, intercept $+2.41$) shows:
    - **Right side (High $\bar{a}_m^{\text{pop}} = 0.69\text{–}1.45$, Low $\bar{b}_m^{\text{pop}}$)**: Factual/declarative subjects (*Marketing `1.45`, Sociology `1.13`, Prehistory `0.95`, HS Geography `0.90`, World History `0.86`*).
    - **Left side (Low $\bar{a}_m^{\text{pop}} = 0.28\text{–}0.55$, High $\bar{b}_m^{\text{pop}}$)**: Quantitative/calculation subjects (*HS Math `0.28`, HS Physics `0.33`, HS Statistics `0.39`, Elementary Math `0.43`, Econometrics `0.44`*).
    - **Result**: 2PL $\hat{\theta}$ weights a *Marketing* item **$5.2\times$ more heavily** than a *High School Math* item!
  - **Hook to Slide 6**: *Now what happens when we overlay SFT's and GRPO's actual gains across those exact same 15 subjects?*

- **Slide 6 — Clue #2 (`Q2g/s1`): GRPO Leads All 8 High-$\bar{a}_m^{\text{pop}}$ Subjects; SFT Leads 5 of 6 Low-$\bar{a}_m^{\text{pop}}$ Subjects**
  - **Visual**: **[`Fig04_subject_sft_vs_grpo_gains.png`](file:///D:/surrogates/gemini/figures/Fig04_subject_sft_vs_grpo_gains.png)**
  - **3-Second Visual Takeaway**: Ordering the 15 subjects by $\bar{a}_m^{\text{pop}}$ (matching `Fig03`):
    - **Top 8 High-$\bar{a}_m^{\text{pop}}$ (`0.69–1.45`) Factual Subjects**: Every bar goes Left (**GRPO leads `8/8`**, with $r(\bar{a}_m^{\text{pop}}, \overline{\Delta L}_{\text{GRPO},m}) = +0.7464$; unseen `Q1` arms $r = 0.64\text{–}0.72$).
    - **Bottom 6 Low-$\bar{a}_m^{\text{pop}}$ (`0.28–0.55`) Quantitative Subjects**: 5 of 6 bars go Right (**CoT SFT leads `5/6`** at a post-hoc $\pm 0.1$ logit threshold: *Elem Math `+1.49` logits, HS Math `+0.77`*, while *HS Statistics `-0.08`* is a tie and *HS Microeconomics `-0.08`, $\bar{a}_m = 0.61$* sits between the two groups).
    - **Mystery Solved**: Across the 15 subject means (`S1` seen states), $\bar{b}_m^{\text{pop}}$ explains **$R^2 = 76.9\%$ ($r = +0.8771$, OLS slope $+0.68$)** and $\bar{a}_m^{\text{pop}}$ explains **$R^2 = 64.2\%$ ($r = -0.8015$)** of $\overline{\Delta L}_{\text{SFT},m} - \overline{\Delta L}_{\text{GRPO},m}$. At the item level (`Q2g/s1/flags.csv`), the Spearman correlation `A3_spearman_c_a` between the IRT residual $c_{s,j}$ and $a_j^{\text{pop}}$ is **negative for SFT ($\rho = -0.184\text{ to }-0.160$)** and **positive for GRPO ($\rho = +0.134\text{ to }+0.149$)**—so 2PL $\hat{\theta}$ downweights SFT's math gains and amplifies GRPO's factual gains!
  - **Hook to Slide 7**: *Wait—is that subject pattern caused by population difficulty $b_j^{\text{pop}}$, or by the model's OWN initial base log-odds $L_{\text{base},j}$?*

- **Slide 7 — Deep Dive (`T9r`, `Q2g`, `T9g`): The Confound Check & Within-Model Prior Scaling**
  - **Visual**: **[`Fig05_within_model_prior_scaling.png`](file:///D:/surrogates/gemini/figures/Fig05_within_model_prior_scaling.png)** + **Table 1**
  - **3-Second Visual Takeaway (`Fig05`)**: Plotting post-training item log-odds $L_{s,j}$ against the model's **own** Base log-odds $L_{\text{base},j}$ (`J = 1,472` items) against the $y=x$ no-change diagonal reveals how the two operators act on the model's prior:
    - **GRPO (Red line, $\hat{\tau} > 1$)**: Steeper than $y=x$—stretches the model's existing prior (items where Base was already positive get pushed higher; items where Base was negative get pushed lower).
    - **CoT SFT (Blue line, $\hat{\tau} < 1$)**: Flatter than $y=x$—lifts the weak/hard items on the left where $L_{\text{base},j} < 0$.
  - **Companion Table 1 (Replacing Cluttered Subplots)**:

| Readout Modality & Operator | Source File (`results/`) | Per-Seed $r(L_s, L_{\text{base}})$ | Dispersion Ratio $\sigma(L_s)/\sigma(L_{\text{base}})$ | Prior-Scaling Slope $\hat{\tau}_s = r \cdot (\sigma_s/\sigma_b)$ | $R^2(\Delta L_s \text{ on } L_{\text{base}})$ & Audit Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Token-0 Forced Read (`SFT`)** | `T9r/items_wide.csv` ($\varepsilon=10^{-4}$) | `0.806–0.835` (mean `0.822`) | `0.440–0.652` (mean `0.523`) | `0.354–0.544` (mean `0.430`, Seed 0 `0.544`) | **`59.2%–87.3%`** (`T9r/summary.md` $\tau$ def: `0.342–0.585`) |
| **Token-0 Forced Read (`GRPO`)** | `T9r/items_wide.csv` ($\varepsilon=10^{-4}$) | `0.900–0.910` (mean `0.904`) | `2.07–2.13` (mean `2.091`) | `1.868–1.920` (mean `1.890`, Seed 0 `1.868`) | **`48.7%–51.4%`** at $\varepsilon=10^{-4}$ (`63.5%–68.5%` at $10^{-6}$; `T9r` $\tau$: `2.089–2.170`) |
| **$K=32$ Sampled Rollouts (`SFT`)** | `Q2g/s1/residuals.csv` (`p_tilde`) | `0.592–0.611` (`0.629` pooled) | `0.82–0.83` (mean `0.823`) | `0.49–0.50` (mean `0.494`) | **`27%–29%`** (`Q2-A` item flags tie classical change: `0/6` in `S1`) |
| **$K=32$ Sampled Rollouts (`GRPO`)** | `Q2g/s1/residuals.csv` (`p_tilde`) | `0.731–0.746` (`0.765` pooled) | `1.91–1.99` (`2.21` pooled) | `1.43–1.46` (mean `1.438`) | **`9.5%–10.1%`** (`Q2-A` item flags: `1/12` in `Q1`) |
| **Free-Text CoT Greedy (`SFT`)** | `T9g/s3/matched.csv` ($\varepsilon=10^{-4}$) | `0.275–0.305` (mean `0.289`) | `2.26–2.39` (mean `2.35`; $\sigma_s \approx 7.44, \sigma_b \approx 3.17$) | `0.62–0.73` (mean `0.68`) | Descriptive recomputation; SFT reorders item difficulty ($r = 0.289$) |
| **Confound Check (`Subject` vs. `Item`)** | `W0` + `Q2g/s1` + `T9r` | — | — | — | • Subject $\bar{b}_m^{\text{pop}}$ correlates **$-0.94$** with $\overline{L}_{\text{base},m}$; partial $r(\bar{b}_m^{\text{pop}} \mid \overline{L}_{\text{base},m}) = \mathbf{0.17}$ (`all`) / $\mathbf{0.25}$ (`parsed`).<br>• Item-level $r(b_j^{\text{pop}}, \Delta L_{\text{SFT},j} - \Delta L_{\text{GRPO},j}) = \mathbf{+0.421}$ ($R^2 = 17.7\%$).<br>• Token-0 $r(a_j^{\text{pop}}, \Delta L_j) = \mathbf{+0.392}$ (`GRPO`, $R^2=15.4\%$) and $\mathbf{-0.470}$ (`SFT`, $R^2=22.1\%$) because $r(a_j^{\text{pop}}, L_{\text{base},j}) = +0.568$. |

- **Slide 8 — Consequence for Evaluation (`Q2c` & `Q2a`): Why High-$a_j^{\text{pop}}$ Adaptive CAT Loses to Random Sampling at $n = 100$**
  - **Visual**: **[`Fig06_cat_sociology_bias.png`](file:///D:/surrogates/gemini/figures/Fig06_cat_sociology_bias.png)**
  - **3-Second Visual Takeaway**: Since `Fig03` showed $a_j^{\text{pop}}$ is concentrated in Sociology and Marketing, what happens when max-information CAT (`irt_adaptive`) greedily selects items maximizing $(a_j^{\text{pop}})^2 p_j(1-p_j)$?
    - **40%–49% of CAT's selected items come from a single subject: Sociology** ($\bar{a} = 1.63\text{–}2.08$, `K.md:47-51`)!
    - In `Q2c` (checkpoint ability band $>0.45$), `irt_adaptive` beats `classical` at $n = 25$ (`0.064 vs. 0.098`) and $n = 50$ (`0.064 vs. 0.070`), but **flatlines at `~0.063`** (`Q2c` adaptive bias `+0.042 to +0.058`, `Q2c-notes.md:44`; item-selection effect `+0.049 to +0.095` at $k=1$, `K.md:106`).
    - Meanwhile, unbiased `classical` random sampling shrinks as $1/\sqrt{n}$ and **crosses below CAT at $n = 100$ (`0.050 vs. 0.063`)**! Across all $k=16$ cells in `Q2a` (`Q2a.md:73`), `classical` RMSE is at most `irt_adaptive` in **19 of 21 cells** (mean ratio `1.65x, 1.53x, 1.40x` at $n = 25, 50, 100$).

---

### Act III: The Multi-Draw Twist — What Repeated Sampling ($K=32$) Reveals About SFT vs. RL (`S1`, `Q1`, `Q2g`, Slides 9–10)

- **Slide 9 — The Multi-Draw Readout Crossover (`S1`): GRPO Wins Single Draw, But SFT Overtakes on Majority Vote**
  - **Visual**: **[`Fig07_s1_readout_crossover.png`](file:///D:/surrogates/gemini/figures/Fig07_s1_readout_crossover.png)**
  - **3-Second Visual Takeaway**: When we sample $K=32$ rollouts per item on the 1,500-item bank (`S1/stats.csv`, `S1/rewrite1_parsed.csv`):
    - **Single-Draw `Mean@16`**: **GRPO leads SFT (`0.479 vs. 0.439`, Base `0.382`)** because GRPO collapses within-item rollout entropy ($H_s: 0.963 \to 0.281$), putting its probability mass on a single modal answer.
    - **Majority-Vote `Maj@16`**: Because GRPO repeats the same answer across draws, voting adds almost nothing (`+0.002` unstarred / `+0.011*` tie-correct) and `Pass@16` drops from Base's **`0.930`** down to **`0.656`**. By contrast, **CoT SFT preserves high entropy ($H_s = 0.943$) and crosses above GRPO on `Maj@16`** (`0.504 vs. 0.481` unstarred [Rule 3 fires in `2/3` seeds, `0/3` on order 1 alone, `S1.md:78`]; `0.555* vs. 0.490*` tie-correct)—a **`+0.065` (`+0.116*`) voting boost**! *(Note: earlier raw mode-change numbers `0.560–0.564 vs. 0.398–0.436` across all 1,500 items are withdrawn as a reordering test because raw mode change scales with dispersion, `state.md:32`.)*
  - **Hook to Slide 10**: *Does this separation between `Base -> RL` and `Base -> SFT` hold across all 19 evaluated model states?*

- **Slide 10 — All 19 States in the $(H_s, \text{Maj@16})$ Plane & Multi-Draw Person Fit ($l_z^{(1)}$ vs. $l_z^{(32)}$)**
  - **Visual**: **[`Fig08_19states_entropy_vs_majority.png`](file:///D:/surrogates/gemini/figures/Fig08_19states_entropy_vs_majority.png)** + **Table 2**
  - **3-Second Visual Takeaway (`Fig08`)**: Plotting all **19 states (`1 Base + 18 post-trained states`: `3 S1 SFT`, `3 S1 GRPO`, `12 Q1 RL`)** in the $(H_s, \text{Maj@16})$ plane shows that **all 15 `Base -> RL` runs are trapped in the bottom-left** ($H_s \in [0.10, 0.42]$, `Maj@16` in `0.467–0.496`, with `q1_ctt_1` at `0.496`), while **only `Base -> SFT` reaches the top-right (`Maj@16 > 0.50`)**!
  - **Companion Table 2 (19-State Readout & Person-Fit Summary)**:

| Model Family (`States`) | Rollout Entropy $H_s$ (nats) | Answer Stability `stable_12` | Single-Draw `Mean@16` | Majority Vote `Maj@16` (`Tie=Wrong` / `Tie=Correct*`) | Coverage `Pass@16` | Voting Boost `Maj@16 - Mean@16` | Single-Draw Fit $l_z^{(1)}$ (`Q2a`, 1 draw) | 32-Trial Fit $l_z^{(32)}$ (`Q2g`, 32 draws) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (`t9_base`, 1 state)** | `0.963` | `0.194` | `0.382` | `0.414` / `0.459*` | `0.930` | `+0.032` / `+0.077*` | **`-3.71`** | **`-212.1`** |
| **S1 CoT SFT (`t9_sft`, 3 seeds)** | `0.943` | `0.219` | `0.439` | **`0.504` / `0.555*`** (`0.529` mid) | **`0.966`** (`+0.036` vs Base) | **`+0.065` / `+0.116*`** | **`-6.94`** (`-8.38 to -4.28`) | **`-225.4`** (`-230.9 to -217.5`) |
| **S1 GRPO 666u (`t9_rl`, 3 seeds)** | `0.281` | `0.785` | **`0.479`** (`0.473–0.486`) | `0.481` / `0.490*` (`0.485` mid) | `0.656` | `+0.002` / `+0.011*` | **`+2.24`** (`+2.03 to +2.52`) | **`-482.3`** (`-498.1 to -471.7`) |
| **Q1 GRPO 333u (`G=5`, 3 seeds)** | `0.320` | `0.757` | `0.4685` | `0.474` | `0.685` | `+0.006` | — | **`-464.5`** |
| **Q1 CTT Critic (`G=1`, 3 seeds)** | `0.216` | `0.866` | `0.4722` | `0.485` (max `0.496`) | `0.618` | `+0.013` | — | **`-524.8`** |
| **Q1 2PL `pop1` (`a_j`-wt `G=1`, 3 seeds)** | `0.197` | `0.881` | `0.4748` | `0.485` | `0.606` | `+0.010` | — | **`-535.3`** |
| **Q1 2PL `pop0` (unwt `G=1`, 3 seeds)** | `0.123` | `0.942` | `0.4690` | `0.482` | `0.546` | `+0.013` | — | **`-576.5`** |

  - **Key Insight from Table 2 ($l_z^{(1)}$ vs. $l_z^{(32)}$)**: `Q2a` $l_z^{(1)}$ (1 order-0 draw at EAP $\hat{\theta}$, `Q2a.md:46`) and `Q2g` $l_z^{(32)}$ (32 trials at `Q2g` $\hat{\theta}$) are two distinct estimators. **Base and SFT are negative on both**, whereas `S1 GRPO` looks Guttman hyper-consistent on a single binary draw ($l_z^{(1)} = +2.24 > +1.645$) because its entropy is low, yet flips to $-482.3$ once 32 trials expose its divergence from the 102-LLM 2PL item curve (`report-q1q2.md:46`).

---

### Act IV: From RL Failure on Base (`Unit Q1`) to Within-Model Curriculum Design (`Unit M1`, Slides 11–14)

- **Slide 11 — `Unit Q1`: Static 2PL Critics Do Not Beat Unweighted GRPO — All 4 Arms Land in a `0.63 pp` Band**
  - **Visual**: **[`Fig09_q1_rl_trajectories.png`](file:///D:/surrogates/gemini/figures/Fig09_q1_rl_trajectories.png)**
  - **3-Second Visual Takeaway**: When we train 12 RL runs from `t9_base` comparing unweighted `grpo` ($G=5$) against three $G=1$ critics (`ctt`, unweighted 2PL `pop0`, and $a_j^{\text{pop}}$-weighted 2PL `pop1`):
    - **Rapid Early Saturation (Seed 0, `Q1/trajectory.csv`)**: In Seed 0 (the only seed with mid-run checkpoint saves), **`83.7%` (`grpo`), `91.5%` (`ctt`), `93.9%` (`pop0`), and `94.5%` (`pop1`)** of the total accuracy gain finishes within the first **83 updates** (`0.382 -> 0.455–0.474`).
    - **Identical Final Plateau (`Q1/stats.csv`)**: At update 333, all 4 arm means finish between **`0.4685` and `0.4748`**—a width of **`0.0063` (`0.63 percentage points`, or `1.3%` of `0.47`; 12-run range `0.459–0.480`)**!
  - **Hook to Slide 12**: *Why do all `Base -> RL` runs stall after update 83 regardless of the critic?*

- **Slide 12 — Why `Base -> RL` Stalls After Update 83: Zero-Advantage Starvation (`G=5`) & Over-Concentration (`G=1`)**
  - **Visual**: **[`Fig10_grpo_zero_advantage_starvation.png`](file:///D:/surrogates/gemini/figures/Fig10_grpo_zero_advantage_starvation.png)** + **Table 3**
  - **3-Second Visual Takeaway (`Fig10`)**: In $G=5$ `grpo`, if all 5 rollouts on a prompt are wrong (`0/5`) or all 5 are right (`5/5`), reward variance is zero and the policy gradient is zero. Starting from **`43.8%` raw zero-variance prompts on Base** (`W1/summary.json`, `wasted_share_g5_raw = 0.438`, shrunk `28.0%`; `grpo` updates `0–9` mean `36.9%–43.8%`), `zero_adv_share` climbs to a **`63.1%` average after update 83 (`84.4%–90.6%` peak)**—wasting nearly two-thirds of all training updates!
  - **Companion Table 3 ($G=5$ `grpo` vs. $G=1$ Critics Mechanism Comparison)**:

| RL Training Arm (`Unit Q1`, 3 Seeds Each) | Rollouts per Prompt $G$ | Advantage Formula $A_{s,j}$ | Uses Static $a_j^{\text{pop}}$? | Post-u83 `zero_adv_share` (`Q1/traces/`) | Length Divisor | Final Answer Stability `stable_12` | Final `Mean@16` (3-Seed Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`q1_grpo` (Standard Group Relative)** | $G = 5$ | $(r_g - \bar{r}_G) / \text{sd}(r_G)$ | **No** | **`63.1%` mean** (`84.4%–90.6%` peak) | Group active length | `0.757` | `0.4685` |
| **`q1_ctt` (Classical Pass-Rate Band Critic)** | $G = 1$ | $r - \hat{p}_j^{\text{CTT}}$ (Rasch 1PL equals `ctt` in expectation, `M.md:97`) | **No** | Empty (`r - \hat{p}_j \neq 0$) | Fixed $L_0 = 78.98$ | `0.866` | `0.4722` |
| **`q1_pop0` (Unweighted 2PL Critic)** | $G = 1$ | $r - P_j(\hat{\theta}; a_j^{\text{pop}}, b_j^{\text{pop}})$ (`report-q1q2.md:13`) | Fixed Fluid $a_j^{\text{pop}}$ in $P_j$ | Empty (`r - \hat{p}_j \neq 0$) | Fixed $L_0 = 78.98$ | **`0.942`** | `0.4690` |
| **`q1_pop1` ($a_j^{\text{pop}}$-Weighted 2PL Critic)** | $G = 1$ | $a_j^{\text{pop}} \cdot \big(r - P_j(\hat{\theta}; a_j^{\text{pop}}, b_j^{\text{pop}})\big)$ | **Yes** (weights high-$a_j$ factual items $5.2\times$ over math) | Empty (`r - \hat{p}_j \neq 0$) | Fixed $L_0 = 78.98$ | `0.881` | `0.4748` |

- **Slide 13 — Connecting the Story (`Unit M1`): Harvesting SFT's `Maj@16` Ceiling Using SFT's Own Pass-Rate Parameter $p_{\text{SFT},j} \in [0.2, 0.8]$**
  - **Visual**: **[`Fig11_w1_pass_rate_and_zpd.png`](file:///D:/surrogates/gemini/figures/Fig11_w1_pass_rate_and_zpd.png)** + **Table 4**
  - **How Every Previous Slide Connects into `Unit M1`**:
    1. **Why Start RL from `SFT` Instead of `Base` (`Fig07` + `Fig08`)**: Starting RL from `Base` traps all 15 runs at `Maj@16 <= 0.496`. `S1` CoT SFT (`t9_sft_0..2_70`) already raised the majority-vote ceiling to **`Maj@16 = 0.504` (`0.555*` tie-correct) and `Pass@16 = 0.966`** at high entropy ($H_s = 0.943$), leaving a **`+0.065 to +0.116*` voting gap** ready to be harvested by RL concentration!
    2. **Why Filter by the SFT Model's OWN Pass Rate $p_{\text{SFT},j} \in [0.2, 0.8]$ (`Fig05` + `Fig10` + `Fig11`)**: In `Fig11` (`W1/pass_rate.csv`), **`48.3%` (`5,164` items)** of the training pool sits on the exterior (`0, 1, 7, 8 of 8`) where Bernoulli variance $p_j(1-p_j)$ is near zero, while **`51.7%` (`5,518` items)** sits in the interior Zone of Proximal Development (`2..6 of 8`, $p_j \in [0.2, 0.8]$) where 1PL Fisher information $p_j(1-p_j)$ peaks. Because `Fig05` & `Table 1` showed that SFT reorders item difficulty relative to Base ($r = 0.289$ in free-text CoT, $0.59\text{–}0.61$ in $K=32$ draws), `Unit M1` (`M1c`) first samples $k=8$ rollouts on `t9_sft_0..2_70` to select **each SFT seed's own interior items $p_{\text{SFT},j} \in [0.2, 0.8]$**!
  - **Companion Table 4 (Committed Baselines vs. Pre-Registered `Unit M1` Design, `notes/units/M1.md`)**:

| Stage / Experimental Arm | Status | Starting Checkpoint | Prompt Curriculum / Filter | Single-Draw `Mean@16` | Majority Vote `Maj@16` (`Tie=Wrong` / `Tie=Correct*`) | Coverage `Pass@16` | Pre-Registered Decision Rule (`notes/units/M1.md`) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **1. Base (`t9_base`)** | Committed (`S1`) | `t9_base` | — | `0.382` | `0.414` / `0.459*` | `0.930` | Baseline reference |
| **2. `Q1` Base $\rightarrow$ RL (`12 runs`, 333u)** | Committed (`Q1`) | `t9_base` | Full pool / Static `W0/W1` critics | `0.471` (`0.459–0.480`) | `0.482` (`0.467–0.496`) | `0.614` | `Base -> RL` benchmark (`grpo` arm `0.4685`) |
| **3. `S1` Base $\rightarrow$ GRPO (`3 runs`, 666u)** | Committed (`S1`) | `t9_base` | Full pool ($G=5$) | `0.479` (`0.473–0.486`) | `0.481` / `0.490*` | `0.656` | 666-update `Base -> GRPO` reference |
| **4. `S1` Base $\rightarrow$ CoT SFT (`3 runs`, 70u)** | Committed (`S1`) | `t9_base` | 70B CoT SFT distillation | `0.439` (`0.435–0.443`) | **`0.504` / `0.555*`** | **`0.966`** | High-entropy starting point for `Unit M1` ($H_s = 0.943$) |
| **5. `M1a`: SFT Low-Temp Readout ($T=0.2$)** | **Running (`M1`)** | `t9_sft_0..2_70` | Zero RL training (inference $T=0.2$ vs. $0.7$) | *Cluster running* | *Cluster running* | *Cluster running* | Tests how much of `Maj@16 - Mean@16` is harvested by temperature scaling alone |
| **6. `M1b`: `t9_sft` $\rightarrow$ Standard GRPO (333u)** | **Running (`M1`)** | `t9_sft_0..2_70` | Unfiltered pool (`Q1` exact settings, save u83 & u333) | *Cluster running* | *Cluster running* | *Cluster running* | **Rule 1**: `+1` if `Mean@16` beats both `t9_sft` & `q1_grpo` in `3/3` seeds ($>2.92\text{ SE}$); `-1` if `Maj@16` drops below `t9_sft` in `3/3` seeds ($<-2.92\text{ SE}$); `0` otherwise |
| **7. `M1c`: `t9_sft` $\rightarrow$ SFT-ZPD GRPO (333u)** | **Running (`M1`)** | `t9_sft_0..2_70` | Filtered to SFT's own **$p_{\text{SFT},j} \in [0.2, 0.8]$ (`2..6 of 8`)** | *Cluster running* | *Cluster running* | *Cluster running* | **Rule 2**: Tests whether filtering by within-model $p_{\text{SFT},j}$ lowers `zero_adv_share` and beats unfiltered `M1b` |

- **Slide 14 — Final Synthesis: What Item Parameters CAN (and CANNOT) Tell Us**
  - **Summary Takeaway**:
    1. **What Static Cross-Respondent $(a_j^{\text{pop}}, b_j^{\text{pop}})$ CANNOT Do**:
       - Cannot transfer from human students to LLMs (`Fig01`, $\rho = +0.089$, `61/61` $Z_h < -1.96$).
       - Cannot isolate item-level learning (`Q2-A`: `0/6` in `S1`, `1/12` in `Q1`) because post-training shifts follow the model's **own** base log-odds $L_{\text{base},j}$ (`Fig05`, `Table 1`).
       - Cannot beat unbiased random sampling at $n = 100$ in greedy CAT (`Fig06`, `40%–49%` Sociology bias).
       - Cannot beat unweighted GRPO when used as static RL critics from Base (`Fig09`, `0.63 pp` band across all 4 `Q1` arms).
    2. **What Item Parameters & Multi-Draw Psychometrics CAN Tell Us**:
       - **Explain Metric Contradictions (`Fig02` $\rightarrow$ `Fig03` $\rightarrow$ `Fig04`)**: Shows why 102-LLM 2PL $\hat{\theta}$ flips stage attribution on seen `S1` states (`59.5/40.5` Accuracy vs. `20.9/79.1` $\hat{\theta}$)—because $a_j^{\text{pop}}$ weights factual recall $5.2\times$ over math (`Fig03`), where GRPO wins `8/8` high-$a_m$ subjects and SFT wins `5/6` low-$a_m$ math subjects (`Fig04`, with $\bar{b}_m^{\text{pop}}$ tracking $\overline{L}_{\text{base},m}$ at $r = -0.94$, `Table 1`).
       - **Diagnose Generative Collapse vs. Latent Coverage (`Fig07`, `Fig08`, `Table 2`)**: Separates GRPO's single-draw sharpening (`Mean@16 = 0.479`, $l_z^{(1)} = +2.24$ vs. $l_z^{(32)} = -482.3$) from SFT's majority-vote ceiling (`Maj@16 = 0.504 / 0.555*`, `Pass@16 = 0.966`).
       - **Guide Within-Model Curriculum Design (`Fig10`, `Fig11`, `Table 4`)**: Replaces static cross-model parameters with the model's **own** item parameters ($L_{\text{base},j}$ and $p_{\text{SFT},j} \in [0.2, 0.8]$) to target active Bernoulli variance $p_j(1-p_j)$ in `Unit M1`.
