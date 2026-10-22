# Story-Driven Slide Plan for Monday, Oct 26 Talk (`What Can and Cannot Item Parameters Tell Us?`)

- **Locked Title**: *Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?*
- **Target Audience**: Measurement, IRT/Psychometrics, and Neuro-Cognitive Science researchers, alongside ML post-training researchers (stepping stone to the Nov 20 40th IRT Workshop: *"Evaluating and Training Large Language Models: What Can Item Parameters Tell Us?"*).
- **Core Interdisciplinary Thesis (`Cannot` $\longrightarrow$ `A helps B` $\longrightarrow$ `B helps A` $\longrightarrow$ `A <-> B` High-Leverage Bet)**:
  - A talk that only lists negative audit verdicts (*"Human IRT failed, 2PL RL critics failed, 2PL item residuals failed"*) is trivially boring.
  - Instead, we **admit the plug-and-play failure cleanly (`What Static Cross-Population Parameters CANNOT Do`)**, **abstract why it failed to the meta-level (Molenaar's Non-Ergodicity & Autoregressive Self-Prior Scaling)**, and show the **three constructive bidirectional insights (`What Item Parameters & Generative Psychometrics CAN Tell Us`)** between **Psychometrics / Neuro-Cognitive Science (`A`)** and **LLM Post-Training (`B`)**:
    1. **`Cannot` (Why Plug-and-Play Cross-Population IRT Fails)**: Between-respondent calibration (across human students `E1` or 102 leaderboard LLMs `W0`) is non-ergodic with within-model post-training (`Q1` RL critics fail; `Q2-A` item-change flags fail; `Q2-B` $\hat{\theta}$ fails on unseen `Q1` models) because item-level post-training shifts are governed by the model's **own** base confidence $L_{\text{base},j}$.
    2. **`A helps B` (What Item Parameters *CAN* Tell Us — Declarative vs. Procedural Dissociation & Attribution Reversal)**: In the 102-LLM population (`W0`), item discrimination $\bar{a}_m$ and difficulty $\bar{b}_m$ ($r = -0.884$) index a macro-level **Cognitive Construct Axis**: **Declarative / Factual Recall** (*Marketing, Sociology*: high $\bar{a}_m = 0.69\text{–}1.45$, where the base model already has positive confidence) vs. **Procedural / Quantitative Reasoning** (*HS Math, Elem Math, Physics*: low $\bar{a}_m = 0.28\text{–}0.55$, where base confidence is low). Outcome RL (`GRPO`) acts as **gain amplification** on existing priors ($\hat{\tau}_s > 1$, favoring high-$a_m$ declarative subjects), while **CoT SFT** acts as **procedural reordering/lifting** ($\hat{\tau}_s < 1$, favoring low-$a_m$ math subjects and transferring $+10.4\text{ to }+11.4$ pts OOD in `signature T6`)—explaining why $a_j$-weighted 2PL $\hat{\theta}$ flips stage attribution from `60% SFT / 40% GRPO` (Accuracy) to `21% SFT / 79% GRPO` ($\hat{\theta}$) on seen `S1` states!
    3. **`B helps A` (What Counterfactual LLM Rollouts Teach Psychometric Theory — The $(\theta_s, \beta_s)$ Generative Respondent Model)**: Sampling $K=32$ counterfactual rollouts per item across 19 states (`1.48M` generations) shows that a reasoning LLM is not a 1-parameter point mass ($\theta_s$ alone), but has two distinct dimensions: **latent item location $\mu_{sj}(\theta_s)$** and **within-item trajectory concentration / inverse temperature $\beta_s$** (rollout entropy $H_s$). This explains both the **Majority-Vote Readout Crossover** (`GRPO` wins 1-draw `47.9% vs. 43.9%` via $\beta_s \uparrow$, while `CoT SFT` wins `Maj@16 = 50.4%–55.5%` and `Pass@16 = 96.6%` via Condorcet voting across diverse paths) and the **Person-Fit Sign Flip** ($l_z(K=1) = +2.24 \to l_z(K=32) = -482$).
    4. **`A <-> B` (The High-Leverage Bet You'd Be Happy Losing — `Unit M1: Operator Composition & Within-Model ZPD Curriculum`)**: All 15 `Base -> RL` runs plateau at `46.9%–47.9%` (`Maj@16 <= 49.6%`) because pure gain amplification ($\beta_s \uparrow$) from `Base` is bounded by `Base`'s modal ceiling (`Maj@16 = 41.4%–45.9%`). Starting RL from `CoT SFT` (`Maj@16 = 50.4%–55.5%`) and filtering prompts by **SFT's own active pass-rate parameter $p_{\text{SFT},j} \in [0.2, 0.8]$ (`2..6 of 8`, where Bernoulli Fisher information $p_j(1-p_j)$ peaks)** tests whether psychometric ZPD targeting harvests SFT's Condorcet voting gap (`+1`: breaks the 15-run `Base -> RL` ceiling) or exposes destructive interference between outcome-RL mode collapse and CoT diversity (`-1`: landmark negative law).

---

## Part 1: Division of Media (`Slide Text/Math` + `2 Schematics` + `4 Tables` + `5 Pure Visual Plots`)

| Medium | Asset | Core Story Role (`Cannot` / `A -> B` / `B -> A` / `A <-> B`) | Source Files (`results/`) | Slide |
| :---: | :--- | :--- | :--- | :---: |
| **Slide Text & Math** | **2PL Equations, Molenaar's Non-Ergodicity & The 4-Part Arc** | Defines how $a_j$ weights evaluation ($\sum_j a_j(y_j - P_j(\theta)) = 0$) and RL critics ($A_j = a_j(r_j - P_j(\theta))$), and frames the transition from cross-population failure (`Cannot`) to within-model generative psychometrics (`Can`). | `report-q1q2.md`<br>`MASTER-PLAN.md` | Slide 2 |
| **Data Plot 1** | [`Plot01_human_vs_llm_difficulty.png`](file:///D:/surrogates/gemini/figures/Plot01_human_vs_llm_difficulty.png) | **`Cannot` #1 (`Unit E1`)**: Human student 2PL difficulty vs. classical LLM error rate across `J = 704` Eedi math items—a flat cloud ($r = +0.139$, 2PL rank $\rho = +0.089$; `61/61` LLMs have $Z_h < -1.96$). | `E1/difficulty.csv`<br>`E1/llm_scores.csv` | Slide 3 |
| **Table 1** | **The Attribution Puzzle & Pre-Registered `Q2-B` Test** | Sets up the central puzzle on seen `S1` states (`Accuracy: 59.5% SFT / 40.5% GRPO` vs. `2PL θ: 20.9% SFT / 79.1% GRPO`) alongside the pre-registered `Q2-B` verdict on the **12 unseen `Q1` models** (`Rule 2 fails: 2/6 B1 pairs`). | `Q2g/s1/personfit.csv`<br>`Q2g/s1/precision.csv`<br>`Q2g/q1/precision.csv` | Slide 4 |
| **Schematic 1** | [`Schematic01_population_vs_posttraining.png`](file:///D:/surrogates/gemini/figures/Plot01_human_vs_llm_difficulty.png) | **`A helps B` Intuition**: 102-LLM 2PL Item Characteristic Curves $P(\theta) = \sigma(\bar{a}_m(\theta - \bar{b}_m))$ showing why Declarative Factual Recall (*Marketing $\bar{a}=1.45$, Sociology $\bar{a}=1.13$*) gets **$5.2\times$ the score weight** of Procedural Math (*HS Math $\bar{a}=0.28$*). | `W0/items_bank.csv` | Slide 5 |
| **Data Plot 2** | [`Plot02_subject_weights_and_gains.png`](file:///D:/surrogates/gemini/figures/Plot02_subject_weights_and_gains.png) | **`A helps B` Empirical Pattern & Confound Check**: 15 seen `S1` subject means sorted by 2PL weight $\bar{a}_m$ ($r(\bar{a}_m, \bar{b}_m) = -0.88$), showing high-weight declarative subjects favoring GRPO and low-weight math subjects favoring SFT, paired with the base-prior confound check ($r(\bar{b}_m, \overline{L}_{\text{base},m}) = -0.94$, partial $r = 0.17$). | `W0/items_bank.csv`<br>`Q2g/s1/flags.csv`<br>`Q2g/s1/personfit.csv` | Slide 6 |
| **Data Plot 3 + Table 2** | [`Plot03_prior_scaling_scatter.png`](file:///D:/surrogates/gemini/figures/Plot03_prior_scaling_scatter.png) | **Within-Model Prior-Scaling Law & `Q2-A` Verdict**: Scatter of $L_{s,j}$ vs. $L_{\text{base},j}$ ($J = 1,472$, `T9r` Seed 0) showing GRPO slope `1.87 > 1` (gain amplification) vs. SFT slope `0.54 < 1` (reordering/lifting). **Table 2** compares `Token-0`, `K=32`, `Free-Text CoT`, and `Q2-A` (`0/6` in `S1`, `1/12` in `Q1`). | `T9r/items_wide.csv`<br>`T9g/s3/matched.csv`<br>`Q2g/s1/reproducibility.csv`<br>`Q2g/q1/reproducibility.csv` | Slide 7 |
| **Data Plot 4** | [`Plot04_readout_crossover.png`](file:///D:/surrogates/gemini/figures/Plot04_readout_crossover.png) | **`B helps A` #1 — The Multi-Draw Readout Crossover (`S1`)**: Under the $(\theta_s, \beta_s)$ model, GRPO ($\beta_s \uparrow$) wins 1-draw `Mean@16` (`47.9% vs. 43.9%`), but CoT SFT ($H_s = 0.94$) crosses above on `Maj@16` (`50.4%–55.5%` vs. `48.1%–49.0%`, with vertical tie-range bars) and `Pass@16` (`96.6% vs. 65.6%`). | `S1/stats.csv`<br>`S1/rewrite1_parsed.csv` | Slide 8 |
| **Table 3** | **`B helps A` #2 — 19-State Readouts & Person-Fit Sign Flip** | Compares all 19 states across `Entropy`, `stable_12`, `Mean@16`, `Maj@16`, `Pass@16`, and shows why GRPO flips from hyper-consistent on 1 draw ($l_z(K=1) = +2.24$) to strong misfit on 32 draws ($l_z(K=32) = -482$). | `S1/stats.csv`<br>`Q1/stats.csv`<br>`Q2a/personfit.csv`<br>`Q2g/s1/personfit.csv`<br>`Q2g/q1/personfit.csv` | Slide 9 |
| **Data Plot 5** | [`Plot05_q1_training_curves.png`](file:///D:/surrogates/gemini/figures/Plot05_q1_training_curves.png) | **`Cannot` #2 (`Unit Q1`) — Static 2PL Critics in RL Training**: Seed 0 trajectories (`0 -> 83 -> 333`) and 3-seed finals (`46.85%–47.48%`) showing ~90% of gain by step 83 and pre-registered `Q1` verdict (`pop0 - ctt`: mean $d = -0.0032$, $\text{SE}_d = 0.0039$). | `Q1/trajectory.csv`<br>`Q1/stats.csv`<br>`Q1/contrasts.csv` | Slide 10 |
| **Schematic 2 + Table 4** | [`Schematic02_multidraw_and_m1_zpd.png`](file:///D:/surrogates/gemini/figures/Schematic02_multidraw_and_m1_zpd.png) | **`A <-> B` High-Leverage Bet (`Unit M1`)**: `W1` pass-rate histogram (`51.7%` interior `2..6 of 8` vs. `48.3%` exterior) + Bernoulli Fisher information $p(1-p)$, paired with **Table 4** (pre-registered `M1a / M1b / M1c` design and two-sided `+1 / -1 / 0` outcomes). | `W1/pass_rate.csv`<br>`notes/units/M1.md` | Slide 11 |
| **Slide Text & Synthesis Table** | **Answering the Title: What Item Parameters CAN (and CANNOT) Tell Us** | Side-by-side synthesis of `What Static Cross-Population Parameters CANNOT Do` vs. `What Within-Model & Generative Psychometrics CAN Tell Us`. | `report-q1q2.md`<br>`MASTER-PLAN.md` | Slide 12 |

---

## Part 2: Slide-by-Slide Narrative Script, Math, and Full Tables (12 Slides)

### Act I: Setting Up the Question & Why Static Cross-Population Parameters Fail (`Cannot`, Slides 1–4)

#### Slide 1 — Title Slide
- **Title**: *Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?*
- **Subtitle**: *From Cross-Sectional Non-Ergodicity to Within-Model Generative Psychometrics Across 4,918 Students, 102 Leaderboard LLMs, and 1.488M Post-Training Rollouts*

#### Slide 2 — Slide Text & Math: The Paradigm Under Test & Molenaar's Non-Ergodicity Question
- **No Figure Needed — Build Intuition with Clean Math & Conceptual Framing**:
  1. **How Item Parameters Enter LLM Evaluation and Training**:
     $$P_j(\theta) = \sigma\!\big(a_j(\theta - b_j)\big), \qquad \text{Eval Score Equation: } \sum_{j=1}^{J} a_j \big(y_j - P_j(\hat{\theta})\big) = 0, \qquad \text{RL Critic: } A_j = w_j \big(r_j - P_j(\theta_t)\big)$$
     - **Difficulty $b_j$** sets the expected baseline $P_j(\theta)$; **Discrimination $a_j$** is the item's **weight** in both ability estimation $\hat{\theta}$ and RL gradient updates.
  2. **The Hidden Ergodicity Assumption (Molenaar, 2004)**:
     - Recent LLM benchmarks (*tinyBenchmarks*, *Fluid Benchmarking*) calibrate $(a_j^{\text{pop}}, b_j^{\text{pop}})$ **between respondents** (across human students or 100+ diverse leaderboard LLMs) and assume those static parameters govern **within-model post-training** (`Base -> SFT` or `Base -> RL`) of a single LLM (`Llama-3.2-1B-Instruct` on the 1,500-item MMLU bank).
  3. **Our Four-Part Question**:
     - **Part I (`Cannot`)**: Can static cross-population item parameters transfer from humans to LLMs (`E1`), locate item-level post-training change (`Q2-A`), or improve RL critics from Base (`Q1`)?
     - **Part II (`A helps B` — What They *CAN* Tell Us)**: Why do 2PL $\hat{\theta}$ and Classical Accuracy give opposite stage attributions on `Base -> SFT -> GRPO`, and what does $(a_m^{\text{pop}}, b_m^{\text{pop}})$ reveal about **Declarative vs. Procedural** post-training?
     - **Part III (`B helps A` — What LLMs Teach Psychometrics)**: When we sample $K=32$ reasoning paths per item, how do **location $\theta_s$** and **trajectory concentration $\beta_s$** interact across `Mean@16`, `Maj@16`, and person fit $l_z$?
     - **Part IV (`A <-> B` — `Unit M1`)**: Can we combine `CoT SFT`'s majority-vote ceiling with **within-model psychometric ZPD filtering** ($p_{\text{SFT},j} \in [0.2, 0.8]$)?

#### Slide 3 — `Cannot` #1 (`Unit E1`): Human Student Item Difficulty Is Orthogonal to LLM Difficulty
- **Visual**: **[`Plot01_human_vs_llm_difficulty.png`](file:///D:/surrogates/gemini/figures/Plot01_human_vs_llm_difficulty.png)**
- **Slide Text Beside Plot**:
  - Across **`J = 704` Eedi diagnostic math items**, plotting **classical LLM error rate (`1 - proportion correct` across 61 LLMs)** against **human student 2PL difficulty $b_j$ (`4,918` students)** produces a uniform scatter cloud:
    - Classical LLM error rate vs. student $b_j$: **Pearson $r = +0.139$, Spearman $\rho = +0.119$** (OLS slope $+3.05\%$ per logit, intercept $57.0\%$).
    - 61-LLM 2PL difficulty rank vs. student $b_j$: **Spearman $\rho = +0.089$** against $0.854$ for 61 synthetic students (`closure.md:29`; the 61-LLM 2PL fit itself was rejected in `E1` with $b_j \in [-1574, +912]$).
    - **`61/61` LLMs** fall below the human 2PL person-fit threshold ($Z_h < -1.96$, `E1/llm_scores.csv`).
  - **Takeaway**: Human psychometric calibrations cannot be transferred to LLMs. *What if we calibrate $(a_j^{\text{pop}}, b_j^{\text{pop}})$ on **102 leaderboard LLMs** (`W0`) instead?*

#### Slide 4 — The Attribution Puzzle (`Table 1`): Why Do Accuracy and 102-LLM 2PL $\hat{\theta}$ Flip Credit on Seen `S1` States, While `Q2-B` Fails on Unseen `Q1` Models?
- **No Figure Needed — Clean `Table 1`**:
  - Evaluating `Llama-3.2-1B-Instruct` across `Base -> CoT SFT -> GRPO` on the 1,500-item MMLU bank (`S1`, $K=32$ draws) reveals a striking puzzle on the seen `S1` states, alongside a negative pre-registered test on the 12 unseen `Q1` models:

| Evaluation Metric & Scope | Base (`t9_base`) | After CoT SFT (`3 seeds`) | After GRPO (`3 seeds`) | Stage 1 Gain (`Base -> SFT`) | Stage 2 Gain (`SFT -> GRPO`) | Share of Total Gain (`SFT` / `GRPO`) | Pre-Registered `Q2-B` Rule 2 Verdict (`report-q1q2.md:50-59`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Classical Accuracy (`S1` seen, `all` draws)** | `38.2%` | `43.9%` | `47.9%` | `+5.8%` ($8.9\text{–}10.5\text{ SE}$) | `+3.9%` ($3.8\text{–}5.9\text{ SE}$) | **`59.5% SFT` / `40.5% GRPO`** (`parsed`: `47% / 53%`) | Equal-weight baseline ($w_j \equiv 1$) |
| **102-LLM 2PL Ability $\hat{\theta}$ (`S1` seen, `all` draws)** | `-1.209` | `-1.045` | `-0.425` | `+0.164` ($2.9\text{–}5.1\text{ SE}$) | `+0.620` ($8.6\text{–}10.5\text{ SE}$) | **`20.9% SFT` / `79.1% GRPO`** (`parsed`: `15% / 85%`) | **Positive on 7 seen `S1` states** (`SFT` vs. `GRPO` $|\Delta/\text{SE}|$: $\hat{\theta}$ `8.6–10.5` vs. acc `3.8–5.9`; reliability `0.983 vs. 0.977`). *Withdrawn as a general finding (`report-q1q2.md:79`) because `S1` was seen.* |
| **12 Unseen `Q1` RL Models (`Q2g/q1/precision.csv`)** | `-1.209` (`38.1%`) | — | `-0.671 to -0.471` (`45.9%–48.0%`) | — | — | — | **Rule 2 FAILS on unseen `Q1` models**: $\hat{\theta}$ is sharper in **`2 of 6` B1 pairs** (`3 of 12` vs. Base; reliability `0.287 vs. 0.167` at no-effect noise level because the 12 `Q1` models span only `2.1%` in accuracy; on the 1 B1 pair with a true accuracy difference, `pop1_2 - pop0_2`, $\hat{\theta}$ $z=5.11$ vs. acc $z=3.69$). |

---

### Act II: `A helps B` — What Item Parameters *CAN* Tell Us (Declarative vs. Procedural Dissociation & Within-Model Prior Scaling, Slides 5–7)

#### Slide 5 — `A helps B` Intuition (`Schematic 1` + Math): The $a_j$-Weighting Attribution Theorem & Declarative vs. Procedural Construct Axis
- **Visual**: **[`Schematic01_population_vs_posttraining.png`](file:///D:/surrogates/gemini/figures/Schematic01_population_vs_posttraining.png)**
- **Slide Text & Math Beside Schematic**:
  - **Why Does 2PL $\hat{\theta}$ Flip Credit from SFT (`60% -> 21%`) to GRPO (`40% -> 79%`) on `S1`?**
    From a first-order expansion of the 2PL score equation $\sum_j a_j^{\text{pop}}(p_{s,j} - P_j(\hat{\theta}_s)) = 0$:
    $$\Delta \hat{\theta}_s \propto \overline{\Delta p}_s + \frac{\operatorname{Cov}_j\!\big(a_j^{\text{pop}}, \, \Delta p_{s,j}\big)}{\bar{a}^{\text{pop}}}$$
    - Whenever a training stage improves **high-$a_j^{\text{pop}}$ items** ($\operatorname{Cov}(a_j^{\text{pop}}, \Delta p_j) > 0$), $\Delta\hat{\theta}$ is amplified relative to classical accuracy $\overline{\Delta p}$; whenever it improves **low-$a_j^{\text{pop}}$ items** ($\operatorname{Cov}(a_j^{\text{pop}}, \Delta p_j) < 0$), $\Delta\hat{\theta}$ is attenuated!
  - **What Does $a_j^{\text{pop}}$ Measure Across the 102 Leaderboard LLMs (`W0`)?**
    - Across the 15 MMLU subjects ($J = 1,472$), $\bar{a}_m^{\text{pop}}$ and $\bar{b}_m^{\text{pop}}$ form a single collinear axis (**$r = -0.8841, R^2 = 78.2\%$**, OLS slope $-2.72$):
      - **Declarative / Semantic Recall (*Marketing $\bar{a}=1.45$, Sociology $\bar{a}=1.13$*)**: Scales smoothly with model size across the 102-LLM leaderboard $\implies$ **steep S-curve ($5.2\times$ score weight)**.
      - **Procedural / Quantitative Reasoning (*HS Math $\bar{a}=0.28$, HS Physics $\bar{a}=0.33$*)**: Noisy and hard across general models $\implies$ **flat S-curve ($1.0\times$ score weight)**.
    - *(Footnote on short-test CAT, `Q2a/Q2c`: because $(a_j^{\text{pop}})^2$ weights Sociology $16\times$–$27\times$ over Math, greedy CAT selects **40%–49% Sociology items**, causing the short-test advantage on checkpoints at $n=50$ [`RMSE ratio 0.909`] to reverse on trained `S1` states [`1.163` at $n=100$, `report-q1q2.md:65-71`].)*

#### Slide 6 — `A helps B` Empirical Pattern & Confound Check (`Plot 2`): Declarative Recall vs. Procedural Math Across the 15 Subjects
- **Visual**: **[`Plot02_subject_weights_and_gains.png`](file:///D:/surrogates/gemini/figures/Plot02_subject_weights_and_gains.png)**
- **Slide Text Beside Plot**:
  - **1. The Construct-Level Pattern on Seen `S1` States**:
    - Sorting the 15 subject means (`Q2g/s1/flags.csv`, `A3_subject_mean_c`) by 2PL weight $\bar{a}_m^{\text{pop}}$ (which is the same axis as $\bar{b}_m^{\text{pop}}$ since $r = -0.88$):
      - **High-weight Declarative subjects (Top, Red)**: Mean log-odds gain is higher under **GRPO** (with *Philosophy* `0.094 vs. 0.079`, *Sociology* `0.081 vs. 0.078`, and *HS Microeconomics* `0.0789 vs. 0.0784` exceeding the 95th-percentile random-label permutation null in `Q2g-notes.md:56`). At the item level, $\rho(c_{\text{GRPO},j}, a_j^{\text{pop}}) = +0.134\text{ to }+0.149$.
      - **Low-weight Procedural Math subjects (Bottom, Blue)**: Mean log-odds gain is higher under **CoT SFT** (with *Elementary Math* `0.211 vs. 0.097` and *High School Math* `0.133 vs. 0.097` exceeding the 95th-percentile permutation null in `Q2g-notes.md:56`). At the item level, $\rho(c_{\text{SFT},j}, a_j^{\text{pop}}) = -0.184\text{ to }-0.160$. *(In `signature T6`, CoT SFT also shifts the Cosine-NTK kernel $\rho = 0.772$ and transfers $+10.4\text{ to }+11.4$ pts OOD, vs. $0$ for GRPO.)*
  - **2. The Critical Within-Model Confound Check (`Why Population Parameters Are Proxies for the Model's Own Prior`)**:
    - Across the 15 subjects, population difficulty $\bar{b}_m^{\text{pop}}$ correlates **$r = -0.94$** with the target model's **own** initial base log-odds $\overline{L}_{\text{base},m}$!
    - Once $\overline{L}_{\text{base},m}$ is controlled, the partial correlation between $\bar{b}_m^{\text{pop}}$ and $\overline{\Delta L}_{\text{SFT},m} - \overline{\Delta L}_{\text{GRPO},m}$ drops to **$\text{partial } r = 0.17$ (`all` draws) / $0.25$ (`parsed` draws)**!
    - **Insight**: Population $(a_m^{\text{pop}}, b_m^{\text{pop}})$ separates SFT from GRPO across domains *because* declarative recall is where the 1B base model already has higher initial confidence $\overline{L}_{\text{base},m}$!

#### Slide 7 — Within-Model Prior-Scaling Law (`Plot 3` + `Table 2`) & Why Pre-Registered `Q2-A` Item Flags Fail
- **Visual**: **[`Plot03_prior_scaling_scatter.png`](file:///D:/surrogates/gemini/figures/Plot03_prior_scaling_scatter.png)** + **Table 2**
- **Slide Text & Math Beside Plot**:
  - **The Within-Model Autoregressive Prior-Scaling Law**:
    $$L_{s,j} = \hat{\tau}_s L_{\text{base},j} + c_s + \varepsilon_{s,j}, \qquad \hat{\tau}_s = r(L_s, L_{\text{base}}) \cdot \frac{\sigma(L_s)}{\sigma(L_{\text{base}})}$$
    - **Outcome RL (`GRPO`, $\hat{\tau}_s > 1$)**: High rank correlation with Base ($r = 0.904$ at Token-0, $0.731\text{–}0.746$ at $K=32$) combined with dispersion expansion ($\sigma_s/\sigma_b \approx 2.09$) yields **regression slope $\hat{\tau}_s = 1.87\text{–}1.92 > 1$** (`1.43–1.46` at $K=32$)—amplifying existing base strengths and weaknesses.
    - **CoT SFT ($\hat{\tau}_s < 1$)**: Regression slope of item log-odds on base log-odds is **`0.35–0.54` at Token-0 (`0.49–0.50` at $K=32$; `0.62–0.73` in free-text CoT)**, lifting items where Base was negative. *(By `Rule 4`, $\hat{\tau}_{\text{SFT}} < 1$ is a regression slope due to item reordering $r = 0.289$ in free-text CoT while $\sigma_s/\sigma_b = 2.35$, NOT a claim that SFT shrinks marginal dispersion!)*
  - **Why Pre-Registered `Q2-A` Fails**: Once global scaling of $L_{\text{base},j}$ is accounted for, subtracting static population $a_j^{\text{pop}}(\hat{\theta}_s - b_j^{\text{pop}})$ does not isolate reproducible item-level learning (`0/6` pairs in `S1`, `1/12` in `Q1`).
- **Table 2 — Within-Model Regression Decomposition Across Readouts & Pre-Registered `Q2-A` Verdict**:

| Readout Modality & Operator | Source (`results/`) | Correlation $r(L_s, L_{\text{base}})$ | SD Ratio $\sigma(L_s)/\sigma(L_{\text{base}})$ | Regression Slope $\hat{\tau}_s$ on $L_{\text{base}}$ | $R^2$ of Item Change $\Delta L_s$ on $L_{\text{base}}$ | Pre-Registered `Q2-A` Verdict (`report-q1q2.md:37-46`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Token-0 Forced Read (`SFT`, 3 seeds)** | `T9r/items_wide.csv` | `0.806–0.835` | `0.44–0.65` | **`0.35–0.54`** (Seed 0: `0.54`) | **`59.2%–87.3%`** ($\Delta R^2$ of $z_j^{\text{pop}} \le 1.04\%$) | — |
| **Token-0 Forced Read (`GRPO`, 3 seeds)** | `T9r/items_wide.csv` | `0.900–0.910` | `2.07–2.13` | **`1.87–1.92`** (Seed 0: `1.87`) | **`48.7%–51.4%`** ($\varepsilon=10^{-4}$; `63.5%–68.5%` at $10^{-6}$) | — |
| **$K=32$ Sampled Draws (`SFT`, 3 seeds)** | `Q2g/s1/` | `0.592–0.611` | `0.82–0.83` | **`0.49–0.50`** | **`27%–29%`** | **Rule 1 FAILS on `S1` (seen)**: top-10% IRT change flags $|c_{s,j}|$ do not repeat across seeds more than classical $|k_{s,j}|$ (`mean diff +0.0091`, interval $>0$ in **`0 of 6` pairs**). |
| **$K=32$ Sampled Draws (`GRPO` / `Q1`)** | `Q2g/s1/` & `Q2g/q1/` | `0.731–0.746` | `1.91–1.99` | **`1.43–1.46`** | **`9.5%–10.1%`** | **Rule 1 FAILS on `Q1` (unseen)**: IRT minus classical reproducibility `+0.0221`, interval $>0$ in **`1 of 12` pairs** (chance `0.26`; planted FDS `0.31` vs. `0.15` limit). |
| **Free-Text CoT Greedy (`SFT`, 3 seeds)** | `T9g/s3/matched.csv` | `0.275–0.305` | `2.26–2.39` | **`0.62–0.73`** | Descriptive recomputation | Proves $\hat{\tau}_{\text{SFT}} < 1$ is regression attenuation from item reordering ($r = 0.289$) while $\sigma_s/\sigma_b = 2.35$. |

---

### Act III: `B helps A` — What Counterfactual LLM Rollouts Teach Psychometric Theory (`S1`, `Q1`, `Q2g`, Slides 8–9)

#### Slide 8 — `B helps A` #1 (`Plot 4`): The $(\theta_s, \beta_s)$ Generative Respondent Model & Majority-Vote Crossover
- **Visual**: **[`Plot04_readout_crossover.png`](file:///D:/surrogates/gemini/figures/Plot04_readout_crossover.png)**
- **Slide Text & Math Beside Plot**:
  - **Why 1-Parameter Point-Mass IRT Fails for Reasoning LLMs**:
    When model state $s$ samples $K$ reasoning paths on item $j$, each rollout has latent margin $Z_{sjk} = \mu_{sj}(\theta_s) + \beta_s^{-1}\varepsilon_{sjk}$, where $\mu_{sj}$ is **item location** and $\beta_s$ is **within-item trajectory concentration / inverse temperature** (measured by rollout entropy $H_s$):
    - **Single-Draw Accuracy (`Mean@16`)** $\mathbb{E}_j[\sigma(\beta_s \mu_{sj})]$ increases when $\beta_s \uparrow$ (for positive-mode items), whereas **Majority Vote (`Maj@16`)** approaches $\mathbb{P}_j(\mu_{sj} > 0)$, which depends on **how many items have a correct plurality mode ($\mu_{sj} > 0$)**!
  - **Empirical Crossover on `Unit S1` (`1,500 items × 16 draws × 2 orders`)**:
    - **Single Draw (`Mean@16`)**: **GRPO beats CoT SFT (`47.9% vs. 43.9%`, Base `38.2%`; ratio `+4.04 to +6.14 SE`)** by collapsing rollout entropy ($H_s: 0.96 \to 0.28$ nats, `stable_12 = 78.5%`).
    - **16-Draw Majority Vote (`Maj@16`, with vertical bars showing `[Tie=Wrong, Tie=Correct]`)**: GRPO gains almost nothing from voting (`+0.2%` / `+1.1%` tie-correct) and drops `Pass@16` from **`93.0%` to `65.6%`**. By contrast, **CoT SFT preserves diversity ($H_s = 0.94$, `Pass@16 = 96.6%`) and crosses above GRPO on `Maj@16`**:
      - `Tie = Wrong` (bottom of bar): **SFT `50.4%` vs. GRPO `48.1%`** (exceeds by $>2\text{ SE}$ in **`2 of 3` seeds pooled** [`-2.77, -0.75, -2.58`], firing `S1` Rule 3; `0 of 3` on order 1 alone, `S1.md:78`).
      - `Tie = Correct` (top of bar): **SFT `55.5%` vs. GRPO `49.0%`** (`4.65 to 7.16 SE` in `3 of 3` seeds, `S1.md:79`).
    - *(Scope note on mode-change test, `S1.md:85-86, state.md:32`: raw $k=16$ mode change is `56.0%–56.4%` SFT vs. `39.8%–43.6%` GRPO, but under the symmetric $k=8$ null, excess mode change scales with dispersion [`M.md:138-145`] and holds in `1 of 3` seeds, so the reorder test was withdrawn.)*

#### Slide 9 — `B helps A` #2 (`Table 3`): All 19 Model States & The Person-Fit Sign-Flip Theorem ($l_z(K=1)$ vs. $l_z(K=32)$)
- **No Figure Needed — Clean `Table 3`**:
  - Across all **19 evaluated states** (`1 Base + 3 S1 SFT + 3 S1 GRPO + 12 Q1 RL`), every `Base -> RL` run is trapped at low entropy ($H_s \in [0.12, 0.32]$) with `Maj@16 <= 49.6%` (max `q1_ctt_1` at `49.6%`), while only `Base -> CoT SFT` reaches `Maj@16 > 50%`.
  - **The Person-Fit Sign-Flip Theorem ($K=1$ vs. $K=32$)**:
    - **`Q2a` $l_z(K=1)$** evaluates **1 binary order-0 draw** at EAP $\hat{\theta}$: because GRPO collapses rollout variance ($\beta_s \uparrow$), its single draw looks hyper-consistent/Guttman (**$l_z(K=1) = +2.24 > +1.645$**).
    - **`Q2g` $l_z(K=32)$** evaluates **32 binomial trials** at `Q2g` $\hat{\theta}$: because expected squared residual scales as $\mathbb{E}[\text{SSR}_{sj}] \approx \frac{p_{sj}q_{sj}}{\hat{p}_{sj}\hat{q}_{sj}} + K \frac{(p_{sj} - \hat{p}_{sj})^2}{\hat{p}_{sj}\hat{q}_{sj}}$, multiplying systematic curve divergence ($L_{\text{base},j} \neq z_j^{\text{pop}}$) by $K=32$ flips GRPO to massive negative misfit (**$l_z(K=32) = -482$ to $-577$**)!

| Model Family (`States`, `Llama-3.2-1B-Instruct`) | Letter Entropy $H_s$ (nats) | Modal Stability `stable_12` ($\ge 12/16$) | Single Draw `Mean@16` | Majority Vote `Maj@16` (`Tie=Wrong` / `Tie=Correct`) | 16-Draw Coverage `Pass@16` | Voting Gain (`Maj@16 - Mean@16`) | 1-Draw Person Fit $l_z(K=1)$ (`Q2a`) | 32-Draw Person Fit $l_z(K=32)$ (`Q2g`, `all` draws) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (`t9_base`, 1 state)** | `0.963` | `19.4%` (`0.269` per-order) | `38.2%` | `41.4%` / `45.9%` | `93.0%` | `+3.2%` / `+7.7%` | **`-3.71`** | **`-212`** (`mssr 11.6`) |
| **`S1` CoT SFT (`t9_sft`, 3 seeds, 70u)** | `0.943` | `21.9%` (`0.245–0.248` per-order) | `43.9%` (`43.4%–44.4%`) | **`50.4%` / `55.5%`** | **`96.6%`** | **`+6.5%` / `+11.6%`** | **`-6.94`** (`-8.38 to -4.28`) | **`-225`** (`-231 to -218`, `mssr 47–81`) |
| **`S1` GRPO (`t9_rl`, 3 seeds, 666u, $G=5$)** | `0.281` | `78.5%` (`0.797–0.836` per-order) | **`47.9%`** (`47.3%–48.6%`) | `48.1%` / `49.0%` | `65.6%` | `+0.2%` / `+1.1%` | **`+2.24`** (`+2.03 to +2.52`) | **`-482`** (`-498 to -472`, `mssr 460–1102`) |
| **`Q1` GRPO (`q1_grpo`, 3 seeds, 333u, $G=5$)** | `0.320` | `75.7%` | `46.85%` | `47.4%` | `68.5%` | `+0.6%` | — | **`-451`** (`-458 to -444`, `mssr 424–513`) |
| **`Q1` CTT Band Critic (`q1_ctt`, 3 seeds, $G=1$)** | `0.216` | `86.6%` | `47.22%` | `48.5%` (max `49.6%`) | `61.8%` | `+1.3%` | — | **`-538`** (`-570 to -518`, `mssr 300–464`) |
| **`Q1` 2PL Unweighted (`q1_pop0`, 3 seeds, $G=1$)** | `0.123` | **`94.2%`** | `46.90%` | `48.2%` | `54.6%` | `+1.3%` | — | **`-577`** (`-612 to -554`, `mssr 150–416`) |
| **`Q1` 2PL Weighted (`q1_pop1`, 3 seeds, $G=1$)** | `0.197` | `88.1%` | `47.48%` | `48.5%` | `60.6%` | `+1.0%` | — | **`-549`** (`-560 to -543`, `mssr 381–465`) |

---

### Act IV: From Static RL Critic Failure (`Unit Q1`) to the `A <-> B` High-Leverage Bet (`Unit M1`, Slides 10–12)

#### Slide 10 — `Cannot` #2 (`Unit Q1`, `Plot 5`): Why Static Population 2PL Critics Fail to Beat Unweighted GRPO or Classical Pass-Rate Bands
- **Visual**: **[`Plot05_q1_training_curves.png`](file:///D:/surrogates/gemini/figures/Plot05_q1_training_curves.png)**
- **Slide Text Beside Plot**:
  - **Pre-Registered `Unit Q1` Verdict (`results/Q1/contrasts.csv`, 3 seeds per arm, 333 updates)**:
    - **`pop0` minus `ctt` (Unweighted 2PL critic vs. classical moving pass-rate band)**: Per-seed $d_s = +0.0045, -0.0061, -0.0079$; **$\text{mean } d = -0.0032$, $\text{SE}_d = 0.0039$** $\implies$ **Rule 1 FAILS** and **Rule 2 FAILS** ($\text{mean } d - 2\text{SE}_d = -0.0109 < -0.01$). *(Mathematically, a 1PL/Rasch critic is identical in expectation to `ctt`, `M.md:97`, and adding static $a_j^{\text{pop}}$ injects an external axis whose online $\hat{\theta}_t$ correlates $-0.20$ with bank `Mean@16` after step 0, `report-q1q2.md:29`.)*
    - **`pop1` minus `pop0` ($a_j^{\text{pop}}$-weighted vs. unweighted 2PL critic)**: Per-seed $d_s = -0.0028, +0.0051, +0.0153$; **$\text{mean } d = +0.0059$, $\text{SE}_d = 0.0052$** $\implies$ **Rule 3 FAILS**.
    - **Rule 4 Holds**: All 4 arms converge to a **`0.63 percentage point` band (`46.85%–47.48%`)**.
  - **Why All `Base -> RL` Runs Plateau After Step 83**:
    - In **Seed 0** (the only seed with mid-run checkpoints), **`83.7%–94.5%` of total gain occurs by step 83**.
    - In $G=5$ `grpo`, `zero_adv_share` (`0/5` or `5/5` correct rollouts) rises from `43.8%` (`W1` raw) to **`63.1%` after step 83 (`84%–91%` peak)**, coinciding with the plateau; meanwhile, $G=1$ critics (`ctt, pop0, pop1`) have no zero-advantage groups (`r - \hat{p}_j \neq 0$, fixed length divisor $L_0 = 78.98$) yet also plateau after step 83 as they over-concentrate onto `Base`'s modal answers (`stable_12 = 86.6%–94.2%`).

#### Slide 11 — `A <-> B` High-Leverage Bet (`Schematic 2` + `Table 4`): Operator Composition & Within-Model ZPD Curriculum (`Unit M1`)
- **Visual**: **[`Schematic02_multidraw_and_m1_zpd.png`](file:///D:/surrogates/gemini/figures/Schematic02_multidraw_and_m1_zpd.png)** + **Table 4**
- **Slide Text Beside Schematic**:
  - **Why `Unit M1` Is a Win–Win High-Leverage Bet ("A Bet You'd Be Happy Losing")**:
    1. **Operator Composition (`SFT -> RL` instead of `Base -> RL`)**: Since RL is a concentration operator ($\beta_s \uparrow$) bounded by the starting state's modal ceiling (`Maj@16 = 41.4%–45.9%` in `Base`), starting RL from **`t9_sft_0..2_70` (`Maj@16 = 50.4%–55.5%`, `Pass@16 = 96.6%`)** tests whether RL can convert SFT's `+6.5% to +11.6%` majority-vote surplus into single-draw `Mean@16` accuracy.
    2. **Within-Model Psychometric ZPD Curriculum (`M1c`)**: Instead of weighting gradients by external 102-LLM $a_j^{\text{pop}}$, 1PL Fisher information for a model's own learning is **Bernoulli reward variance $I_{s,j} = p_{s,j}(1 - p_{s,j})$**, which peaks at $p_{s,j} = 0.50$. On `W1` ($N = 10,682$), **`48.3%` (`5,164` prompts)** sit at the exterior (`0, 1, 7, 8 of 8`) where $p(1-p) \approx 0$. Because SFT reorders item difficulty relative to Base ($r = 0.29\text{–}0.60$), `M1c` samples $k=8$ rollouts on `t9_sft_0..2_70` first to filter training to **SFT's own interior ZPD band $p_{\text{SFT},j} \in [0.2, 0.8]$ (`2..6 of 8`)**!
    3. **Why Both Outcomes (`+1` or `-1`) Are Landmark Findings**:
       - **If `+1` (Synergistic Composition)**: `SFT -> RL` beats both `t9_sft` (`43.9%`) and `q1_grpo` (`46.85%`) in `3/3` seeds ($>2.92\text{ SE}_d$), breaking the 15-run `Base -> RL` wall via operator composition + within-model ZPD targeting.
       - **If `-1` (Destructive Interference Law)**: Standard outcome RL collapses SFT's multi-path CoT diversity onto short-path attractors and **destroys SFT's `Maj@16` ceiling** ($<-2.92\text{ SE}_d$ in `3/3` seeds)—proving that preserving reasoning diversity requires entropy-regularized or process-level credit assignment.
- **Table 4 — Pre-Registered `Unit M1` Experimental Matrix (`Design Only — Running on Cluster`)**:

| Arm | Starting State | Training / Inference Procedure | Prompt Curriculum Filter | Pre-Registered Decision Rule (`notes/units/M1.md`) |
| :--- | :---: | :--- | :--- | :--- |
| **`M1a` (Inference Control)** | `t9_sft_0..2_70` | **Zero RL training**; evaluate `t9_sft` at low temperature ($T = 0.2$ vs. $T = 0.7$) | — | Measures how much of SFT's `Maj@16 - Mean@16` gap (`+6.5% to +11.6%`) is harvested by temperature scaling alone without RL updates. |
| **`M1b` (Unfiltered `SFT -> GRPO`)** | `t9_sft_0..2_70` | $G=5$ GRPO for 333 updates (`Q1` exact settings; save & sample step 83 & step 333) | Full unfiltered `W1` prompt pool ($N = 10,682$) | **Rule 1**: **`+1`** if `Mean@16` beats both `t9_sft` & `q1_grpo` in `3/3` seeds ($>2.92\text{ SE}_d$); **`-1`** if `Maj@16` or `Pass@16` drops below `t9_sft` in `3/3` seeds ($<-2.92\text{ SE}_d$); **`0`** otherwise. |
| **`M1c` (ZPD-Filtered `SFT -> GRPO`)** | `t9_sft_0..2_70` | Sample $k=8$ rollouts on `t9_sft_0..2_70`, then run $G=5$ GRPO for 333 updates (save step 83 & 333) | **SFT's own interior pass-rate band $p_{\text{SFT},j} \in [0.2, 0.8]$ (`2..6 of 8` correct)** | **Rule 2**: Tests whether within-model ZPD curriculum filtering (`p_SFT(1 - p_SFT)`) beats unfiltered `M1b` on bank `Mean@16` in `3/3` seeds ($>2.92\text{ SE}_d$). |

#### Slide 12 — Answering the Title: What Item Parameters CAN (and CANNOT) Tell Us
- **Complete Synthesis Table (`Cannot` vs. `Can`)**:

| Dimension | What Static Cross-Population Parameters $(a_j^{\text{pop}}, b_j^{\text{pop}})$ **CANNOT** Do (And Why) | What Item Parameters & Multi-Draw Generative Psychometrics **CAN** Tell Us (`A <-> B`) |
| :--- | :--- | :--- |
| **1. Cross-Population Transfer (`Unit E1`)** | **Cannot transfer human student difficulty to LLMs** (`Plot 1`: `61/61` LLMs have $Z_h < -1.96$; 2PL rank $\rho = +0.089$, classical error rate $r = +0.139$ over 704 items). | **Diagnoses Population Non-Ergodicity**: Proves LLMs and human students do not share a common item-difficulty hierarchy (`closure.md`). |
| **2. Stage Attribution & Construct Structure (`Q2-B`, `W0`, `S1`)** | **Cannot provide a construct-neutral 1D ranking on unseen `Q1` models** (`Table 1`: Rule 2 fails on unseen `Q1` models [`2/6` B1 pairs], which span only `2.1%` in accuracy). | **`A helps B` — Explains the Attribution Reversal via Declarative vs. Procedural Weighting (`Schematic 1`, `Plot 2`)**: Shows why 2PL $\hat{\theta}$ flips `S1` credit (`60/40` Accuracy $\to$ `21/79` $\hat{\theta}$): $a_j^{\text{pop}}$ weights declarative recall $5.2\times$ over math, where GRPO amplifies high-$a_m$ priors and CoT SFT lifts low-$a_m$ math (with $\bar{b}_m^{\text{pop}}$ tracking $\overline{L}_{\text{base},m}$ at $r = -0.94$, partial $r = 0.17$). |
| **3. Item-Level Change Detection (`Q2-A`, `T9r`, `T9g`)** | **Cannot isolate reproducible item-level learning** (`Table 2`: Rule 1 fails in `0/6` seen `S1` pairs and `1/12` unseen `Q1` pairs; planted FDS `0.31 > 0.15`). | **Establishes the Within-Model Prior-Scaling Law (`Plot 3`, `Table 2`)**: Item shifts regress on the model's **own** base confidence $L_{\text{base},j}$ ($L_{s,j} = \hat{\tau}_s L_{\text{base},j} + c_s$), separating RL gain amplification ($\hat{\tau}_s = 1.87 > 1$) from CoT SFT item reordering ($\hat{\tau}_s = 0.54 < 1$, $r = 0.289$ in free-text CoT). |
| **4. Multi-Draw Evaluation (`S1`, `Q1`, `Q2a`, `Q2g`)** | **Single-draw point-mass IRT ($\theta_s$ alone) cannot capture reasoning diversity** or reconcile $l_z(K=1) = +2.24$ with $l_z(K=32) = -482$. | **`B helps A` — The $(\theta_s, \beta_s)$ Generative Respondent Model (`Plot 4`, `Table 3`)**: Separates **item location $\theta_s$** from **trajectory concentration $\beta_s$**, explaining both the **Majority-Vote Crossover** (SFT `Maj@16 = 50.4%–55.5%` overtaking GRPO `48.1%–49.0%`) and the **$l_z(K=1) \to l_z(K=32)$ sign flip**. |
| **5. Post-Training & Curriculum Design (`Unit Q1` $\to$ `Unit M1`)** | **Static 102-LLM 2PL critics (`pop0, pop1`) cannot beat unweighted GRPO or classical `ctt`** (`Plot 5`: `pop0 - ctt` mean $d = -0.0032$, $\text{SE}_d = 0.0039$; all 4 arms plateau in `46.85%–47.48%` after step 83). | **`A <-> B` — Guides Operator Composition & Within-Model ZPD Curriculum (`Schematic 2`, `Table 4`)**: Replaces static cross-model parameters with **SFT's own pass-rate parameter $p_{\text{SFT},j} \in [0.2, 0.8]$ (`2..6 of 8`)** to maximize active Bernoulli variance $p_j(1-p_j)$ when harvesting SFT's majority-vote ceiling in `Unit M1`. |
