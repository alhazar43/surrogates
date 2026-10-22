# Detailed Slide & Figure/Table Plan for Monday, Oct 26 Talk

- **Locked Title**: *Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?*
- **Target Audience**: Measurement, IRT / Psychometrics, and Neuro-Cognitive Science researchers.
- **Core Thread (Anchored to Item Parameters $(a_j, b_j)$ on Every Slide)**:
  - Distinguish **cross-model static 2PL item parameters $(a_j^{\text{pop}}, b_j^{\text{pop}})$** (calibrated across 102 LLMs in `W0` or 4,918 humans in `E1`) from **within-model item parameters** (the model's own base difficulty $L_{\text{base},j}$, trajectory discrimination $a_j^{\text{traj}}$, and SFT pass-rate difficulty $p_{\text{SFT},j}$).
  - Show **both `CANNOT` and `CAN` with 100% existing committed data on Slides 1–10**, and design **Slide 11 (`Unit M1`)** so it works seamlessly **without** waiting for `M1b/M1c` (using existing `Q1/trajectory.csv`, `S1/stats.csv`, and today's 1-GPU `M1a` + `sft_pass_rate_0..2.csv`), while keeping a drop-in slot if `M1b/M1c` (`u83` or `u333`) land by Oct 25–26.

---

## Part 1: Inventory of Figures & Tables That Can Be Built RIGHT NOW (Zero Cluster Wait)

All 8 core figures (**F1–F8**) and 5 core tables (**T1–T5**) below can be generated **today** directly from committed CSVs in `item-params-llm/results/` and `signature/results/`:

| ID | Type | Title & Visual Design | Exact Source CSVs (100% Committed Now) | Slide |Status |
| :---: | :---: | :--- | :--- | :---: | :---: |
| **F1** | **Figure** (2-panel scatter) | **Molenaar's Non-Ergodicity of Item Parameters**:<br>• **Panel A**: Human 2PL $b_j^{\text{human}}$ (4,918 students) vs. LLM $b_j^{\text{LLM}}$ (61 LLMs) on 600 Eedi math items ($r = -0.013, \rho = 0.089$; `61/61` LLMs $l_z < -1.96$).<br>• **Panel B**: Cross-model 102-LLM discrimination $a_j^{\text{pop}}$ vs. single-model RL trajectory slope $a_j^{\text{traj}}$ ($r = 0.04$ on MMLU; $r = -0.22$ on GSM8K). | `results/E1/difficulty.csv`<br>`results/E1/llm_scores.csv`<br>`results/W0/items_bank.csv`<br>`results/T9r/items_wide.csv` | Slide 2 (`CANNOT #1`) | **Ready Now** |
| **F2** | **Figure** (2-panel regression) | **Why Static $(a_j^{\text{pop}}, b_j^{\text{pop}})$ Cannot Isolate Item Learning (`Q2-A`) — The Within-Model Prior-Scaling Law**:<br>• **Panel A**: Post-training log-odds $L_{s,j}$ vs. the model's OWN Base difficulty $L_{\text{base},j}$ ($R^2 = 59\%\text{–}87\%$ for SFT, $49\%\text{–}69\%$ for GRPO; residual $\Delta R^2$ of population $z_j^{\text{pop}} \le 1.04\%$).<br>• **Panel B**: Slope decomposition $\hat{\tau}_s = r(L_s, L_{\text{base}}) \cdot (\sigma_s / \sigma_{\text{base}})$ contrasting **SFT item reordering ($r = 0.289$)** vs. **RL gain scaling ($r = 0.908, \sigma_s/\sigma_{\text{base}} = 1.53\text{–}2.09$)**. | `results/T9r/items_wide.csv`<br>`results/T9g/s3/matched.csv`<br>`results/Q2g/s1/flags.csv`<br>`results/W0/items_bank.csv` | Slide 3 (`CANNOT #2`) | **Ready Now** |
| **T1** | **Table** | **Table 1.1: Verification of $\hat{\tau}_s = r \cdot (\sigma_s/\sigma_{\text{base}})$ & Population $R^2$ Across All 3 Readout Modalities** (Token-0 Logits, Free-Text CoT Greedy Logits, $K=32$ Sampled Rollouts). | `verify_master_plan.py` (`Part 1 & 2`) | Slide 3 | **Ready Now** |
| **F3** | **Figure** (2-panel domain map + bar chart) | **`CAN #1` (Headline Figure): Item Parameters $(a_m^{\text{pop}}, b_m^{\text{pop}})$ Predict SFT vs. RL Domain Gains ($R^2 = 76.9\%$ Across 15 MMLU Subjects)**:<br>• **Panel A**: 15 MMLU subjects in the $(\bar{a}_m^{\text{pop}}, \bar{b}_m^{\text{pop}})$ item-parameter plane ($r = -0.8841$), cleanly separating **Low-$a$/High-$b$ Procedural Quantitative domains** from **High-$a$/Low-$b$ Declarative Factual domains**.<br>• **Panel B**: Operator advantage $\overline{\Delta L}_{\text{SFT},m} - \overline{\Delta L}_{\text{GRPO},m}$ vs. domain item difficulty $\bar{b}_m^{\text{pop}}$ ($r = +0.8771, R^2 = 76.9\%, p = 1.8\times 10^{-5}$) and discrimination $\bar{a}_m^{\text{pop}}$ ($r = -0.8015, R^2 = 64.2\%$). | `results/W0/items_bank.csv`<br>`results/Q2g/s1/flags.csv`<br>`results/Q2g/q1/flags.csv` | Slides 4 & 5 (`CAN #1` / `A helps B`) | **Ready Now** |
| **T2** | **Table** | **Table 1.2: Full 15-Subject Neuro-Cognitive Double Dissociation Table** (Sorted by $\bar{a}_m^{\text{pop}}$ from `0.278` High School Math to `1.448` Marketing: SFT wins `5/6` procedural domains; GRPO wins `8/8` declarative domains). | `verify_master_plan.py` (`Part 4`) | Slide 5 (`CAN #1`) | **Ready Now** |
| **F4** | **Figure** (crossover + entropy-ceiling plane) | **The $(\theta_s, \beta_s)$ Generative Respondent Duality & `S1` Readout Crossover**:<br>• **Panel A**: Evaluation functional crossover across $K=32$ rollouts: GRPO wins single-draw `Mean@16` (`0.479` vs. `0.439`), while SFT overtakes on `Maj@16` (`0.504 / 0.529 / 0.555*` vs. `0.481 / 0.485 / 0.490*`) and `Pass@16` (`0.966` vs. `0.656`), exposing SFT's **`+0.065 to +0.116` unharvested Condorcet voting gap**.<br>• **Panel B**: All 19 states plotted in the `(Rollout Entropy H_s, Modal Ceiling Maj@16)` plane. | `results/S1/stats.csv`<br>`results/S1/rewrite1_parsed.csv`<br>`results/Q1/stats.csv` | Slide 6 (`B helps A`) | **Ready Now** |
| **T3** | **Table** | **Table 2.1: Master 19-State $(\theta_s, \beta_s)$ Table** (`stable_12`, $H_s$, `Mean@16`, `Maj@16`, `Pass@16`, Condorcet Gap, $l_z^{(1)}$, $l_z^{(32)}$). | `verify_master_plan.py` (`Part 3`) | Slide 6 | **Ready Now** |
| **F5** | **Figure** (stacked bar + item-weight curve) | **`CAN #2`: How Item Discrimination $a_j^{\text{pop}}$ Explains the `60%/40%` vs. `21%/79%` Attribution Reversal (`Q2-B`)**:<br>• **Panel A**: Side-by-side post-training credit split (`Base -> SFT -> GRPO`): Classical Accuracy gives **`60% SFT / 40% GRPO`**, whereas 2PL $\hat{\theta}$ flips it to **`21% SFT / 79% GRPO`**.<br>• **Panel B**: Item gain $\Delta p_j$ vs. $a_j^{\text{pop}}$ showing $\operatorname{Cov}(a_j^{\text{pop}}, \Delta p_{\text{SFT}}) < 0$ ($\rho = -0.17$, downweighting SFT's low-$a_j$ math gains) vs. $\operatorname{Cov}(a_j^{\text{pop}}, \Delta p_{\text{GRPO}}) > 0$ ($\rho = +0.14$, amplifying GRPO's high-$a_j$ factual sharpening). | `results/Q2g/s1/personfit.csv`<br>`results/Q2g/s1/precision.csv`<br>`results/Q2g/s1/flags.csv` | Slide 7 (`CAN #2` / `B helps A`) | **Ready Now** |
| **F6** | **Figure** (2-panel curve) | **Two Multi-Draw Psychometric Theorems (`l_z` Sign Flip & CAT vs. Random RMSE Crossover)**:<br>• **Panel A**: Person-fit $l_z^{(1)} = +2.24$ (single draw looks Guttman hyper-consistent) flipping to $l_z^{(32)} = -482$ to $-577$ at $K=32$ draws, tracking concentration $\beta_s$ monotonically.<br>• **Panel B (`CANNOT #3`)**: Short-test RMSE as $n: 25 \to 50 \to 100$ (`Q2a/Q2c`): greedy CAT picks high-$a_j^{\text{pop}}$ items (`40%–49%` Sociology), locking in a `+0.04 to +0.09` asymptote bias and losing to $1/\sqrt{n}$ Classical random sampling by $n=100$. | `results/Q2a/personfit.csv`<br>`results/Q2g/s1/personfit.csv`<br>`results/Q2g/q1/personfit.csv`<br>`results/Q2a/cells.csv`<br>`results/Q2c/cells.csv` | Slides 8 & 9 (`B helps A` & `CANNOT #3`) | **Ready Now** |
| **F7** | **Figure** (2-panel training trajectory) | **`CANNOT #4`: Why Weighting RL by Static $(a_j^{\text{pop}}, b_j^{\text{pop}})$ Fails & All 15 `Base -> RL` Runs Hit the `0.479` Wall by Update 83 (`Q1`)**:<br>• **Panel A**: Training curves across updates `0 -> 83 -> 166 -> 250 -> 333` (`Q1/trajectory.csv`) for `grpo, ctt, pop0, pop1`: `84.4%–100%` of the gain finishes by update 83, and all 4 arms converge to `0.4685–0.4748` (`0.63%` band), bounded by Base's `Maj@16` (`0.414 / 0.459*`).<br>• **Panel B**: Zero-advantage share (`Q1/traces/*.csv`) rising from `35.0%` (`W1`) to `63.1%` mean after update 83 (`84%–91%` peak), plus why $G=1$ critics (`pop0: 0.942`) collapse diversity more than $G=5$ `grpo` (`0.757`). | `results/Q1/trajectory.csv`<br>`results/Q1/traces/grpo_0..2.csv`<br>`results/Q1/stats.csv`<br>`results/Q1b/length_check.csv` | Slide 10 (`CANNOT #4`) | **Ready Now** |
| **F8** | **Figure** (modular 2-panel + slot) | **`CAN #3` / `A <-> B`: Using the SFT Model's OWN Item Difficulty Parameter $p_{\text{SFT},j}$ to Guide `SFT -> RL` (`Unit M1`)**:<br>• **Panel A (Ready Now)**: The `Base -> RL` wall (`0.479`, bounded by Base `Maj@16 = 0.414 / 0.459*`) vs. SFT's unlocked modal ceiling (`Maj@16 = 0.504 / 0.529 / 0.555*`, `Pass@16 = 0.966`) and `+0.065 to +0.116` Condorcet gap.<br>• **Panel B (Ready Today from 1-GPU Chain)**: Base pass-rate distribution `W1/pass_rate.csv` ($p_{\text{base},j}$) vs. SFT pass-rate distribution `M1/sft_pass_rate_0..2.csv` ($p_{\text{SFT},j}$, highlighting the SFT ZPD band $p_{\text{SFT},j} \in [0.2, 0.8]$) + `M1a` ($T=0.2$) readout.<br>• **Panel C (Modular Slot)**: Pre-registered `+1 / -1 / 0` outcome matrix for `M1b` (`t9_sft -> GRPO`) and `M1c` (`t9_sft -> ZPD-GRPO`) (plug in numbers if finished by Oct 25–26, or present as live pre-registered test). | `results/S1/stats.csv`<br>`results/Q1/trajectory.csv`<br>`results/W1/pass_rate.csv`<br>*(+ today's `sft_pass_rate_0..2.csv` & `M1a`)* | Slide 11 (`CAN #3` / `A <-> B`) | **Panel A Ready Now; Panel B Today; Panel C Modular** |
| **T4 & T5** | **Tables** | **T4**: Full-Bank Evaluation Readings A & B Summary (`Table 2.2`).<br>**T5**: Synthesis Matrix of `What Item Parameters CAN vs. CANNOT Tell Us` Across Evaluation & Training. | `verify_master_plan.py` | Slides 7 & 12 | **Ready Now** |

---

## Part 2: Slide-by-Slide Detailed Plan (12 Slides, ~30–40 Min Talk)

### Act I: The Hook & What Static Cross-Model Item Parameters $(a_j^{\text{pop}}, b_j^{\text{pop}})$ CANNOT Do (Slides 1–3)

- **Slide 1 — Title & Framing the Psychometric Question**
  - **Title**: *Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?*
  - **Key Message for Audience**: Recent ML benchmarks (*Fluid Benchmarking*, *tinyBenchmarks*, *SEELE*, *RIDE*) fit 2PL item parameters $(a_j^{\text{pop}}, b_j^{\text{pop}})$ across a crowd of 100+ LLMs and use them as drop-in replacements for test scoring (CAT / $\hat{\theta}$) and RL reward weighting. We audit this paradigm across `1.48M+` rollouts (`E1`: 4,918 humans vs. 61 LLMs; `W0`: 102 LLMs; `S1`: 7 SFT/RL states; `Q1`: 12 RL runs; `Unit M1`).
  - **Visual**: Clean 2×2 matrix previewing the talk's answer: **Static Cross-Model $(a_j^{\text{pop}}, b_j^{\text{pop}})$ vs. Within-Model $(L_{\text{base},j}, p_{s,j}, \beta_s)$** across **Evaluation** and **Training**.

- **Slide 2 — `CANNOT #1`: Molenaar's Non-Ergodicity (Between-Population Item Parameters $\neq$ Within-Model Trajectory)**
  - **Key Message**: In psychometrics, Molenaar (2004) showed that cross-sectional between-subject structure is only valid within an individual if the process is ergodic. Static item parameters violate ergodicity at two levels:
    1. **Human vs. LLM (`E1`)**: On 600 Eedi diagnostic math items, human 2PL difficulty $b_j^{\text{human}}$ (4,918 students, reliability $0.985$) is orthogonal to LLM difficulty $b_j^{\text{LLM}}$ ($r = -0.013, \rho = 0.089$), and `61/61` LLMs reject the human 2PL response model ($l_z < -1.96$).
    2. **LLM Zoo vs. Single-Model Post-Training (`W0` vs. `S1/Q1` & `Gate 1`)**: Across 102 LLMs, discrimination $a_j^{\text{pop}}$ measures *how strongly an item separates 0.5B from 70B models across pretraining corpora*, which is orthogonal to a single 1B model's RL learning trajectory slope $a_j^{\text{traj}}$ ($r = 0.04$ on MMLU; $r = -0.22$ on GSM8K).
  - **Visual**: **Figure F1** (2-panel scatter: Panel A Human vs. LLM $b_j$; Panel B $a_j^{\text{pop}}$ vs. $a_j^{\text{traj}}$).

- **Slide 3 — `CANNOT #2`: Why Static $(a_j^{\text{pop}}, b_j^{\text{pop}})$ Cannot Isolate Item-Level Learning (`Q2-A`) — The Within-Model Prior-Scaling Law**
  - **Key Message**: When we tried to use 102-LLM item parameters $z_j^{\text{pop}} = a_j^{\text{pop}}(\theta_s - b_j^{\text{pop}})$ to detect which items changed during post-training (`Q2-A`), IRT item residuals $c_{s,j}$ tied classical raw change (`0/6` in `S1`, `1/12` in `Q1`). Why? Because **`48.7%–87.3%` of post-training item change is governed by the model's OWN initial item difficulty $L_{\text{base},j}$**:
    $$\Delta L_{s,j} = (\tau_s - 1)L_{\text{base},j} + \delta_s + \gamma_s z_j^{\text{pop}} + \varepsilon_{s,j}, \qquad \hat{\tau}_s = r(L_s, L_{\text{base}}) \cdot \frac{\sigma(L_s)}{\sigma(L_{\text{base}})}$$
    Adding population $(a_j^{\text{pop}}, b_j^{\text{pop}})$ adds $\Delta R^2 \le 1.04\%$! Moreover, decomposing $\hat{\tau}_s$ separates our two post-training operators:
    - **Outcome RL (`GRPO`)**: Preserves item difficulty order ($r = 0.908$) while stretching dispersion ($\sigma_s/\sigma_{\text{base}} = 1.53\text{–}2.09 \implies \hat{\tau}_{\text{GRPO}} > 1$).
    - **CoT SFT (70B Distillation)**: Reorders item difficulties ($r = 0.289$ in free-text CoT, $r = 0.733$ in $K=32$ rollouts), causing regression attenuation ($\hat{\tau}_{\text{SFT}} < 1$) even though answer-token dispersion increases ($\sigma_s/\sigma_{\text{base}} = 2.35$).
  - **Visual**: **Figure F2** + **Table T1 (`Table 1.1`)**.

---

### Act II: `A helps B` — What Item Parameters $(a_j^{\text{pop}}, b_j^{\text{pop}})$ CAN Tell Us ($R^2 = 76.9\%$ Double Dissociation, Slides 4–6)

- **Slide 4 — `CAN #1` (Part A): What Do $(a_m^{\text{pop}}, b_m^{\text{pop}})$ Actually Measure? Declarative vs. Procedural Cognitive Constructs**
  - **Key Message**: If $(a_j^{\text{pop}}, b_j^{\text{pop}})$ are not within-model trajectory slopes, what *do* they tell us? Sorting the 15 MMLU subjects by their mean 102-LLM discrimination $\bar{a}_m^{\text{pop}}$ and difficulty $\bar{b}_m^{\text{pop}}$ ($r = -0.8841$) reveals a textbook **Cognitive Construct Axis** (Squire, 1992; Anderson's ACT-R):
    - **High $\bar{a}_m^{\text{pop}}$ (`0.69–1.45`), Low $\bar{b}_m^{\text{pop}}$ (`-0.99 to +0.72`) = Declarative / Semantic Recall** (*Marketing, Sociology, Prehistory, Geography, World History, Biology, Philosophy, Clinical Knowledge*): Factual associations scale monotonically with pretraining size across the 102-LLM zoo, producing steep 2PL discrimination $a_j^{\text{pop}}$.
    - **Low $\bar{a}_m^{\text{pop}}$ (`0.28–0.55`), High $\bar{b}_m^{\text{pop}}$ (`+0.25 to +2.30`) = Procedural / Multi-Step Quantitative Reasoning** (*HS Math, Physics, Statistics, Elementary Math, Econometrics, Macroeconomics*): Multi-step symbolic computation is noisy across standard base models, producing flat $a_j^{\text{pop}}$ and high $b_j^{\text{pop}}$.
  - **Visual**: **Figure F3 (Panel A)** — The 15 MMLU domains plotted in the $(\bar{a}_m^{\text{pop}}, \bar{b}_m^{\text{pop}})$ plane, color-coded by cognitive construct type.

- **Slide 5 — `CAN #1` (Part B): Item Parameters Predict Which Domains Need SFT vs. RL ($R^2 = 76.9\%$ Double Dissociation)**
  - **Key Message**: Projecting our $K=32$ post-training gains ($\overline{\Delta L}_{\text{SFT},m}$ vs. $\overline{\Delta L}_{\text{GRPO},m}$) onto $(\bar{a}_m^{\text{pop}}, \bar{b}_m^{\text{pop}})$ yields a **Neuro-Cognitive Double Dissociation**:
    - **High-$a_m^{\text{pop}}$, Low-$b_m^{\text{pop}}$ (Declarative) $\implies$ RL (`GRPO`) wins `8/8` domains**: Outcome RL acts as **Neural Gain Modulation** (Aston-Jones & Cohen, 2005), sharpening existing factual associations ($r(\bar{a}_m^{\text{pop}}, \overline{\Delta L}_{\text{GRPO},m}) = +0.7464, p = 0.0014$; `0` OOD transfer).
    - **Low-$a_m^{\text{pop}}$, High-$b_m^{\text{pop}}$ (Procedural) $\implies$ CoT SFT wins `5/6` domains (`1 tie`)**: While RL stalls or regresses on math (*Econometrics `-0.575` logits, Macroeconomics `-0.192`, Physics `-0.035`*), 70B CoT SFT injects new multi-step procedures (*Elementary Math `+1.491` logits over GRPO, HS Math `+0.772`*, Cosine-NTK shift $\rho = 0.772$, `+10.4 to +11.4` OOD transfer).
    - **Predictive Power of Item Parameters**: Across all 15 domains, $\bar{b}_m^{\text{pop}}$ explains **`76.9%` of the variance ($r = +0.8771, p = 1.8\times 10^{-5}$)** and $\bar{a}_m^{\text{pop}}$ explains **`64.2%` ($r = -0.8015, p = 0.0003$)** in $\overline{\Delta L}_{\text{SFT},m} - \overline{\Delta L}_{\text{GRPO},m}$!
  - **Visual**: **Figure F3 (Panel B)** + **Table T2 (`Table 1.2`)**.

- **Slide 6 — Formalizing the Respondent: The $(\theta_s, \beta_s)$ Generative Model & The `S1` Readout Crossover**
  - **Key Message**: To capture both location shifts ($\mu_{sj}$) and neural gain concentration ($\beta_s$), we replace point-mass IRT with a two-parameter generative respondent law $Z_{sjk} = \mu_{sj} + \beta_s^{-1}\varepsilon_{sjk} \implies p_{sj} = \sigma(\beta_s \mu_{sj})$. Differentiating single-draw `Mean@K` vs. majority-vote `Maj@K` with respect to concentration $\beta_s$:
    $$\frac{\partial \mathcal{M}_{\text{maj}}}{\partial \beta_s} = 0, \qquad \frac{\partial \mathcal{M}_1}{\partial \beta_s} = \mathbb{E}_j\big[\mu_{sj} p_{sj}(1-p_{sj})\big] > 0$$
    - **Why GRPO wins `Mean@16` (`0.479` vs. `0.439`)**: Collapsing rollout entropy ($H_s: 0.963 \to 0.281$) harvests right-mode items into single draws, leaving almost zero Condorcet voting gain (`Maj@16 - Mean@16 = +0.002` / `+0.011*`) and collapsing `Pass@16` (`0.656`).
    - **Why SFT wins `Maj@16` (`0.504 / 0.529 / 0.555*`) and `Pass@16` (`0.966`)**: SFT shifts procedural item modes (`raw mode change 0.560–0.564` vs. `0.398–0.436` GRPO across the 1,500 bank items) while keeping rollout entropy high ($H_s = 0.943$), creating a massive **`+0.065 to +0.116` Condorcet voting boost**.
  - **Visual**: **Figure F4** + **Table T3 (`Table 2.1`)**.

---

### Act III: `B helps A` — What Counterfactual LLM Rollouts Teach Psychometric Theory (Slides 7–9)

- **Slide 7 — `CAN #2` (Theorem 1): How $a_j^{\text{pop}}$ Explains the `60%/40%` vs. `21%/79%` Attribution Reversal (`Q2-B`)**
  - **Key Message**: If a practitioner evaluates `Base -> SFT -> GRPO` using Classical Accuracy vs. 2PL $\hat{\theta}$, they get opposite conclusions about where learning happened:
    - **Classical Accuracy**: **`60%` of the gain comes from SFT** (`+0.058`) and **`40%` from GRPO** (`+0.039`).
    - **2PL $\hat{\theta}$**: **`21%` comes from SFT** (`+0.164`) and **`79%` from GRPO** (`+0.620`)!
    - **Why? Item Discrimination $a_j^{\text{pop}}$ Explains It Mathematically**: By first-order Taylor expansion of the 2PL score equation:
      $$\Delta\hat{\theta} \propto \overline{\Delta p} + \frac{\operatorname{Cov}_j(a_j^{\text{pop}}, \, \Delta p_j)}{\bar{a}^{\text{pop}}}$$
      Because $a_j^{\text{pop}}$ weights high-$a_j$ declarative items (`1.45`) **$5\times$ more heavily** than low-$a_j$ procedural math items (`0.28`), 2PL $\hat{\theta}$ **downweights SFT's math gains ($\rho = -0.17$, losing `0/3` to Accuracy on `Base vs. SFT`)** and **over-amplifies GRPO's factual sharpening ($\rho = +0.14$, winning `9/9` on `SFT vs. GRPO`)**!
  - **Visual**: **Figure F5** + **Table T4 (`Table 2.2` excerpt)**.

- **Slide 8 — Theorem 2: The Multi-Draw Person-Fit Sign Flip ($l_z^{(1)} = +2.24 \to l_z^{(32)} = -482$)**
  - **Key Message**: In human testing ($k=1$ draw per item), a high-gain RL respondent ($\beta_s > 1$) has $l_z^{(1)} = +2.24 > +1.645$—looking like a hyper-consistent Guttman respondent. Yet when we draw $K=32$ counterfactual rollouts per item, person fit flips to $l_z^{(32)} = -482$ (`S1 grpo`) and `-577` (`Q1 pop0`):
    $$\mathbb{E}[\text{SSR}_{sj}] = \frac{p_{sj}(1-p_{sj})}{\hat{p}_{sj}^{\text{2PL}}(1-\hat{p}_{sj}^{\text{2PL}})} + K \cdot \frac{(p_{sj} - \hat{p}_{sj}^{\text{2PL}})^2}{\hat{p}_{sj}^{\text{2PL}}(1-\hat{p}_{sj}^{\text{2PL}})}$$
    Because $L_{\text{base},j} \neq z_j^{\text{pop}}$, multiplying by $K=32$ exposes the structural divergence between the model's own prior and the population 2PL curve.
  - **Visual**: **Figure F6 (Panel A)**.

- **Slide 9 — `CANNOT #3` (Theorem 3): Option-Order Testlets & Why High-$a_j^{\text{pop}}$ Adaptive CAT Loses to Random Sampling (`Q2a/Q2c`)**
  - **Key Message**:
    1. **Option-Order Testlets (`signature F6–F8`)**: Cyclic option-order variance across just 2 permutations is **$16\times$ larger than $K=32$ binomial sampling variance** ($z_{\text{sd}} = 4.02$), and RL gain modulation ($\tau_{\text{GRPO}} > 1$) amplifies the option-order noise ratio by **$4.96\times$** (`4.962` vs. `1.356` SFT).
    2. **Why Adaptive CAT Fails on Level Estimation (`Q2a/Q2c`)**: Max-information CAT greedily selects items maximizing $(a_j^{\text{pop}})^2 \hat{p}_j(1-\hat{p}_j)$, picking **`40%–49%` Sociology** ($\bar{a} = 1.63\text{–}2.08$). Because high-$a_j^{\text{pop}}$ declarative items are where model-specific prior scaling diverges most from the population curve, CAT locks in a **`+0.042 to +0.092` asymptotic bias**, losing to unbiased $1/\sqrt{n}$ Classical random sampling by $n=100$ (`0.063` vs. `0.050` in `Q2c`, and `1.40x–1.65x` worse RMSE across all $k=16$ cells in `Q2a`).
  - **Visual**: **Figure F6 (Panel B)**.

---

### Act IV: From `CANNOT` to `CAN` in RL Training — The `0.479` Wall & `Unit M1` (Slides 10–12)

- **Slide 10 — `CANNOT #4`: Why Static $(a_j^{\text{pop}}, b_j^{\text{pop}})$ RL Critics Fail & All 15 `Base -> RL` Runs Hit the `0.479` Wall by Update 83 (`Q1`)**
  - **Key Message**: In `Q1`, we trained 12 RL runs comparing standard `grpo` ($G=5$) against three $G=1$ critics (`ctt` pass-rate band, unweighted 2PL `pop0`, and $a_j^{\text{pop}}$-weighted 2PL `pop1`). All 4 arms converged into a **`0.0063` (`0.63%`) band (`0.4685–0.4748`)** (`S1 grpo` at 666u: `0.479`). Why?
    1. **Algebraic Identity (`M7`) + Orthogonality**: Unweighted 1PL/Rasch is identical in expectation to a classical pass-rate band (`ctt`), while static $a_j^{\text{pop}}$ is orthogonal to trajectory slope ($r = 0.04$) and **anti-correlates with procedural math**!
    2. **Variance Starvation by Update 83**: **`84.4%–100%` of the accuracy gain occurs within the first 83 updates** (`0.382 -> 0.455–0.474`), after which `63.1%` of prompts (`84%–91%` peak) have zero reward variance (`zero_adv_share`). Multiplying zero variance by $a_j^{\text{pop}}$ produces zero signal.
    3. **The Structural Ceiling**: Because RL is pure gain modulation ($\Delta\beta_s \gg 0$), starting from `t9_base` bounds `Mean@16` by Base's modal boundary (`Maj@16 = 0.414` unstarred / `0.459*` tie-correct).
  - **Visual**: **Figure F7** (Panel A: 12-run `Q1` trajectory curve showing the update-83 elbow and `0.479` wall; Panel B: `zero_adv_share` saturation + $G=5$ vs. $G=1$ diversity collapse `stable_12`).

- **Slide 11 — `CAN #3` / `A <-> B`: Using the SFT Model's OWN Item Difficulty Parameter $p_{\text{SFT},j}$ to Guide `SFT -> RL` (`Unit M1`)**
  - **Key Message**: Even if Cross-Model $(a_j^{\text{pop}}, b_j^{\text{pop}})$ cannot weight RL from Base, **Within-Model Item Parameters** tell us exactly how to break the wall:
    1. **Operator Composition (`SFT -> RL`)**: `t9_sft_0..2_70` *already* acquired procedural math schemas (`+1.20 to +1.65` logits) and raised the modal boundary to **`Maj@16 = 0.504` (`0.529` random tie / `0.555*` tie-correct) and `Pass@16 = 0.966`**, leaving a **`+0.065 to +0.116` unharvested Condorcet gap** due to high entropy ($H_s = 0.943$).
    2. **Within-Model Psychometric ZPD Parameter ($p_{\text{SFT},j} \in [0.2, 0.8]$)**: Under within-model IRT, Fisher information for binary RL is $I_j(\text{SFT}) = p_{\text{SFT},j}(1 - p_{\text{SFT},j})$. Using the Base model's `W1` pass rate is wrong because SFT reordered item difficulties ($r = 0.289\text{–}0.733$); we must calibrate **$p_{\text{SFT},j}$ ($k=8$ rollouts per prompt on `t9_sft_0..2_70`)** and target SFT's own interior ZPD band (`[0.2, 0.8]`, `2..6 of 8`).
    3. **Modular Presentation (Zero Risk from Cluster Queue)**:
       - **Already Ready for Monday**: Show (i) the 15-run `0.479` wall vs. SFT `Maj@16 = 0.504 / 0.555*` gap (**Panel A**), (ii) the shift from Base item pass rate $p_{\text{base},j}$ (`W1`) to SFT item pass rate $p_{\text{SFT},j}$ (`sft_pass_rate_0..2.csv`) plus **`M1a` ($T=0.2$ inference temperature scaling on `t9_sft_0..2_70`)** (**Panel B**, finishing today on the 1-GPU chain!), and (iii) the pre-registered `+1 / -1 / 0` decision table for **`M1b` (`SFT -> GRPO`)** and **`M1c` (`SFT -> ZPD-GRPO`)** (**Panel C** — plug in `u83`/`u333` numbers if Slurm finishes on Oct 25–26, or present as the live pre-registered test leading into the Nov 20 IRT Workshop).
  - **Visual**: **Figure F8**.

- **Slide 12 — Synthesis: What Item Parameters CAN (and CANNOT) Tell Us**
  - **Key Message**: A clean, memorable takeaway table for Measurement, Psychometrics, and ML researchers connecting all findings to our accepted `JEDM 2026` paper and the Nov 20 40th IRT Workshop:
    - **What Cross-Model Static $(a_j^{\text{pop}}, b_j^{\text{pop}})$ CANNOT Do**: (1) Weight single-model RL rewards (`Q1`), (2) Debias short-test CAT (`Q2a/Q2c`), (3) Isolate item learning without removing $L_{\text{base},j}$ (`Q2-A`).
    - **What Item Parameters CAN Tell Us**:
      1. **`CAN #1` (`A helps B`, $R^2 = 76.9\%$)**: Domain $(\bar{a}_m^{\text{pop}}, \bar{b}_m^{\text{pop}})$ separates Declarative Recall from Procedural Math and predicts whether **RL Gain Modulation** (`8/8` high-$a_m$ domains) or **CoT SFT Schema Acquisition** (`5/6` high-$b_m$ domains) wins.
      2. **`CAN #2` (`B helps A`, Attribution Reversal)**: Item discrimination $a_j^{\text{pop}}$ explains why 2PL $\hat{\theta}$ flips post-training credit from **`60% SFT / 40% GRPO` (Accuracy)** to **`21% SFT / 79% GRPO` ($\hat{\theta}$)**, and why $l_z^{(1)} = +2.24$ flips to $l_z^{(32)} = -482$.
      3. **`CAN #3` (`A <-> B`, `Unit M1`)**: The model's **own** item difficulty parameter ($L_{\text{base},j}$ and $p_{\text{SFT},j} \in [0.2, 0.8]$) governs within-model scaling $\hat{\tau}_s$ and defines the true psychometric ZPD for `SFT -> RL` composition.
  - **Visual**: **Table T5 (Master `CAN` vs. `CANNOT` Summary Matrix)**.

---

## Part 3: Recommended Next Execution Steps (Once You Approve This Plan)

1. **Write a single deterministic plotting script (`src/irt4llm/make_oct26_figures.py`)** that reads the committed CSVs in `results/` and outputs publication-grade vector PDFs + PNGs for **Figures F1–F8** and formatted LaTeX/Markdown tables for **Tables T1–T5** in `<10s`.
2. **Plug in `M1a` and `sft_pass_rate_0..2.csv`** into **Figure F8 (Panel B)** as soon as today's 1-GPU chain finishes (and plug in `M1b/M1c` into **Panel C** if the 2-GPU jobs finish on Oct 25–26).
