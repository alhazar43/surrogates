# 05 — Flattened Storyline & Win-Win Breakout Experiments for Monday (Oct 26)

> **Purpose**: Flatten all verified empirical assets across `item-params-llm`, `signature`, and `irt4llm` into **one cohesive, non-trivial narrative** for Monday's presentation (*"Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?"*), honestly diagnose the remaining thin spots, and specify **3 win-win breakout experiments** (where either a positive or negative result strengthens the paper and talk).

---

## 1. Honest Diagnosis: What Is Trivial vs. Non-Trivial If We Only Present `item-params-llm` Alone?

If Monday's presentation only shows `E1`, `Q2c/Q2a`, `W0/S1/Q2g`, and `Q1` from `item-params-llm` in isolation, a critical audience (both LLM researchers and IRT psychometricians) will immediately spot three weaknesses:

1. **Weakness A (`E1` alone is borderline trivial)**:
   - Saying *"British middle-school math difficulty $b_i$ doesn't correlate with LLM difficulty ($\rho = 0.091$)"* is unsurprising on its own in 2026.
   - **How flattening across repos makes it non-trivial (`E1` + `irt4llm/P2b` + `signature/U2-F15`)**:
     - While **1D item difficulty is orthogonal** between humans and LLMs ($\rho = 0.091$, and `61/61` LLMs fail 1D person-fit with median $Z_h = -6.12$), once we center out item/option difficulty (`P2b`), **human and LLM wrong-answer distractor co-occurrence geometries across $245,390$ item pairs align at $\rho = 0.1477$ ($p = 0.0$, $29.8\%$ of the student split-half ceiling $0.4953$)** (synthetic difficulty-only null $= 0.0197$, 8 independent null seeds $= -0.0026$).
     - Furthermore, layerwise probing on `Qwen3-8B` (`signature/U2-F15`) shows this misconception alignment is near zero through Layers 2–22 and **emerges abruptly in Layers 24–26** (`0.2007 vs 0.1112`), proving it is a deep semantic property rather than surface token overlap.

2. **Weakness B (`Q2c vs Q2a` and `Q2-B` look mechanically obvious unless tied to the training operator geometry)**:
   - Telling an IRT audience *"2PL $\hat{\theta}$ weights items by $a_i$"* or *"CAT works in-distribution (`Q2c`) and fails when item ranks shift (`Q2a`)"* is a mathematical tautology of the 2PL score equation $\sum_i a_i(y_i - P_i(\theta)) = 0$ **unless** we explain **why LLM training operators split into rank-preserving vs. rank-breaking families**.
   - **How flattening across repos makes it genuinely non-trivial (`signature/T1, T3, T9` + `item-params-llm/W0, S1, Q2c, Q2a, Q2g`)**:
     - Across the 102 Open LLM Leaderboard models (`W0`), cross-model discrimination $\bar{a}_m$ and difficulty $\bar{b}_m$ across the 15 MMLU subjects are strongly negatively correlated ($r = -0.8841$): **factual/declarative subjects (*Marketing* $\bar{a}=1.45$, *Sociology* $\bar{a}=0.75$, *World History* $\bar{a}=1.12$) scale tightly with base model family/size (high $a_i$, low $b_i$), whereas multi-step quantitative subjects (*High School Math* $\bar{a}=0.28$, *Elementary Math* $\bar{a}=0.48$) have flat cross-model slopes (low $a_i$, high $b_i$)**.
     - **LLM adaptation operators split cleanly into two regimes with respect to this pretraining manifold**:
       1. **Manifold-Preserving / Sharpening Operators (`Pretraining Q2c`, `5-shot ICL & LoRA in T1/T3`, `Outcome RL / GRPO in S1/Q1/T9`)**:
          - In `signature/T1 + T3`, **LoRA stays inside the base model's Layer-18 PCA subspace (`0.875` variance capture) and preserves item difficulty ranks in `4/5` subjects** (`5-shot ICL` also preserves item ranks in `4/5` subjects).
          - In `S1` & `T9`, **GRPO preserves Base item ranks ($r(\text{RL}, \text{Base}) = 0.9083$)** with zero forgetting (`-0.90` pts in `T9`), collapsing rollout entropy (`0.956 -> 0.287` nats) to convert Base's existing majority-vote answers into single-draw accuracy. Consequently, GRPO wins on **`8/8` low-$b_i$, high-$a_i$ subjects**, **2PL CAT works on pretraining checkpoints (`Q2c`: `35%` RMSE reduction at $n=25$)** and even on GRPO at $n=25$ (`6.95%` vs `7.99%` random), **but all 15 `Base -> RL` runs hit a hard accuracy ceiling at `46.9%–47.9%` (`Mean@16 = 47.9%` $\approx$ `Maj@16 = 48.1%`)**.
       2. **Manifold-Breaking / Schema-Injecting Operators (`Full FT in T1/T3`, `70B CoT SFT in S1/T9`)**:
          - In `signature/T1 + T3`, **Full FT leaves the Layer-18 PCA subspace (`0.239–0.290`) and breaks item rank invariance in `4/5` subjects** (with OOD damage predicted by trained-subject Forward KL at $\rho = 0.870$ in `T8-P3` and Cosine-NTK at $\rho = 0.772$ in `T6`).
          - In `T9` & `S1`, **70B CoT SFT breaks Base item ranks ($r(\text{SFT}, \text{Base}) = 0.2895$)**:
            - Under **greedy $T=0$ evaluation (`signature/T9`)**, SFT's `+5.01%` raw gain is **100% format/parse repair (`gain_parsed = -0.55%`, with `10.82%` forgetting)**! Standard greedy evaluation completely misses what SFT learned.
            - Under **$T=1.0$ multi-draw evaluation (`S1`) + 2D $(b_i, a_i)$ decomposition**, SFT preserves diversity (`H = 0.938` nats, `Pass@16 = 90.1%`, `Pass@32 = 96.6%`) and **beats GRPO on `5/6` high-$b_i$, low-$a_i$ math/quantitative subjects** (*Elementary Math `+38.3% vs +3.8%`*, *High School Math `+23.8% vs +7.1%`*), raising **`Maj@16` to `50.4%` (`55.5%*` tie-correct) — above every RL run**!
            - Because SFT's gains sit on low-$a_i$ math subjects while GRPO's gains sit on high-$a_i$ recall subjects ($r(\bar{b}_m, \Delta_{\text{SFT}} - \Delta_{\text{GRPO}}) = +0.850$ in `%` gain, $+0.877 / R^2 = 76.9\%$ in log-odds), **collapsing evaluation into 1D 2PL $\hat{\theta}$ (`Q2-B`) weights *Marketing* $5\times$ as much as *High School Math* and flips post-training credit from `60% SFT / 40% GRPO` (Accuracy) to `21% SFT / 79% GRPO` (2PL $\hat{\theta}$)**, while **2PL CAT reverses on SFT models (`Q2a`: `11.4%` vs `7.3%` random at $n=25$)**.

3. **Weakness C (Our Biggest Vulnerability Right Now — The `Training` Half of the Title Has Only Negative/Diagnostic Results in `Q1`)**:
   - In `Unit Q1`, putting static 2PL parameters $(a_i, b_i)$ into the RL critic (`pop0`, `pop1`) **did not beat `grpo` (`all 12 runs in 46.85%–47.48%`)**, and 1-draw critics (`ctt`, `pop0`, `pop1`) **collapsed `Pass@32` from `69.4%` (`grpo`) down to `52.4%–58.7%`** because they lack `grpo`'s 5-draw zero-variance shutoff gate ($A=0$ when all 5 rollouts agree).
   - While `Q1` gives two useful diagnostics (**0 pre-samples needed** to match `ctt`'s 85,456-draw `W1` calibration, and **`>85%` of RL gain finishes by update 83** because `63.1%` of groups after update 83 have zero reward variance), **we need constructive Training experiments where either a `+1` or `-1` outcome delivers a non-trivial insight**.

---

## 2. The Flattened End-to-End Storyline (4 Acts, Zero Artificial Tiering)

### **Act 1: Human vs. LLM Calibration (`E1` + `P2b` + `U2-F15`)**
- **Question**: Can human student item parameters evaluate LLMs?
- **Cannot**: 1D human student difficulty $b_i$ (`4,918` students, `704` Eedi math items) does not transfer to `61` LLMs ($\rho = 0.091$, $r = -0.013$; `61/61` LLMs fail 1D person-fit, median $Z_h = -6.12$).
- **Can**: Once 1D difficulty is centered out (`P2b`), human and LLM wrong-answer distractor RDMs across $245,390$ item pairs align at $\rho = 0.1477$ ($29.8\%$ of student split-half ceiling), and this error geometry emerges abruptly in the final third of LLM layers (`Qwen3-8B` Layers 24–26, `U2-F15`).
- **Bridge**: To evaluate or train LLMs with item parameters, we must calibrate $(a_i, b_i)$ on an LLM population (`102` leaderboard LLMs across `1,472` MMLU items, `W0`).

### **Act 2: What LLM Item Parameters $(a_i, b_i)$ Measure — And Why Training Operators Split Into Two Regimes (`W0` + `T1/T3` + `T9/S1` + `Q2c/Q2a/Q2g`)**
- **Core Mechanism**: Cross-model 2PL parameters $(a_i, b_i)$ capture the **pretraining scaling manifold**: high-$a_i$, low-$b_i$ items test factual/declarative recall; low-$a_i$, high-$b_i$ items test multi-step quantitative reasoning ($r(\bar{a}_m, \bar{b}_m) = -0.884$).
- **Regime 1 — Rank-Preserving Operators (`Pretraining Q2c`, `ICL & LoRA T1/T3`, `GRPO S1/T9`)**:
  - Stay inside the base feature subspace (`LoRA 87.5%` in `T3`) and preserve item difficulty order ($r(\text{RL}, \text{Base}) = 0.908$).
  - **What IRT CAN do here**: 2PL adaptive testing (`Q2c`) cuts short-test error by **`35%` at $n=25$** (`6.4%` vs `9.8%` random; 25 adaptive items beat 50 random items).
  - **What RL CANNOT do here**: Because GRPO only sharpens existing modes (`entropy 0.96 -> 0.28`), `Mean@16` (`47.9%`) hits Base's majority-vote boundary (`Maj@16 = 48.1%`).
- **Regime 2 — Rank-Breaking Operators (`Full FT T1/T3/T8`, `CoT SFT T9/S1`)**:
  - Leave the base subspace (`Full FT 24%–29%` in `T3`) and reorder item difficulties ($r(\text{SFT}, \text{Base}) = 0.289$), accompanied by OOD forgetting predicted by Forward KL ($\rho = 0.870$, `T8-P3`).
  - **What Standard Greedy Eval (`T9`) & 1D 2PL (`Q2-B`, `Q2a`) CANNOT do here**:
    - Greedy $T=0$ eval (`T9`) sees `-0.55%` net change on parsed items for SFT (`100%` of SFT's `+5.01%` greedy gain is format repair!).
    - 1D 2PL $\hat{\theta}$ (`Q2-B`) weights high-$a_i$ recall $5\times$ over low-$a_i$ math, flipping SFT/GRPO credit from **`60% / 40%` (Accuracy) to `21% / 79%` (2PL $\hat{\theta}$)**.
    - 2PL CAT (`Q2a`) reverses on SFT models (`11.4%` vs `7.3%` random at $n=25$).
  - **What 2D $(b_i, a_i)$ + Multi-Draw (`S1`) CAN reveal**:
    - Plotting gains across $(b_i, a_i)$ reveals the **X-shaped crossover ($r = +0.85$ in `%`, $R^2 = 76.9\%$ in log-odds)**: GRPO wins `8/8` low-$b_i$, high-$a_i$ subjects, whereas SFT wins `5/6` high-$b_i$, low-$a_i$ math/STEM subjects and raises **`Maj@16` to `50.4%` (`55.5%*` tie-correct) and `Pass@16` to `90.1%` (`Pass@32 = 96.6%`)**.

### **Act 3: Item Parameters Inside RL Training (`Q1` + `W1` + Breakout Experiment 2)**
- **What Online 2PL Critics (`pop0`, `pop1`) CAN do**:
  - Match 5-draw `grpo` (`46.9%`) and 1-draw `ctt` (`47.2%`) at **`46.9%–47.5%` with `0` pre-training rollouts** (saving `ctt`'s `85,456` pre-samples in `W1`).
  - Diagnose why **`>85%` of RL learning finishes by step 83**: `63.1%` of 5-draw groups after step 83 (up from `33.2%` at step 0) have zero reward variance (`0/5` or `5/5` correct).
- **What Static 1-Draw Critics CANNOT do (and why)**:
  - Static cross-model $a_i$ weights (`pop1`) cannot raise the `47%` accuracy plateau because cross-model $a_i$ is orthogonal ($r = 0.04$) to within-run trajectory changes.
  - Because 1-draw critics lack 5-draw `grpo`'s **zero-variance shutoff gate ($A=0$)**, they keep pushing $r_i - \hat{p}_i \neq 0$ on solved/impossible items, collapsing `Pass@32` from `69.4%` (`grpo`) to `58.7%` (`pop1`), `58.6%` (`ctt`), and `52.4%` (`pop0`).
  - *(Breakout Experiment 2 tests whether adding an explicit **IRT probability shutoff gate $\hat{P}_i \in [0.2, 0.8]$** to 1-draw critics restores `Pass@32` diversity without needing 5 draws!)*

### **Act 4: Escaping the `47.9%` RL Wall via Operator Composition & Within-Model Targeting (`Unit M1`)**
- Since `Base -> RL` is trapped at `47.9%` because RL only sharpens existing modes, while `SFT` raises `Maj@16` to `50.4%–55.5%` on high-$b_i$ math items without sharpening (`Mean@16 = 43.9%`), **`Unit M1`** tests:
  - **`M1a` (`t9_sft` at decoding $T = 0.2$, 0 training steps)**: Does inference-time temperature sharpening alone harvest SFT's `Maj@16` advantage?
  - **`M1b` (`t9_sft -> GRPO`, 83 & 333 updates)**: Does composing `SFT -> GRPO` break the 15-run `47.9%` wall (`+1`), or does short-response RL (`~113` tokens) overwrite SFT's `310`-token math CoT traces (`-1`)?
  - **`M1c` (`t9_sft -> Within-Model Pass-Rate [0.2, 0.8] Targeted GRPO`)**: Does filtering training items to the SFT checkpoint's own intermediate pass-rate band (`2/8` to `6/8` correct, where `76%–94%` of 5-draw groups have non-zero variance) beat unfiltered `M1b`?

---

## 3. Three Win-Win Breakout Experiments Before Monday (Oct 26)

Every experiment below is designed so that **both `+1` (positive) and `-1` (negative) outcomes produce a clean, non-trivial finding**:

| Experiment | Compute Cost | Exact Protocol | Why `+1` Helps Us | Why `-1` Helps Us |
| :--- | :---: | :--- | :--- | :--- |
| **Exp 1: `Unit M1` (`M1a`, `M1b`, `M1c`)**<br>*(Spec in `gemini/M1.md` & `item-params-llm/notes/units/M1.md`)* | `M1a`: ~20 GPU min<br>`M1b/c`: ~6–8 GPU hrs (with `u83` early save in ~2 hrs) | 1. `M1a`: Sample `t9_sft_0..2_70` at `--temperature 0.2`.<br>2. `M1b`: `t9_sft_0..2_70 -> GRPO` (Q1 settings, save at `u83` and `u333`).<br>3. `M1c`: Sample $k=8$ pass rates on `t9_sft_0..2_70` (`W1` protocol), filter to $p_i^{\text{SFT}} \in [0.2, 0.8]$, run `GRPO` (`u83` & `u333`). | **`+1` (`M1b/c > 47.9%`, `M1c > M1b`)**: Constructive proof that IRT/pass-rate diagnosis breaks the 15-run RL ceiling by combining SFT's `Maj@16` expansion with within-model ZPD targeting. | **`-1` on `M1a` (`M1a` at $T=0.2$ matches RL)**: Proves trajectory RL concentration after SFT is replicated at **zero training cost** by token temperature scaling.<br>**`-1` on `M1b` (`Maj@16` drops on math)**: Proves **short-path RL interference** (`113`-token RL overwrites `310`-token SFT math CoT). |
| **Exp 2: Gated 1-Draw Critic (`Q1-Gate`)**<br>*(Direct fix for `Q1` mode collapse)* | ~40 GPU min per run (only `83` updates needed, since `>85%` of Q1 gain happens by `u83`!) | In `core.training.rl`, add a shutoff gate to 1-draw `pop0` and `ctt`: set advantage $A_i = 0$ whenever predicted pass probability $\hat{P}_i \notin [0.2, 0.8]$ (i.e., $\hat{P}_i < 0.2$ or $\hat{P}_i > 0.8$), mimicking 5-draw GRPO's zero-variance gate with **1 draw**. Run to update `83` (`seed 0` or `seeds 0..2`) and sample bank. | **`+1` (`Pass@32` recovers from `52.4%` toward `65%–70%` while `Mean@16` stays `~47%`)**: Turns `Q1` from a negative result into a **constructive 1-draw algorithm**: 2PL/CTT probability gating prevents mode collapse at $1/5$ the rollout cost per prompt. | **`-1` (`pop0-gate` still collapses, `ctt-gate` recovers)**: Clean causal proof that static cross-model $(a_i, b_i)$ misidentifies which items are saturated for the current model, whereas within-model $p_{0,i}$ (`ctt-gate`) correctly gates them. |
| **Exp 3: Cross-Operator Rank-Invariance vs. Utility Synthesis Plot**<br>*(Zero GPU cost, local script on existing CSVs)* | **0 GPU min** (10 min local Python script) | Combine existing CSVs from `signature/T1, T3, T8, T9` and `item-params-llm/Q2c, Q2a, S1, Q1` into one unified table/plot: x-axis = **Item Rank Preservation $\rho(b_{\text{pre}}, b_{\text{post}})$**, y-axis = **2PL CAT Error Reduction (`Q2c/Q2a`) & Forgetting (`T8/T9`) & `Maj@16 - Mean@16` Gap (`S1`)**. | Visually unifies `signature` and `item-params-llm` into a single law: **Rank-preserving operators (`Pretraining, ICL, LoRA, GRPO`) obey static IRT (`CAT` works, 0 forgetting, ceiling-bound), while rank-breaking operators (`Full FT, CoT SFT`) break static 1D IRT (`CAT` & 1D $\hat{\theta}$ fail) precisely because they inject new high-$b_i$ capabilities.** | Zero risk (100% existing `[VERIFIED-RAW]` data). |
