# Project & Agent Operating Rules

These rules govern how we formulate hypotheses, design experiments, interpret results, and collaborate across human-AI research sessions. They exist specifically to prevent two failure modes observed in long-context LLM workflows: **the SGD Local-Minimum Trap** (narrowing scope as context grows until we are micro-optimizing a dead end) and **the `0.01` Noise Swamp** (spending days of GPU time trying to rescue a weak, dataset-dependent scalar signal).

---

## Rule 1: The `-1 / 0 / +1` Principle & How to Turn `0` from a Noise Swamp into an Exact Invariance Law
Every empirical outcome falls into one of three states:
- **`+1` (Strong Positive)**: A clean, macro-scale confirmation (`r > 0.35` or `> 5 SE` separation, robust across seeds and datasets).
- **`-1` (Strong Negative / Structural Falsification)**: A clean, undeniable demonstration that a widely assumed property fails or inverts (`> 5 SE` in the opposite direction, or `r ≈ 0` where classical theory demands `r > 0.7`).
- **`0` (Zero Effect / Flat Delta)**: In poorly designed experiments, `0` (`±0.01`) is a **Bad `0` (The Noise Swamp)** caused by comparing a single unanchored tweak against a single baseline on a composite scalar where opposing forces cancel out. In properly designed experiments, `0` is a **Good `0` (An Exact Invariance / Conservation Law)**.

**Mandatory Experimental Design Gate (Two-Endpoint Bracketing & Orthogonal Decomposition)**:
1. **Never Measure Only an Un-Decomposed Composite Scalar**:
   - A composite metric $M(\lambda) = f(\lambda) + g(\lambda)$ (such as total single-state $\text{RMSE} = \sqrt{\text{Bias}^2 + \text{SD}^2}$) will produce a flat "Bad `0`" whenever $f(\lambda)$ increases and $g(\lambda)$ decreases. Always decompose the estimator into orthogonal components (e.g., $\text{Bias}(\lambda)$ vs. $\text{SD}(\lambda)$; location shift $\Delta\theta$ vs. concentration shift $\Delta\beta$) before interpreting a flat composite curve.
2. **Always Bracket a New Intervention $X$ Between Two Pre-Verified Endpoints ($E_0, E_1$) Separated by $>10\text{ SE}$**:
   - Because $(X - E_0) + (E_1 - X) = E_1 - E_0 = \Delta_{\text{total}} \gg 10\text{ SE}$, it is **algebraically impossible** for $X$ to be `0` relative to $E_0$ and `0` relative to $E_1$ simultaneously!
   - If $X - E_0 = 0$ (the `0` outcome), then $E_1 - X = \Delta_{\text{total}} \gg 10\text{ SE}$ by exact conservation—proving with $>10\text{ SE}$ power that the factor varied between $E_0$ and $X$ is an **exact invariant (`0%` contribution)** and that `100%` of $\Delta_{\text{total}}$ arises from the complementary mechanism between $X$ and $E_1$.
3. **Never Chase a Bad `0`**:
   - If an unbracketed exploratory direction yields `0.01` noise, do not tweak hyperparameters to rescue it; either reformulate it as a bracketed conservation decomposition or kill the branch within 24 hours.

---

## Rule 2: Escape the Context-Narrowing SGD Trap
As conversation history and experimental logs grow, AI agents naturally tend to narrow their scope—fixing local bugs, rewriting narrow gate criteria (`Q2 -> Q2g -> T9 -> T9r -> T9g`), and sinking deeper into a local basin without questioning whether the basin itself is worth exploring.

**Mandatory Gate**:
- Before proposing any "Rewrite", patch, or follow-up run (`T_next`), explicitly ask: **"If this run works as intended, does it answer our high-level philosophical question, or is it just patching a broken proxy?"**
- Maintain a Dual Horizon at all times:
  1. **Tactical Horizon (Immediate Deadlines)**: What is the cleanest, most honest story we can tell *right now* from existing artifacts?
  2. **Strategic Horizon (Thesis / Taste)**: How does this experiment advance our understanding of latent representations and psychometric measurement?

---

## Rule 3: Two-Stage Contingent Funnel (Diagnose First, Train Second)
Never launch multi-GPU training runs (SFT / GRPO / curriculum sweeps) to test a hypothesis that has not first shown a macro-scale signal at the representation or distribution level.

- **Stage 0 (Zero-Training Diagnostic — 1 GPU / CPU, <= 1 Day)**:
  - Probe existing checkpoints, hidden states, logits, or rollout distributions (`k` samples per item).
  - Verify that the proposed latent quantity or psychometric signature exhibits a **macro-scale effect** (`> 0.30`) and is not an artifact of prompt formatting or evaluation position.
- **Stage 1 (Targeted Intervention — 2x H100, <= 2 Days)**:
  - Only after Stage 0 confirms a strong structural phenomenon, run the minimal training intervention (e.g., SFT/RL curation or representation-guided weighting) that directly exploits that verified structure.

---

## Rule 4: White-Box Probing & Mandatory Confounder Decomposition
Classical psychometrics was forced to use binary `0/1` response matrices because human brains cannot be inspected directly. Applying black-box `0/1` IRT to open-weight LLMs discards 99% of the information we actually have access to.

However, when moving from black-box `0/1` responses to white-box logits, temperatures, or hidden states, **every dramatic scalar jump must be audited for statistical and format confounders before claiming a discovery**:

1. **Always Decompose Regression Slopes (`tau = r * (sigma_y / sigma_x)`)**:
   - Never interpret an OLS regression slope (`tau`) of trained logits on base logits as pure "confidence flattening" or "temperature sharpening."
   - Mathematically, `tau = r(l_t, l_0) * (sigma(l_t) / sigma(l_0))`. If a training run (such as SFT on an external teacher) reorders item preferences and drops correlation `r` from `1.0` to `0.42`, `tau` automatically shrinks by more than half due to **regression attenuation**, even if logit variance `sigma(l_t)` did not flatten at all. Always report `r` (rank/preference invariance) and `sigma_y / sigma_x` (scale/temperature ratio) separately.

2. **Audit Readout Position vs. Generation Trajectory Format**:
   - Never compare forced-prefix token-0 probabilities (`"The answer is **"`) across models trained with different generation lengths without checking the free-generation trajectory.
   - *Canonical Warning (`T9r` vs. `T9g`)*: SFT trained on 310-token Chain-of-Thought (CoT) traces looked "uncalibrated/flattened" at token 0 (`tau = 0.34–0.58`, letter mass `0.755`) only because it was forced to answer before generating its reasoning trace. Once allowed to generate its 310-token CoT (`T9g`), its confidence at the actual answer token (`0.511–0.534`) was *higher* than Base (`0.422`). SFT did not lose confidence; it relocated the decision from token 0 into the reasoning trajectory. Conversely, RL (GRPO) preserved Base's item ordering (`r ≈ 0.85`), shortened responses, and sharpened token-0 temperature (`tau > 2.0`).

3. **Distinguish Trajectory Reorganization from Surface Temperature Scaling**:
   - When evaluating post-training (SFT vs. RL), always separate **where the decision happens** (token-0 vs. CoT trajectory rollout) and **whether item ordering changed** (`r` with base) from **how sharp the output softmax is** (`sigma`).

---

## Rule 5: Brutal Intellectual Honesty & The "Too-Good-To-Be-True" Audit
- Do not flatter the user or dress up a failed experiment, a noisy `0.01` delta, or a methodological artifact as a "breakthrough."
- **Mandatory "Wait a Minute" Audit**: Whenever a single summary statistic looks dramatically clean or theatrical (e.g., `0.34 vs. 2.17`), the agent's first instinct must be **skepticism, not celebration**. Immediately pull the underlying raw tables, write out the estimator's mathematical formula, and check for trivial confounders (attenuation, format mismatch, length bias, denominator shifts) *before* pitching a narrative to the user.
- When the user expresses skepticism about a result, immediately audit the raw numbers and code to falsify the claim rather than defending it.
- A fast, honest falsification saves weeks of wasted PhD time; artificial optimism destroys research taste.
