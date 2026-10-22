# Research Philosophy & Intellectual Identity

## 1. Core Identity: Self-Explainability Through Latent Representations
My background is in applied mathematics and machine learning, operating at the intersection of educational AI, psychometrics, and large language models. Across these domains, my core intellectual obsession is **self-explainability through latent representations**: understanding when a model's internal geometry and behavioral distributions genuinely encode the structure of a domain (difficulty, discrimination, skill state, uncertainty) versus when they merely optimize a surface predictive loss.

In both Educational AI and LLM research, the field largely splits into two extremes that I explicitly reject:
1. **Compute-Driven Leaderboard Hounding & Dataset-as-a-Bar Papers**: Chasing marginal `0.01` AUC/accuracy gains with increasingly opaque architectures or massive GPU clusters, without asking *what* the model has actually learned or whether its internal coordinates mean anything.
2. **Dogmatic Psychometric Pedantry**: Treating classical 1980s IRT/2PL equations as sacred ground truth and mechanically fitting scalar curves to modern neural systems while ignoring their internal representations, token dynamics, and non-monotonic failure modes.

My comparative advantage lies between these two poles: **bringing the philosophical and mathematical lens of psychometrics (measurement invariance, identifiability, calibration vs. ranking, item-respondent duality) into deep learning and LLM representations**, without being bound to literal IRT equations or competing in brute-force compute races.

---

## 2. What Makes "Good Research"?

### A Bet You Would Be Happy Losing
Good research is a bet you are genuinely happy losing. However, there is a critical difference between a **structural bet** and an **unanchored surrogate**:
- **Why Unanchored Surrogates Fail**: In the past, aiming for indirect "surrogates" (hoping serendipity would lead back to the core philosophical question) often pulled the research further away from the central phenomenon. Worse, when a surrogate produces a noisy `0.01` effect, it traps the researcher in a cycle of post-hoc self-justification—spending months trying to separate a weak signal from dataset noise.
- **How to Make a Bet Worth Losing**: A bet is only worth losing if **both `+1` (confirmation) and `-1` (falsification) reveal a clean, macro-scale structural truth** (`|effect| > 0.30`), and the experimental design makes it impossible to get stuck in the `0.01` noise swamp.

### Showing How We Learned, Not Just What We Learned
A great paper is not a lecture that hands the reader a static conclusion ("what I learned"); it is an **ingestive and inspirational journey that shows the reader *how* I learned it**.
- It starts from a natural, high-stakes conceptual assumption that the community implicitly holds (e.g., *"better next-step prediction implies better parameter recovery"*, or *"standard item difficulty parameters computed across an LLM zoo tell us which training questions are hard or useful for RL/SFT"*).
- It tests that assumption at its structural root, exposes where and why the naive intuition breaks down, and follows the failure mode to uncover the deeper latent mechanism.
- Even when an intuitive hypothesis fails (`-1`), showing *why* it fails—and what latent coupling or distributional structure actually governs the model—is vastly more inspiring than reporting a `+0.008` benchmark improvement.

### Separating Philosophical/Motivational Novelty from Engineering Novelty
Leaderboard engineering novelty asks: *"How do we squeeze 1.5% more accuracy out of GRPO or KT?"*
Philosophical and motivational novelty asks: *"What does it mean for a model to 'know' an item, what does an item parameter actually measure when applied to a generative model, and how do training objectives reshape latent space?"*
When compute is constrained (e.g., 2x H100s rather than a 256-GPU cluster), competing on pure engineering scale is a losing game. Competing on **conceptual clarity, diagnostic elegance, and latent representation geometry** turns a compute constraint into a taste advantage.

---

## 3. The Unifying Thesis Arc (14-Month Horizon)

Rather than forcing every project into a narrow question ("What is learning?") or an exclusively LLM-centric box, the broader thesis backbone asks:

> **Prediction, Calibration, and Latent Geometry: What Psychometric Measurement Can (and Cannot) Reveal About Neural Learning Systems**

This overarching question naturally connects both human-facing educational models and large language models under one coherent intellectual roof:
1. **In Knowledge Tracing (JEDM)**: Optimizing predictive accuracy couples latent ability and item parameters through shared history encoders (`Prediction Is Not Calibration`), destroying item parameter recovery unless structural measurement constraints or decoupled architectures are respected.
2. **In LLM Evaluation & Post-Training (`irt4llm` / `signature`)**: Fitting classical static IRT parameters (`b`, `a`) to binarized `0/1` LLM benchmark matrices fails to predict item transfer or guide SFT/RL curation (`0.00–0.01` noise), because compressing a generative policy into a single binary outcome throws away the model's actual psychometric state—its **internal logit geometry, token-level confidence dynamics, and multi-rollout survival distribution** (`p_j(\theta)`).
3. **The Forward Path**: Moving from **extrinsic black-box item parameters** (treating LLMs as black-box human test-takers) to **intrinsic representation and distributional psychometrics** (reading difficulty, discrimination, and learnability directly from the model's hidden representations, rollout distributions, and training dynamics).
