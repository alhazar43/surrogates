# Verified Mathematical Propositions & Derivations Across All Repos (`03-VERIFIED-THEORY-AND-MATH.md`)

Every mathematical derivation below has been cross-checked against its independent review file (`D:\surrogates\theory\*-review.md`, `D:\surrogates\notes\units\RevE..RevS.md`, `item-params-llm/results/review-2026-10-19/{M,K}.md`, and `learner-state/base/sources/identification.md`).

---

## 1. `D:\surrogates\theory\{E, G, M, P, R, S}.md` (Audited Against `*-review.md` & `RevE..RevS`)

> **Note**: All 6 theory files in `D:\surrogates\theory\` were revised on Oct 19–20 (`RevE..RevS`) in response to the independent Opus reviews (`*-review.md`). Below is the exact status of each proposition and the required mathematical scope conditions.

### 1.1 Track E (`theory/E.md` & `E-review.md`): Short-Test & CAT Evaluation Efficiency
- **`[SURVIVED REVIEW]` `E7, E8, E9` (Misspecification Crossover $n^*$ & High-$a$ Selection Bias)**:
  - **Decomposition (`E7`)**: Under model misspecification $\delta_i(\theta) = P_{\text{true},i} - \sigma(a_i(\theta - b_i))$, plug-in IRT test-characteristic curve (TCC) estimation error decomposes as $\operatorname{MSE}(\hat{\mu}_{\text{IRT}}) = B(\theta^*)^2 + \frac{\bar{P}'(\theta^*)^2}{n\bar{J}(\theta^*) + \sigma_0^{-2}} + O(n^{-1})$, whereas random-item classical proportion correct (`CTT`) is unbiased ($B_{\text{CTT}} = 0$) with variance $V_{\text{CTT}}/n$.
  - **Crossover Sample Size (`E8`)**: Because IRT variance shrinks as $O(1/n)$ while squared misspecification bias $B(\theta^*)^2$ is constant in $n$, classical random sampling overtakes IRT at the exact threshold:
    $$n^* = \frac{n\,(V_{\text{CTT}} - V_{\text{IRT}})}{B(\theta^*)^2}$$
  - **Adaptive Selection Bias (`E9` & `review-2026-10-19/K.md K2`)**: Maximum-information CAT selects items $i \in S$ maximizing $a_i^2 p_i(1-p_i)$, inducing a first-order pseudo-true bias shift:
    $$B_S(\theta^*_S) \approx \frac{\bar{P}'(\theta)}{\bar{J}_S(\theta)} \cdot \frac{1}{n}\sum_{i \in S} a_i\,\delta_i(\theta)$$
    Whenever high-$a_i$ items carry systematic positive residuals $\delta_i(\theta) > 0$ (as in `Q2a/Q2c`, where `40%–49%` of selected items come from *Sociology* with mean $a = 1.63\text{–}2.08$), CAT inflates $B_S^2$ and loses to random sampling by $n=100$.
- **`[NARROWED BY REVIEW]` `E5` (Cauchy–Schwarz Variance Ratio $\le \bar{a}^2/\overline{a^2}$)**:
  - Holds strictly when $p_i(1-p_i)$ is constant across items ($b_i = b$); otherwise $\operatorname{Cov}(a_i, p_i(1-p_i))$ enters (`E-review.md` Issue 4).
- **`[NARROWED BY REVIEW]` `E10` (Anchor Shift Gauge Invariance)**:
  - Under flat-prior ML (`M8` / `K2`), shifting $b_i \to b_i - \Delta$ shifts $\hat{\theta}_{\text{ML}} \to \hat{\theta}_{\text{ML}} - \Delta$ and **cancels identically** in $a_i(\hat{\theta} - b_i + \Delta)$. Under an informative EAP prior $\mathcal{N}(\mu_0, \sigma_0^2)$, it changes the prior pull by $O(\sigma_0^{-2}/(n\bar{J}))$ (`E-review.md` Issue 1).

### 1.2 Track G (`theory/G.md` & `G-review.md`): Geometry of Post-Training Log-Odds Changes
- **`[SURVIVED REVIEW]` `G R1–R3, R5, R7, R8, R10` (Three-Part Step Decomposition)**:
  - For centered option log-odds $\eta_{si} \in \mathbb{R}^K$ (or binary log-odds $l_{si}$ in `R10`), any post-training update from base state $0$ to state $s$ decomposes as:
    $$\Delta \eta_{si} = (\tau_s - 1)\eta_{0i} + \Lambda_i \Delta\theta_s + \Delta\varepsilon_{si}, \qquad \Delta l_{si} = (\tau_s - 1)l_{0i} + a_i\Delta\theta_s + \Delta e_{si}$$
  - Under uniform positive scaling ($\tau_s > 0, \Delta\theta_s = 0, \Delta\varepsilon_{si} = 0$), greedy argmax accuracy is strictly invariant (`R5`), while KL divergence scales quadratically as $\frac{1}{2}(\tau_s - 1)^2 \sum_i \operatorname{Var}_{p_{0i}}(\eta_{0i})$ (`R7`).
- **`[NARROWED BY REVIEW]` `G R4, R6, R9`**:
  1. **Non-Orthogonality of $\eta_{0i}$ and $\Lambda_i$ (`R4`, `G-review.md` Major Issue 1)**: Because the base model already has non-zero ability $\theta_0$, $\eta_{0i}$ is strongly collinear with $\Lambda_i$ (and in 1D 2PL, $r(L_{\text{base},i}, b_i^{\text{pop}}) = -0.56$ item-level and $-0.94$ subject-level!). Orthogonal decomposition requires projecting onto the **deflated subspace** $\tilde{\Lambda} = (I - P_{\eta_0})\Lambda$.
  2. **Multi-Option Scaling Sign Condition (`R6`, `G-review.md` Major Issue 2)**: For $K > 2$ options, $\partial \mu_s/\partial \tau_s > 0$ on item $i$ iff $\eta_{si,k_i} > \sum_c p_{si,c}\eta_{si,c}$ (guaranteed when the key $k_i$ is the mode $\arg\max_c \eta_{si,c}$).
  3. **CoT Marginal Jensen Curvature (`R9`, `G-review.md` Major Issue 3)**: Marginalizing over reasoning chains $p_{si}^{\text{CoT}}(c) = \sum_z \pi_s(z|i)P_s(c|i,z)$ introduces a positive Jensen variance term $+\frac{1}{2}L\operatorname{Var}_z(\tilde{\eta}_i(z))$ into the marginal log-odds contrast.

### 1.3 Track M (`theory/M.md` & `M-review.md`): Measurement Foundations for Generative Respondents
- **`[SURVIVED REVIEW]` `M P1, P4, P6, P7, P8`**:
  - Defines the protocol-indexed marginal response distribution $p_{mi}^{(\tau, u)}(c)$, $\operatorname{GL}(d)$ gauge identification (`P4`), Jeffreys $O(1/k)$ smoothing bias (`P6`), within-respondent variance decomposition into template + CoT path + numerical precision (`P7`, explaining `T9r`'s greedy `bfloat16` flips), and Eckart–Young SVD truncation (`P8`).
- **`[DEMOLISHED AS INVARIANCE / NARROWED]` `M P2, P3/P3a, P5`**:
  1. **Protocol Sensitivity (`P2`, `M-review.md` Major Issue 1)**: $p_{mi}^{(\tau, u)}$ is **not** invariant to temperature $\tau$ or prompt template $u$.
  2. **4-Option Rank vs. Binary Rank (`P3/P3a`, `M-review.md` Major Issues 2, 3, 5)**: A $d$-dimensional respondent space has option-contrast matrix rank $d(K-1)$, not $d$, unless option loadings factor as $\lambda_{ic} = a_i u_c$; greedy decoding ($\tau \to 0$) produces $\{0,1\}$ boundary corners where logits are $\pm\infty$; and conditional rank-$d$ per CoT path does not imply marginal rank-$d$.

### 1.4 Track P (`theory/P.md` & `P-review.md`): Predictability Limits of Item Parameters
- **`[SURVIVED REVIEW]` `P D4, D6–D9`** (Cold-start Fisher information attenuation $\tilde{J}_i = J_i/(1 + a_i^2\sigma_{e,b}^2 J_i^*)^2$ and cold-start CAT regret bound $\le \frac{\bar{a}^2}{2}\operatorname{MSE}_b$).
- **`[NARROWED BY REVIEW]` `P D1, D2, D3, D5`**:
  - In `D1`, $-\bar{\eta}_i$ in a 2PL model equals $a_i(b_i - \bar{\theta})$ (the intercept $-d_i = a_i b_i$, not $b_i$; `P-review.md` Issue 1).
  - In `D3`, out-of-sample CV MSE has `1x` estimation noise variance $\bar{v}_b$ in the numerator (not `2x`; `P-review.md` Issue 3).
  - In `D5`, the Fano/mutual-information lower bound $\operatorname{MSE}_b \ge \sigma_b^2 e^{-2I(b;\phi(X))}$ holds when $b_i$ depends on unobserved training-corpus exposure $E_i$ not present in item text $X_i$ (`P-review.md` Issue 4; empirically confirmed by `Surrogates P1` $R^2_{\text{oos}} = 0.056$).

### 1.5 Track R & Track S (`theory/R.md`, `theory/S.md`, `review-2026-10-19/M.md`): RL Training & Critic Mechanics
- **`[VERIFIED-MATH & SURVIVED REVIEW]` The Per-Token $1/L(y)$ Length-Bias Theorem (`R R2`, `S Prop 2`, `review-2026-10-19/M.md M1`)**:
  - When a policy gradient objective normalizes token losses by sequence length $L(y)$, subtracting any non-zero prompt baseline $\hat{p}_j$ **fails to vanish in expectation**:
    $$\mathbb{E}_{y \sim \pi_\phi}\!\left[\frac{r(y) - \hat{p}_j}{L(y)}\sum_{t=1}^{L(y)}\nabla_\phi \log \pi_\phi(y_t \mid y_{<t}, x_j)\right] = \nabla_\phi \mathbb{E}\!\left[\frac{r(y)}{L(y)}\right] - \hat{p}_j\,\nabla_\phi \mathbb{E}\!\left[\frac{1}{L(y)}\right]$$
  - Whenever $\hat{p}_j > 0$, the spurious term $-\hat{p}_j \nabla_\phi \mathbb{E}[1/L(y)]$ pushes the policy toward shorter sequences (explaining why `Q1b` collapsed length from `89.6 -> 6.4` tokens in 1 epoch until replaced by constant $L_0 = 78.98$ normalization in `R R1` / `S Prop 1`).
- **`[VERIFIED-MATH & SURVIVED REVIEW]` Effective KL Shift Under Gradient Clipping (`R R3`, `S Prop 5`, `M.md M3`)**:
  - Scaling prompt advantages by $w_j$ changes the gradient norm $\|g\|$, which under gradient clipping $\min(1, c/\|g\|)$ alters the effective ratio between the reward gradient and the unclipped KL penalty unless normalized to unit RMS $\frac{1}{N}\sum_{j=1}^N w_j^2 = 1$.
- **`[VERIFIED-MATH & SURVIVED REVIEW]` Unbiased Hypergeometric Zero-Variance Estimator (`R R4`, `M.md M4`, `K.md K3`)**:
  - Given $c$ successes in $K$ independent binomial draws, the unique minimum-variance unbiased estimator (UMVUE) of the probability $W_G(p) = p^G + (1-p)^G$ that a group of size $G$ has zero reward variance is:
    $$\hat{W}_G(c) = \frac{\binom{c}{G} + \binom{K-c}{G}}{\binom{K}{G}}$$
    On `W1` ($K=8, G=5, N=10,682$), $\frac{1}{N}\sum_j \hat{W}_5(c_j) = \mathbf{0.3496}$ (`25.9%` all-wrong + `9.1%` all-right).
- **`[VERIFIED-MATH & SURVIVED REVIEW]` Equivalence of Self-Initialized Rasch Critic and Classical Moving Pass-Rate Band (`S Prop 9`, `M.md M5/M7`)**:
  - A 1PL Rasch critic initialized from the model's own pre-training pass rates $b_j = -\operatorname{logit}(p_{0j})$ predicts $\hat{p}_j(t) = \sigma(\theta_t + \operatorname{logit}(p_{0j}))$, which is **algebraically identical** to the classical moving pass-rate band (`ctt`) $\sigma(\operatorname{logit}(p_{0j}) + \delta_t)$ and uses zero population IRT information.
- **`[VERIFIED-MATH & SURVIVED REVIEW]` 3-Seed Minimum Detectable Effect (`R R9`, `M.md M6`)**:
  - With $S=3$ seeds per arm ($t_{2,0.95} = 2.92$) and seed-to-seed SD $\tau_{\text{seed}} \approx 0.0064\text{–}0.0091$, the minimum detectable effect at 80% power is $\text{MDE} \approx \mathbf{0.015\text{–}0.023}$ (`1.5–2.3` percentage points).

---

## 2. `irt4llm/notes/theory-kt-irt/{math,cs}.md` & `plan-within-run.md`: Within-Run Person-Slope & Option-Level Theory (`[VERIFIED-MATH]`)

*(Note: Per `irt4llm/CLAUDE.md`, `learner-state` is strictly excluded and never opened, cited, or copied here.)*

1. **Key-Contrast Decomposition & Consistent Errors-in-Variables Gain Estimator (`math.md §1.1`, `plan-within-run.md §3`)**:
   - Let $\ell_{sjr}$ be the key logit minus mean distractor logit in option order $r \in \{1,\dots,R\}$, $\mu_{sj} = \mathbb{E}_r \ell_{sjr}$, and $\nu_{sjr} = \ell_{sjr} - \mu_{sj}$:
     $$\mu_{sj} = \tau_s \mu_{0j} + \delta_s + \kappa_{sj}, \qquad \kappa_{sj} = \tilde{\phi}_j + g_s \tilde{a}_j + \varepsilon_{sj}, \qquad \nu_{sjr} = \omega_s \nu_{0jr} + \eta_{sjr}.$$
   - With $R \ge 2$ option orders in both states, $\omega_s$ and the order noise $\sigma_0^2$ are identified, and the consistent errors-in-variables person-slope estimator is:
     $$\hat{\tau}_s = \frac{\operatorname{Cov}_j(\bar{L}_s, \bar{L}_0) - \hat{\omega}_s \hat{\sigma}_0^2 / R}{\operatorname{Var}_j(\bar{L}_0) - \hat{\sigma}_0^2 / R}, \qquad \operatorname{Var}(\hat{\kappa}_{sj} \mid \kappa_{sj}) = \frac{\sigma_s^2 + (\omega_s - \tau_s)^2 \sigma_0^2}{R}.$$
   - Separating item-specific gain $\tilde{a}_j$ from offset $\tilde{\phi}_j$ requires $T \ge 2$ states of distinct $g_s$, and non-affine residual $\varepsilon_{sj}$ requires $T \ge 3$ (`math.md Prop 1`).

2. **Nested Option-Level Models (`N0–N4`) & Modal-Answer Invariance (`math.md §5.1`, `plan-within-run.md §3`)**:
   - Let $\eta_{0jo} = \log \pi_0(o\mid j) - \frac{1}{4}\sum_{o'} \log \pi_0(o'\mid j)$ be the centered log-probability of option $o$ at state $s=0$:
     - **`N0` (1PL shift)**: $\pi_s(o\mid j) \propto \exp(\eta_{0jo} + \delta_s \mathbf{1}[o=k_j])$.
     - **`N1` (Person slope)**: $\pi_s(o\mid j) \propto \exp(\kappa_s \eta_{0jo})$. Under `N1` with $\kappa_s > 0$, $\arg\max_o \pi_s(o\mid j) = \arg\max_o \eta_{0jo}$ for every item, so the modal-key fraction satisfies $A_\infty(s) = A_\infty(0)$ and accuracy approaches $A_\infty(0)$ as $\kappa_s \to \infty$ (while finite-$\kappa_s$ accuracy is not bounded by $A_\infty(0)$).
     - **`N2` (Person slope + shift)**: $\pi_s(o\mid j) \propto \exp(\kappa_s \eta_{0jo} + \delta_s \mathbf{1}[o=k_j])$. Under `N2`, `pass@k` at every $k$ is determined by two person parameters $(\kappa_s, \delta_s)$ via $\operatorname{mean}_j[1 - (1 - \pi_s(k_j\mid j))^k]$ (`Prediction P11`).
     - **`N3` (Within-run option slopes $\lambda_{jo}$)**: $\pi_s(o\mid j) \propto \exp(\kappa_s \eta_{0jo} + \delta_s \mathbf{1}[o=k_j] + \theta_s \lambda_{jo})$, identified up to sign under $\sum_o \lambda_{jo} = 0$, $\lambda \perp \eta_0$, $\lambda \perp \mathbf{1}[o=k_j]$, $\sum_{j,o}\lambda_{jo}^2 = J$.
   - **Keyed Margin Sign Law (`math.md §2.3, §5.1`, `Prediction P9`)**: With 4 options, the sign of an item's key-probability change under a gain $\kappa_s > 1$ is the sign of its **keyed margin** $\eta_{0jk} - \sum_{o\neq k}\frac{q_o}{1-q_k}\eta_{0jo}$, which lies between $\operatorname{logit} p_{\text{key}} + \log 3$ (equal distractors, zero at $p_{\text{key}} = 1/4$) and $\operatorname{logit} p_{\text{key}}$ (one dominant distractor).

3. **Expected Policy Gradient Along the Gain Direction (`math.md §2.1–2.4`, `cs.md §3.3`)**:
   - For binary reward, $G$ on-policy rollouts, and group-normalized advantage:
     $$\mathbb{E}\!\left[\frac{1}{G}\sum_{g=1}^G A_g \nabla_w \log \pi_w(y_g)\right] = c_G(p_j)\nabla_w z_j, \qquad c_G(p) = G^{-1}\mathbb{E}_{n\sim\text{Bin}(G,p)}\sqrt{n(G-n)}.$$
   - In a Llama-type network with final RMSNorm scale $g_n$, scaling $g_n$ by $\gamma$ scales every logit by $\gamma$. The component of one GRPO group's score along that direction is $\sqrt{n(G-n)}(\bar{c}_1 - \bar{c}_0)$, where $c(y) = \sum_t (z_t(y_t) - \mathbb{E}_{v\sim\pi_t} z_t(v))$ is the inverse-temperature derivative of $\log \pi(y)$. Thus GRPO sharpens ($\tau_s > 1$) when correct rollouts are more typical of the policy than wrong ones ($\bar{c}_1 > \bar{c}_0$), whereas SFT on unfamiliar teacher traces $y^*$ has push $c(y^*) < 0$ and flattens ($\tau_s < 1$, `Prediction P12`).
   - Under 4 options, prompts with $p_{\text{key}} < 1/4$ give a negative push on $\tau$. Therefore, filtering out `0, 1, 7, 8 of 8` in `M1c` **raises** the expected stationary gain ($\log \tau_c / \log \tau_b = 1.18\text{ to }1.30$, `Prediction P5`).

4. **Within-Run Discrimination as a Trajectory Derivative (`math.md §4.1–4.2`, `Predictions P7, P8`)**:
   - Let $u$ be the first right singular vector of the centered weight trajectory $\Delta w(t)$, $g(t)$ its score, and $\hat{a}_j$ the OLS slope of $z_j(w(t))$ on $g(t)$. To first order:
     $$\hat{a}_j = J_j u + R_j = \frac{u \cdot e_\gamma}{\|t_\gamma\|} z_j(w(0)) + J_j u_\perp + R_j, \qquad J_j = \nabla_w z_j(w(0)),$$
     where the first term is the common gain along the final-norm tangent $t_\gamma$ ($J_j t_\gamma = z_j(w(0))$) and $\tilde{a}_j \approx J_j u_\perp$ is the item-specific within-run discrimination (out-of-sample by construction).

---

## 3. Structural Isomorphism with `JEDM` Section 3 (*Prediction Is Not Calibration*)

In your completed JEDM manuscript (`C:\Users\steph\Documents\deep-mirt\overleaf-sync`), Section 3 proves two foundational properties of neural psychometric architectures:
1. **Fisher Information Asymmetry Between Slope $\alpha_q$ and Location $\beta_q$**:
   - The score and Fisher information for difficulty $\beta_q$ are $\mathcal{O}(1)$ at first order ($I_{\beta\beta} = \alpha_q^2 p(1-p)$), whereas the Fisher information for discrimination $\alpha_q$ is scaled by the squared latent residual $I_{\alpha\alpha} = (\theta_i - \beta_q)^2 p(1-p)$ and vanishes near $\theta_i \approx \beta_q$, requiring wide respondent spread to identify.
   - **LLM Counterpart**: Explains why in `E1` ($R=61$ LLMs) and `Gate Within-Run`, difficulty $b_j$ is recovered reliably within a single run while slope $a_j$ fails at $R=61$ (`E1`) and requires averaging across 6 runs (`Gate Within-Run`).
2. **Affine Logit Absorption (`JEDM Eq. 7–9`)**:
   - Because $\alpha_q(\hat{\theta}_{i,t}^- - \beta_q) = (\alpha_q s_q)(\theta_i^{\text{ref}} - \frac{\beta_q - \phi_q}{s_q})$, an unconstrained sequential encoder leaks item-difficulty and history statistics into $\hat{\theta}_{i,t}^-$ ($r = -0.84$ on fixed-order data), which $(\alpha_q, \beta_q)$ absorb—destroying parameter recovery (`0.35` vs `0.96`) even though next-step test accuracy differs by only `0.004`.
   - **LLM Counterpart (`Track G R3/R10`, `S1`, `Q2g`)**: Compressing a post-trained generative policy into a 1D static 2PL readout $\sigma(a_j^{\text{pop}}(\theta_s - b_j^{\text{pop}}))$ suffers the **exact same affine & dimensional absorption**: it collapses preference reordering ($r_{s,\text{base}}$), logit dispersion ($\sigma_s/\sigma_b$), and within-item sampling entropy ($H_s$) into a single scalar weighted by $a_j^{\text{pop}}$, flipping SFT:GRPO credit from `60:40` to `21:79` while hiding the `mean@16` vs `maj@16` distribution crossing.
