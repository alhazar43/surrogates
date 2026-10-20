# Surrogates used by others for two probes (SweepC)

Notation. w is the parameter vector, l(w, z) the loss on example z, eta the step size. Every source below was opened as PDF full text on 2026-10-19, except Jain et al., opened earlier (reading.md). Each row cites its section or equation. No surrogate is ranked here.

## Probe 1. Revival: a few steps of retraining bring back a capability that fine-tuning hid

Exact target. Let w_ft be the fine-tuned model and C a capability with test cases S_C. Revival is the smallest number of steps k such that accuracy on S_C, after k steps of training on data that carries no S_C answers, reaches the pre-fine-tuning level, with k much smaller than for a model that never had C.

Common failure. Rows S1 to S4 have no built-in baseline for how fast a model that truly lacks C would recover (Deeb and Roger 2025, related work). Each needs that control as a second run.

### S1. Reverse fine-tuning time
- Computes: iterations k of training on the original task until its accuracy reaches the pretrained level, compared with the iterations of a control that never had the capability.
- Source: Jain, Kirk, Lubana, Dick, Tanaka, Grefenstette, Rocktaschel, Krueger 2024, Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks, arXiv 2311.12786, Section 5.1, Figure 9.
- Tracks the target if: the control has the same size, optimizer and learning rate, and differs from the test model only in the capability.
- Misleads if: the fine-tuning learning rate was large (no revival there, Figure 43), or the control trains on a different task pair, so the comparison differs in more than the capability.
- Cost, small model: one training run per setting, 0.1K to 4.5K iterations on a 3M parameter model (Appendix C.2). Minutes on one GPU.

### S2. Recovery rate after retraining on T
- Computes: Recovery Rate = Acc_V(unlearned model after RTT) / Acc_V(original model after RTT). RTT fine-tunes on facts T and tests on facts V that are independent of T.
- Source: Deeb and Roger 2025, Do unlearning methods remove information from language model weights?, arXiv 2410.08827v3, Section 3.2 (min-max objective), Section 4.3 (RTT, Recovery Rate), Section 7.1 (limits).
- Tracks the target if: T and V share no information, so a gain on V comes from undoing a hiding step and not from teaching V, and the hiding acts on T and V alike.
- Misleads if: hiding is local to each fact (for example in different layers). RTT then recovers little although the information remains (Section 7.1). It also needs low-leakage facts, which do not exist for a skill such as coding.
- Cost, small model: fine-tuning with a learning rate search. The paper uses 628 facts in T, 157 in V, two runs (Section 4.3). It costs more than one benchmark evaluation (Section 7.1).

### S3. Attack success rate of benign relearning
- Computes: ASR = fraction of queries q whose completion gen(w, q) contains the target keyword (eval(o, q)), after fine-tuning the unlearned model on related text that does not hold the answer. Reported against relearning steps and unlearning depth.
- Source: Hu, Fu, Wu, Smith 2025, Unlearning or obfuscating? Jogging the memory of unlearned LLMs via benign relearning, arXiv 2406.13356, Section 2.3 (eval), Section 3 and Figure 3 (ASR).
- Tracks the target if: the relearn set gives no direct information on the evaluation queries, so a hit is a recovered association and not a copy.
- Misleads if: unlearning is deep (ASR falls as unlearning steps grow, Section 3), or relearning runs too long. ASR drops after about 40 steps on TOFU for all checkpoints (Section 3).
- Cost, small model: LoRA relearning, batch size 8, learning rate 2e-4 (appendix hyperparameters). Tens of steps per cell, over a grid of unlearning checkpoints.

### S4. Epochs to regain F1 after pruning
- Computes: F1 of the retrained model against epochs, after pruning the most salient concept neurons. Revival is the epoch at which F1 reaches the base model F1.
- Source: Lo, Cohen, Barez 2024, Large language models relearn removed concepts, arXiv 2401.01814, Section 5.3, Section 6.2 (F1 0.443 base, 0.330 after 2 epochs of retraining).
- Tracks the target if: the pruned concept is the only change, so a return of F1 reflects the concept returning.
- Misleads if: recovery comes from re-learning the task, not the concept. F1 on named entities is not accuracy on a test set of the capability. The paper shows one model (DistilBERT) and one task.
- Cost, small model: 2 to 8 epochs of retraining a DistilBERT. Cheap.

### S5. Linear probe accuracy on the hidden capability
- Computes: accuracy of a linear readout of the capability from a layer of w_ft.
- Source: Jain et al. 2024 (as S1), Figure 8; limit stated in Deeb and Roger 2025, related work.
- Tracks the target if: the capability is linearly decodable at the probed layer and the probe trains on data without the test answers.
- Misleads if: the information sits in weights and not in activations. Probes on RMU models fail while the models still yield to jailbreaks (Deeb and Roger).
- Cost, small model: one forward pass and one linear fit per layer. It does not measure retraining steps.

## Probe 2. One-step response: how a learner reacts to one example predicts its later learning

Exact target. Let z be an example and w_1 = w_0 - eta grad l(w_0, z) the model after one step. The target is the relation, across z, between the one-step response R(z) (change in loss, output or parameters at w_1) and the later change in loss on a test set over the steps that follow.

### T1. Idealized TracIn (loss drop of the step)
- Computes: TracInIdeal(z, z') = sum over steps t with z_t = z of [l(w_t, z') - l(w_{t+1}, z')]. For a minibatch B_t of size b the first-order form is (eta_t / b) grad l(w_t, z') . grad l(w_t, z).
- Source: Pruthi, Liu, Kale, Sundararajan 2020, Estimating training data influence by tracing gradient descent, arXiv 2002.08484v3, Section 3.1 (idealized), Section 3.2 (first-order, minibatches).
- Tracks the target if: the step is small enough that the O(eta^2) remainder is small. Remark 3.3 gives the check, the real loss change against the first-order term.
- Misleads if: the order of the data matters. The score belongs to the run that visited z at that step, and another run changes it (footnote 5). It also counts only steps in which z was used.
- Cost, small model: two gradients per step, or two forward passes if the loss drop itself is used. The cheapest exact version.

### T2. TracInCP (checkpoint sum)
- Computes: TracInCP(z, z') = sum over checkpoints i of eta_i grad l(w_ti, z) . grad l(w_ti, z') (Eq. 1).
- Source: Pruthi et al. 2020, Section 3.3, Eq. 1.
- Tracks the target if: each example is visited about once between checkpoints and the step size is constant between them. It can also score an example outside the training set (Remark 3.5).
- Misleads if: checkpoints are sparse, so the checkpoint gradient differs from the step gradient. Different checkpoints rank different classes first (Section 4.2), so the checkpoint set changes the answer.
- Cost, small model: k gradient pairs per (z, z'). The paper uses the last layer only for ResNet-56.

### T3. Influence function
- Computes: I_up,loss(z, z_test) = - grad l(z_test, w_hat)^T H^-1 grad l(z, w_hat), with H the Hessian of the empirical risk at the minimum w_hat (Eq. 2).
- Source: Koh and Liang 2017, Understanding black-box predictions via influence functions, arXiv 1703.04730v3, Section 2.1, Eq. 2; Section 4.2 for non-convexity.
- Tracks the target if: w_hat minimizes a twice differentiable, strictly convex risk. It then approximates leave-one-out retraining, an effect at convergence, not a one-step response.
- Misleads if: the model is not at a minimum or the loss is non-convex, so H can have negative eigenvalues. A damping term is then added (Section 4.2). Non-differentiable losses give wrong values until smoothed (Figure 3). It does not describe the early path of training.
- Cost, small model: one Hessian-vector product solve per example (Section 3). Exact H^-1 is feasible only for a small parameter count.

### T4. LESS influence with Adam (cosine of gradients)
- Computes: Inf_Adam(z, z') = sum over epochs i of eta_i cos(grad l(z'; w_i), Gamma(z, w_i)), with Gamma the Adam update direction m_{t+1} / (sqrt(v_{t+1}) + eps) (Definition 3.1).
- Source: Xia, Malladi, Gururangan, Arora, Chen 2024, LESS: Selecting influential data for targeted instruction tuning, arXiv 2402.04333v3, Section 3.1, Definition 3.1; Section 4.1 for the LoRA projection.
- Tracks the target if: a first-order expansion of one Adam step holds, and the optimizer state (m, v) at the warmup checkpoints stands for the state of the later run (the paper calls this circular).
- Misleads if: gradient norm depends on sequence length. The dot product then favors short examples, so the paper uses the cosine, which discards the size of the step (Section 3).
- Cost, small model: LoRA plus a 8192 dimensional random projection. Gradient features cost 48 A100 hours for a 7B model (Table 4). Much less at small scale, where exact gradients fit.

## Check against the unit
- Probe 1 has five sourced surrogates (S1 to S5). Probe 2 has four (T1 to T4). No row is abstract only.
- Read in part: Hu (Sections 2.3 and 3, appendix hyperparameters; Section 5 theory not read), Lo (Sections 5.3 and 6.2; the neuron saliency analysis not read), Koh (Sections 2 and 4.2; Section 3 skimmed).
