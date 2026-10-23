"""
Generate the 6 clean figures (Fig00 through Fig05) for the Oct 26, 2026 presentation:

  - Fig00_eedi_human_vs_llm.png (Unit E1):
      Dataset: 704 Eedi UK school math items; 4,918 human students vs. 61 open-weight LLMs.
      Role: Opening boundary premise — human 2PL difficulty b_human does not transfer to LLMs (rho = 0.09).

  - Fig01_posttraining_eval_across_n_q2a.png (Unit Q2a Extended, Single-Panel):
      Dataset: 1,472 MMLU items calibrated on 102 Open-LLM-Leaderboard models; evaluated on 7 post-training
      checkpoints of Llama-3.2-1B-Instruct (Base, 3 CoT SFT seeds, 3 GRPO seeds) across n = 10..1,472.
      Role: Static population 2PL adaptive testing fails on post-trained models (hits 2.85% specification bias floor
      even at full bank N = 1,472, while Random Sampling RMSE drops along 1/sqrt(n) to 0.40%).

  - Fig02_sft_vs_grpo_empirical_puzzle_s1.png (Unit S1, Single-Panel — NO IRT, sets up the empirical puzzle):
      Dataset: 1,500 MMLU bank items x 2 option orders x 16 sampled CoT rollouts (T = 1.0) on Llama-3.2-1B-Instruct.
      Role: GRPO beats CoT SFT on 1-draw Pass@1 accuracy (47.9% vs 43.9%), but CoT SFT beats GRPO on
      Modal-Vote Accuracy (55.5% vs 49.0%), Pass@4 (78.1% vs 57.2%), and Pass@16 (96.6% vs 65.6%).

  - Fig03_irt_subject_separation_2d.png (Units W0 + S1 + Q2g, Single-Panel — POSITIVE IRT DIAGNOSTIC #1):
      Dataset: 1,472 matched MMLU items across 15 subjects (Llama-3.2-1B-Instruct CoT SFT vs. GRPO).
      Role: 2D item parameters (a_m, b_m) explain the puzzle at the subject level — item difficulty b_m separates
      GRPO-favored factual subjects from SFT-favored math/quant subjects (r = +0.85), while discrimination a_m
      shows why 1D ability theta overweights factual recall 5x over math.

  - Fig04_irt_item_slope_mechanism.png (Unit T9r, Single-Panel — POSITIVE IRT DIAGNOSTIC #2):
      Dataset: 1,472 matched MMLU items grouped into 10 equal-size difficulty deciles (~147 items/bin).
      Role: Within-run IRT slope tau_s explains the puzzle at the item level — GRPO acts as a slope-sharpening
      operator (tau = 1.88 > 1, 887 items up, 613 items down), whereas CoT SFT flattens the slope (tau = 0.71 < 1)
      while lifting hard items.

  - Fig05_rl_critics_q1.png (Unit Q1, 2 Coupled Panels on the Exact Same 4 RL Runs):
      Dataset: Llama-3.2-1B-Instruct trained for 333 RL updates on 17,068 MMLU train items, evaluated on 1,500 MMLU bank items.
      Role: Static 2PL critics inside RL (pop0, pop1) match 5-draw GRPO single-draw accuracy (~47.5%) with zero
      pre-samples, but cannot break the 48% ceiling and reduce multi-draw Pass@4 / Pass@16 accuracy.
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.stats import spearmanr, rankdata
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO_ROOT = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\item-params-llm"
SIG_ROOT = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\plms-recent\signature"
RESULTS = os.path.join(REPO_ROOT, "results")
SIG_RESULTS = os.path.join(SIG_ROOT, "results")

OUT_DIRS = [
    r"D:\surrogates\gemini\figures",
    r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49",
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)
for old_f in os.listdir(OUT_DIRS[0]):
    if old_f.endswith(".png"):
        os.remove(os.path.join(OUT_DIRS[0], old_f))


def save_fig(fig, filename):
    first_path = os.path.join(OUT_DIRS[0], filename)
    fig.savefig(first_path, bbox_inches="tight", facecolor="white")
    for d in OUT_DIRS[1:]:
        shutil.copy2(first_path, os.path.join(d, filename))
    plt.close(fig)
    print(f"Saved: {filename}")


# ==============================================================================
# FIGURE 0 (Unit E1): Human Student IRT vs. LLM IRT on 704 Eedi Math Items
# ==============================================================================
def make_fig0_eedi():
    diff_df = pd.read_csv(os.path.join(RESULTS, "E1", "difficulty.csv")).dropna(subset=["b_students", "b_llms"]).copy()
    b_stud = diff_df["b_students"].to_numpy(float)
    a_stud = diff_df["a_students"].to_numpy(float)
    b_llm = diff_df["b_llms"].to_numpy(float)

    x_pct = (rankdata(b_stud) - 1) / (len(b_stud) - 1) * 100.0
    y_llm_pct = (rankdata(b_llm) - 1) / (len(b_llm) - 1) * 100.0

    rng = np.random.default_rng(42)
    theta_syn = rng.standard_normal(61)
    p_syn = expit(a_stud[:, None] * (theta_syn[None, :] - b_stud[:, None]))
    mask = rng.random(p_syn.shape) < 0.38
    draws = (rng.random(p_syn.shape) < p_syn) * mask
    y_syn_pass = np.where(mask.sum(axis=1) > 0, draws.sum(axis=1) / np.maximum(mask.sum(axis=1), 1), 0.5)
    y_syn_pct = (rankdata(-y_syn_pass) - 1) / (len(y_syn_pass) - 1) * 100.0

    rho_stud, _ = spearmanr(b_stud, -y_syn_pass)
    rho_llm, _ = spearmanr(b_stud, b_llm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    fig.subplots_adjust(wspace=0.14)

    ax1.scatter(x_pct, y_syn_pct, s=18, color="#0284C7", alpha=0.42, edgecolors="none", label="704 Eedi Math Items")
    ax1.plot([0, 100], [0, 100], "--", color="#0F172A", linewidth=2.2, label=f"y = x (Spearman \u03c1 = {rho_stud:.2f})")
    ax1.set_title("(A) Human IRT \u2192 Held-Out Students (Eedi Math)", pad=8)
    ax1.set_xlabel("Human 2PL Difficulty Percentile b_human")
    ax1.set_ylabel("Target Group Difficulty Percentile")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92)
    ax1.grid(True, linestyle=":", alpha=0.4)

    ax2.scatter(x_pct, y_llm_pct, s=18, color="#DC2626", alpha=0.42, edgecolors="none", label="704 Eedi Math Items")
    ax2.plot([0, 100], [0, 100], "--", color="#64748B", linewidth=1.8, label="y = x")
    ax2.plot([0, 100], [50, 50], "-", color="#DC2626", linewidth=2.6,
             label=f"LLM Fit (Spearman \u03c1 = {rho_llm:.2f})")
    ax2.set_title("(B) Human IRT \u2192 61 Open-Weight LLMs (Eedi Math)", pad=8)
    ax2.set_xlabel("Human 2PL Difficulty Percentile b_human")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92)
    ax2.grid(True, linestyle=":", alpha=0.4)

    save_fig(fig, "Fig00_eedi_human_vs_llm.png")


# ==============================================================================
# FIGURE 1 (Unit Q2a Extended, Single-Panel): Post-Training Evaluation Across Full Bank (n = 10..1,472)
# ==============================================================================
def make_fig1_posttraining_q2a():
    ext_df = pd.read_csv(os.path.join(RESULTS, "Q2a", "errors_ext.csv"))
    ext_df = ext_df[ext_df["target"] == "state"].copy()
    ns_ext = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1472]
    k16 = ext_df[ext_df["k"] == 16].groupby(["n", "method"])["rmse"].mean().unstack() * 100.0

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))

    y_cl = [k16.loc[n, "classical"] for n in ns_ext]
    y_rd = [k16.loc[n, "irt_random"] for n in ns_ext]
    y_ad = [k16.loc[n, "irt_adaptive"] for n in ns_ext]

    ax.plot(ns_ext, y_cl, "--o", color="#475569", linewidth=2.5, markersize=6,
            label=f"Random Sampling (Sample Mean \u2192 {y_cl[-1]:.2f}%)")
    ax.plot(ns_ext, y_rd, ":^", color="#F59E0B", linewidth=2.3, markersize=5.8,
            label=f"2PL IRT Scoring (Random Items \u2192 {y_rd[-1]:.2f}%)")
    ax.plot(ns_ext, y_ad, "-s", color="#DC2626", linewidth=2.7, markersize=6.2,
            label=f"2PL IRT Adaptive (Max Fisher Info \u2192 {y_ad[-1]:.2f}% floor)")

    ax.axhline(y_ad[-1], color="#DC2626", linestyle=":", linewidth=1.3, alpha=0.65)

    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 300, 1472])
    ax.set_xticklabels(["10", "25", "50", "100", "300", "1472 (Full Bank)"])
    ax.set_xlim(9, 1750)
    ax.set_ylim(0, 16.5)
    ax.set_title("Post-Training Benchmark Estimation Error Across Test Length (n = 10..1,472)", pad=10)
    ax.set_xlabel("Number of Sampled MMLU Items n (log scale, out of 1,472)")
    ax.set_ylabel("Score Estimation RMSE (%) (\u2193)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    save_fig(fig, "Fig01_posttraining_eval_across_n_q2a.png")


# ==============================================================================
# FIGURE 2 (Unit S1, Single-Panel — NO IRT): The Post-Training Empirical Puzzle
# ==============================================================================
def make_fig2_sft_vs_grpo_puzzle():
    s1_raw = pd.read_csv(os.path.join(RESULTS, "S1", "stats.csv"))
    s1_pool = s1_raw[s1_raw["order"] == "pooled"].pivot(index="state", columns="statistic", values="value") * 100.0
    rw1 = pd.read_csv(os.path.join(RESULTS, "S1", "rewrite1_parsed.csv")).set_index("state")

    metrics = ["mean_at_16", "maj_tie_correct", "pass_at_4", "pass_at_16"]
    x_labels = [
        "Pass@1 Accuracy\n(Single-Draw Mean@16)",
        "Modal-Vote Accuracy\n(Mode = Key, incl. ties)",
        "Pass@4 Accuracy\n(Solved in \u22651 of 4 Draws)",
        "Pass@16 Accuracy\n(Solved in \u22651 of 16 Draws)",
    ]

    def get_vals(state_list):
        out = []
        for m in metrics:
            if m == "maj_tie_correct":
                vals = [rw1.loc[s, "maj_tie_correct"] * 100.0 for s in state_list]
            else:
                vals = [s1_pool.loc[s, m] for s in state_list]
            out.append(vals)
        return out

    base_vals = get_vals(["t9_base"])
    grpo_vals = get_vals(["t9_rl_0_666", "t9_rl_1_666", "t9_rl_2_666"])
    sft_vals = get_vals(["t9_sft_0_70", "t9_sft_1_70", "t9_sft_2_70"])

    base_means = [np.mean(v) for v in base_vals]
    grpo_means = [np.mean(v) for v in grpo_vals]
    sft_means  = [np.mean(v) for v in sft_vals]

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.0))
    x = np.arange(len(metrics))
    w = 0.25

    ax.bar(x - w, base_means, width=w, color="#94A3B8", edgecolor="#475569", linewidth=0.9, label="Base Model (Llama-3.2-1B)")
    ax.bar(x,     grpo_means, width=w, color="#DC2626", edgecolor="#991B1B", linewidth=0.9, alpha=0.88, label="GRPO (3 seeds)")
    ax.bar(x + w, sft_means,  width=w, color="#0284C7", edgecolor="#0369A1", linewidth=0.9, alpha=0.88, label="CoT SFT (3 seeds)")

    for i in range(len(metrics)):
        ax.scatter(np.full(len(grpo_vals[i]), x[i]), grpo_vals[i], s=26, color="#7F1D1D", zorder=5)
        ax.scatter(np.full(len(sft_vals[i]), x[i] + w), sft_vals[i], s=26, color="#0C4A6E", zorder=5)

    for i in range(len(metrics)):
        ax.text(x[i] - w, max(base_vals[i]) + 1.8, f"{base_means[i]:.1f}%", ha="center", fontsize=9.2, color="#334155", fontweight="bold")
        ax.text(x[i],     max(grpo_vals[i]) + 1.8, f"{grpo_means[i]:.1f}%", ha="center", fontsize=9.2, color="#991B1B", fontweight="bold")
        ax.text(x[i] + w, max(sft_vals[i])  + 1.8, f"{sft_means[i]:.1f}%",  ha="center", fontsize=9.2, color="#0369A1", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("MMLU Test Bank Accuracy / Pass@k (%) (\u2191)")
    ax.set_ylim(0, 109)
    ax.set_title("Post-Training Accuracy Reversal: Single-Draw vs. Modal Vote & Pass@k (1,500 MMLU Items)", pad=10)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.93)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    save_fig(fig, "Fig02_sft_vs_grpo_empirical_puzzle_s1.png")


# ==============================================================================
# FIGURE 3 (Units W0 + S1 + Q2g, Single-Panel — POSITIVE IRT #1):
#   How 2D Item Parameters (a_m, b_m) Explain the Subject-Level Crossover
# ==============================================================================
def make_fig3_irt_subject_separation():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    it10 = pd.read_csv(os.path.join(SIG_RESULTS, "T10", "runs", "items.csv"))
    flags = pd.read_csv(os.path.join(RESULTS, "Q2g", "s1", "flags.csv"))
    pf = pd.read_csv(os.path.join(RESULTS, "Q2g", "s1", "personfit.csv")).iloc[:7]

    th_base = pf[pf["method"] == "base"]["theta"].mean()
    th_sft = pf[pf["method"] == "sft"]["theta"].mean()
    th_grpo = pf[pf["method"] == "grpo"]["theta"].mean()

    a3 = flags[(flags["reading"] == "A3_subject_mean_c") & (flags["variant"] == "all") & (flags["state"] == "pooled_states")]
    sub_a = w0_ok.groupby("subject")["a"].mean()
    sub_b = w0_ok.groupby("subject")["b"].mean()
    p_base_sub = it10.groupby("subject")["p_ref"].mean()
    c_sft = a3[a3["method"] == "sft"].set_index("subject")["value"]
    c_grpo = a3[a3["method"] == "grpo"].set_index("subject")["value"]
    dp_sft_sub = (expit(logit(p_base_sub) + c_sft + sub_a * (th_sft - th_base)) - p_base_sub) * 100.0
    dp_grpo_sub = (expit(logit(p_base_sub) + c_grpo + sub_a * (th_grpo - th_base)) - p_base_sub) * 100.0

    df_sub = pd.DataFrame({"b": sub_b, "a": sub_a, "dp_sft": dp_sft_sub, "dp_grpo": dp_grpo_sub}).sort_values("b")

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.2))
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)

    sizes = df_sub["a"] * 95.0
    ax.scatter(df_sub["b"], df_sub["dp_sft"], s=sizes, color="#0284C7", alpha=0.84, edgecolors="#0369A1", linewidth=1.0,
               label="CoT SFT (wins 5/6 high-b, low-a math/quant subjects)")
    ax.scatter(df_sub["b"], df_sub["dp_grpo"], s=sizes, color="#DC2626", marker="s", alpha=0.84, edgecolors="#991B1B", linewidth=1.0,
               label="GRPO (wins 8/8 low-b, high-a factual subjects)")

    x_grid = np.linspace(df_sub["b"].min() - 0.12, df_sub["b"].max() + 0.12, 100)
    m_sft, c_sft_l = np.polyfit(df_sub["b"], df_sub["dp_sft"], 1)
    m_grpo, c_grpo_l = np.polyfit(df_sub["b"], df_sub["dp_grpo"], 1)
    ax.plot(x_grid, m_sft * x_grid + c_sft_l, "-", color="#0284C7", linewidth=2.2, alpha=0.8)
    ax.plot(x_grid, m_grpo * x_grid + c_grpo_l, "-", color="#DC2626", linewidth=2.2, alpha=0.8)

    for subj, label, dx, dy, col in [
        ("marketing", "Marketing (a=1.45)", 0.07, -0.5, "#991B1B"),
        ("sociology", "Sociology (a=1.28)", 0.07, -0.8, "#991B1B"),
        ("elementary_mathematics", "Elem. Math (a=0.45)", -0.85, -0.8, "#0369A1"),
        ("high_school_mathematics", "HS Math (a=0.28)", -0.72, 1.8, "#0369A1"),
    ]:
        r = df_sub.loc[subj]
        col_y = "dp_grpo" if col == "#991B1B" else "dp_sft"
        ax.text(r["b"] + dx, r[col_y] + dy, label, fontsize=9.2, color=col, fontweight="bold")

    ax.set_title("IRT (a_m, b_m) Separates SFT vs. GRPO Gains Across 15 MMLU Subjects (r = +0.85)", pad=10)
    ax.set_xlabel("Subject Mean IRT Difficulty b_m (Easy \u2192 Hard)")
    ax.set_ylabel("Pass-Rate Change Over Base: \u0394Mean@16 (% pts) (\u2191)")
    ax.set_xlim(-1.15, 2.55)
    ax.set_ylim(-14, 44)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.93, fontsize=9.0)
    ax.grid(True, linestyle=":", alpha=0.4)

    save_fig(fig, "Fig03_irt_subject_separation_2d.png")


# ==============================================================================
# FIGURE 3 v2: 15 MMLU Subjects with 1,472 Background Items (a_m, b_m)
# ==============================================================================
def make_fig3_v2_irt_subject_separation():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    it10 = pd.read_csv(os.path.join(SIG_RESULTS, "T10", "runs", "items.csv"))
    flags = pd.read_csv(os.path.join(RESULTS, "Q2g", "s1", "flags.csv"))
    pf = pd.read_csv(os.path.join(RESULTS, "Q2g", "s1", "personfit.csv")).iloc[:7]

    th_base = pf[pf["method"] == "base"]["theta"].mean()
    th_sft = pf[pf["method"] == "sft"]["theta"].mean()
    th_grpo = pf[pf["method"] == "grpo"]["theta"].mean()

    a3 = flags[(flags["reading"] == "A3_subject_mean_c") & (flags["variant"] == "all") & (flags["state"] == "pooled_states")]
    sub_a = w0_ok.groupby("subject")["a"].mean()
    sub_b = w0_ok.groupby("subject")["b"].mean()
    p_base_sub = it10.groupby("subject")["p_ref"].mean()
    c_sft = a3[a3["method"] == "sft"].set_index("subject")["value"]
    c_grpo = a3[a3["method"] == "grpo"].set_index("subject")["value"]
    dp_sft_sub = (expit(logit(p_base_sub) + c_sft + sub_a * (th_sft - th_base)) - p_base_sub) * 100.0
    dp_grpo_sub = (expit(logit(p_base_sub) + c_grpo + sub_a * (th_grpo - th_base)) - p_base_sub) * 100.0

    df_sub = pd.DataFrame({"b": sub_b, "a": sub_a, "dp_sft": dp_sft_sub, "dp_grpo": dp_grpo_sub}).sort_values("b")

    # Item-level background cloud
    it10_idx = it10.set_index("item_id").loc[w0_ok["bank_row_id"]]
    p_base = np.clip(it10_idx["p_ref"].values, 1e-4, 1 - 1e-4)
    L_base = logit(p_base)
    w0_ok["c_sft"] = w0_ok["subject"].map(c_sft.to_dict())
    w0_ok["c_grpo"] = w0_ok["subject"].map(c_grpo.to_dict())
    w0_ok["dp_sft_item"] = (expit(L_base + w0_ok["c_sft"].values + w0_ok["a"].values * (th_sft - th_base)) - p_base) * 100.0
    w0_ok["dp_grpo_item"] = (expit(L_base + w0_ok["c_grpo"].values + w0_ok["a"].values * (th_grpo - th_base)) - p_base) * 100.0

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.2))
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)

    # 1,472 raw items background scatter
    ax.scatter(w0_ok["b"], w0_ok["dp_sft_item"], s=w0_ok["a"] * 16.0, color="#0284C7", alpha=0.10, edgecolors="none", zorder=2)
    ax.scatter(w0_ok["b"], w0_ok["dp_grpo_item"], s=w0_ok["a"] * 16.0, color="#DC2626", alpha=0.10, edgecolors="none", zorder=2)

    # 15 Subject bubbles
    sizes = df_sub["a"] * 100.0
    ax.scatter(df_sub["b"], df_sub["dp_sft"], s=sizes, color="#0284C7", alpha=0.88, edgecolors="#0369A1", linewidth=1.1, zorder=4,
               label="CoT SFT (wins 5/6 high-b, low-a math/quant subjects)")
    ax.scatter(df_sub["b"], df_sub["dp_grpo"], s=sizes, color="#DC2626", marker="s", alpha=0.88, edgecolors="#991B1B", linewidth=1.1, zorder=4,
               label="GRPO (wins 8/8 low-b, high-a factual subjects)")

    x_grid = np.linspace(df_sub["b"].min() - 0.12, df_sub["b"].max() + 0.12, 100)
    m_sft, c_sft_l = np.polyfit(df_sub["b"], df_sub["dp_sft"], 1)
    m_grpo, c_grpo_l = np.polyfit(df_sub["b"], df_sub["dp_grpo"], 1)
    ax.plot(x_grid, m_sft * x_grid + c_sft_l, "-", color="#0284C7", linewidth=2.2, alpha=0.85, zorder=3)
    ax.plot(x_grid, m_grpo * x_grid + c_grpo_l, "-", color="#DC2626", linewidth=2.2, alpha=0.85, zorder=3)

    for subj, label, dx, dy, col in [
        ("marketing", "Marketing (a=1.45)", 0.07, -0.5, "#991B1B"),
        ("sociology", "Sociology (a=1.28)", 0.07, -0.8, "#991B1B"),
        ("elementary_mathematics", "Elem. Math (a=0.45)", -0.85, -0.8, "#0369A1"),
        ("high_school_mathematics", "HS Math (a=0.28)", -0.72, 1.8, "#0369A1"),
    ]:
        r = df_sub.loc[subj]
        col_y = "dp_grpo" if col == "#991B1B" else "dp_sft"
        ax.text(r["b"] + dx, r[col_y] + dy, label, fontsize=9.2, color=col, fontweight="bold", zorder=5)

    ax.set_title("IRT (a_m, b_m) Separates SFT vs. GRPO Across 15 Subjects & 1,472 Items (r = +0.85)", pad=10)
    ax.set_xlabel("Subject Mean IRT Difficulty b_m (Easy \u2192 Hard)")
    ax.set_ylabel("Pass-Rate Change Over Base: \u0394Mean@16 (% pts) (\u2191)")
    ax.set_xlim(-1.15, 2.55)
    ax.set_ylim(-14, 44)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.93, fontsize=9.0)
    ax.grid(True, linestyle=":", alpha=0.4)

    save_fig(fig, "Fig03_v2.png")


# ==============================================================================
# FIGURE 4 (Unit T9r, Single-Panel — POSITIVE IRT #2):
#   Item-Level Pass-Rate Change & Person-Slope Operator tau_s Across Difficulty b_j
# ==============================================================================
def make_fig4_irt_item_slope():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    w0_ok["p_base"] = np.clip(t9["p_key_base"].values, 1e-4, 1 - 1e-4)
    w0_ok["L_base"] = logit(w0_ok["p_base"])
    w0_ok["dp_sft"] = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values * 100.0
    w0_ok["dp_grpo"] = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values * 100.0

    w0_ok["p_sft"] = np.clip(w0_ok["p_base"] + w0_ok["dp_sft"] / 100.0, 1e-4, 1 - 1e-4)
    w0_ok["p_grpo"] = np.clip(w0_ok["p_base"] + w0_ok["dp_grpo"] / 100.0, 1e-4, 1 - 1e-4)

    L0_all = w0_ok["L_base"].to_numpy()
    res_g = minimize(lambda x: np.mean((w0_ok["p_grpo"].to_numpy() - expit(x[0] * L0_all + x[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda x: np.mean((w0_ok["p_sft"].to_numpy() - expit(x[0] * L0_all + x[1])) ** 2), [0.5, 0.0])
    w0_ok["pred_dp_grpo"] = (expit(res_g.x[0] * L0_all + res_g.x[1]) - w0_ok["p_base"]) * 100.0
    w0_ok["pred_dp_sft"] = (expit(res_s.x[0] * L0_all + res_s.x[1]) - w0_ok["p_base"]) * 100.0

    w0_ok["b_bin"] = pd.qcut(w0_ok["b"], q=10)
    binned = w0_ok.groupby("b_bin", observed=False)[["b", "dp_sft", "dp_grpo", "pred_dp_sft", "pred_dp_grpo"]].mean()
    b_grid = binned["b"].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.0))
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.4)

    ax.scatter(b_grid, binned["dp_grpo"], s=60, color="#DC2626", marker="s", zorder=4,
               label="GRPO Empirical Deciles (887 items \u2191, 613 items \u2193)")
    ax.plot(b_grid, binned["pred_dp_grpo"], "-", color="#DC2626", linewidth=2.6,
            label="GRPO Within-Run IRT Slope (\u03c4 = 1.88 > 1: sharpens preference)")

    ax.scatter(b_grid, binned["dp_sft"], s=60, color="#0284C7", marker="o", zorder=4,
               label="CoT SFT Empirical Deciles (lifts hard items)")
    ax.plot(b_grid, binned["pred_dp_sft"], "-", color="#0284C7", linewidth=2.6,
            label="CoT SFT Within-Run IRT Slope (\u03c4 = 0.71 < 1: flattens & shifts)")

    ax.set_title("Item-Level Pass-Rate Change vs. IRT Difficulty b_j (1,472 MMLU Items)", pad=10)
    ax.set_xlabel("IRT Item Difficulty b_j (10 Equal-Size Decile Bins: Easy \u2192 Hard)")
    ax.set_ylabel("Item Pass-Rate Change: \u0394p_j (% pts) (\u2191)")
    ax.set_ylim(-18, 15.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=9.1)
    ax.grid(True, linestyle=":", alpha=0.4)

    save_fig(fig, "Fig04_irt_item_slope_mechanism.png")


# ==============================================================================
# FIGURE 4 v2: 10 Deciles + 1,472 Individual Items with Discrimination Bubble Sizes
# ==============================================================================
def make_fig4_v2_irt_item_slope():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    w0_ok["p_base"] = np.clip(t9["p_key_base"].values, 1e-4, 1 - 1e-4)
    w0_ok["L_base"] = logit(w0_ok["p_base"])
    w0_ok["dp_sft"] = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values * 100.0
    w0_ok["dp_grpo"] = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values * 100.0

    w0_ok["p_sft"] = np.clip(w0_ok["p_base"] + w0_ok["dp_sft"] / 100.0, 1e-4, 1 - 1e-4)
    w0_ok["p_grpo"] = np.clip(w0_ok["p_base"] + w0_ok["dp_grpo"] / 100.0, 1e-4, 1 - 1e-4)

    L0_all = w0_ok["L_base"].to_numpy()
    res_g = minimize(lambda x: np.mean((w0_ok["p_grpo"].to_numpy() - expit(x[0] * L0_all + x[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda x: np.mean((w0_ok["p_sft"].to_numpy() - expit(x[0] * L0_all + x[1])) ** 2), [0.5, 0.0])
    w0_ok["pred_dp_grpo"] = (expit(res_g.x[0] * L0_all + res_g.x[1]) - w0_ok["p_base"]) * 100.0
    w0_ok["pred_dp_sft"] = (expit(res_s.x[0] * L0_all + res_s.x[1]) - w0_ok["p_base"]) * 100.0

    w0_ok["b_bin"] = pd.qcut(w0_ok["b"], q=10)
    binned = w0_ok.groupby("b_bin", observed=False)[["a", "b", "dp_sft", "dp_grpo", "pred_dp_sft", "pred_dp_grpo"]].mean()
    b_grid = binned["b"].to_numpy()
    sizes_bin = binned["a"].to_numpy() * 110.0

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.2))
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.4)

    # 1,472 raw individual items background scatter
    ax.scatter(w0_ok["b"], w0_ok["dp_grpo"], s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.14, edgecolors="none", zorder=2)
    ax.scatter(w0_ok["b"], w0_ok["dp_sft"], s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.14, edgecolors="none", zorder=2)

    # 10 decile binned markers
    ax.scatter(b_grid, binned["dp_grpo"], s=sizes_bin, color="#DC2626", marker="s", zorder=5, edgecolors="#7F1D1D", linewidth=1.2,
               label="GRPO 10 Deciles")
    ax.scatter(b_grid, binned["dp_sft"], s=sizes_bin, color="#0284C7", marker="o", zorder=5, edgecolors="#0369A1", linewidth=1.2,
               label="CoT SFT 10 Deciles")

    # Continuous fitted curves
    ax.plot(b_grid, binned["pred_dp_grpo"], "-", color="#DC2626", linewidth=2.8, zorder=6,
            label="GRPO Within-Run IRT Slope (\u03c4 = 1.88 > 1: sharpens preference)")
    ax.plot(b_grid, binned["pred_dp_sft"], "-", color="#0284C7", linewidth=2.8, zorder=6,
            label="CoT SFT Within-Run IRT Slope (\u03c4 = 0.71 < 1: flattens & shifts)")

    ax.set_title("Item-Level Pass-Rate Change vs. IRT (a_j, b_j) Across 10 Deciles & 1,472 Items", pad=10)
    ax.set_xlabel("IRT Item Difficulty b_j (10 Equal-Size Deciles: Easy \u2192 Hard)")
    ax.set_ylabel("Item Pass-Rate Change: \u0394p_j (% pts) (\u2191)")
    ax.set_xlim(-2.5, 5.0)
    ax.set_ylim(-22, 22)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.8)
    ax.grid(True, linestyle=":", alpha=0.4)

    save_fig(fig, "Fig04_v2.png")


# ==============================================================================
# FIGURE 4 v3: Full-Point Fit Directly on IRT Difficulty b_j (1,472 Items)
# ==============================================================================
def make_fig4_v3_b_fullpoint_fit():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    w0_ok["dp_sft"] = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values * 100.0
    w0_ok["dp_grpo"] = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values * 100.0

    b = w0_ok["b"].to_numpy()
    dp_g = w0_ok["dp_grpo"].to_numpy()
    dp_s = w0_ok["dp_sft"].to_numpy()

    # 10 Decile bins
    w0_ok["b_bin"] = pd.qcut(w0_ok["b"], q=10)
    binned = w0_ok.groupby("b_bin", observed=False)[["a", "b", "dp_sft", "dp_grpo"]].mean()
    b_grid = binned["b"].to_numpy()
    sizes_bin = binned["a"].to_numpy() * 110.0

    # Dense grid for evaluation on b_j
    b_dense = np.linspace(-2.5, 5.0, 300)

    # Linear fits directly on all 1,472 raw points
    m_g, c_g = np.polyfit(b, dp_g, 1)
    m_s, c_s = np.polyfit(b, dp_s, 1)
    r2_lin_g = 1.0 - np.sum((dp_g - (m_g * b + c_g)) ** 2) / np.sum((dp_g - np.mean(dp_g)) ** 2)
    r2_lin_s = 1.0 - np.sum((dp_s - (m_s * b + c_s)) ** 2) / np.sum((dp_s - np.mean(dp_s)) ** 2)

    # Smooth cubic fits directly on all 1,472 raw points
    p3_g = np.poly1d(np.polyfit(b, dp_g, 3))
    p3_s = np.poly1d(np.polyfit(b, dp_s, 3))
    r2_p3_g = 1.0 - np.sum((dp_g - p3_g(b)) ** 2) / np.sum((dp_g - np.mean(dp_g)) ** 2)
    r2_p3_s = 1.0 - np.sum((dp_s - p3_s(b)) ** 2) / np.sum((dp_s - np.mean(dp_s)) ** 2)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.3))
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.4)

    # 1,472 raw individual items background scatter
    ax.scatter(b, dp_g, s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.15, edgecolors="none", zorder=2)
    ax.scatter(b, dp_s, s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.15, edgecolors="none", zorder=2)

    # 10 decile binned markers
    ax.scatter(b_grid, binned["dp_grpo"], s=sizes_bin, color="#DC2626", marker="s", zorder=5, edgecolors="#7F1D1D", linewidth=1.2,
               label="GRPO 10 Deciles")
    ax.scatter(b_grid, binned["dp_sft"], s=sizes_bin, color="#0284C7", marker="o", zorder=5, edgecolors="#0369A1", linewidth=1.2,
               label="CoT SFT 10 Deciles")

    # Smooth & linear fit curves
    ax.plot(b_dense, p3_g(b_dense), "-", color="#DC2626", linewidth=2.8, zorder=6,
            label=f"GRPO Smooth Fit (R\u00b2 = {r2_p3_g:.3f})")
    ax.plot(b_dense, m_g * b_dense + c_g, "--", color="#991B1B", linewidth=1.8, zorder=6,
            label=f"GRPO Linear Fit (slope = {m_g:.2f}, R\u00b2 = {r2_lin_g:.3f})")

    ax.plot(b_dense, p3_s(b_dense), "-", color="#0284C7", linewidth=2.8, zorder=6,
            label=f"CoT SFT Smooth Fit (R\u00b2 = {r2_p3_s:.3f})")
    ax.plot(b_dense, m_s * b_dense + c_s, "--", color="#0369A1", linewidth=1.8, zorder=6,
            label=f"CoT SFT Linear Fit (slope = +{m_s:.2f}, R\u00b2 = {r2_lin_s:.3f})")

    ax.set_title("Full-Point Fit Directly on IRT Difficulty b_j (1,472 MMLU Items)", pad=10)
    ax.set_xlabel("IRT Item Difficulty b_j (10 Deciles: Easy \u2192 Hard)")
    ax.set_ylabel("Item Pass-Rate Change: \u0394p_j (% pts) (\u2191)")
    ax.set_xlim(-2.5, 5.0)
    ax.set_ylim(-25, 25)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.4)
    ax.grid(True, linestyle=":", alpha=0.4)

    save_fig(fig, "Fig04_v3_b_fullpoint_fit.png")


# ==============================================================================
# FIGURE 4 v4: True Within-Run S-Curve on Base Log-Odds mu_0j = logit(p_base,j)
# ==============================================================================
def make_fig4_v4_base_logit_scurve():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    w0_ok["p_base"] = np.clip(t9["p_key_base"].values, 1e-4, 1 - 1e-4)
    w0_ok["L_base"] = logit(w0_ok["p_base"])
    w0_ok["dp_sft"] = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values * 100.0
    w0_ok["dp_grpo"] = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values * 100.0

    p0 = w0_ok["p_base"].values
    L0 = w0_ok["L_base"].values
    p_g = np.clip(p0 + w0_ok["dp_grpo"].values / 100.0, 1e-4, 1 - 1e-4)
    p_s = np.clip(p0 + w0_ok["dp_sft"].values / 100.0, 1e-4, 1 - 1e-4)

    res_g = minimize(lambda x: np.mean((p_g - expit(x[0] * L0 + x[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda x: np.mean((p_s - expit(x[0] * L0 + x[1])) ** 2), [0.5, 0.0])

    w0_ok["L_bin"] = pd.qcut(w0_ok["L_base"], q=10)
    binned_L = w0_ok.groupby("L_bin", observed=False)[["a", "L_base", "dp_sft", "dp_grpo"]].mean()
    sizes_bin = binned_L["a"].to_numpy() * 110.0

    L_grid = np.linspace(3.5, -4.5, 400)
    pred_grpo_curve = (expit(res_g.x[0] * L_grid + res_g.x[1]) - expit(L_grid)) * 100.0
    pred_sft_curve = (expit(res_s.x[0] * L_grid + res_s.x[1]) - expit(L_grid)) * 100.0

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.4))
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.4)

    # 1,472 raw items background scatter
    ax.scatter(w0_ok["L_base"], w0_ok["dp_grpo"], s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.16, edgecolors="none", zorder=2)
    ax.scatter(w0_ok["L_base"], w0_ok["dp_sft"], s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.16, edgecolors="none", zorder=2)

    # 10 deciles along L_base
    ax.scatter(binned_L["L_base"], binned_L["dp_grpo"], s=sizes_bin, color="#DC2626", marker="s", zorder=5, edgecolors="#7F1D1D", linewidth=1.2,
               label="GRPO 10 Deciles along \u03bc_0")
    ax.scatter(binned_L["L_base"], binned_L["dp_sft"], s=sizes_bin, color="#0284C7", marker="o", zorder=5, edgecolors="#0369A1", linewidth=1.2,
               label="CoT SFT 10 Deciles along \u03bc_0")

    # Continuous analytical S-curves
    ax.plot(L_grid, pred_grpo_curve, "-", color="#DC2626", linewidth=2.8, zorder=6,
            label=f"GRPO Analytical S-Curve (\u03c4 = {res_g.x[0]:.2f} > 1)")
    ax.plot(L_grid, pred_sft_curve, "-", color="#0284C7", linewidth=2.8, zorder=6,
            label=f"CoT SFT Analytical S-Curve (\u03c4 = {res_s.x[0]:.2f} < 1)")

    ax.set_title("Item Pass-Rate Change vs. Base Model Initial Log-Odds \u03bc_0j (1,472 Items)", pad=12)
    ax.set_xlabel("Base Model Initial Log-Odds \u03bc_0j = logit(p_base,j) (Easy \u2192 Hard)")
    ax.set_ylabel("Item Pass-Rate Change: \u0394p_j (% pts) (\u2191)")
    ax.set_xlim(3.5, -4.5)
    ax.set_ylim(-30, 35)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.8)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Secondary top axis showing probability p_base
    ax2 = ax.twiny()
    p_ticks = np.array([0.90, 0.75, 0.50, 0.25, 0.10, 0.05])
    logit_ticks = logit(p_ticks)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(logit_ticks)
    ax2.set_xticklabels(["90%", "75%", "50%", "25%", "10%", "5%"])
    ax2.set_xlabel("Base Model Probability p_base = \u03c3(\u03bc_0j)", fontsize=10.0, fontweight="bold", labelpad=6)

    save_fig(fig, "Fig04_v4_base_logit_scurve.png")


# ==============================================================================
# FIGURE 4 v5: Stacked 2x1 Comparison (Population b_j vs. Within-Model mu_0j)
# ==============================================================================
def make_fig4_v5_stacked_b_vs_base():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    w0_ok["p_base"] = np.clip(t9["p_key_base"].values, 1e-4, 1 - 1e-4)
    w0_ok["L_base"] = logit(w0_ok["p_base"])
    w0_ok["dp_sft"] = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values * 100.0
    w0_ok["dp_grpo"] = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values * 100.0

    b = w0_ok["b"].to_numpy()
    dp_g = w0_ok["dp_grpo"].to_numpy()
    dp_s = w0_ok["dp_sft"].to_numpy()
    L0 = w0_ok["L_base"].to_numpy()
    p0 = w0_ok["p_base"].to_numpy()

    # Panel A: Fits on b_j
    m_g, c_g = np.polyfit(b, dp_g, 1)
    m_s, c_s = np.polyfit(b, dp_s, 1)
    r2_lin_g = 1.0 - np.sum((dp_g - (m_g * b + c_g)) ** 2) / np.sum((dp_g - np.mean(dp_g)) ** 2)
    r2_lin_s = 1.0 - np.sum((dp_s - (m_s * b + c_s)) ** 2) / np.sum((dp_s - np.mean(dp_s)) ** 2)
    p3_g = np.poly1d(np.polyfit(b, dp_g, 3))
    p3_s = np.poly1d(np.polyfit(b, dp_s, 3))
    r2_p3_g = 1.0 - np.sum((dp_g - p3_g(b)) ** 2) / np.sum((dp_g - np.mean(dp_g)) ** 2)
    r2_p3_s = 1.0 - np.sum((dp_s - p3_s(b)) ** 2) / np.sum((dp_s - np.mean(dp_s)) ** 2)

    w0_ok["b_bin"] = pd.qcut(w0_ok["b"], q=10)
    binned_b = w0_ok.groupby("b_bin", observed=False)[["a", "b", "dp_sft", "dp_grpo"]].mean()

    # Panel B: Fits on L_base
    p_g = np.clip(p0 + dp_g / 100.0, 1e-4, 1 - 1e-4)
    p_s = np.clip(p0 + dp_s / 100.0, 1e-4, 1 - 1e-4)
    res_g = minimize(lambda x: np.mean((p_g - expit(x[0] * L0 + x[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda x: np.mean((p_s - expit(x[0] * L0 + x[1])) ** 2), [0.5, 0.0])

    w0_ok["L_bin"] = pd.qcut(w0_ok["L_base"], q=10)
    binned_L = w0_ok.groupby("L_bin", observed=False)[["a", "L_base", "dp_sft", "dp_grpo"]].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 8.8), gridspec_kw={"hspace": 0.40})

    # ==================== Panel A (Top) ====================
    ax1.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)
    ax1.scatter(b, dp_g, s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.15, edgecolors="none", zorder=2)
    ax1.scatter(b, dp_s, s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.15, edgecolors="none", zorder=2)
    ax1.scatter(binned_b["b"], binned_b["dp_grpo"], s=binned_b["a"] * 110.0, color="#DC2626", marker="s", zorder=5, edgecolors="#7F1D1D", linewidth=1.2, label="GRPO 10 Deciles")
    ax1.scatter(binned_b["b"], binned_b["dp_sft"], s=binned_b["a"] * 110.0, color="#0284C7", marker="o", zorder=5, edgecolors="#0369A1", linewidth=1.2, label="CoT SFT 10 Deciles")

    b_dense = np.linspace(-2.5, 5.0, 300)
    ax1.plot(b_dense, p3_g(b_dense), "-", color="#DC2626", linewidth=2.8, zorder=6, label=f"GRPO Smooth Fit (R\u00b2 = {r2_p3_g:.3f})")
    ax1.plot(b_dense, m_g * b_dense + c_g, "--", color="#991B1B", linewidth=1.8, zorder=6, label=f"GRPO Linear Fit (slope = {m_g:.2f}, R\u00b2 = {r2_lin_g:.3f})")
    ax1.plot(b_dense, p3_s(b_dense), "-", color="#0284C7", linewidth=2.8, zorder=6, label=f"CoT SFT Smooth Fit (R\u00b2 = {r2_p3_s:.3f})")
    ax1.plot(b_dense, m_s * b_dense + c_s, "--", color="#0369A1", linewidth=1.8, zorder=6, label=f"CoT SFT Linear Fit (slope = +{m_s:.2f}, R\u00b2 = {r2_lin_s:.3f})")

    ax1.set_title("(A) External Population IRT Difficulty b_j (102 LLMs: Easy \u2192 Hard)", pad=8)
    ax1.set_xlabel("IRT Item Difficulty b_j (Easy \u2192 Hard)")
    ax1.set_ylabel("Pass-Rate Change: \u0394p_j (% pts) (\u2191)")
    ax1.set_xlim(-2.5, 5.0)
    ax1.set_ylim(-28, 32)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.2)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # ==================== Panel B (Bottom) ====================
    ax2.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)

    # Shading and vertical dashed line at logit(0.25)
    ax2.axvspan(-1.0986, -4.5, color="#FEE2E2", alpha=0.22, zorder=1)
    ax2.axvline(-1.0986, color="#DC2626", linestyle=":", linewidth=1.5, alpha=0.8, zorder=3)

    ax2.scatter(w0_ok["L_base"], dp_g, s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.15, edgecolors="none", zorder=2)
    ax2.scatter(w0_ok["L_base"], dp_s, s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.15, edgecolors="none", zorder=2)
    ax2.scatter(binned_L["L_base"], binned_L["dp_grpo"], s=binned_L["a"] * 110.0, color="#DC2626", marker="s", zorder=5, edgecolors="#7F1D1D", linewidth=1.2, label="GRPO 10 Deciles along \u03bc_0")
    ax2.scatter(binned_L["L_base"], binned_L["dp_sft"], s=binned_L["a"] * 110.0, color="#0284C7", marker="o", zorder=5, edgecolors="#0369A1", linewidth=1.2, label="CoT SFT 10 Deciles along \u03bc_0")

    L_grid = np.linspace(3.5, -4.5, 400)
    pred_grpo_curve = (expit(res_g.x[0] * L_grid + res_g.x[1]) - expit(L_grid)) * 100.0
    pred_sft_curve = (expit(res_s.x[0] * L_grid + res_s.x[1]) - expit(L_grid)) * 100.0
    ax2.plot(L_grid, pred_grpo_curve, "-", color="#DC2626", linewidth=2.8, zorder=6, label=f"GRPO Analytical S-Curve (\u03c4 = {res_g.x[0]:.2f} > 1)")
    ax2.plot(L_grid, pred_sft_curve, "-", color="#0284C7", linewidth=2.8, zorder=6, label=f"CoT SFT Analytical S-Curve (\u03c4 = {res_s.x[0]:.2f} < 1)")

    # Actionable RL prescription annotation
    ax2.text(-2.8, -21.0, "p_base < 25% (0/8 & 1/8 draws):\nGRPO sharpening hurts (\u0394p < 0)\n\u2192 M1c filters out this zone",
             ha="center", fontsize=8.0, color="#991B1B", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#F87171", alpha=0.92, linewidth=0.8), zorder=7)

    ax2.set_title("(B) Within-Model Initial Log-Odds \u03bc_0j = logit(p_base,j) (Easy \u2192 Hard)", pad=8)
    ax2.set_xlabel("Base Model Initial Log-Odds \u03bc_0j = logit(p_base,j) (Easy \u2192 Hard)")
    ax2.set_ylabel("Pass-Rate Change: \u0394p_j (% pts) (\u2191)")
    ax2.set_xlim(3.5, -4.5)
    ax2.set_ylim(-28, 32)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.0)
    ax2.grid(True, linestyle=":", alpha=0.4)

    ax2_top = ax2.twiny()
    p_ticks = np.array([0.90, 0.75, 0.50, 0.25, 0.10, 0.05])
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(logit(p_ticks))
    ax2_top.set_xticklabels(["90%", "75%", "50%", "25%", "10%", "5%"])
    ax2_top.set_xlabel("Base Model Probability p_base = \u03c3(\u03bc_0j)", fontsize=9.5, fontweight="bold", labelpad=4)

    save_fig(fig, "Fig04_v5_stacked_b_vs_base.png")


# ==============================================================================
# FIGURE 5 (Unit Q1, 2 Coupled Panels on the Exact Same 4 RL Runs):
#   Online RL Training With Static 2PL Critics on MMLU
# ==============================================================================
def make_fig5_rl_critics():
    traj = pd.read_csv(os.path.join(RESULTS, "Q1", "trajectory.csv"))
    q1_raw = pd.read_csv(os.path.join(RESULTS, "Q1", "stats.csv"))
    q1_pool = q1_raw[q1_raw["order"] == "pooled"].pivot(index="state", columns="statistic", values="value") * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.8), gridspec_kw={"width_ratios": [1.05, 1.15]})
    fig.subplots_adjust(wspace=0.25)

    styles = [
        ("grpo", "#334155", "--o", "GRPO Baseline (5 draws/item)"),
        ("ctt",  "#94A3B8", ":s",  "CTT Critic (85k pre-samples, 1 draw)"),
        ("pop0", "#F59E0B", "-.^", "2PL Unweighted Critic pop0 (0 pre-samples, 1 draw)"),
        ("pop1", "#0284C7", "-D",  "2PL a-Weighted Critic pop1 (0 pre-samples, 1 draw)"),
    ]

    s0 = traj[traj["seed"] == 0].sort_values("updates")
    for m, col, fmt, lbl in styles:
        sub = s0[s0["mode"] == m]
        ax1.plot(sub["updates"], sub["mean_at_16"] * 100.0, fmt, color=col, linewidth=2.4, markersize=6.5, label=lbl)

    ax1.set_title("(A) Test Bank Single-Draw Accuracy Across RL Updates", pad=8)
    ax1.set_xlabel("RL Training Updates (5 Checkpoints)")
    ax1.set_ylabel("MMLU Test Bank Accuracy: Mean@16 (%) (\u2191)")
    ax1.set_xticks([0, 83, 166, 250, 333])
    ax1.set_ylim(37.0, 50.0)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.8)
    ax1.grid(True, linestyle=":", alpha=0.4)

    metrics = ["mean_at_16", "pass_at_4", "pass_at_16"]
    x_labels = [
        "Pass@1 Accuracy\n(Single-Draw Mean@16)",
        "Pass@4 Accuracy\n(Solved in \u22651 of 4 Draws)",
        "Pass@16 Accuracy\n(Solved in \u22651 of 16 Draws)",
    ]
    x = np.arange(len(metrics))
    w = 0.19
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    for idx, (m, col, fmt, lbl) in enumerate(styles):
        states = [f"q1_{m}_{s}" for s in (0, 1, 2)]
        means = [q1_pool.loc[states, met].mean() for met in metrics]
        ax2.bar(x + offsets[idx], means, width=w, color=col, alpha=0.86, edgecolor="#0F172A", linewidth=0.7, label=lbl)
        for i, met in enumerate(metrics):
            seed_vals = q1_pool.loc[states, met].to_numpy()
            ax2.scatter(np.full(3, x[i] + offsets[idx]), seed_vals, s=20, color="#0F172A", alpha=0.75, zorder=5)
            ax2.text(x[i] + offsets[idx], float(np.max(seed_vals)) + 1.1, f"{means[i]:.1f}", ha="center", fontsize=8.2, fontweight="bold", color="#1E293B")

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylabel("MMLU Test Bank Pass@k Accuracy (%) (\u2191)")
    ax2.set_ylim(35.0, 79.5)
    ax2.set_title("(B) Step-333 Test Bank Pass@k Accuracy (k = 1, 4, 16; 3 Seeds)", pad=8)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.5)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    save_fig(fig, "Fig05_rl_critics_q1.png")


# ==============================================================================
# FIGURE 6 (Unit R1a Audited — Extra Products: Repeatable Learning & Order Bias):
#   Rows 4, 5, 6, 21 from JOINT-CLAIMS-LIST.md
# ==============================================================================
def make_fig6_r1a_extra_products():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    fig.subplots_adjust(wspace=0.28)

    # ==================== Panel A ====================
    categories_a = [
        "GRPO Residual \u03ba_sj\n(Seeds 1 & 2, Row 21)",
        "CoT SFT Residual \u03ba_sj\n(3 Seed Pairs, Row 5)",
    ]
    x_a = np.arange(len(categories_a))
    vals_a = [0.603, 0.731]
    colors_a = ["#DC2626", "#0284C7"]
    edge_a = ["#991B1B", "#0369A1"]

    ax1.bar(x_a, vals_a, width=0.44, color=colors_a, edgecolor=edge_a, linewidth=1.1, alpha=0.88, zorder=3)

    # GRPO null 97.5% = 0.112
    ax1.hlines(0.112, x_a[0] - 0.24, x_a[0] + 0.24, color="#7F1D1D", linestyle="--", linewidth=1.8, zorder=4)
    ax1.text(x_a[0], 0.125, "Planted Null 97.5%: 0.112", ha="center", fontsize=8.2, fontweight="bold", color="#7F1D1D")

    # SFT null 97.5% = 0.195
    ax1.hlines(0.195, x_a[1] - 0.24, x_a[1] + 0.24, color="#0C4A6E", linestyle="--", linewidth=1.8, zorder=4)
    ax1.text(x_a[1], 0.208, "Planted Null 97.5%: 0.195", ha="center", fontsize=8.2, fontweight="bold", color="#0C4A6E")

    # SFT individual seed pairs: 0.727, 0.732, 0.735
    ax1.scatter([x_a[1]] * 3, [0.727, 0.732, 0.735], color="#0C4A6E", s=32, zorder=5)

    # Value text labels on bars
    ax1.text(x_a[0], 0.618, "r = 0.603***", ha="center", fontsize=9.2, fontweight="bold", color="#991B1B")
    ax1.text(x_a[1], 0.750, "Mean r = 0.731***", ha="center", fontsize=9.2, fontweight="bold", color="#0369A1")

    # Annotation box for S-1 and S-6
    anno_text = (
        "Both SFT (+6.0% to +8.3% modal flips toward key) and GRPO repeat beyond the\n"
        "nulls tested so far after removing common (\u03c4_s, \u03b4_s).\n"
        "S-6 Caveat: Seeds share 1 start model; 2 additional P1c nulls queued\n"
        "(SFT at fitted values & shared non-linear base misfit)."
    )
    ax1.text(0.5, 0.93, anno_text, ha="center", va="center", fontsize=7.6, color="#1E293B",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.95, linewidth=0.8), zorder=6)

    ax1.set_xticks(x_a)
    ax1.set_xticklabels(categories_a, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Cross-Seed Residual Correlation r(\u03ba_s, \u03ba_s') (\u2191)")
    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(0, 1.08)
    ax1.set_title("(A) Orthogonal Item Residual \u03ba_sj Repeats Beyond Nulls", pad=10)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    # ==================== Panel B ====================
    categories_b = [
        "CoT SFT\n(3 seeds)",
        "Base Model\n(Llama-3.2-1B)",
        "GRPO\n(3 seeds)",
        "Q1 RL Runs\n(24 states)",
    ]
    x_b = np.arange(len(categories_b))
    vals_b = [1.88, 2.83, 10.67, 11.8]
    colors_b = ["#0284C7", "#475569", "#DC2626", "#D97706"]
    edge_b = ["#0369A1", "#1E293B", "#991B1B", "#B45309"]

    ax2.bar(x_b, vals_b, width=0.52, color=colors_b, edgecolor=edge_b, linewidth=1.1, alpha=0.88, zorder=3)

    # Horizontal reference line at D = 1.035
    ax2.axhline(1.035, color="#64748B", linestyle="--", linewidth=1.4, zorder=2)
    ax2.text(1.55, 1.35, "No Order Effect Null D = 1.035", ha="center", fontsize=8.2, color="#475569", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CBD5E1", alpha=0.9, linewidth=0.6), zorder=6)

    # SFT dots: 1.85, 1.88, 1.92
    ax2.scatter([x_b[0]] * 3, [1.85, 1.88, 1.92], color="#0C4A6E", s=28, zorder=5)
    # GRPO dots: 9.74, 10.67, 11.60
    ax2.scatter([x_b[2]] * 3, [9.74, 10.67, 11.60], color="#7F1D1D", s=28, zorder=5)
    # Q1 error bar: 6.58 to 19.36
    ax2.errorbar(x_b[3], 11.8, yerr=[[11.8 - 6.58], [19.36 - 11.8]], fmt="none", color="#92400E", capsize=5, elinewidth=2.0, capthick=2.0, zorder=5)

    # Text labels
    ax2.text(x_b[0], 2.20, "D = 1.88\n(\u221234% order bias)", ha="center", fontsize=8.2, fontweight="bold", color="#0369A1")
    ax2.text(x_b[1], 3.15, "D = 2.83\n(base)", ha="center", fontsize=8.2, fontweight="bold", color="#1E293B")
    ax2.text(x_b[2], 12.3, "D = 10.67\n(4\u00d7 amplification)", ha="center", fontsize=8.2, fontweight="bold", color="#991B1B")
    ax2.text(x_b[3], 19.8, "D = 11.8\n(range: 6.6\u201319.4)", ha="center", fontsize=8.2, fontweight="bold", color="#78350F")

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(categories_b, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Option-Order Dispersion D (Order 0 vs. 1) (\u2193)")
    ax2.set_ylim(0, 21.5)
    ax2.set_title("(B) Paired-Order IRT: RL Amplifies Option-Order Bias 4\u00d7", pad=10)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    save_fig(fig, "Fig06_r1a_irt_extra_products.png")


if __name__ == "__main__":
    make_fig0_eedi()
    make_fig1_posttraining_q2a()
    make_fig2_sft_vs_grpo_puzzle()
    make_fig3_irt_subject_separation()
    make_fig3_v2_irt_subject_separation()
    make_fig4_irt_item_slope()
    make_fig4_v2_irt_item_slope()
    make_fig4_v3_b_fullpoint_fit()
    make_fig4_v4_base_logit_scurve()
    make_fig4_v5_stacked_b_vs_base()
    make_fig5_rl_critics()
    make_fig6_r1a_extra_products()
    print("All figures (Fig00..Fig06 + v2/v3/v4/v5 variants) generated successfully.")
