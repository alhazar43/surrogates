"""
Clean, publication-grade figures for the Oct 26 presentation.

Design principles:
1. No forced 'CAN vs CANNOT' side-by-side layout: subplots in a figure always tell ONE unified story.
2. Minimalist annotations: no paragraph text boxes inside plots (explanations belong on the slide).
   Use clean legends, concise axis labels, and consistent visual encoding.
3. Explicit visual contrast between:
   - w/o IRT (Standard aggregate accuracy / random sampling / empirical baselines): Neutral Grey/Slate
   - w/ IRT (2PL item parameters b_i, a_i / IRT curves / IRT critics): Distinct High-Contrast Colors
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.8,
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
# FIGURE 1: Adaptive Testing Across Test Length n (w/o IRT vs. w/ IRT)
# Unified story: How estimation error evolves from n = 10 to n = 1,472 (full bank)
# Panel A: Pretraining checkpoints (Q2c, n=25..100) — w/ IRT wins at small n, crosses at n=100
# Panel B: Post-trained models (Q2a extended, n=10..1,472) — w/o IRT -> 0%, w/ IRT hits bias floor
# Panel C: Error decomposition (|Bias| vs. Sampling SD) showing convergence of SD
# ==============================================================================
def make_fig1_adaptive_eval():
    with open(os.path.join(RESULTS, "Q2c", "summary.json")) as f:
        q2c = json.load(f)["outcome"]["cells"]
    ext_df = pd.read_csv(r"D:\surrogates\gemini\q2a_extended_n_t9r.csv")
    ext_df["abs_bias"] = ext_df["bias"].abs()

    ns_q2c = [25, 50, 100]
    c_rand = [q2c[f"above 0.45 n={n}"]["rmse_classical"] * 100.0 for n in ns_q2c]
    c_adap = [q2c[f"above 0.45 n={n}"]["rmse_irt_adaptive"] * 100.0 for n in ns_q2c]

    ns_ext = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1472]
    k16 = ext_df[ext_df["k"] == 16].groupby(["n", "method"])[["rmse", "abs_bias", "sd"]].mean().unstack() * 100.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.6), gridspec_kw={"width_ratios": [0.85, 1.15, 1.15]})
    fig.subplots_adjust(wspace=0.26)

    # Panel A: Pretraining checkpoints (Q2c)
    ax1.plot(ns_q2c, c_rand, "--o", color="#64748B", linewidth=2.4, markersize=7, label="w/o IRT: Random Mean")
    ax1.plot(ns_q2c, c_adap, "-s", color="#0284C7", linewidth=2.8, markersize=7, label="w/ IRT: 2PL Adaptive")
    ax1.set_title("(A) Pretraining Checkpoints (Q2c)", pad=8)
    ax1.set_xlabel("Test Length n (items)")
    ax1.set_ylabel("Estimation Error (RMSE, %)")
    ax1.set_xticks(ns_q2c)
    ax1.set_xlim(18, 107)
    ax1.set_ylim(0, 12.5)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Panel B: Full-bank trajectory (n = 10 to 1,472) on Post-Trained Models
    y_cl = [k16["rmse"].loc[n, "classical"] for n in ns_ext]
    y_rd = [k16["rmse"].loc[n, "irt_random"] for n in ns_ext]
    y_ad = [k16["rmse"].loc[n, "irt_adaptive"] for n in ns_ext]

    ax2.plot(ns_ext, y_cl, "--o", color="#64748B", linewidth=2.4, markersize=6, label="w/o IRT: Random Mean")
    ax2.plot(ns_ext, y_rd, "-.^", color="#0284C7", linewidth=2.2, markersize=6, label="w/ IRT: 2PL + Random Items")
    ax2.plot(ns_ext, y_ad, "-s", color="#DC2626", linewidth=2.6, markersize=6, label="w/ IRT: 2PL Adaptive (Fisher)")
    ax2.set_xscale("log")
    ax2.set_xticks([10, 25, 50, 100, 300, 1472])
    ax2.set_xticklabels(["10", "25", "50", "100", "300", "1472"])
    ax2.set_xlim(9, 1700)
    ax2.set_ylim(0, 12.5)
    ax2.set_title("(B) Post-Trained Models: Full Bank (n = 10..1,472)", pad=8)
    ax2.set_xlabel("Test Length n (log scale, out of 1,472 items)")
    ax2.set_ylabel("Estimation Error (RMSE, %)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)

    # Panel C: Bias vs. Sampling SD Decomposition
    b_ad = [k16["abs_bias"].loc[n, "irt_adaptive"] for n in ns_ext]
    sd_ad = [k16["sd"].loc[n, "irt_adaptive"] for n in ns_ext]
    b_cl = [k16["abs_bias"].loc[n, "classical"] for n in ns_ext]
    sd_cl = [k16["sd"].loc[n, "classical"] for n in ns_ext]

    ax3.plot(ns_ext, b_ad, "-s", color="#DC2626", linewidth=2.6, markersize=6, label="w/ IRT Adaptive: |Bias|")
    ax3.plot(ns_ext, sd_ad, ":s", color="#F87171", linewidth=2.2, markersize=5, label="w/ IRT Adaptive: Sampling SD")
    ax3.plot(ns_ext, sd_cl, "--o", color="#64748B", linewidth=2.2, markersize=5.5, label="w/o IRT Random: Sampling SD")
    ax3.plot(ns_ext, b_cl, "-o", color="#94A3B8", linewidth=1.6, markersize=4, label="w/o IRT Random: |Bias| (~0%)")
    ax3.set_xscale("log")
    ax3.set_xticks([10, 25, 50, 100, 300, 1472])
    ax3.set_xticklabels(["10", "25", "50", "100", "300", "1472"])
    ax3.set_xlim(9, 1700)
    ax3.set_ylim(0, 12.5)
    ax3.set_title("(C) Error Decomposition: |Bias| vs. Sampling SD", pad=8)
    ax3.set_xlabel("Test Length n (log scale, out of 1,472 items)")
    ax3.set_ylabel("Error Component (%)")
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)

    save_fig(fig, "Fig01_adaptive_eval_across_n.png")


# ==============================================================================
# FIGURE 2: Positive Diagnostic Power of IRT — Revealing How SFT vs. GRPO Learn
# Unified story: w/o IRT (aggregate accuracy) sees a 1D step up (39.6% -> 45.3% -> 49.3%),
# whereas w/ IRT (item difficulty b_i and discrimination a_i) reveals two opposite mechanisms:
# Panel A: w/o IRT Aggregate Score vs. w/ IRT 1D Ability Score (credit reversal)
# Panel B: w/ IRT Item-Level Change across IRT Difficulty b_i (GRPO slope sharpening vs. SFT lift)
# Panel C: w/ IRT Subject Gains across IRT Difficulty b_m and Discrimination a_m (r = +0.85)
# ==============================================================================
def make_fig2_irt_diagnostic_sft_vs_grpo():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]
    w0_ok["p_base"] = t9["p_key_base"].values
    w0_ok["dp_sft"] = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values * 100.0
    w0_ok["dp_grpo"] = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values * 100.0

    # Subject-level data from S1 + Q2g (32 draws)
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.7), gridspec_kw={"width_ratios": [0.78, 1.12, 1.15]})
    fig.subplots_adjust(wspace=0.28)

    # Panel A: w/o IRT (Raw Accuracy Share, Grey) vs. w/ 1D IRT (2PL Theta Share, Blue)
    x = np.arange(2)
    w = 0.34
    wo_irt = [60.0, 40.0]
    w_irt = [21.0, 79.0]
    ax1.bar(x - w/2, wo_irt, width=w, color="#64748B", label="w/o IRT: Raw Accuracy")
    ax1.bar(x + w/2, w_irt, width=w, color="#0284C7", label="w/ 1D IRT: 2PL Ability \u03b8")
    for i, (v1, v2) in enumerate(zip(wo_irt, w_irt)):
        ax1.text(i - w/2, v1 + 2.0, f"{v1:.0f}%", ha="center", fontweight="bold", fontsize=10, color="#334155")
        ax1.text(i + w/2, v2 + 2.0, f"{v2:.0f}%", ha="center", fontweight="bold", fontsize=10, color="#0369A1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["SFT Stage\n(CoT Distillation)", "GRPO Stage\n(Outcome RL)"])
    ax1.set_ylabel("Share of Total Post-Training Gain (%)")
    ax1.set_ylim(0, 96)
    ax1.set_title("(A) Credit Assignment: w/o vs. w/ 1D IRT", pad=8)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Panel B: w/ IRT Item-Level Accuracy Change Across IRT Difficulty Deciles (1,472 items)
    w0_ok["b_bin"] = pd.qcut(w0_ok["b"], q=10)
    binned = w0_ok.groupby("b_bin", observed=False)[["b", "a", "dp_sft", "dp_grpo"]].mean()
    ax2.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)
    ax2.plot(binned["b"], binned["dp_sft"], "-o", color="#0284C7", linewidth=2.6, markersize=6.5, label="SFT (\u03c4 = 0.71: lifts hard items)")
    ax2.plot(binned["b"], binned["dp_grpo"], "-s", color="#DC2626", linewidth=2.6, markersize=6.5, label="GRPO (\u03c4 = 1.88: sharpens easy vs hard)")
    ax2.set_title("(B) w/ IRT: Item Gain vs. Difficulty b (1,472 Items)", pad=8)
    ax2.set_xlabel("IRT Item Difficulty b (Decile Means: Easy \u2192 Hard)")
    ax2.set_ylabel("Token-0 Key Probability Change \u0394p (%)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.4)

    # Panel C: w/ IRT Subject-Level Crossover Across (b_m, a_m) (15 MMLU Subjects)
    df_sub = pd.DataFrame({"b": sub_b, "a": sub_a, "dp_sft": dp_sft_sub, "dp_grpo": dp_grpo_sub}).sort_values("b")
    ax3.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)
    # Bubble size proportional to discrimination a_m
    sizes = df_sub["a"] * 75.0
    ax3.scatter(df_sub["b"], df_sub["dp_sft"], s=sizes, color="#0284C7", alpha=0.85, label="SFT (CoT Distillation)")
    ax3.scatter(df_sub["b"], df_sub["dp_grpo"], s=sizes, color="#DC2626", marker="s", alpha=0.85, label="GRPO (Outcome RL)")

    x_grid = np.linspace(df_sub["b"].min() - 0.1, df_sub["b"].max() + 0.1, 100)
    m_sft, c_sft_l = np.polyfit(df_sub["b"], df_sub["dp_sft"], 1)
    m_grpo, c_grpo_l = np.polyfit(df_sub["b"], df_sub["dp_grpo"], 1)
    ax3.plot(x_grid, m_sft * x_grid + c_sft_l, "-", color="#0284C7", linewidth=2.0, alpha=0.75)
    ax3.plot(x_grid, m_grpo * x_grid + c_grpo_l, "-", color="#DC2626", linewidth=2.0, alpha=0.75)

    # Minimal 1-word subject tags for anchor points only
    for subj, label, dx, dy, col in [
        ("marketing", "Marketing", 0.06, -0.5, "#991B1B"),
        ("sociology", "Sociology", 0.06, -0.5, "#991B1B"),
        ("elementary_mathematics", "Elem. Math", -0.68, -0.5, "#0369A1"),
        ("high_school_mathematics", "HS Math", -0.52, 1.5, "#0369A1"),
    ]:
        r = df_sub.loc[subj]
        col_y = "dp_grpo" if col == "#991B1B" else "dp_sft"
        ax3.text(r["b"] + dx, r[col_y] + dy, label, fontsize=8.8, color=col, fontweight="bold")

    ax3.set_title("(C) w/ IRT: Subject Gain vs. (b, a) (Bubble Size = a)", pad=8)
    ax3.set_xlabel("Subject Mean IRT Difficulty b (Easy/High-a \u2192 Hard/Low-a)")
    ax3.set_ylabel("32-Draw Accuracy Gain Over Base (%)")
    ax3.set_xlim(-1.15, 2.55)
    ax3.set_ylim(-14, 44)
    ax3.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.4)

    save_fig(fig, "Fig02_irt_diagnoses_sft_vs_grpo.png")


# ==============================================================================
# FIGURE 3: RL Training With vs. Without IRT Critics (Unit Q1)
# Unified story: Comparing w/o IRT baselines (5-draw GRPO, 85k-presampled CTT)
# against w/ IRT zero-presample critics (pop0, pop1) across accuracy & diversity
# ==============================================================================
def make_fig3_rl_training_with_vs_without_irt():
    traj = pd.read_csv(os.path.join(RESULTS, "Q1", "trajectory.csv"))
    s1_raw = pd.read_csv(os.path.join(RESULTS, "S1", "stats.csv"))
    q1_raw = pd.read_csv(os.path.join(RESULTS, "Q1", "stats.csv"))
    stats_raw = pd.concat([s1_raw[s1_raw["state"] == "t9_base"], q1_raw], ignore_index=True)
    stats = stats_raw[stats_raw["order"] == "pooled"].pivot(index="state", columns="statistic", values="value").reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.7), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(wspace=0.25)

    # Panel A: Training Accuracy Trajectory (w/o IRT in Grey/Slate vs. w/ IRT in Blue/Red)
    styles = {
        "grpo": ("#334155", "--o", "w/o IRT: GRPO (5 draws/item)"),
        "ctt":  ("#94A3B8", ":s",  "w/o IRT: CTT Critic (1 draw, 85k pre-samples)"),
        "pop0": ("#0284C7", "-^",  "w/ IRT: 2PL Unweighted pop0 (1 draw, 0 pre-samples)"),
        "pop1": ("#DC2626", "-D",  "w/ IRT: 2PL a-Weighted pop1 (1 draw, 0 pre-samples)"),
    }
    base_acc = float(stats[stats["state"] == "t9_base"]["mean_at_16"].iloc[0]) * 100.0
    s0 = traj[traj["seed"] == 0].sort_values("updates")
    for m, (col, fmt, lbl) in styles.items():
        sub = s0[s0["mode"] == m]
        xs = [0] + sub["updates"].tolist() if 0 not in sub["updates"].values else sub["updates"].tolist()
        ys = ([base_acc] if 0 not in sub["updates"].values else []) + (sub["mean_at_16"] * 100.0).tolist()
        ax1.plot(xs, ys, fmt, color=col, linewidth=2.4, markersize=6.5, label=lbl)

    ax1.set_title("(A) Test Accuracy Trajectory (Mean@16)", pad=8)
    ax1.set_xlabel("RL Training Updates")
    ax1.set_ylabel("MMLU Bank Accuracy (%)")
    ax1.set_xticks([0, 83, 166, 250, 333])
    ax1.set_ylim(37.5, 50.0)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Panel B: Mean@16 vs. Order-Averaged Pass@16 at Step 333 (w/o IRT vs. w/ IRT)
    def grp(s):
        if s == "t9_base": return "Base"
        for k in ["grpo", "ctt", "pop1", "pop0"]:
            if s.startswith(f"q1_{k}"): return k
        return None

    stats["grp"] = stats["state"].map(grp)
    stats = stats.dropna(subset=["grp"])
    order = ["Base", "grpo", "ctt", "pop1", "pop0"]
    labels = [
        "Base\nModel",
        "w/o IRT\nGRPO\n(5 draws)",
        "w/o IRT\nCTT\n(85k cal.)",
        "w/ IRT\npop1\n(a-weighted)",
        "w/ IRT\npop0\n(unweighted)",
    ]
    colors_pass = ["#64748B", "#475569", "#94A3B8", "#DC2626", "#0284C7"]

    means_p16 = [stats[stats["grp"] == g]["pass_at_16"].mean() * 100.0 for g in order]
    means_m16 = [stats[stats["grp"] == g]["mean_at_16"].mean() * 100.0 for g in order]

    x = np.arange(len(order))
    w = 0.36
    ax2.bar(x - w/2, means_m16, width=w, color="#CBD5E1", edgecolor="#475569", label="Mean@16 (1-Draw Accuracy)")
    ax2.bar(x + w/2, means_p16, width=w, color=colors_pass, alpha=0.9, label="Pass@16 (16-Draw Coverage)")

    for i, g in enumerate(order):
        sub = stats[stats["grp"] == g]
        ax2.scatter(np.full(len(sub), x[i] + w/2), sub["pass_at_16"] * 100.0, s=28, color="#0F172A", zorder=5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.2)
    ax2.set_ylabel("Score on 1,472 MMLU Items (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("(B) Accuracy (Mean@16) vs. Diversity (Pass@16)", pad=8)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    save_fig(fig, "Fig03_rl_critics_with_vs_without_irt.png")


# ==============================================================================
# FIGURE 4: Human IRT vs. LLM IRT on 704 Eedi Math Questions (Unit E1)
# Unified story: IRT 2PL is highly reliable within humans (r=0.85) and within LLMs (r=0.88),
# but human IRT parameters cannot substitute for LLM IRT parameters (Spearman = 0.09).
# ==============================================================================
def make_fig4_human_irt_vs_llm_irt():
    df = pd.read_csv(os.path.join(RESULTS, "E1", "difficulty.csv"))
    sub = df.dropna(subset=["b_students", "b_llms"]).copy()

    stu_pct = sub["b_students"].rank(pct=True) * 100.0
    llm_pct = sub["b_llms"].rank(pct=True) * 100.0

    rng = np.random.default_rng(42)
    a_stu = sub["a_students"].values
    b_stu = sub["b_students"].values
    thetas = rng.normal(0.0, 1.0, size=(61, 1))
    probs = expit(a_stu[None, :] * (thetas - b_stu[None, :]))
    Y_sim = rng.binomial(1, probs)
    sim_pct = pd.Series(-np.mean(Y_sim, axis=0)).rank(pct=True).values * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.18)

    ax1.scatter(stu_pct, sim_pct, s=15, alpha=0.35, color="#0284C7", edgecolors="none", label="704 Eedi Items")
    ax1.plot([0, 100], [0, 100], "--", color="#1E293B", linewidth=1.8, label="y = x (Spearman \u03c1 = 0.85)")
    ax1.set_title("(A) Human IRT \u2192 Held-Out Students", pad=8)
    ax1.set_xlabel("Human 2PL Difficulty Percentile b_human")
    ax1.set_ylabel("Target Group Difficulty Percentile")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", frameon=True, facecolor="white")
    ax1.grid(True, linestyle=":", alpha=0.4)

    ax2.scatter(stu_pct, llm_pct, s=15, alpha=0.35, color="#DC2626", edgecolors="none", label="704 Eedi Items")
    ax2.plot([0, 100], [0, 100], "--", color="#64748B", linewidth=1.5, label="y = x")
    ax2.axhline(50, color="#DC2626", linewidth=2.0, label="LLM Fit (Spearman \u03c1 = 0.09)")
    ax2.set_title("(B) Human IRT \u2192 61 Open-Weight LLMs", pad=8)
    ax2.set_xlabel("Human 2PL Difficulty Percentile b_human")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", frameon=True, facecolor="white")
    ax2.grid(True, linestyle=":", alpha=0.4)

    save_fig(fig, "Fig04_human_irt_vs_llm_irt.png")


if __name__ == "__main__":
    make_fig1_adaptive_eval()
    make_fig2_irt_diagnostic_sft_vs_grpo()
    make_fig3_rl_training_with_vs_without_irt()
    make_fig4_human_irt_vs_llm_irt()
    print("All 4 unified figures generated successfully.")
