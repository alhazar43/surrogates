"""
make_oct26_v5_talk_figures.py

Generates the 7 science-first empirical presentation figures (talk_Fig01 through talk_Fig07 + talk_FigB2)
for the Oct 26 presentation into both:
  - D:\\surrogates\\gemini\\slides_oct26\\figures\\
  - C:\\Users\\steph\\.gemini\\antigravity-cli\\brain\\d00b65ad-c24f-4025-8dd4-1c0acb407e49\\

Strict Design Principles:
  - 300 DPI, Segoe UI / sans-serif, clean presentation layout.
  - ZERO bbox multi-line text cards inside data charts.
  - ZERO internal code names (no Q2a, Q1, W0, W1, T9r, S1, R6, M1, Unit P, pop0, pop1, ctt, JEDM).
  - Clean continuous curves, horizontal lollipops, 2D phase planes, and surgical direct labels.
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Ellipse

# Global Presentation Typography & Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.titlesize": 12.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.0,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.8,
    "ytick.labelsize": 9.8,
    "legend.fontsize": 9.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO_ROOT = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\item-params-llm"
SIG_ROOT  = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\plms-recent\signature"
RESULTS   = os.path.join(REPO_ROOT, "results")
SIG_RESULTS = os.path.join(SIG_ROOT, "results")

OUT_DIRS = [
    r"D:\surrogates\gemini\slides_oct26\figures",
    r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49",
]

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save_fig(fig, filename):
    first_path = os.path.join(OUT_DIRS[0], filename)
    fig.savefig(first_path, bbox_inches="tight", facecolor="white")
    for d in OUT_DIRS[1:]:
        shutil.copy2(first_path, os.path.join(d, filename))
    plt.close(fig)
    print(f"[OK] Saved: {filename}")


# ==============================================================================
# FIGURE 1: Standalone 1x1 — Pass@k Puzzle and Modal Ceiling
# ==============================================================================
def make_talk_fig01_puzzle_and_modal_ceiling():
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    p0 = np.clip(t9["p_key_base"].values, 1e-4, 1 - 1e-4)
    dp_g = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values
    dp_s = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values
    p_g = np.clip(p0 + dp_g, 1e-4, 1 - 1e-4)
    p_s = np.clip(p0 + dp_s, 1e-4, 1 - 1e-4)

    cal_p0 = expit(0.927 * logit(p0) - 0.292)
    cal_pg = expit(3.292 * logit(p_g) + 1.018)
    cal_ps = expit(1.501 * logit(p_s) + 0.754)

    k_dense = np.linspace(1, 16, 300)
    x_log2 = np.log2(k_dense)

    curve_base = (1.0 - np.mean((1.0 - cal_p0[:, None]) ** k_dense[None, :], axis=0)) * 100.0
    curve_grpo = (1.0 - np.mean((1.0 - cal_pg[:, None]) ** k_dense[None, :], axis=0)) * 100.0
    curve_sft  = (1.0 - np.mean((1.0 - cal_ps[:, None]) ** k_dense[None, :], axis=0)) * 100.0

    idx_cross = np.where(curve_sft > curve_grpo)[0][0]
    x_cross = x_log2[idx_cross]
    y_cross = curve_sft[idx_cross]

    # Scissor Shading
    ax.fill_between(
        x_log2[:idx_cross + 1], curve_sft[:idx_cross + 1], curve_grpo[:idx_cross + 1],
        color="#FEE2E2", alpha=0.45, zorder=1
    )
    ax.fill_between(
        x_log2[idx_cross:], curve_grpo[idx_cross:], curve_sft[idx_cross:],
        color="#E0F2FE", alpha=0.45, zorder=1
    )
    ax.axvline(x_cross, color="#64748B", linestyle=":", linewidth=1.4, alpha=0.75, zorder=2)

    # Reference Modal Ceilings across the plot
    ax.plot([0.0, 4.0], [47.6, 47.6], linestyle="--", color="#991B1B", linewidth=1.5, alpha=0.80, zorder=2)
    ax.plot([1.4, 4.0], [54.6, 54.6], linestyle="--", color="#0284C7", linewidth=1.5, alpha=0.80, zorder=2)

    # Direct Modal Ceiling Labels in the completely open bottom-right space (x in [2.1, 3.9])
    ax.text(2.15, 51.8, "CoT-SFT Majority Vote (Maj@16 = 54.6%)",
            fontsize=8.8, fontweight="bold", color="#0369A1", zorder=7)
    ax.text(2.15, 44.5, "Base & GRPO Majority Vote (Maj@16 = 47.6% vs. 47.5%)",
            fontsize=8.8, fontweight="bold", color="#991B1B", zorder=7)

    # Continuous Curves
    ax.plot(
        x_log2, curve_sft, "-", color="#0284C7", linewidth=3.0, zorder=4,
        label="Supervised Fine-Tuning (CoT-SFT): 43.9% → 96.6%"
    )
    ax.plot(
        x_log2, curve_base, "--", color="#475569", linewidth=2.3, zorder=3,
        label="Base Model (Llama-3.2-1B): 38.2% → 93.0%"
    )
    ax.plot(
        x_log2, curve_grpo, "-", color="#DC2626", linewidth=3.0, zorder=4,
        label="Reinforcement Learning (GRPO): 47.9% → 65.6%"
    )

    # Empirical 3-Seed Dots at k = 1, 4, 16
    ax.scatter([0.0] * 3, [47.785, 48.644, 47.262], color="#991B1B", marker="s", s=32, zorder=5)
    ax.scatter([2.0] * 3, [56.64, 57.52, 57.36], color="#991B1B", marker="s", s=32, zorder=5)
    ax.scatter([4.0] * 3, [64.87, 65.97, 66.07], color="#991B1B", marker="s", s=32, zorder=5)

    ax.scatter([0.0] * 3, [44.41, 43.38, 43.88], color="#0369A1", marker="o", s=32, zorder=5)
    ax.scatter([2.0] * 3, [78.82, 77.88, 77.72], color="#0369A1", marker="o", s=32, zorder=5)
    ax.scatter([4.0] * 3, [97.10, 96.87, 95.97], color="#0369A1", marker="o", s=32, zorder=5)

    ax.scatter([0.0, 2.0, 4.0], [38.18, 70.20, 93.00], color="#475569", marker="o", s=36, zorder=5)

    # Crossover Point
    ax.scatter([x_cross], [y_cross], s=95, color="#FDE047", edgecolors="#0F172A", linewidth=2.0, zorder=6)

    # Clean Direct Annotations
    ax.annotate(
        "Crossover (k ≈ 1.3)",
        xy=(x_cross, y_cross), xytext=(x_cross + 0.35, y_cross - 7.2),
        arrowprops=dict(arrowstyle="-|>", color="#0F172A", lw=1.2),
        fontsize=9.0, fontweight="bold", color="#0F172A", zorder=7
    )
    ax.annotate(
        "RL Wins Pass@1 (+4.0%)",
        xy=(0.0, 48.8), xytext=(-0.14, 66.0),
        arrowprops=dict(arrowstyle="-|>", color="#991B1B", lw=1.2),
        ha="left", fontsize=9.0, fontweight="bold", color="#991B1B", zorder=7
    )

    # Endpoint Labels at k = 16
    ax.text(4.08, 96.6, "96.6% (SFT)", va="center", ha="left", fontsize=9.2, fontweight="bold", color="#0369A1")
    ax.text(4.08, 92.5, "93.0% (Base)", va="center", ha="left", fontsize=9.0, fontweight="bold", color="#475569")
    ax.text(4.08, 65.6, "65.6% (RL)", va="center", ha="left", fontsize=9.2, fontweight="bold", color="#991B1B")

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["k = 1\n(Pass@1)", "k = 2", "k = 4\n(Pass@4)", "k = 8", "k = 16\n(Pass@16)"])
    ax.set_xlim(-0.22, 4.78)
    ax.set_ylim(28.0, 103.0)
    ax.set_title("Single-Draw Accuracy (Pass@1) vs. Multi-Draw Coverage (Pass@k)", pad=12)
    ax.set_xlabel("Number of Test-Time Rollout Attempts k (log₂ scale)")
    ax.set_ylabel("MMLU Test Accuracy: Pass@k (%)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.8)
    ax.grid(True, linestyle=":", alpha=0.45)

    save_fig(fig, "talk_Fig01_puzzle_and_modal_ceiling.png")


# ==============================================================================
# FIGURE 2: 1x2 — Adaptive CAT Evaluation Bias Floor & 2D Discrimination Bias
# ==============================================================================
def make_talk_fig02_eval_cat_and_2d_domains():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.8, 5.3), gridspec_kw={"width_ratios": [1.0, 1.02], "wspace": 0.25}
    )

    # -------------------------------------------------------------------------
    # Panel A: Adaptive Evaluation (CAT) Hits a 2.4% Bias Floor
    # -------------------------------------------------------------------------
    ext_df = pd.read_csv(os.path.join(RESULTS, "Q2a", "errors_ext.csv"))
    ext_df = ext_df[ext_df["target"] == "state"].copy()
    ns_ext = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1472]
    k16 = ext_df[ext_df["k"] == 16].groupby(["n", "method"])["rmse"].mean().unstack() * 100.0

    y_cl = [k16.loc[n, "classical"] for n in ns_ext]
    y_rd = [k16.loc[n, "irt_random"] for n in ns_ext]
    y_ad = [k16.loc[n, "irt_adaptive"] for n in ns_ext]

    ax1.axhspan(0, y_ad[-1], color="#FEE2E2", alpha=0.38, zorder=1)
    ax1.axhline(y_ad[-1], color="#DC2626", linestyle="--", linewidth=1.8, alpha=0.9, zorder=2)

    ax1.plot(
        ns_ext, y_cl, "--o", color="#1E293B", linewidth=2.5, markersize=6.5, zorder=4,
        label="Simple Random Average (1/√n → 0.8% error)"
    )
    ax1.plot(
        ns_ext, y_rd, ":^", color="#D97706", linewidth=2.4, markersize=6.5, zorder=4,
        label="Population IRT Scoring (Random Items → 2.4% floor)"
    )
    ax1.plot(
        ns_ext, y_ad, "-s", color="#DC2626", linewidth=2.8, markersize=6.8, zorder=5,
        label="Population IRT Adaptive Selection (CAT → 2.4% floor)"
    )

    ax1.text(12.0, 1.55, "2.4% Static IRT Bias Floor", fontsize=9.0, fontweight="bold", color="#991B1B", zorder=6)
    ax1.annotate(
        "0.8% Simple Average",
        xy=(1472, y_cl[-1]), xytext=(420, 4.3),
        arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=1.2, connectionstyle="arc3,rad=-0.15"),
        fontsize=8.8, fontweight="bold", color="#1E293B", zorder=6
    )

    ax1.set_xscale("log")
    ax1.set_xticks([10, 25, 50, 100, 300, 750, 1472])
    ax1.set_xticklabels(["10", "25", "50", "100", "300", "750", "1472\n(Full Bank)"])
    ax1.set_xlim(9, 1750)
    ax1.set_ylim(0, 16.8)
    ax1.set_title("(A) Adaptive Testing (CAT) vs. Random Sampling", pad=11)
    ax1.set_xlabel("Number of Sampled Test Items n (log scale)")
    ax1.set_ylabel("Estimation Error: RMSE (%)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.8)
    ax1.grid(True, which="both", linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Why CAT & 1D θ Fail: 5× Discrimination Bias in 2D (a, b)
    # -------------------------------------------------------------------------
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    it10 = pd.read_csv(os.path.join(SIG_RESULTS, "T10", "runs", "items.csv"))
    flags = pd.read_csv(os.path.join(RESULTS, "Q2g", "s1", "flags.csv"))
    pf = pd.read_csv(os.path.join(RESULTS, "Q2g", "s1", "personfit.csv")).iloc[:7]

    th_base = pf[pf["method"] == "base"]["theta"].mean()
    th_sft  = pf[pf["method"] == "sft"]["theta"].mean()
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

    # Translucent 1,472 background items
    it10_idx = it10.set_index("item_id").loc[w0_ok["bank_row_id"]]
    p_base_it = np.clip(it10_idx["p_ref"].values, 1e-4, 1 - 1e-4)
    L_base_it = logit(p_base_it)
    w0_ok["c_sft"] = w0_ok["subject"].map(c_sft.to_dict())
    w0_ok["c_grpo"] = w0_ok["subject"].map(c_grpo.to_dict())
    dp_sft_item = (expit(L_base_it + w0_ok["c_sft"].values + w0_ok["a"].values * (th_sft - th_base)) - p_base_it) * 100.0
    dp_grpo_item = (expit(L_base_it + w0_ok["c_grpo"].values + w0_ok["a"].values * (th_grpo - th_base)) - p_base_it) * 100.0

    ax2.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)
    ax2.scatter(w0_ok["b"], dp_sft_item, s=w0_ok["a"] * 15.0, color="#0284C7", alpha=0.09, edgecolors="none", zorder=2)
    ax2.scatter(w0_ok["b"], dp_grpo_item, s=w0_ok["a"] * 15.0, color="#DC2626", alpha=0.09, edgecolors="none", zorder=2)

    sizes = df_sub["a"] * 105.0
    ax2.scatter(
        df_sub["b"], df_sub["dp_sft"], s=sizes, color="#0284C7", alpha=0.88,
        edgecolors="#0369A1", linewidth=1.1, zorder=4,
        label=r"CoT-SFT (Wins hard, low-$a$ math/STEM subjects)"
    )
    ax2.scatter(
        df_sub["b"], df_sub["dp_grpo"], s=sizes, color="#DC2626", marker="s", alpha=0.88,
        edgecolors="#991B1B", linewidth=1.1, zorder=4,
        label=r"RL / GRPO (Wins easy, high-$a$ factual subjects)"
    )

    x_grid = np.linspace(df_sub["b"].min() - 0.15, df_sub["b"].max() + 0.15, 100)
    m_sft, c_sft_l = np.polyfit(df_sub["b"], df_sub["dp_sft"], 1)
    m_grpo, c_grpo_l = np.polyfit(df_sub["b"], df_sub["dp_grpo"], 1)
    ax2.plot(x_grid, m_sft * x_grid + c_sft_l, "-", color="#0284C7", linewidth=2.3, alpha=0.85, zorder=3)
    ax2.plot(x_grid, m_grpo * x_grid + c_grpo_l, "-", color="#DC2626", linewidth=2.3, alpha=0.85, zorder=3)

    # Clean Subject Callouts
    r_mkt = df_sub.loc["marketing"]
    ax2.annotate(
        "Marketing (a = 1.45)",
        xy=(r_mkt["b"], r_mkt["dp_grpo"]), xytext=(-1.18, 13.8),
        arrowprops=dict(arrowstyle="-|>", color="#991B1B", lw=1.1),
        fontsize=8.6, fontweight="bold", color="#991B1B", zorder=6
    )
    r_soc = df_sub.loc["sociology"]
    ax2.text(r_soc["b"] + 0.08, r_soc["dp_grpo"] + 0.8, "Sociology (a = 1.28)", fontsize=8.6, fontweight="bold", color="#991B1B", zorder=6)

    r_em = df_sub.loc["elementary_mathematics"]
    ax2.text(r_em["b"] - 0.98, r_em["dp_sft"] + 2.0, "Elementary Math (a = 0.45)", fontsize=8.6, fontweight="bold", color="#0369A1", zorder=6)

    r_hm = df_sub.loc["high_school_mathematics"]
    ax2.text(r_hm["b"] - 0.88, r_hm["dp_sft"] + 2.0, "High School Math (a = 0.28)", fontsize=8.6, fontweight="bold", color="#0369A1", zorder=6)

    ax2.text(
        -1.18, 36.5, r"Bubble area $\propto$ IRT discrimination $a$" + "\n(Factual recall weighted 5× over math)",
        fontsize=8.6, fontweight="bold", color="#334155"
    )

    ax2.set_title("(B) Subject-Level Gains vs. 2PL Parameters (a, b)", pad=10)
    ax2.set_xlabel("Subject Mean 2PL Difficulty b (Easy → Hard)")
    ax2.set_ylabel("Accuracy Gain Over Base: ΔPass@1 (% pts)")
    ax2.set_xlim(-1.25, 2.65)
    ax2.set_ylim(-14, 44)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=8.3)
    ax2.grid(True, linestyle=":", alpha=0.45)

    save_fig(fig, "talk_Fig02_eval_cat_and_2d_domains.png")


# ==============================================================================
# FIGURE 3: 1x2 — RL Critics Ceiling & Multi-Draw Pass@k Collapse (EQ Test)
# ==============================================================================
def make_talk_fig03_rl_critics_and_eq_divergence():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.8, 5.3), gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.25}
    )

    # -------------------------------------------------------------------------
    # Panel A: All 4 Static IRT Critics Tie at the ~47.5% Modal Ceiling
    # -------------------------------------------------------------------------
    traj = pd.read_csv(os.path.join(RESULTS, "Q1", "trajectory.csv"))
    styles = [
        ("grpo", "#1E293B", "-o",  "Standard 5-Draw GRPO"),
        ("ctt",  "#64748B", ":s",  "Base Pass-Rate Baseline"),
        ("pop1", "#0284C7", "-.D", "2PL Discrimination-Weighted Critic"),
        ("pop0", "#D97706", "--^", "2PL Unweighted Critic"),
    ]

    ax1.axhspan(46.8, 48.0, color="#FEF3C7", alpha=0.60, zorder=1)
    ax1.axhline(47.5, color="#D97706", linestyle=":", linewidth=1.4, alpha=0.85, zorder=2)

    s0 = traj[traj["seed"] == 0].sort_values("updates")
    for m, col, fmt, lbl in styles:
        sub = s0[s0["mode"] == m]
        ax1.plot(
            sub["updates"], sub["mean_at_16"] * 100.0, fmt, color=col,
            linewidth=2.4, markersize=5.8, label=lbl, zorder=4
        )

    ax1.text(166, 48.45, "All 4 Critics Converge to ~47.5%", ha="center", fontsize=8.8, fontweight="bold", color="#92400E", zorder=6)
    ax1.set_title("(A) Single-Draw Pass@1 Across RL Critics", pad=11)
    ax1.set_xlabel("RL Training Updates")
    ax1.set_ylabel("MMLU Test Accuracy: Pass@1 (%)")
    ax1.set_xticks([0, 83, 166, 250, 333])
    ax1.set_ylim(37.2, 50.0)
    ax1.legend(loc="lower right", bbox_to_anchor=(0.98, 0.06), frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.4)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Multi-draw Pass@k Collapse vs Base & SFT + EQ Divergence
    # -------------------------------------------------------------------------
    q1_raw = pd.read_csv(os.path.join(RESULTS, "Q1", "stats.csv"))
    q1_pool = q1_raw[q1_raw["order"] == "pooled"].pivot(index="state", columns="statistic", values="value") * 100.0

    k_points = np.array([1, 4, 16])
    x_points = np.log2(k_points)
    k_dense = np.linspace(1, 16, 200)
    x_dense = np.log2(k_dense)

    # Base & SFT Curves
    p_base = PchipInterpolator(x_points, [38.18, 70.20, 93.00])
    p_sft  = PchipInterpolator(x_points, [43.89, 78.10, 96.60])

    ax2.plot(x_dense, p_sft(x_dense), "-", color="#0284C7", linewidth=2.8, label="Supervised Fine-Tuning (CoT-SFT): 43.9% → 96.6%", zorder=4)
    ax2.plot(x_dense, p_base(x_dense), "--", color="#475569", linewidth=2.2, label="Base Model (Llama-3.2-1B): 38.2% → 93.0%", zorder=3)

    for m, col, fmt, lbl in styles:
        states = [f"q1_{m}_{s}" for s in range(3)]
        p1 = q1_pool.loc[states, "pass_at_1"].mean()
        p4 = q1_pool.loc[states, "pass_at_4"].mean()
        p16 = q1_pool.loc[states, "pass_at_16"].mean()
        p_interp = PchipInterpolator(x_points, [p1, p4, p16])
        ax2.plot(x_dense, p_interp(x_dense), fmt[0], color=col, linewidth=2.0, label=f"{lbl}: {p1:.1f}% → {p16:.1f}%", zorder=5)
        # Scatter 3 empirical seeds at k = 1, 4, 16
        for i, met in enumerate(["pass_at_1", "pass_at_4", "pass_at_16"]):
            s_vals = q1_pool.loc[states, met].to_numpy()
            ax2.scatter([x_points[i]] * 3, s_vals, s=18, color=col, alpha=0.7, zorder=6)

    # Direct Endpoint Labels at k = 16
    ax2.text(4.06, 96.6, "96.6% (SFT)", va="center", ha="left", fontsize=8.4, fontweight="bold", color="#0284C7")
    ax2.text(4.06, 93.0, "93.0% (Base)", va="center", ha="left", fontsize=8.4, fontweight="bold", color="#475569")
    ax2.text(4.06, 69.4, "69.4% (GRPO)", va="center", ha="left", fontsize=8.4, fontweight="bold", color="#1E293B")
    ax2.text(4.06, 58.7, "58.7% (2PL-W & Base-Rate)", va="center", ha="left", fontsize=8.4, fontweight="bold", color="#0284C7")
    ax2.text(4.06, 52.4, "52.4% (Unweighted 2PL)", va="center", ha="left", fontsize=8.4, fontweight="bold", color="#D97706")

    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(["k = 1", "k = 2", "k = 4", "k = 8", "k = 16"])
    ax2.set_xlim(-0.15, 4.95)
    ax2.set_ylim(32.0, 103.0)
    ax2.set_title("(B) Multi-Draw Pass@k at End of Training", pad=11)
    ax2.set_xlabel("Number of Test-Time Rollout Attempts k (log₂ scale)")
    ax2.set_ylabel("MMLU Test Accuracy: Pass@k (%)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.0)
    ax2.grid(True, linestyle=":", alpha=0.45)

    save_fig(fig, "talk_Fig03_rl_critics_and_eq_divergence.png")


# ==============================================================================
# FIGURE 4: 2x1 Vertical Stack — Coordinate Translation (Population vs Base Model)
# ==============================================================================
def make_talk_fig04_coordinate_translation_stacked():
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

    # Panel A fits on external population difficulty b_j
    r2_p3_g = 1.0 - np.sum((dp_g - np.poly1d(np.polyfit(b, dp_g, 3))(b)) ** 2) / np.sum((dp_g - np.mean(dp_g)) ** 2)
    r2_p3_s = 1.0 - np.sum((dp_s - np.poly1d(np.polyfit(b, dp_s, 3))(b)) ** 2) / np.sum((dp_s - np.mean(dp_s)) ** 2)
    p3_g = np.poly1d(np.polyfit(b, dp_g, 3))
    p3_s = np.poly1d(np.polyfit(b, dp_s, 3))

    w0_ok["b_bin"] = pd.qcut(w0_ok["b"], q=10)
    binned_b = w0_ok.groupby("b_bin", observed=False)[["a", "b", "dp_sft", "dp_grpo"]].mean()

    # Panel B fits on within-model initial log-odds L_base
    p_g = np.clip(p0 + dp_g / 100.0, 1e-4, 1 - 1e-4)
    p_s = np.clip(p0 + dp_s / 100.0, 1e-4, 1 - 1e-4)
    res_g = minimize(lambda x: np.mean((p_g - expit(x[0] * L0 + x[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda x: np.mean((p_s - expit(x[0] * L0 + x[1])) ** 2), [0.5, 0.0])

    w0_ok["L_bin"] = pd.qcut(w0_ok["L_base"], q=10)
    binned_L = w0_ok.groupby("L_bin", observed=False)[["a", "L_base", "dp_sft", "dp_grpo"]].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.2), gridspec_kw={"hspace": 0.62})

    # ==================== Panel A (Top: External Difficulty) ====================
    ax1.axhline(0, color="#64748B", linestyle="--", linewidth=1.2)
    ax1.scatter(b, dp_g, s=w0_ok["a"] * 14.0, color="#DC2626", alpha=0.12, edgecolors="none", zorder=2)
    ax1.scatter(b, dp_s, s=w0_ok["a"] * 14.0, color="#0284C7", alpha=0.12, edgecolors="none", zorder=2)
    ax1.scatter(
        binned_b["b"], binned_b["dp_grpo"], s=binned_b["a"] * 90.0, color="#DC2626", marker="s",
        zorder=5, edgecolors="#7F1D1D", linewidth=1.1, label="RL (GRPO) — 10 Deciles"
    )
    ax1.scatter(
        binned_b["b"], binned_b["dp_sft"], s=binned_b["a"] * 90.0, color="#0284C7", marker="o",
        zorder=5, edgecolors="#0369A1", linewidth=1.1, label="SFT (CoT-SFT) — 10 Deciles"
    )

    b_dense = np.linspace(-2.5, 5.0, 300)
    ax1.plot(b_dense, p3_g(b_dense), "-", color="#DC2626", linewidth=2.5, zorder=6, label=f"RL Smooth Trend (R² = {r2_p3_g:.2f})")
    ax1.plot(b_dense, p3_s(b_dense), "-", color="#0284C7", linewidth=2.5, zorder=6, label=f"SFT Smooth Trend (R² = {r2_p3_s:.2f})")

    ax1.set_title("(A) Ordered by External 102-LLM Difficulty: Blurry Cloud (R² ≈ 0.08)", pad=9, fontsize=11.5)
    ax1.set_xlabel("External Population Item Difficulty b (Calibrated Across 102 LLMs: Easy → Hard)", fontsize=10.0)
    ax1.set_ylabel("Accuracy Change Δp (% pts) (↑)", fontsize=10.0)
    ax1.set_xlim(-2.5, 5.0)
    ax1.set_ylim(-30, 35)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.2)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # ==================== Panel B (Bottom: Base Model Pass Rate) ====================
    ax2.axhline(0, color="#64748B", linestyle="--", linewidth=1.2)

    # Shaded Random Guessing Zone (<25%, logit(0.25) = -1.0986)
    ax2.axvspan(-1.0986, -4.5, color="#FEE2E2", alpha=0.35, zorder=1)
    ax2.axvline(-1.0986, color="#DC2626", linestyle=":", linewidth=1.5, alpha=0.85, zorder=3)

    ax2.scatter(w0_ok["L_base"], dp_g, s=w0_ok["a"] * 14.0, color="#DC2626", alpha=0.12, edgecolors="none", zorder=2)
    ax2.scatter(w0_ok["L_base"], dp_s, s=w0_ok["a"] * 14.0, color="#0284C7", alpha=0.12, edgecolors="none", zorder=2)
    ax2.scatter(
        binned_L["L_base"], binned_L["dp_grpo"], s=binned_L["a"] * 90.0, color="#DC2626", marker="s",
        zorder=5, edgecolors="#7F1D1D", linewidth=1.1, label="RL Deciles"
    )
    ax2.scatter(
        binned_L["L_base"], binned_L["dp_sft"], s=binned_L["a"] * 90.0, color="#0284C7", marker="o",
        zorder=5, edgecolors="#0369A1", linewidth=1.1, label="SFT Deciles"
    )

    L_grid = np.linspace(3.5, -4.5, 400)
    pred_grpo_curve = (expit(res_g.x[0] * L_grid + res_g.x[1]) - expit(L_grid)) * 100.0
    pred_sft_curve  = (expit(res_s.x[0] * L_grid + res_s.x[1]) - expit(L_grid)) * 100.0
    classical_shift = (expit(L_grid + 0.5) - expit(L_grid)) * 100.0

    ax2.plot(L_grid, classical_shift, "--", color="#059669", linewidth=1.8, zorder=4, label=r"Classical Shift ($\Delta\theta = +0.5, l_z = 0$)")
    ax2.plot(L_grid, pred_grpo_curve, "-", color="#DC2626", linewidth=2.6, zorder=6, label=rf"RL Consolidation ($\tau = {res_g.x[0]:.2f}, l_z = +7.55$)")
    ax2.plot(L_grid, pred_sft_curve, "-", color="#0284C7", linewidth=2.6, zorder=6, label=rf"SFT Teacher Imitation ($\tau = {res_s.x[0]:.2f}, l_z = -16.99$)")

    # Surgical Direct Labels
    ax2.text(-2.8, 32.0, "Random Guessing Zone (<25%)", ha="center", fontsize=8.4, fontweight="bold", color="#991B1B", zorder=7)
    ax2.text(-2.8, -17.0, "RL Hurts Hard Items (-613 items)", ha="center", fontsize=8.2, fontweight="bold", color="#991B1B", zorder=7)
    ax2.annotate(
        "RL Consolidates Easy Items\n(l_z = +7.55, 0% easy misses)",
        xy=(0.45, 23.0), xytext=(-0.05, 34.0),
        arrowprops=dict(arrowstyle="-|>", color="#15803D", lw=1.1),
        ha="center", fontsize=8.1, fontweight="bold", color="#15803D", zorder=7
    )
    ax2.text(1.75, -24.5, "SFT Guttman Inversions\n(l_z = -16.99, 16.6% easy misses)", ha="center", fontsize=8.0, fontweight="bold", color="#0369A1", zorder=7)
    ax2.text(-2.8, 12.5, "SFT Lifts Hard Items", ha="center", fontsize=8.2, fontweight="bold", color="#0369A1", zorder=7)

    ax2.set_title("(B) Re-Ordered by Base Pass Rate: Hyper-Consistency (l_z = +7.55) vs. Inversions (l_z = -16.99)", pad=18, fontsize=11.0)
    ax2.set_xlabel(r"Base Model Initial Log-Odds $\mu_{0j} = \mathrm{logit}(p_{\mathrm{base},j})$ (Easy $\rightarrow$ Hard)", fontsize=10.0)
    ax2.set_ylabel("Accuracy Change Δp (% pts) (↑)", fontsize=10.0)
    ax2.set_xlim(3.5, -4.5)
    ax2.set_ylim(-32, 45)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=1, borderpad=0.3, labelspacing=0.22, frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=7.4)
    ax2.grid(True, linestyle=":", alpha=0.45)

    ax2_top = ax2.twiny()
    p_ticks = np.array([0.90, 0.75, 0.50, 0.25, 0.10, 0.05])
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(logit(p_ticks))
    ax2_top.set_xticklabels(["90%", "75%", "50%", "25%", "10%", "5%"])
    ax2_top.set_xlabel(r"Base Model Initial Pass Probability $p_{\mathrm{base},j}$", fontsize=9.4, fontweight="bold", labelpad=4)

    save_fig(fig, "talk_Fig04_coordinate_translation_stacked.png")


# ==============================================================================
# FIGURE 5: 1x2 — 3PL Information Trap & Nominal Distractor Split (p_top vs p_key)
# ==============================================================================
def make_talk_fig05_prf_slope_and_distractor_pivot():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.8, 5.3), gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.25}
    )

    # -------------------------------------------------------------------------
    # Panel A: 2PL/GRPO Variance vs. Birnbaum's 3PL Information (Guessing Floor c = 25%)
    # -------------------------------------------------------------------------
    p_grid = np.linspace(0.01, 0.99, 500)
    c = 0.25
    var_2pl = p_grid * (1.0 - p_grid)
    var_2pl_norm = var_2pl / np.max(var_2pl)  # Peak = 1.0 at p = 0.50

    info_3pl = np.where(p_grid > c, ((p_grid - c) ** 2) * (1.0 - p_grid) / (p_grid * ((1.0 - c) ** 2)), 0.0)
    info_3pl_norm = info_3pl / np.max(info_3pl)  # Peak = 1.0 at p ≈ 0.68

    # Empirical W1 active training prompt distribution across pass rates (k = 1..7 out of 8)
    w1 = pd.read_csv(os.path.join(RESULTS, "W1", "pass_rate.csv"))
    w1_active = w1[(w1["correct"] > 0) & (w1["correct"] < w1["k"])].copy()
    counts = w1_active["correct"].value_counts().sort_index()
    p_bins = counts.index.to_numpy() / 8.0
    shares = counts.to_numpy() / len(w1_active)  # Fraction of active GRPO prompts
    bar_colors = ["#DC2626" if p <= 0.25 else "#94A3B8" for p in p_bins]

    ax1.axvspan(0.0, 26.5, color="#FEE2E2", alpha=0.45, zorder=1)
    ax1.axvline(25.0, color="#DC2626", linestyle=":", linewidth=1.6, zorder=3)
    ax1.bar(p_bins * 100.0, shares * 1.75, width=7.2, color=bar_colors, alpha=0.35, edgecolor="#475569", linewidth=0.8, zorder=2,
            label="Active GRPO Training Prompts (Distribution)")
    ax1.plot(p_grid * 100.0, var_2pl_norm, "--", color="#DC2626", linewidth=2.6, zorder=5,
             label=r"2PL / GRPO Variance $p_j(1-p_j)$ (75% of peak at 25%)")
    ax1.plot(p_grid * 100.0, info_3pl_norm, "-", color="#7C3AED", linewidth=3.0, zorder=6,
             label=r"Birnbaum 3PL Info $I_j^{\mathrm{3PL}}$ ($=0$ at $\leq 25\%$)")

    ax1.annotate(
        "53.9% of Active Prompts\nin <=25% Guessing Region\n(3PL Info = 0, GRPO +2σ)",
        xy=(12.5, 0.52), xytext=(15.5, 0.99),
        arrowprops=dict(arrowstyle="-|>", color="#991B1B", lw=1.3),
        ha="center", fontsize=7.9, fontweight="bold", color="#991B1B", zorder=7
    )
    ax1.annotate(
        "3PL True-Ability Info\nPeaks in [2..6/8] Band",
        xy=(68.0, 0.99), xytext=(62.5, 0.48),
        arrowprops=dict(arrowstyle="-|>", color="#5B21B6", lw=1.2),
        ha="center", fontsize=8.1, fontweight="bold", color="#5B21B6", zorder=7
    )

    ax1.set_title("(A) 2PL Variance vs. 3PL Information (Guessing Floor c = 25%)", pad=11)
    ax1.set_xlabel("Base Model Initial Pass Rate p_base,j (%)")
    ax1.set_ylabel("Normalized Information / Signal Weight")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.24)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=7.6)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Nominal Distractor Split (p_top_base vs. p_key_base)
    # -------------------------------------------------------------------------
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    p_key = t9["p_key_base"].values
    p_top = t9["p_top_base"].values
    dp_sft = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values * 100.0
    dp_rl  = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values * 100.0

    key_is_top = np.isclose(p_key, p_top, atol=1e-4)
    wrong_is_top = p_top > p_key + 1e-4

    masks = [
        key_is_top,
        wrong_is_top & (p_top >= 0.25) & (p_top < 0.50),
        wrong_is_top & (p_top >= 0.50) & (p_top < 0.75),
        wrong_is_top & (p_top >= 0.75),
    ]
    labels = [
        f"Correct Key\nIs Base Mode\n(n = {masks[0].sum()})",
        f"Weak Wrong\nDistractor\np_top in [25,50%)\n(n = {masks[1].sum()})",
        f"Moderate Wrong\nDistractor\np_top in [50,75%)\n(n = {masks[2].sum()})",
        f"Strong Wrong\nMisconception\np_top >= 75%\n(n = {masks[3].sum()})",
    ]

    rl_means  = [dp_rl[m].mean() for m in masks]
    sft_means = [dp_sft[m].mean() for m in masks]

    x = np.arange(len(labels))
    width = 0.34

    ax2.axvspan(0.5, 3.5, color="#F8FAFC", alpha=0.9, zorder=0)
    ax2.axvline(0.5, color="#64748B", linestyle="--", linewidth=1.3, zorder=2)
    ax2.axhline(0.0, color="#334155", linewidth=1.2, zorder=2)

    rects_rl  = ax2.bar(x - width / 2, rl_means, width, color="#DC2626", edgecolor="#7F1D1D", linewidth=1.0, zorder=4,
                        label="GRPO (RL): Sharpens Existing Mode (Locks In Wrong Distractor)")
    rects_sft = ax2.bar(x + width / 2, sft_means, width, color="#0284C7", edgecolor="#0369A1", linewidth=1.0, zorder=4,
                        label="CoT-SFT: Flips 56.2% of Modes (Overthrows Wrong Distractor)")

    for r in rects_rl:
        h = r.get_height()
        va = "bottom" if h >= 0 else "top"
        y_pos = h + 0.6 if h >= 0 else h - 0.6
        ax2.text(r.get_x() + r.get_width() / 2, y_pos, f"{h:+.1f}%", ha="center", va=va, fontsize=8.6, fontweight="bold", color="#991B1B", zorder=6)

    for r in rects_sft:
        h = r.get_height()
        va = "bottom" if h >= 0 else "top"
        y_pos = h + 0.6 if h >= 0 else h - 0.6
        ax2.text(r.get_x() + r.get_width() / 2, y_pos, f"{h:+.1f}%", ha="center", va=va, fontsize=8.6, fontweight="bold", color="#0369A1", zorder=6)

    ax2.text(2.0, 15.6, "Wrong Distractor Is Base Mode (51.4% of Bank, n = 756)\nSFT Overthrows Misconceptions (+11.7%); RL Locks Them In (-4.9%)",
             ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1E293B", zorder=6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.6, fontweight="bold")
    ax2.set_title("(B) Nominal Distractor Split: Lock-In (RL) vs. Overthrow (SFT)", pad=11)
    ax2.set_ylabel("Accuracy Change on Correct Key: Δp_key (% pts)")
    ax2.set_xlim(-0.55, 3.55)
    ax2.set_ylim(-20.5, 19.5)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.0)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.45)

    save_fig(fig, "talk_Fig05_prf_slope_and_distractor_pivot.png")


# ==============================================================================
# FIGURE 6: 1x2 — Local Independence Violation & Paired Option-Order DIF
# ==============================================================================
def make_talk_fig06_local_independence_and_order_dif():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.8, 5.3), gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.26}
    )

    # -------------------------------------------------------------------------
    # Panel A: Cross-Seed Item Residual Repeatability (r = 0.81)
    # -------------------------------------------------------------------------
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]
    p0 = np.clip(t9["p_key_base"].values, 1e-4, 1 - 1e-4)
    L0 = logit(p0)

    res_grpo_seeds = []
    res_sft_seeds = []
    for s in range(3):
        dp_g_s = t9[f"dp_key_rl_{s}_666"].values
        p_g_s = np.clip(p0 + dp_g_s, 1e-4, 1 - 1e-4)
        fit_g = minimize(lambda c: np.mean((p_g_s - expit(c[0] * L0 + c[1])) ** 2), [1.8, 0.0])
        pred_dp_g = expit(fit_g.x[0] * L0 + fit_g.x[1]) - p0
        res_grpo_seeds.append((dp_g_s - pred_dp_g) * 100.0)

        dp_s_s = t9[f"dp_key_sft_{s}_70"].values
        p_s_s = np.clip(p0 + dp_s_s, 1e-4, 1 - 1e-4)
        fit_s = minimize(lambda c: np.mean((p_s_s - expit(c[0] * L0 + c[1])) ** 2), [0.5, 0.0])
        pred_dp_s = expit(fit_s.x[0] * L0 + fit_s.x[1]) - p0
        res_sft_seeds.append((dp_s_s - pred_dp_s) * 100.0)

    r_grpo = np.corrcoef(res_grpo_seeds[0], res_grpo_seeds[1])[0, 1]
    r_sft  = np.corrcoef(res_sft_seeds[0], res_sft_seeds[1])[0, 1]

    ax1.axhline(0, color="#94A3B8", linestyle=":", linewidth=1.1, zorder=1)
    ax1.axvline(0, color="#94A3B8", linestyle=":", linewidth=1.1, zorder=1)
    ax1.plot([-42, 42], [-42, 42], "--", color="#1E293B", linewidth=1.8, alpha=0.75, zorder=3, label="Exact Cross-Seed Identity (y = x)")

    null_circle = Ellipse(
        (0, 0), width=36, height=36, edgecolor="#64748B", facecolor="#F1F5F9",
        linestyle=":", linewidth=1.6, alpha=0.55, zorder=2, label="Random Noise Null Circle (r ≤ 0.19)"
    )
    ax1.add_patch(null_circle)

    ax1.scatter(res_grpo_seeds[0], res_grpo_seeds[1], s=14, color="#DC2626", alpha=0.18, edgecolors="none", zorder=3)
    ax1.scatter(res_sft_seeds[0], res_sft_seeds[1], s=14, color="#0284C7", alpha=0.18, edgecolors="none", zorder=3)

    df_res = pd.DataFrame({"g0": res_grpo_seeds[0], "g1": res_grpo_seeds[1], "s0": res_sft_seeds[0], "s1": res_sft_seeds[1]})
    df_res["g_bin"] = pd.qcut(df_res["g0"], q=10)
    df_res["s_bin"] = pd.qcut(df_res["s0"], q=10)
    bin_g = df_res.groupby("g_bin", observed=False)[["g0", "g1"]].mean()
    bin_s = df_res.groupby("s_bin", observed=False)[["s0", "s1"]].mean()

    ax1.plot(bin_g["g0"], bin_g["g1"], "-s", color="#DC2626", linewidth=2.8, markersize=6.8, markeredgecolor="#7F1D1D", zorder=6,
             label=f"RL Item Residuals Across Seeds (r = {r_grpo:.2f})")
    ax1.plot(bin_s["s0"], bin_s["s1"], "-o", color="#0284C7", linewidth=2.8, markersize=6.8, markeredgecolor="#0369A1", zorder=6,
             label=f"SFT Item Residuals Across Seeds (r = {r_sft:.2f})")

    ax1.set_xlim(-42, 42)
    ax1.set_ylim(-42, 42)
    ax1.set_title("(A) Cross-Seed Residual Correlation (r = 0.81)", pad=10)
    ax1.set_xlabel("Seed 1 Item Residual Gain (% pts)")
    ax1.set_ylabel("Seed 2 Item Residual Gain (% pts)")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.4)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Clean Horizontal Dot-and-Range / Lollipop Chart for Order Sensitivity
    # -------------------------------------------------------------------------
    categories = [
        "CoT-SFT\n(3 seeds)",
        "Base Model\n(Llama-3.2-1B)",
        "GRPO (G = 5)\n(3 seeds)",
        "GRPO + 2PL Critics\n(Online 2PL)",
    ]
    y_pos = np.arange(len(categories))
    vals = [1.88, 2.83, 10.67, 11.80]
    colors = ["#0284C7", "#475569", "#DC2626", "#D97706"]
    edge_colors = ["#0369A1", "#1E293B", "#7F1D1D", "#92400E"]

    # Reference lines
    ax2.axvline(1.035, color="#64748B", linestyle="--", linewidth=1.5, zorder=2)
    ax2.axvline(2.83, color="#94A3B8", linestyle=":", linewidth=1.4, zorder=2)
    ax2.text(3.15, 3.62, "← Sampling Noise Floor (D = 1.04)", ha="left", fontsize=8.4, fontweight="bold", color="#475569")

    # Horizontal Stems
    for i in range(len(categories)):
        ax2.hlines(y_pos[i], 1.035, vals[i], color=colors[i], linewidth=2.4, alpha=0.75, zorder=3)

    # Empirical Seed Points
    ax2.scatter([1.85, 1.88, 1.92], [y_pos[0]] * 3, color="#0C4A6E", s=32, zorder=5)
    ax2.scatter([9.74, 10.67, 11.60], [y_pos[2]] * 3, color="#7F1D1D", s=32, zorder=5)
    ax2.errorbar(vals[3], y_pos[3], xerr=[[vals[3] - 6.58], [19.36 - vals[3]]], fmt="none", color="#92400E", capsize=4, elinewidth=2.0, capthick=2.0, zorder=5)

    # Large Value Markers
    for i in range(len(categories)):
        ax2.scatter(vals[i], y_pos[i], s=120, color=colors[i], edgecolors=edge_colors[i], linewidth=1.6, zorder=6)

    # Direct Labels Beside Markers (Zero Overlap)
    ax2.text(vals[0] + 0.65, y_pos[0], "D = 1.88% (-34% vs. Base)", va="center", ha="left", fontsize=8.8, fontweight="bold", color="#0284C7")
    ax2.text(vals[1] + 0.65, y_pos[1], "D = 2.83% (Base)", va="center", ha="left", fontsize=8.8, fontweight="bold", color="#475569")
    ax2.text(11.60 + 0.65, y_pos[2], "D = 10.67% (3.8× Base)", va="center", ha="left", fontsize=8.8, fontweight="bold", color="#DC2626")
    ax2.text(19.36 + 0.55, y_pos[3], "D = 11.80%", va="center", ha="left", fontsize=8.8, fontweight="bold", color="#D97706")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=9.2, fontweight="bold")
    ax2.set_xlabel("Option-Order Sensitivity D (% pts)")
    ax2.set_xlim(0, 26.0)
    ax2.set_ylim(-0.6, 4.0)
    ax2.set_title("(B) Sensitivity to Option Order (Swapping A/B/C/D)", pad=10)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.45)

    save_fig(fig, "talk_Fig06_local_independence_and_order_dif.png")


# ==============================================================================
# FIGURE 7: 1x2 — Predictions, Coverage Phase Space & 2-Tier Slope Regulation
# ==============================================================================
def make_talk_fig07_predictions_and_fisher_band():
    fig = plt.figure(figsize=(14.8, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.27)

    # -------------------------------------------------------------------------
    # Panel A: Accuracy vs. Coverage Phase Space
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    m1_raw = pd.read_csv(os.path.join(RESULTS, "M1", "stats.csv"))
    m1_p = m1_raw.pivot(index="state", columns="statistic", values="value") * 100.0

    # 1. Base Model
    base_x = m1_p.loc["t9_base", "mean_at_16"]
    base_y = m1_p.loc["t9_base", "pass_at_16"]
    ax1.scatter([base_x], [base_y], s=120, color="#475569", marker="o", edgecolors="#0F172A", linewidth=1.5, zorder=6, label="Base Model (38.2%, 93.0%)")
    ax1.text(base_x - 0.2, base_y + 1.6, "Base (38.2%, 93.0%)", ha="left", fontsize=8.4, fontweight="bold", color="#334155")

    # 2. Standalone RL (u333)
    grpo_states = ["q1_grpo_0", "q1_grpo_1", "q1_grpo_2"]
    grpo_xs = m1_p.loc[grpo_states, "mean_at_16"].to_numpy()
    grpo_ys = m1_p.loc[grpo_states, "pass_at_16"].to_numpy()
    ax1.scatter(grpo_xs, grpo_ys, color="#DC2626", s=30, alpha=0.45, zorder=5)
    grpo_cx, grpo_cy = np.mean(grpo_xs), np.mean(grpo_ys)
    ax1.scatter([grpo_cx], [grpo_cy], s=120, color="#DC2626", marker="s", edgecolors="#7F1D1D", linewidth=1.5, zorder=6, label="Standalone RL (46.9%, 69.4%)")

    ax1.axhspan(64.5, 73.2, color="#FEE2E2", alpha=0.38, zorder=1)
    ax1.text(
        46.65, 66.2,
        "Standalone RL Mode Collapse (-23.6% vs. Base)",
        ha="center", va="center", fontsize=8.4, fontweight="bold", color="#991B1B", zorder=6
    )

    arrow_grpo = FancyArrowPatch(
        (base_x, base_y), (grpo_cx, grpo_cy),
        connectionstyle="arc3,rad=-0.12", color="#DC2626",
        arrowstyle="-|>", mutation_scale=16, linewidth=2.2, zorder=5
    )
    ax1.add_patch(arrow_grpo)

    # 3. CoT-SFT Base
    sft_states = ["t9_sft_0_70", "t9_sft_1_70", "t9_sft_2_70"]
    sft_xs = m1_p.loc[sft_states, "mean_at_16"].to_numpy()
    sft_ys = m1_p.loc[sft_states, "pass_at_16"].to_numpy()
    sft_cx, sft_cy = np.mean(sft_xs), np.mean(sft_ys)
    ax1.scatter([sft_cx], [sft_cy], s=120, color="#0284C7", marker="o", edgecolors="#0369A1", linewidth=1.5, zorder=6, label="CoT-SFT (T=0.7) (43.9%, 96.6%)")
    ax1.text(sft_cx - 0.35, sft_cy + 1.5, "CoT-SFT (43.9%, 96.6%)", ha="right", fontsize=8.4, fontweight="bold", color="#0369A1")

    arrow_base_sft = FancyArrowPatch(
        (base_x, base_y), (sft_cx, sft_cy),
        connectionstyle="arc3,rad=0.06", color="#0284C7",
        arrowstyle="-|>", mutation_scale=16, linewidth=2.2, zorder=5
    )
    ax1.add_patch(arrow_base_sft)

    # 4. Zero-Training SFT T=0.2
    t02_states = ["m1a_sft_t02_0", "m1a_sft_t02_1", "m1a_sft_t02_2"]
    t02_xs = m1_p.loc[t02_states, "mean_at_16"].to_numpy()
    t02_ys = m1_p.loc[t02_states, "pass_at_16"].to_numpy()
    t02_cx, t02_cy = np.mean(t02_xs), np.mean(t02_ys)
    ax1.scatter([t02_cx], [t02_cy], s=125, color="#059669", marker="D", edgecolors="#047857", linewidth=1.5, zorder=6, label="SFT + Low Temp T=0.2 (0 RL steps: 47.5%, 87.9%)")

    arrow_t02 = FancyArrowPatch(
        (sft_cx, sft_cy), (t02_cx, t02_cy),
        connectionstyle="arc3,rad=-0.12", color="#059669",
        arrowstyle="-|>", mutation_scale=16, linewidth=2.2, zorder=5
    )
    ax1.add_patch(arrow_t02)

    # 5. Two-Stage SFT -> RL & Fisher Informative Band
    m1b_u333_states = [f"m1b_sft_grpo_{s}" for s in range(3)]
    m1b_33_x, m1b_33_y = m1_p.loc[m1b_u333_states, "mean_at_16"].mean(), m1_p.loc[m1b_u333_states, "pass_at_16"].mean()
    ax1.scatter([m1b_33_x], [m1b_33_y], s=120, color="#7C3AED", marker="s", edgecolors="#5B21B6", linewidth=1.5, zorder=6, label="Two-Stage SFT → RL (46.7%, 94.9%)")

    m1c_u333_states = [f"m1c_sft_zpd_{s}" for s in range(3)]
    m1c_33_x, m1c_p_y = m1_p.loc[m1c_u333_states, "mean_at_16"].mean(), m1_p.loc[m1c_u333_states, "pass_at_16"].mean()
    ax1.scatter([m1c_33_x], [m1c_p_y], s=120, color="#D97706", marker="^", edgecolors="#B45309", linewidth=1.5, zorder=6, label="Two-Stage SFT → RL [2..6/8 Band] (46.5%, 94.2%)")

    arrow_sft_rl = FancyArrowPatch(
        (sft_cx, sft_cy), (m1b_33_x, m1b_33_y),
        connectionstyle="arc3,rad=-0.06", color="#7C3AED",
        arrowstyle="-|>", mutation_scale=15, linewidth=2.2, zorder=5
    )
    ax1.add_patch(arrow_sft_rl)

    # Shaded Optimal Region
    ax1.fill_between([44.9, 48.2], 93.0, 97.2, color="#F5F3FF", alpha=0.65, zorder=1)
    pareto_box = patches.Rectangle((44.9, 93.0), 3.3, 4.2, fill=False, edgecolor="#7C3AED", linestyle="--", linewidth=1.4, zorder=2)
    ax1.add_patch(pareto_box)

    ax1.text(46.55, 98.2, "Optimal Region (High Pass@1 + Pass@16)", ha="center", fontsize=8.4, fontweight="bold", color="#5B21B6")
    ax1.text(46.9, 83.6, "Zero-Training (T=0.2):\nBeats RL in 3/3 seeds", ha="center", fontsize=8.2, fontweight="bold", color="#047857")

    ax1.set_xlim(36.8, 49.0)
    ax1.set_ylim(64.5, 102.5)
    ax1.set_title("(A) Single-Draw Pass@1 vs. Multi-Draw Pass@16", pad=11)
    ax1.set_xlabel("Single-Draw Accuracy: Pass@1 (%)")
    ax1.set_ylabel("Multi-Draw Coverage: Pass@16 (%)")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=7.8)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Controlled Latent Slope τ_t on a Clean 2-Tier Broken Y-Axis
    # -------------------------------------------------------------------------
    gs_b = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.13)
    ax2_top = fig.add_subplot(gs_b[0])
    ax2_bot = fig.add_subplot(gs_b[1], sharex=ax2_top)

    r6_updates = [0, 83, 166, 250, 333]
    tau_r6 = np.array([1.000, 2.716, 3.215, 3.234, 3.275])
    tau_r6_se = np.array([0.100, 0.179, 0.183, 0.186, 0.202])

    t_comp = [0, 83, 333]
    tau_band = [1.000, 1.118, 1.233]
    tau_unf  = [1.000, 1.080, 1.192]

    # --- Top Tier: Unconstrained RL (y in [2.2, 3.65]) ---
    ax2_top.fill_between(r6_updates, tau_r6 - tau_r6_se, tau_r6 + tau_r6_se, color="#DC2626", alpha=0.18, zorder=2)
    ax2_top.plot(r6_updates, tau_r6, "-o", color="#DC2626", linewidth=2.6, markersize=5.5, zorder=5)
    ax2_top.scatter([333, 333, 333], [3.275, 3.216, 3.303], color="#7F1D1D", s=26, zorder=6)
    ax2_top.text(250, 2.45, "Unconstrained RL: τ → 3.28\n(Over-sharpening into mode collapse)", ha="center", fontsize=8.2, fontweight="bold", color="#991B1B")

    ax2_top.set_ylim(2.20, 3.65)
    ax2_top.set_yticks([2.5, 3.0, 3.5])
    ax2_top.spines["bottom"].set_visible(False)
    ax2_top.tick_params(bottom=False, labelbottom=False)
    ax2_top.grid(True, linestyle=":", alpha=0.45)
    ax2_top.set_title("(B) Response Curve Slope τ Across RL Updates", pad=11)

    # Clean Combined Legend in Top Tier (2 columns, perfectly open)
    handles = [
        plt.Line2D([], [], color="#DC2626", marker="o", lw=2.2, label="Unconstrained RL (Shoots to τ = 3.28)"),
        plt.Line2D([], [], color="#059669", ls="--", lw=2.0, label="SFT + Low Temp T=0.2 (τ = 1.66)"),
        plt.Line2D([], [], color="#D97706", marker="s", lw=2.2, label="SFT → RL [2..6/8 Informative Band]"),
        plt.Line2D([], [], color="#7C3AED", ls="--", marker="^", lw=2.0, label="SFT → RL [All Items, Unfiltered]"),
    ]
    ax2_top.legend(handles=handles, loc="upper left", ncol=2, fontsize=7.2, frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96)

    # --- Bottom Tier: Controlled Regimes (y in [0.92, 1.78]) ---
    ax2_bot.plot(r6_updates, tau_r6, "-o", color="#DC2626", linewidth=2.2, markersize=5.2, alpha=0.75, zorder=4)
    ax2_bot.axhline(1.664, color="#059669", linestyle="--", linewidth=2.0, zorder=4)
    ax2_bot.scatter([0, 0, 0], [1.663, 1.710, 1.618], color="#047857", marker="D", s=26, zorder=6)

    ax2_bot.plot(t_comp, tau_band, "-s", color="#D97706", linewidth=2.4, markersize=5.5, zorder=5)
    ax2_bot.plot(t_comp, tau_unf,  "--^", color="#7C3AED", linewidth=2.1, markersize=5.5, zorder=5)

    ax2_bot.scatter([83, 83, 83], [1.1028, 1.1441, 1.1079], color="#B45309", s=18, zorder=6)
    ax2_bot.scatter([333, 333, 333], [1.2278, 1.2938, 1.1759], color="#B45309", s=18, zorder=6)
    ax2_bot.scatter([83, 83, 83], [1.0712, 1.1047, 1.0650], color="#5B21B6", s=18, zorder=6)
    ax2_bot.scatter([333, 333, 333], [1.2126, 1.2177, 1.1470], color="#5B21B6", s=18, zorder=6)

    ax2_bot.text(42, 1.69, "CoT-SFT @ T=0.2 (τ = 1.66, 0 RL steps)", fontsize=8.2, fontweight="bold", color="#047857")

    ax2_bot.annotate(
        "Informative Band [2..6/8]:\nΔτ = +0.04 (p < 0.001)",
        xy=(333, 1.233), xytext=(215, 1.42),
        arrowprops=dict(arrowstyle="-|>", color="#B45309", lw=1.2),
        ha="center", fontsize=8.2, fontweight="bold", color="#B45309"
    )

    ax2_bot.set_ylim(0.92, 1.78)
    ax2_bot.set_yticks([1.0, 1.2, 1.4, 1.6])
    ax2_bot.spines["top"].set_visible(False)
    ax2_bot.grid(True, linestyle=":", alpha=0.45)
    ax2_bot.set_xticks([0, 83, 166, 250, 333])
    ax2_bot.set_xlim(-15, 365)
    ax2_bot.set_xlabel("RL Training Updates")

    fig.text(0.505, 0.52, "IRT Slope τ (Policy Steepening Factor)", va="center", rotation="vertical", fontsize=10.5, fontweight="bold")

    d_slash = 0.015
    kwargs = dict(transform=ax2_top.transAxes, color="#475569", clip_on=False, linewidth=1.2)
    ax2_top.plot((-d_slash, +d_slash), (-d_slash * 1.5, +d_slash * 1.5), **kwargs)
    kwargs.update(transform=ax2_bot.transAxes)
    ax2_bot.plot((-d_slash, +d_slash), (1 - d_slash * 1.5, 1 + d_slash * 1.5), **kwargs)

    save_fig(fig, "talk_Fig07_predictions_and_fisher_band.png")


# ==============================================================================
# FIGURE 8 (BACKUP B2): Copy FP32 vs BF16 Numeric Audit
# ==============================================================================
def copy_talk_figb2_fp32_vs_bf16_audit():
    src = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\v2_Fig06_backup_b2_fp32_vs_bf16.png"
    if os.path.exists(src):
        for d in OUT_DIRS:
            dest = os.path.join(d, "talk_FigB2_fp32_vs_bf16_audit.png")
            shutil.copy2(src, dest)
            print(f"[OK] Copied {src} -> {dest}")
    else:
        print(f"[WARN] Source file not found: {src}")


if __name__ == "__main__":
    print("=" * 70)
    print("Generating 7 Science-First Presentation Figures (Oct 26 Presentation)...")
    print("=" * 70)
    make_talk_fig01_puzzle_and_modal_ceiling()
    make_talk_fig02_eval_cat_and_2d_domains()
    make_talk_fig03_rl_critics_and_eq_divergence()
    make_talk_fig04_coordinate_translation_stacked()
    make_talk_fig05_prf_slope_and_distractor_pivot()
    make_talk_fig06_local_independence_and_order_dif()
    make_talk_fig07_predictions_and_fisher_band()
    copy_talk_figb2_fp32_vs_bf16_audit()
    print("=" * 70)
    print("All figures generated successfully without errors!")
    print("=" * 70)
