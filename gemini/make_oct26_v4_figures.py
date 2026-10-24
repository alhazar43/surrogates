"""
make_oct26_v4_figures.py

Generates the 7 V4 "Less is More" Presentation Figures for the Oct 26 Presentation:
  1. v4_Fig01_empirical_passk_puzzle.png        (Standalone 1x1: Empirical Pass@k Scaling Dilemma)
  2. v4_Fig02_static_population_irt_fails.png   (1x2: Static Benchmark Evaluation Floor & RL Reward Ceiling)
  3. v4_Fig03_coordinate_translation_stacked.png (Standalone 2x1: External Population Difficulty vs Base Model Initial Pass Rate)
  4. v4_Fig04_icc_slope_translation.png         (Standalone 1x1: Item Characteristic Curve Rotation & IRT <-> LLM Translation)
  5. v4_Fig05_irt_added_value_diagnostics.png   (1x2: 15-Subject 2D Separation & Cross-Seed Residual Repeatability)
  6. v4_Fig06_actionable_control_payoff.png     (1x2: Accuracy-Coverage Tradeoff Plane & 2-Tier Online Slope Regulation)
  7. v4_Fig07_dynamic_irt_schematic.png         (Standalone 1x1: Minimalist Closed-Loop Dynamic IRT Schematic)

Strict Design Principles ("Less is More"):
  - ZERO internal code names or jargon (no Q2a, Q1, W0, W1, T9r, S1, R6, M1, Unit P, pop0, pop1, ctt, JEDM, DUET, GSL, Doob).
  - ZERO multi-line text annotation boxes (no bbox paragraph cards inside data charts).
  - ZERO bar plots (continuous curves, 2D trajectories, vertical stacks, clean schematic).
  - Different figures help each other sequentially to bridge the IRT <-> LLM translational gap.
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

# Global Presentation Typography & Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.titlesize": 12.4,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.2,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.4,
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
    r"D:\surrogates\gemini\figures_v4",
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
# FIGURE 1: Standalone 1x1 — The Empirical Pass@k Scaling Dilemma
# ==============================================================================
def make_v4_fig01_empirical_passk_puzzle():
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

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

    # Subtle Scissor Shading
    ax.fill_between(
        x_log2[:idx_cross + 1], curve_sft[:idx_cross + 1], curve_grpo[:idx_cross + 1],
        color="#FEE2E2", alpha=0.45, zorder=1
    )
    ax.fill_between(
        x_log2[idx_cross:], curve_grpo[idx_cross:], curve_sft[idx_cross:],
        color="#E0F2FE", alpha=0.45, zorder=1
    )
    ax.axvline(x_cross, color="#64748B", linestyle=":", linewidth=1.4, alpha=0.75, zorder=2)

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

    # Minimalist Direct Labels
    ax.annotate(
        "Crossover (k ≈ 1.3)",
        xy=(x_cross, y_cross), xytext=(x_cross + 0.35, y_cross - 6.8),
        arrowprops=dict(arrowstyle="-|>", color="#0F172A", lw=1.2),
        fontsize=9.2, fontweight="bold", color="#0F172A", zorder=7
    )
    ax.annotate(
        "RL Wins Pass@1 (+4.0%)",
        xy=(0.0, 49.0), xytext=(-0.12, 60.5),
        arrowprops=dict(arrowstyle="-|>", color="#991B1B", lw=1.2),
        ha="left", fontsize=9.2, fontweight="bold", color="#991B1B", zorder=7
    )

    # Endpoint Labels at k = 16
    ax.text(4.08, 96.6, "96.6% (SFT)", va="center", ha="left", fontsize=9.4, fontweight="bold", color="#0369A1")
    ax.text(4.08, 92.5, "93.0% (Base)", va="center", ha="left", fontsize=9.2, fontweight="bold", color="#475569")
    ax.text(4.08, 65.6, "65.6% (RL)", va="center", ha="left", fontsize=9.4, fontweight="bold", color="#991B1B")

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["k = 1\n(Pass@1)", "k = 2", "k = 4\n(Pass@4)", "k = 8", "k = 16\n(Pass@16)"])
    ax.set_xlim(-0.22, 4.78)
    ax.set_ylim(30.0, 103.0)
    ax.set_title("The Post-Training Puzzle: RL Wins at 1 Draw but Collapses Multi-Draw Reach", pad=12)
    ax.set_xlabel("Number of Test-Time Rollout Attempts k (log₂ scale)")
    ax.set_ylabel("MMLU Test Bank Accuracy: Pass@k (%) (↑)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=9.2)
    ax.grid(True, linestyle=":", alpha=0.45)

    save_fig(fig, "v4_Fig01_empirical_passk_puzzle.png")


# ==============================================================================
# FIGURE 2: 1x2 — Why Naive Population IRT Fails in Evaluation & RL Training
# ==============================================================================
def make_v4_fig02_static_population_irt_fails():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.0, 5.3), gridspec_kw={"width_ratios": [1.0, 1.02], "wspace": 0.25}
    )

    # -------------------------------------------------------------------------
    # Panel A: Benchmark Evaluation Error Floor
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

    # Clean direct labels without line collision
    ax1.text(12.0, 1.55, "2.4% Static IRT Error Floor", fontsize=9.0, fontweight="bold", color="#991B1B", zorder=6)
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
    ax1.set_title("(A) Benchmark Evaluation: Adaptive IRT Hits a 2.4% Error Floor", pad=11)
    ax1.set_xlabel("Number of Sampled MMLU Test Items n (log scale)")
    ax1.set_ylabel("Score Estimation Error: RMSE (%) (↓)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.8)
    ax1.grid(True, which="both", linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: RL Training Reward Weighting Trajectories (Llama-3.2-1B on MMLU)
    # -------------------------------------------------------------------------
    traj = pd.read_csv(os.path.join(RESULTS, "Q1", "trajectory.csv"))
    styles = [
        ("grpo", "#1E293B", "-o",  "Unweighted Binary Reward (Standard GRPO)"),
        ("ctt",  "#64748B", ":s",  "Difficulty-Weighted Reward (Empirical Pass Rate)"),
        ("pop1", "#0284C7", "-.D", "Information-Weighted Reward (2PL Discrimination)"),
        ("pop0", "#D97706", "--^", "Surrogate IRT Reward (Population Difficulty)"),
    ]

    ax2.axhspan(46.8, 48.0, color="#FEF3C7", alpha=0.60, zorder=1)
    ax2.axhline(47.2, color="#D97706", linestyle=":", linewidth=1.4, alpha=0.85, zorder=2)

    s0 = traj[traj["seed"] == 0].sort_values("updates")
    for m, col, fmt, lbl in styles:
        sub = s0[s0["mode"] == m]
        ax2.plot(
            sub["updates"], sub["mean_at_16"] * 100.0, fmt, color=col,
            linewidth=2.4, markersize=5.8, label=lbl, zorder=4
        )

    ax2.text(205, 48.45, "Identical ~47% Single-Draw Ceiling Across All 4 Critics", ha="center", fontsize=9.0, fontweight="bold", color="#92400E", zorder=6)

    ax2.set_title("(B) RL Post-Training: Population IRT Critics Do Not Lift Accuracy", pad=11)
    ax2.set_xlabel("RL Training Updates (Llama-3.2-1B on MMLU)")
    ax2.set_ylabel("MMLU Test Bank Accuracy: Pass@1 (%) (↑)")
    ax2.set_xticks([0, 83, 166, 250, 333])
    ax2.set_ylim(37.2, 50.0)
    # Place legend in the completely open lower-right pocket (no curve overlap!)
    ax2.legend(loc="lower right", bbox_to_anchor=(0.98, 0.06), frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.6)
    ax2.grid(True, linestyle=":", alpha=0.45)

    save_fig(fig, "v4_Fig02_static_population_irt_fails.png")


# ==============================================================================
# FIGURE 3: 2x1 Vertical Stack — Coordinate Translation (Population vs Base Model)
# ==============================================================================
def make_v4_fig03_coordinate_translation_stacked():
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

    # Panel A: Fits on external population difficulty b_j
    r2_p3_g = 1.0 - np.sum((dp_g - np.poly1d(np.polyfit(b, dp_g, 3))(b)) ** 2) / np.sum((dp_g - np.mean(dp_g)) ** 2)
    r2_p3_s = 1.0 - np.sum((dp_s - np.poly1d(np.polyfit(b, dp_s, 3))(b)) ** 2) / np.sum((dp_s - np.mean(dp_s)) ** 2)
    p3_g = np.poly1d(np.polyfit(b, dp_g, 3))
    p3_s = np.poly1d(np.polyfit(b, dp_s, 3))

    w0_ok["b_bin"] = pd.qcut(w0_ok["b"], q=10)
    binned_b = w0_ok.groupby("b_bin", observed=False)[["a", "b", "dp_sft", "dp_grpo"]].mean()

    # Panel B: Fits on within-model initial log-odds L_base
    p_g = np.clip(p0 + dp_g / 100.0, 1e-4, 1 - 1e-4)
    p_s = np.clip(p0 + dp_s / 100.0, 1e-4, 1 - 1e-4)
    res_g = minimize(lambda x: np.mean((p_g - expit(x[0] * L0 + x[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda x: np.mean((p_s - expit(x[0] * L0 + x[1])) ** 2), [0.5, 0.0])

    w0_ok["L_bin"] = pd.qcut(w0_ok["L_base"], q=10)
    binned_L = w0_ok.groupby("L_bin", observed=False)[["a", "L_base", "dp_sft", "dp_grpo"]].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 9.6), gridspec_kw={"hspace": 0.44})

    # ==================== Panel A (Top: External Difficulty) ====================
    ax1.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)
    ax1.scatter(b, dp_g, s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.13, edgecolors="none", zorder=2)
    ax1.scatter(b, dp_s, s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.13, edgecolors="none", zorder=2)
    ax1.scatter(
        binned_b["b"], binned_b["dp_grpo"], s=binned_b["a"] * 110.0, color="#DC2626", marker="s",
        zorder=5, edgecolors="#7F1D1D", linewidth=1.2, label="RL (GRPO) — 10 Deciles"
    )
    ax1.scatter(
        binned_b["b"], binned_b["dp_sft"], s=binned_b["a"] * 110.0, color="#0284C7", marker="o",
        zorder=5, edgecolors="#0369A1", linewidth=1.2, label="SFT (CoT-SFT) — 10 Deciles"
    )

    b_dense = np.linspace(-2.5, 5.0, 300)
    ax1.plot(b_dense, p3_g(b_dense), "-", color="#DC2626", linewidth=2.8, zorder=6, label=f"RL Smooth Trend (R² = {r2_p3_g:.2f})")
    ax1.plot(b_dense, p3_s(b_dense), "-", color="#0284C7", linewidth=2.8, zorder=6, label=f"SFT Smooth Trend (R² = {r2_p3_s:.2f})")

    ax1.set_title("(A) Ordered by External 102-LLM Difficulty: Blurry Cloud (R² ≈ 0.09)", pad=10)
    ax1.set_xlabel("External Population Item Difficulty b (Calibrated Across 102 LLMs: Easy → Hard)")
    ax1.set_ylabel("Accuracy Change Δp (% pts) (↑)")
    ax1.set_xlim(-2.5, 5.0)
    ax1.set_ylim(-31, 36)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.6)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # ==================== Panel B (Bottom: Base Model's Own Initial Pass Rate) ====================
    ax2.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)

    # 25% 4-choice random guessing threshold (logit(0.25) = -1.0986)
    ax2.axvspan(-1.0986, -4.5, color="#FEE2E2", alpha=0.38, zorder=1)
    ax2.axvline(-1.0986, color="#DC2626", linestyle=":", linewidth=1.6, alpha=0.85, zorder=3)

    ax2.scatter(w0_ok["L_base"], dp_g, s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.13, edgecolors="none", zorder=2)
    ax2.scatter(w0_ok["L_base"], dp_s, s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.13, edgecolors="none", zorder=2)
    ax2.scatter(
        binned_L["L_base"], binned_L["dp_grpo"], s=binned_L["a"] * 110.0, color="#DC2626", marker="s",
        zorder=5, edgecolors="#7F1D1D", linewidth=1.2, label="RL Deciles"
    )
    ax2.scatter(
        binned_L["L_base"], binned_L["dp_sft"], s=binned_L["a"] * 110.0, color="#0284C7", marker="o",
        zorder=5, edgecolors="#0369A1", linewidth=1.2, label="SFT Deciles"
    )

    L_grid = np.linspace(3.5, -4.5, 400)
    pred_grpo_curve = (expit(res_g.x[0] * L_grid + res_g.x[1]) - expit(L_grid)) * 100.0
    pred_sft_curve  = (expit(res_s.x[0] * L_grid + res_s.x[1]) - expit(L_grid)) * 100.0
    classical_shift = (expit(L_grid + 0.5) - expit(L_grid)) * 100.0

    ax2.plot(L_grid, classical_shift, "--", color="#059669", linewidth=2.0, zorder=4, label=r"Classical Ability Shift ($\Delta\theta = +0.5$)")
    ax2.plot(L_grid, pred_grpo_curve, "-", color="#DC2626", linewidth=2.9, zorder=6, label=rf"RL Steepening ($\tau = {res_g.x[0]:.2f} > 1$)")
    ax2.plot(L_grid, pred_sft_curve, "-", color="#0284C7", linewidth=2.9, zorder=6, label=rf"SFT Flattening ($\tau = {res_s.x[0]:.2f} < 1$)")

    # Short, surgical direct labels
    ax2.text(-2.8, 31.5, "Random Guessing Zone (<25%)", ha="center", fontsize=8.8, fontweight="bold", color="#991B1B", zorder=7)
    ax2.text(-2.8, -17.5, "RL Hurts Hard Items\n(613 items drop)", ha="center", fontsize=8.6, fontweight="bold", color="#991B1B", zorder=7)
    ax2.annotate(
        "RL Boosts Known Items\n(+887 items rise)",
        xy=(0.6, 23.5), xytext=(0.45, 30.5),
        arrowprops=dict(arrowstyle="-|>", color="#15803D", lw=1.2),
        ha="center", fontsize=8.6, fontweight="bold", color="#15803D", zorder=7
    )
    ax2.text(-2.8, 12.5, "SFT Lifts Hard Items", ha="center", fontsize=8.6, fontweight="bold", color="#0369A1", zorder=7)

    ax2.set_title("(B) Re-Ordered by Base Model's Own Initial Pass Rate: Opposite S-Curves Emerge", pad=16)
    ax2.set_xlabel(r"Base Model Initial Log-Odds $\mu_{0j} = \mathrm{logit}(p_{\mathrm{base},j})$ (Easy $\rightarrow$ Hard)")
    ax2.set_ylabel("Accuracy Change Δp (% pts) (↑)")
    ax2.set_xlim(3.5, -4.5)
    ax2.set_ylim(-31, 38)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), borderpad=0.35, labelspacing=0.28, frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.0)
    ax2.grid(True, linestyle=":", alpha=0.45)

    ax2_top = ax2.twiny()
    p_ticks = np.array([0.90, 0.75, 0.50, 0.25, 0.10, 0.05])
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(logit(p_ticks))
    ax2_top.set_xticklabels(["90% (Easy)", "75%", "50%", "25% (Guessing)", "10%", "5% (Hard)"])
    ax2_top.set_xlabel(r"Base Model Initial Pass Probability $p_{\mathrm{base},j}$", fontsize=10.0, fontweight="bold", labelpad=6)

    save_fig(fig, "v4_Fig03_coordinate_translation_stacked.png")


# ==============================================================================
# FIGURE 4: Standalone 1x1 — Item Characteristic Curve Rotation (IRT <-> LLM Bridge)
# ==============================================================================
def make_v4_fig04_icc_slope_translation():
    fig, ax1 = plt.subplots(figsize=(9.2, 5.8))

    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    p0 = np.clip(t9["p_key_base"].values, 1e-4, 1 - 1e-4)
    L0 = logit(p0)
    dp_g = t9[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean(axis=1).values
    dp_s = t9[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean(axis=1).values
    p_g = np.clip(p0 + dp_g, 1e-4, 1 - 1e-4)
    p_s = np.clip(p0 + dp_s, 1e-4, 1 - 1e-4)

    res_g = minimize(lambda c: np.mean((p_g - expit(c[0] * L0 + c[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda c: np.mean((p_s - expit(c[0] * L0 + c[1])) ** 2), [0.5, 0.0])

    w0_ok["L_base"] = L0
    w0_ok["L_bin"] = pd.qcut(w0_ok["L_base"], q=10)
    binned_L = w0_ok.groupby("L_bin", observed=False)[["a", "L_base"]].mean()
    binned_L["p_g"] = np.clip(t9.groupby(w0_ok["L_bin"], observed=False)[["dp_key_rl_0_666", "dp_key_rl_1_666", "dp_key_rl_2_666"]].mean().mean(axis=1) + t9.groupby(w0_ok["L_bin"], observed=False)["p_key_base"].mean(), 0, 1) * 100.0
    binned_L["p_s"] = np.clip(t9.groupby(w0_ok["L_bin"], observed=False)[["dp_key_sft_0_70", "dp_key_sft_1_70", "dp_key_sft_2_70"]].mean().mean(axis=1) + t9.groupby(w0_ok["L_bin"], observed=False)["p_key_base"].mean(), 0, 1) * 100.0

    L_grid = np.linspace(3.5, -4.5, 400)
    base_icc = expit(L_grid) * 100.0
    classical_icc = expit(L_grid + 0.5) * 100.0
    grpo_icc = expit(res_g.x[0] * L_grid + res_g.x[1]) * 100.0
    sft_icc = expit(res_s.x[0] * L_grid + res_s.x[1]) * 100.0

    L_pivot = res_g.x[1] / (1.0 - res_g.x[0])
    p_pivot = expit(L_pivot) * 100.0

    # Shaded Wedges
    mask_easy = L_grid >= L_pivot
    mask_hard = L_grid <= L_pivot
    ax1.fill_between(L_grid[mask_easy], base_icc[mask_easy], grpo_icc[mask_easy], color="#DCFCE7", alpha=0.55, zorder=1)
    ax1.fill_between(L_grid[mask_hard], grpo_icc[mask_hard], base_icc[mask_hard], color="#FEE2E2", alpha=0.65, zorder=1)
    ax1.axvspan(L_pivot, -4.5, color="#FEE2E2", alpha=0.18, zorder=0)
    ax1.axvline(L_pivot, color="#DC2626", linestyle=":", linewidth=1.6, alpha=0.85, zorder=2)

    # Curves with explicit IRT <-> LLM Translation in the legend
    ax1.plot(L_grid, base_icc, "-", color="#475569", linewidth=2.3, zorder=3,
             label=r"Base Model ($\tau = 1.00$)")
    ax1.plot(L_grid, classical_icc, "--", color="#059669", linewidth=2.1, zorder=4,
             label=r"Classical IRT Shift ($\Delta\theta = +0.5, \tau = 1.0$) $\leftrightarrow$ Uniform Gain")
    ax1.plot(L_grid, grpo_icc, "-", color="#DC2626", linewidth=3.0, zorder=5,
             label=rf"RL Steepening ($\tau = {res_g.x[0]:.2f} > 1$) $\leftrightarrow$ Policy Sharpening ($1/T$)")
    ax1.scatter(binned_L["L_base"], binned_L["p_g"], s=48, color="#DC2626", marker="s", edgecolors="#7F1D1D", linewidth=1.1, zorder=6)
    ax1.plot(L_grid, sft_icc, "-", color="#0284C7", linewidth=3.0, zorder=5,
             label=rf"SFT Flattening ($\tau = {res_s.x[0]:.2f} < 1$) $\leftrightarrow$ Distribution Broadening")
    ax1.scatter(binned_L["L_base"], binned_L["p_s"], s=48, color="#0284C7", marker="o", edgecolors="#0369A1", linewidth=1.1, zorder=6)

    # Pivot Marker
    ax1.scatter([L_pivot], [p_pivot], s=105, color="#FDE047", marker="o", edgecolors="#0F172A", linewidth=2.0, zorder=8)

    # Clean, direct labels on the wedges
    ax1.annotate(
        "Known Items Sharpened (+887 items)\n→ Wins Single-Draw Pass@1",
        xy=(1.05, 84.0), xytext=(-0.35, 93.0),
        arrowprops=dict(arrowstyle="-|>", color="#166534", lw=1.3),
        ha="center", fontsize=8.8, fontweight="bold", color="#166534", zorder=8
    )
    ax1.text(
        -2.95, 43.0,
        "Hard Items Crushed (-613 items)\n→ Collapses Multi-Draw Pass@16",
        ha="center", fontsize=8.8, fontweight="bold", color="#991B1B", zorder=8
    )
    ax1.annotate(
        "Random-Guessing Pivot (~25%)",
        xy=(L_pivot, p_pivot), xytext=(0.35, 25.5),
        arrowprops=dict(arrowstyle="-|>", color="#0F172A", lw=1.3),
        ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A", zorder=8
    )

    ax1.set_title("Bridging IRT & LLMs: Post-Training Rotates the S-Curve Slope (τ ↔ 1/T)", pad=16)
    ax1.set_xlabel(r"Base Model Initial Log-Odds $\mu_{0j} = \mathrm{logit}(p_{\mathrm{base},j})$ (Easy $\rightarrow$ Hard)")
    ax1.set_ylabel(r"Post-Training Pass Probability $p_j$ (%)")
    ax1.set_xlim(3.5, -4.5)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.45)

    ax1_top = ax1.twiny()
    p_ticks = np.array([0.90, 0.75, 0.50, 0.25, 0.10, 0.05])
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(logit(p_ticks))
    ax1_top.set_xticklabels(["90% (Easy)", "75%", "50%", "25% (Guessing)", "10%", "5% (Hard)"])
    ax1_top.set_xlabel(r"Base Model Initial Pass Probability $p_{\mathrm{base},j}$", fontsize=10.0, fontweight="bold", labelpad=6)

    save_fig(fig, "v4_Fig04_icc_slope_translation.png")


# ==============================================================================
# FIGURE 5: 1x2 — IRT's Added Value Beyond 1D Global Slope
# ==============================================================================
def make_v4_fig05_irt_added_value_diagnostics():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.4, 5.6), gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.26}
    )

    # -------------------------------------------------------------------------
    # Panel A: Subject-Level Separation Across 15 MMLU Subjects + 1,472 Background Items
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

    # Background 1,472 items
    it10_idx = it10.set_index("item_id").loc[w0_ok["bank_row_id"]]
    p_base_it = np.clip(it10_idx["p_ref"].values, 1e-4, 1 - 1e-4)
    L_base_it = logit(p_base_it)
    w0_ok["c_sft"] = w0_ok["subject"].map(c_sft.to_dict())
    w0_ok["c_grpo"] = w0_ok["subject"].map(c_grpo.to_dict())
    dp_sft_item = (expit(L_base_it + w0_ok["c_sft"].values + w0_ok["a"].values * (th_sft - th_base)) - p_base_it) * 100.0
    dp_grpo_item = (expit(L_base_it + w0_ok["c_grpo"].values + w0_ok["a"].values * (th_grpo - th_base)) - p_base_it) * 100.0

    ax1.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)
    ax1.scatter(w0_ok["b"], dp_sft_item, s=w0_ok["a"] * 15.0, color="#0284C7", alpha=0.09, edgecolors="none", zorder=2)
    ax1.scatter(w0_ok["b"], dp_grpo_item, s=w0_ok["a"] * 15.0, color="#DC2626", alpha=0.09, edgecolors="none", zorder=2)

    sizes = df_sub["a"] * 105.0
    ax1.scatter(
        df_sub["b"], df_sub["dp_sft"], s=sizes, color="#0284C7", alpha=0.88,
        edgecolors="#0369A1", linewidth=1.1, zorder=4,
        label=r"CoT-SFT (Wins hard, low-$a$ math/STEM subjects)"
    )
    ax1.scatter(
        df_sub["b"], df_sub["dp_grpo"], s=sizes, color="#DC2626", marker="s", alpha=0.88,
        edgecolors="#991B1B", linewidth=1.1, zorder=4,
        label=r"RL / GRPO (Wins easy, high-$a$ factual subjects)"
    )

    x_grid = np.linspace(df_sub["b"].min() - 0.15, df_sub["b"].max() + 0.15, 100)
    m_sft, c_sft_l = np.polyfit(df_sub["b"], df_sub["dp_sft"], 1)
    m_grpo, c_grpo_l = np.polyfit(df_sub["b"], df_sub["dp_grpo"], 1)
    ax1.plot(x_grid, m_sft * x_grid + c_sft_l, "-", color="#0284C7", linewidth=2.3, alpha=0.85, zorder=3)
    ax1.plot(x_grid, m_grpo * x_grid + c_grpo_l, "-", color="#DC2626", linewidth=2.3, alpha=0.85, zorder=3)

    # Clean non-overlapping subject labels
    r_mkt = df_sub.loc["marketing"]
    ax1.annotate(
        "Marketing (a = 1.45)",
        xy=(r_mkt["b"], r_mkt["dp_grpo"]), xytext=(-1.18, 13.8),
        arrowprops=dict(arrowstyle="-|>", color="#991B1B", lw=1.1),
        fontsize=8.6, fontweight="bold", color="#991B1B", zorder=6
    )
    r_soc = df_sub.loc["sociology"]
    ax1.text(r_soc["b"] + 0.08, r_soc["dp_grpo"] + 0.8, "Sociology (a = 1.28)", fontsize=8.6, fontweight="bold", color="#991B1B", zorder=6)

    r_em = df_sub.loc["elementary_mathematics"]
    ax1.text(r_em["b"] - 0.98, r_em["dp_sft"] + 2.0, "Elementary Math (a = 0.45)", fontsize=8.6, fontweight="bold", color="#0369A1", zorder=6)

    r_hm = df_sub.loc["high_school_mathematics"]
    ax1.text(r_hm["b"] - 0.88, r_hm["dp_sft"] + 2.0, "High School Math (a = 0.28)", fontsize=8.6, fontweight="bold", color="#0369A1", zorder=6)

    ax1.text(
        -1.18, 36.5, r"Bubble area $\propto$ IRT discrimination $a$" + "\n(Factual recall weighted 5× over math)",
        fontsize=8.6, fontweight="bold", color="#334155"
    )

    ax1.set_title("(A) Macro 2D IRT: Why Scalar Scores Overweight Factual Recall", pad=10)
    ax1.set_xlabel("Subject Mean IRT Difficulty b (Easy → Hard)")
    ax1.set_ylabel("Accuracy Gain Over Base: ΔPass@1 (% pts) (↑)")
    ax1.set_xlim(-1.25, 2.65)
    ax1.set_ylim(-14, 44)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=8.3)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Cross-Seed Item Residual Repeatability (Pure & Self-Explanatory)
    # -------------------------------------------------------------------------
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

    ax2.axhline(0, color="#94A3B8", linestyle=":", linewidth=1.1, zorder=1)
    ax2.axvline(0, color="#94A3B8", linestyle=":", linewidth=1.1, zorder=1)
    ax2.plot([-42, 42], [-42, 42], "--", color="#1E293B", linewidth=1.8, alpha=0.75, zorder=3, label="Exact Cross-Seed Identity (y = x)")

    null_circle = Ellipse(
        (0, 0), width=36, height=36, edgecolor="#64748B", facecolor="#F1F5F9",
        linestyle=":", linewidth=1.6, alpha=0.55, zorder=2, label="Random Noise Null Circle (r ≤ 0.19)"
    )
    ax2.add_patch(null_circle)

    ax2.scatter(res_grpo_seeds[0], res_grpo_seeds[1], s=14, color="#DC2626", alpha=0.18, edgecolors="none", zorder=3)
    ax2.scatter(res_sft_seeds[0], res_sft_seeds[1], s=14, color="#0284C7", alpha=0.18, edgecolors="none", zorder=3)

    df_res = pd.DataFrame({"g0": res_grpo_seeds[0], "g1": res_grpo_seeds[1], "s0": res_sft_seeds[0], "s1": res_sft_seeds[1]})
    df_res["g_bin"] = pd.qcut(df_res["g0"], q=10)
    df_res["s_bin"] = pd.qcut(df_res["s0"], q=10)
    bin_g = df_res.groupby("g_bin", observed=False)[["g0", "g1"]].mean()
    bin_s = df_res.groupby("s_bin", observed=False)[["s0", "s1"]].mean()

    ax2.plot(bin_g["g0"], bin_g["g1"], "-s", color="#DC2626", linewidth=2.8, markersize=6.8, markeredgecolor="#7F1D1D", zorder=6,
             label=f"RL Item Residuals Across Seeds (r = {r_grpo:.2f})")
    ax2.plot(bin_s["s0"], bin_s["s1"], "-o", color="#0284C7", linewidth=2.8, markersize=6.8, markeredgecolor="#0369A1", zorder=6,
             label=f"SFT Item Residuals Across Seeds (r = {r_sft:.2f})")

    ax2.set_xlim(-42, 42)
    ax2.set_ylim(-42, 42)
    ax2.set_title("(B) Micro Item Residuals: Reproducible Skill Clusters Across Seeds", pad=10)
    ax2.set_xlabel("Seed 1 Item Residual Gain After Subtracting 1D Curve (% pts)")
    ax2.set_ylabel("Seed 2 Item Residual Gain After Subtracting 1D Curve (% pts)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.4)
    ax2.grid(True, linestyle=":", alpha=0.45)

    save_fig(fig, "v4_Fig05_irt_added_value_diagnostics.png")


# ==============================================================================
# FIGURE 6: 1x2 — Actionable Control Payoff & 2-Tier Online Slope Regulation
# ==============================================================================
def make_v4_fig06_actionable_control_payoff():
    fig = plt.figure(figsize=(15.6, 5.7))
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
    ax1.scatter([base_x], [base_y], s=130, color="#475569", marker="o", edgecolors="#0F172A", linewidth=1.6, zorder=6, label="Base Model (38.2%, 93.0%)")
    ax1.text(base_x - 0.3, base_y + 1.3, "Base (38.2%, 93.0%)", fontsize=8.6, fontweight="bold", color="#334155")

    # 2. Standalone RL (u333)
    grpo_states = ["q1_grpo_0", "q1_grpo_1", "q1_grpo_2"]
    grpo_xs = m1_p.loc[grpo_states, "mean_at_16"].to_numpy()
    grpo_ys = m1_p.loc[grpo_states, "pass_at_16"].to_numpy()
    ax1.scatter(grpo_xs, grpo_ys, color="#DC2626", s=34, alpha=0.45, zorder=5)
    grpo_cx, grpo_cy = np.mean(grpo_xs), np.mean(grpo_ys)
    ax1.scatter([grpo_cx], [grpo_cy], s=130, color="#DC2626", marker="s", edgecolors="#7F1D1D", linewidth=1.6, zorder=6, label="Standalone RL (46.9%, 69.4%)")

    ax1.axhspan(64.5, 73.2, color="#FEE2E2", alpha=0.38, zorder=1)
    ax1.text(
        46.65, 66.2,
        "Standalone RL Mode Collapse (-23.6% vs. Base)",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#991B1B", zorder=6
    )

    arrow_grpo = FancyArrowPatch(
        (base_x, base_y), (grpo_cx, grpo_cy),
        connectionstyle="arc3,rad=-0.12", color="#DC2626",
        arrowstyle="-|>", mutation_scale=17, linewidth=2.4, zorder=5
    )
    ax1.add_patch(arrow_grpo)

    # 3. CoT-SFT Base
    sft_states = ["t9_sft_0_70", "t9_sft_1_70", "t9_sft_2_70"]
    sft_xs = m1_p.loc[sft_states, "mean_at_16"].to_numpy()
    sft_ys = m1_p.loc[sft_states, "pass_at_16"].to_numpy()
    sft_cx, sft_cy = np.mean(sft_xs), np.mean(sft_ys)
    ax1.scatter([sft_cx], [sft_cy], s=130, color="#0284C7", marker="o", edgecolors="#0369A1", linewidth=1.6, zorder=6, label="CoT-SFT (T=0.7) (43.9%, 96.6%)")
    ax1.text(sft_cx - 0.3, sft_cy + 1.3, "CoT-SFT (43.9%, 96.6%)", ha="right", fontsize=8.6, fontweight="bold", color="#0369A1")

    # 4. Zero-Training SFT T=0.2
    t02_states = ["m1a_sft_t02_0", "m1a_sft_t02_1", "m1a_sft_t02_2"]
    t02_xs = m1_p.loc[t02_states, "mean_at_16"].to_numpy()
    t02_ys = m1_p.loc[t02_states, "pass_at_16"].to_numpy()
    t02_cx, t02_cy = np.mean(t02_xs), np.mean(t02_ys)
    ax1.scatter([t02_cx], [t02_cy], s=135, color="#059669", marker="D", edgecolors="#047857", linewidth=1.6, zorder=6, label="SFT + Low Temp T=0.2 (0 RL steps: 47.5%, 87.9%)")

    arrow_t02 = FancyArrowPatch(
        (sft_cx, sft_cy), (t02_cx, t02_cy),
        connectionstyle="arc3,rad=-0.12", color="#059669",
        arrowstyle="-|>", mutation_scale=17, linewidth=2.4, zorder=5
    )
    ax1.add_patch(arrow_t02)

    # 5. Two-Stage SFT -> RL & Informative Band
    m1b_u333_states = [f"m1b_sft_grpo_{s}" for s in range(3)]
    m1b_33_x, m1b_33_y = m1_p.loc[m1b_u333_states, "mean_at_16"].mean(), m1_p.loc[m1b_u333_states, "pass_at_16"].mean()
    ax1.scatter([m1b_33_x], [m1b_33_y], s=125, color="#7C3AED", marker="s", edgecolors="#5B21B6", linewidth=1.5, zorder=6, label="Two-Stage SFT → RL (46.7%, 94.9%)")

    m1c_u333_states = [f"m1c_sft_zpd_{s}" for s in range(3)]
    m1c_33_x, m1c_p_y = m1_p.loc[m1c_u333_states, "mean_at_16"].mean(), m1_p.loc[m1c_u333_states, "pass_at_16"].mean()
    ax1.scatter([m1c_33_x], [m1c_p_y], s=125, color="#D97706", marker="^", edgecolors="#B45309", linewidth=1.5, zorder=6, label="Two-Stage SFT → RL [2..6/8 Band] (46.5%, 94.2%)")

    arrow_sft_rl = FancyArrowPatch(
        (sft_cx, sft_cy), (m1b_33_x, m1b_33_y),
        connectionstyle="arc3,rad=-0.06", color="#7C3AED",
        arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=5
    )
    ax1.add_patch(arrow_sft_rl)

    # Shaded Optimal Tradeoff Region (Top Right)
    ax1.fill_between([44.9, 48.2], 93.0, 97.2, color="#F5F3FF", alpha=0.65, zorder=1)
    pareto_box = patches.Rectangle((44.9, 93.0), 3.3, 4.2, fill=False, edgecolor="#7C3AED", linestyle="--", linewidth=1.5, zorder=2)
    ax1.add_patch(pareto_box)

    ax1.text(46.55, 98.2, "Optimal Region (High Pass@1 + Pass@16)", ha="center", fontsize=8.6, fontweight="bold", color="#5B21B6")
    ax1.text(46.9, 83.6, "Zero-Training (T=0.2):\nBeats RL in 3/3 seeds", ha="center", fontsize=8.4, fontweight="bold", color="#047857")

    ax1.set_xlim(36.8, 49.0)
    ax1.set_ylim(64.5, 102.5)
    ax1.set_title("(A) Accuracy vs. Coverage: Escaping RL Mode Collapse", pad=11)
    ax1.set_xlabel("Single-Draw Accuracy: Pass@1 (%) (↑)")
    ax1.set_ylabel("Multi-Draw Coverage: Pass@16 (%) (↑)")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=8.0)
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
    ax2_top.plot(r6_updates, tau_r6, "-o", color="#DC2626", linewidth=2.8, markersize=5.8, zorder=5)
    ax2_top.scatter([333, 333, 333], [3.275, 3.216, 3.303], color="#7F1D1D", s=28, zorder=6)
    ax2_top.text(215, 2.58, "Unconstrained RL: τ → 3.28\n(Over-sharpening into mode collapse)", fontsize=8.6, fontweight="bold", color="#991B1B")

    ax2_top.set_ylim(2.20, 3.65)
    ax2_top.set_yticks([2.5, 3.0, 3.5])
    ax2_top.spines["bottom"].set_visible(False)
    ax2_top.tick_params(bottom=False, labelbottom=False)
    ax2_top.grid(True, linestyle=":", alpha=0.45)
    ax2_top.set_title("(B) Tracking & Controlling IRT Slope τ Across RL Steps", pad=11)

    # --- Bottom Tier: Controlled Regimes (y in [0.95, 1.78]) ---
    # Also plot Unconstrained RL starting from (0, 1.00) so the trajectory across the break is continuous!
    ax2_bot.plot(r6_updates, tau_r6, "-o", color="#DC2626", linewidth=2.4, markersize=5.5, alpha=0.75, zorder=4, label="Unconstrained RL (Shoots to τ = 3.28)")
    ax2_bot.axhline(1.664, color="#059669", linestyle="--", linewidth=2.1, zorder=4, label="SFT + Low Temp (T=0.2): τ = 1.66")
    ax2_bot.scatter([0, 0, 0], [1.663, 1.710, 1.618], color="#047857", marker="D", s=28, zorder=6)

    ax2_bot.plot(t_comp, tau_band, "-s", color="#D97706", linewidth=2.5, markersize=5.8, zorder=5, label="SFT → RL [2..6/8 Informative Band]")
    ax2_bot.plot(t_comp, tau_unf,  "--^", color="#7C3AED", linewidth=2.2, markersize=5.8, zorder=5, label="SFT → RL [All Items, Unfiltered]")

    ax2_bot.scatter([83, 83, 83], [1.1028, 1.1441, 1.1079], color="#B45309", s=20, zorder=6)
    ax2_bot.scatter([333, 333, 333], [1.2278, 1.2938, 1.1759], color="#B45309", s=20, zorder=6)
    ax2_bot.scatter([83, 83, 83], [1.0712, 1.1047, 1.0650], color="#5B21B6", s=20, zorder=6)
    ax2_bot.scatter([333, 333, 333], [1.2126, 1.2177, 1.1470], color="#5B21B6", s=20, zorder=6)

    ax2_bot.text(195, 1.69, "Zero-Training Temp T=0.2 (τ = 1.66)", fontsize=8.4, fontweight="bold", color="#047857")
    ax2_bot.annotate(
        "[2..6/8] Band Steepens τ (+0.04, p < 0.001)\n& Cuts Dead Rollouts 34% → 23%",
        xy=(333, 1.233), xytext=(245, 1.03),
        arrowprops=dict(arrowstyle="-|>", color="#B45309", lw=1.2),
        ha="center", fontsize=8.1, fontweight="bold", color="#B45309"
    )

    ax2_bot.set_ylim(0.92, 1.78)
    ax2_bot.set_yticks([1.0, 1.2, 1.4, 1.6])
    ax2_bot.spines["top"].set_visible(False)
    ax2_bot.grid(True, linestyle=":", alpha=0.45)
    ax2_bot.set_xticks([0, 83, 166, 250, 333])
    ax2_bot.set_xlim(-15, 365)
    ax2_bot.set_xlabel("RL Training Updates")
    ax2_bot.legend(loc="upper left", bbox_to_anchor=(0.08, 0.84), frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=7.8)

    fig.text(0.505, 0.52, "IRT Slope τ (Policy Steepening Factor)", va="center", rotation="vertical", fontsize=10.8, fontweight="bold")

    d_slash = 0.015
    kwargs = dict(transform=ax2_top.transAxes, color="#475569", clip_on=False, linewidth=1.2)
    ax2_top.plot((-d_slash, +d_slash), (-d_slash * 1.5, +d_slash * 1.5), **kwargs)
    kwargs.update(transform=ax2_bot.transAxes)
    ax2_bot.plot((-d_slash, +d_slash), (1 - d_slash * 1.5, 1 + d_slash * 1.5), **kwargs)

    save_fig(fig, "v4_Fig06_actionable_control_payoff.png")


# ==============================================================================
# FIGURE 7: Standalone 1x1 — Minimalist Closed-Loop Dynamic IRT Schematic
# ==============================================================================
def make_v4_fig07_dynamic_irt_schematic():
    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    canvas = FancyBboxPatch((1.0, 1.0), 98.0, 98.0, boxstyle="round,pad=0.6,rounding_size=2.0", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2, zorder=1)
    ax.add_patch(canvas)

    # Title with generous clearance above top banners
    ax.text(50.0, 95.8, "From Static Population IRT to Dynamic Within-Model IRT", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#0F172A", zorder=3)

    # Top Comparative Banners (y = 80.5 to 90.5)
    card_static = FancyBboxPatch((4.0, 80.5), 44.0, 9.8, boxstyle="round,pad=0.4,rounding_size=1.2", facecolor="#FEF2F2", edgecolor="#F87171", linewidth=1.3, zorder=2)
    ax.add_patch(card_static)
    ax.text(26.0, 87.3, "Classical Population IRT (Static)", ha="center", fontsize=9.8, fontweight="bold", color="#991B1B", zorder=3)
    ax.text(26.0, 83.0, "Fixed 102-LLM difficulty b  |  Single scalar ability θ", ha="center", fontsize=8.8, color="#7F1D1D", zorder=3)

    card_dynamic = FancyBboxPatch((52.0, 80.5), 44.0, 9.8, boxstyle="round,pad=0.4,rounding_size=1.2", facecolor="#F0FDF4", edgecolor="#4ADE80", linewidth=1.3, zorder=2)
    ax.add_patch(card_dynamic)
    ax.text(74.0, 87.3, "Dynamic Within-Model IRT (Ours)", ha="center", fontsize=9.8, fontweight="bold", color="#166534", zorder=3)
    ax.text(74.0, 83.0, "Anchored on Base Model  |  Tracks slope τ, shift δ & skills s", ha="center", fontsize=8.8, color="#14532D", zorder=3)

    # Central State Hub
    hub = FancyBboxPatch((35.5, 33.5), 29.0, 23.0, boxstyle="round,pad=0.5,rounding_size=2.0", facecolor="#0F172A", edgecolor="#38BDF8", linewidth=2.0, zorder=5)
    ax.add_patch(hub)
    ax.text(50.0, 51.0, "DYNAMIC IRT STATE", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#38BDF8", zorder=6)
    ax.text(50.0, 44.5, r"$x_t = (\tau_t, \, \delta_t, \, s_t)$", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#F8FAFC", zorder=6)
    ax.text(50.0, 38.0, "Slope τ  |  Shift δ  |  Skill Profile s", ha="center", va="center", fontsize=8.6, color="#CBD5E1", zorder=6)

    # 4 Minimalist Outer Nodes (Big, Readable Slide Text — Max 2 Short Lines Each)
    # 1. Top-Left
    n1 = FancyBboxPatch((4.0, 48.0), 27.5, 25.5, boxstyle="round,pad=0.5,rounding_size=1.5", facecolor="#FFFFFF", edgecolor="#0284C7", linewidth=1.8, zorder=3)
    ax.add_patch(n1)
    ax.text(17.75, 68.5, "1. Base Model Coordinate", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0369A1", zorder=4)
    ax.text(17.75, 60.5, r"$\mu_{0j} = \mathrm{logit}(p_{\mathrm{base},j})$", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A", zorder=4)
    ax.text(17.75, 53.0, "Replaces external difficulty;\nisolates <25% guessing zone", ha="center", va="center", fontsize=8.8, color="#334155", zorder=4)

    # 2. Top-Right
    n2 = FancyBboxPatch((68.5, 48.0), 27.5, 25.5, boxstyle="round,pad=0.5,rounding_size=1.5", facecolor="#FFFFFF", edgecolor="#4F46E5", linewidth=1.8, zorder=3)
    ax.add_patch(n2)
    ax.text(82.25, 68.5, "2. Practice Update (Write)", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#4338CA", zorder=4)
    ax.text(82.25, 60.5, "Gradient Rollout Update", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#0F172A", zorder=4)
    ax.text(82.25, 53.0, "Updates slope τ and skill profile s\nfrom batch rollout outcomes", ha="center", va="center", fontsize=8.8, color="#334155", zorder=4)

    # 3. Bottom-Right
    n3 = FancyBboxPatch((68.5, 6.5), 27.5, 25.5, boxstyle="round,pad=0.5,rounding_size=1.5", facecolor="#FFFFFF", edgecolor="#059669", linewidth=1.8, zorder=3)
    ax.add_patch(n3)
    ax.text(82.25, 27.0, "3. Benchmark Probe (Read)", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#047857", zorder=4)
    ax.text(82.25, 19.0, r"$p_{j,t} = \sigma(\tau_t \mu_{0j} + \delta_t + a_j^\top s_t)$", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#0F172A", zorder=4)
    ax.text(82.25, 11.5, "Predicts unseen item accuracy\nonline (73.5% accuracy)", ha="center", va="center", fontsize=8.8, color="#334155", zorder=4)

    # 4. Bottom-Left
    n4 = FancyBboxPatch((4.0, 6.5), 27.5, 25.5, boxstyle="round,pad=0.5,rounding_size=1.5", facecolor="#FFFFFF", edgecolor="#D97706", linewidth=1.8, zorder=3)
    ax.add_patch(n4)
    ax.text(17.75, 27.0, "4. Active Closed-Loop Control", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#B45309", zorder=4)
    ax.text(17.75, 19.0, r"Band [2..6/8] & Temp $T \propto 1/\tau$", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#0F172A", zorder=4)
    ax.text(17.75, 11.5, "Drops <25% zone; cuts dead\nrollouts 34% → 23%", ha="center", va="center", fontsize=8.8, color="#334155", zorder=4)

    # Arrows
    ax.add_patch(FancyArrowPatch((32.0, 63.0), (68.0, 63.0), color="#0284C7", arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=7))
    ax.text(50.0, 65.2, "Initial Log-Odds & Skill Loadings", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#0369A1", zorder=8)

    ax.add_patch(FancyArrowPatch((82.25, 47.5), (82.25, 32.5), color="#4F46E5", arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=7))
    ax.text(84.2, 40.0, "Online State\nFilter", ha="left", va="center", fontsize=8.4, fontweight="bold", color="#4338CA", zorder=8)

    ax.add_patch(FancyArrowPatch((68.0, 17.0), (32.0, 17.0), color="#059669", arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=7))
    ax.text(50.0, 19.2, "Updated Item Pass Probabilities", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#047857", zorder=8)

    ax.add_patch(FancyArrowPatch((17.75, 32.5), (17.75, 47.5), color="#D97706", arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=7))
    ax.text(15.8, 40.0, "Next Item Batch\n& Temp T", ha="right", va="center", fontsize=8.4, fontweight="bold", color="#B45309", zorder=8)

    save_fig(fig, "v4_Fig07_dynamic_irt_schematic.png")


if __name__ == "__main__":
    print("Generating 7 Supervisor-Polished V4 Presentation Figures...")
    make_v4_fig01_empirical_passk_puzzle()
    make_v4_fig02_static_population_irt_fails()
    make_v4_fig03_coordinate_translation_stacked()
    make_v4_fig04_icc_slope_translation()
    make_v4_fig05_irt_added_value_diagnostics()
    make_v4_fig06_actionable_control_payoff()
    make_v4_fig07_dynamic_irt_schematic()
    print("All 7 V4 presentation figures generated successfully!")
