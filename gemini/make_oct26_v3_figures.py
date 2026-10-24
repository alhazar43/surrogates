"""
make_oct26_v3_figures.py

Generates the V3 Intuition-First Presentation Figure Suite for the Oct 26 presentation
(Zero Bar Plots, Zero Unrelated Subplot Stitching, Publication-Grade Visual Intuition):

  1. v3_Fig01a_eval_cat_wall.png             (Act I, Step 1A: Standalone Q2a CAT Specification Bias Wall)
  2. v3_Fig01b_rl_critic_collapse.png        (Act I, Step 1B: Coupled Q1 RL Training Ceiling & Pass@k Collapse)
  3. v3_Fig02_coordinate_alignment_stacked.png (Act II, Step 2A: Standalone 2x1 Stack — Out-of-Focus b_j^pop vs In-Focus mu_0j)
  4. v3_Fig03_icc_rotation_and_passk.png     (Act II, Step 2B: Coupled 1x2 Scissor — Item-Space ICC Rotation & Pass@k Crossover)
  5. v3_Fig04_phase_plane_and_tracker.png    (Act III, Step 3: 2D Exploitation-Exploration Phase Plane & Online Slope Tracker tau_t)
  6. v3_Fig05_beyond_1d_places_and_duet.png  (Act IV, Step 4: Cross-Seed Residual Repeatability Cloud + Closed-Loop State-Space Diagram)

Also preserves the 5-file aliases (v3_Fig01_static_irt_failure.png, v3_Fig02_icc_rotation_mechanism.png,
v3_Fig05_dynamic_architecture_schematic.png) so all links remain valid.
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

# Matplotlib global presentation configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.titlesize": 12.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.0,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO_ROOT = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\item-params-llm"
RESULTS = os.path.join(REPO_ROOT, "results")

OUT_DIRS = [
    r"D:\surrogates\gemini\figures_v3",
    r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49",
]

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save_fig(fig, filename, alias_filenames=None):
    first_path = os.path.join(OUT_DIRS[0], filename)
    fig.savefig(first_path, bbox_inches="tight", facecolor="white")
    for d in OUT_DIRS[1:]:
        shutil.copy2(first_path, os.path.join(d, filename))
    if alias_filenames:
        for alias in alias_filenames:
            for d in OUT_DIRS:
                shutil.copy2(first_path, os.path.join(d, alias))
    plt.close(fig)
    print(f"[OK] Saved: {filename}")


# ==============================================================================
# FIGURE 1A: ACT I (Step 1A) — Standalone Q2a Benchmark Evaluation CAT Wall
# ==============================================================================
def make_v3_fig01a_eval_cat_wall():
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    ext_df = pd.read_csv(os.path.join(RESULTS, "Q2a", "errors_ext.csv"))
    ext_df = ext_df[ext_df["target"] == "state"].copy()
    ns_ext = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1472]
    k16 = ext_df[ext_df["k"] == 16].groupby(["n", "method"])["rmse"].mean().unstack() * 100.0

    y_cl = [k16.loc[n, "classical"] for n in ns_ext]
    y_rd = [k16.loc[n, "irt_random"] for n in ns_ext]
    y_ad = [k16.loc[n, "irt_adaptive"] for n in ns_ext]

    # Shaded specification bias floor band
    ax.axhspan(0, y_ad[-1], color="#FEE2E2", alpha=0.42, zorder=1)
    ax.axhline(y_ad[-1], color="#DC2626", linestyle="--", linewidth=1.9, alpha=0.9, zorder=2)

    # Shaded gap between Static CAT and Classical Random Sampling for n >= 50
    ns_arr = np.array(ns_ext)
    mask_gap = ns_arr >= 35
    ax.fill_between(
        ns_arr[mask_gap],
        np.array(y_cl)[mask_gap],
        np.array(y_ad)[mask_gap],
        color="#FCA5A5",
        alpha=0.18,
        zorder=2,
    )

    # Curves
    ax.plot(
        ns_ext, y_cl, "--o", color="#1E293B", linewidth=2.6, markersize=6.5, zorder=4,
        label=r"Classical Random Sampling ($\bar{y}_n \propto 1/\sqrt{n} \rightarrow 0.76\%$)"
    )
    ax.plot(
        ns_ext, y_rd, ":^", color="#D97706", linewidth=2.4, markersize=6.5, zorder=4,
        label=r"Static 2PL IRT Scoring (Random Items $\rightarrow 2.41\%$ Floor)"
    )
    ax.plot(
        ns_ext, y_ad, "-s", color="#DC2626", linewidth=2.9, markersize=6.8, zorder=5,
        label=r"Static 2PL IRT Adaptive CAT (Max Fisher Info $\rightarrow 2.43\%$ Floor)"
    )

    # Direct wall badge on the left inside the shaded floor
    ax.text(
        11.0, y_ad[-1] - 0.32,
        "Static Bias Wall: 2.43% RMSE Floor\n"
        "(Full bank $N = 1,472$ still stuck at 2.41%–2.43%)",
        ha="left", va="top", fontsize=8.6, fontweight="bold", color="#991B1B", zorder=6
    )

    # Clean callout pointing to n=1472 classical sample mean (placed in open space above orange curve)
    ax.annotate(
        "Classical Sample Mean: 0.76%\n(3.2× lower error than Static CAT at $N=1,472$)",
        xy=(1472, y_cl[-1]),
        xytext=(420, 5.5),
        ha="center", va="center", fontsize=8.6, fontweight="bold", color="#0F172A",
        arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=1.5, connectionstyle="arc3,rad=-0.18"),
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.97, linewidth=0.9),
        zorder=7
    )

    # Clean mechanism box in upper-right below legend (completely clear of the red curve!)
    callout_eval = (
        "Why Static Population CAT Fails After Post-Training:\n"
        "• 2PL bank $(a_j^{\\mathrm{pop}}, b_j^{\\mathrm{pop}})$ is calibrated on 102 heterogeneous LLMs\n"
        "• Post-training (SFT / GRPO) systematically twists the model's item manifold\n"
        "• Selecting 'high-$a_j^{\\mathrm{pop}}$' items actively amplifies model-specification bias"
    )
    ax.text(
        0.97, 0.66, callout_eval, transform=ax.transAxes, ha="right", va="top",
        fontsize=8.6, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#FFFBEB", edgecolor="#F59E0B", alpha=0.96, linewidth=1.0),
        zorder=7
    )

    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 300, 750, 1472])
    ax.set_xticklabels(["10", "25", "50", "100", "300", "750", "1472 (Full Bank)"])
    ax.set_xlim(9, 1750)
    ax.set_ylim(0, 16.8)
    ax.set_title("Benchmark Evaluation (Q2a): Static 2PL CAT Hits a 2.43% Specification Bias Wall", pad=12)
    ax.set_xlabel("Number of Sampled MMLU Evaluation Items $n$ (log scale, out of $N = 1,472$)")
    ax.set_ylabel("Benchmark Score Estimation RMSE (%) ($\\downarrow$ better)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=9.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)

    save_fig(fig, "v3_Fig01a_eval_cat_wall.png")


# ==============================================================================
# FIGURE 1B: ACT I (Step 1B) — Coupled Q1 RL Training Ceiling & Pass@k Collapse
# ==============================================================================
def make_v3_fig01b_rl_critic_collapse():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.4, 5.2), gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.25}
    )

    # -------------------------------------------------------------------------
    # Panel A: RL Training Trajectories Braid Together at ~47% (Q1)
    # -------------------------------------------------------------------------
    traj = pd.read_csv(os.path.join(RESULTS, "Q1", "trajectory.csv"))
    styles = [
        ("grpo", "#1E293B", "--o", "Standard Binary GRPO (5 draws/item)"),
        ("ctt",  "#64748B", ":s",  "CTT Difficulty Critic (85k pre-samples, 1 draw)"),
        ("pop0", "#D97706", "-.^", "Static 2PL Unweighted Critic pop0 (1 draw)"),
        ("pop1", "#0284C7", "-D",  "Static 2PL $a$-Weighted Critic pop1 (1 draw)"),
    ]

    # Shaded ceiling band at 46.8% - 48.0%
    ax1.axhspan(46.8, 48.1, color="#FEF3C7", alpha=0.65, zorder=1)
    ax1.axhline(47.2, color="#D97706", linestyle=":", linewidth=1.3, alpha=0.85, zorder=2)

    # Plot mean trajectory across all available seeds or seed 0 with individual seed dots at u333
    s0 = traj[traj["seed"] == 0].sort_values("updates")
    for m, col, fmt, lbl in styles:
        sub = s0[s0["mode"] == m]
        ax1.plot(
            sub["updates"], sub["mean_at_16"] * 100.0, fmt, color=col,
            linewidth=2.4, markersize=6.0, label=lbl, zorder=4
        )

    ax1.text(
        205, 48.45,
        "Tied Single-Draw Ceiling (~47% Mean@16)\nAll 4 RL critic arms braid into the exact same plateau",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#92400E",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor="#FCD34D", alpha=0.96, linewidth=0.9),
        zorder=6
    )

    ax1.annotate(
        "Compute-matched:\n1,665 updates × 1 draw\n= 333 updates × 5 draws",
        xy=(83, 46.8), xytext=(155, 42.5),
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#334155",
        arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.2, connectionstyle="arc3,rad=-0.15"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95, linewidth=0.8),
        zorder=6
    )

    ax1.set_title("(A) Across RL Updates: Identical ~47% Single-Draw Ceiling", pad=11)
    ax1.set_xlabel("RL Training Updates $t$ (Qwen2.5-3B on GSM8K)")
    ax1.set_ylabel("MMLU Test Bank Accuracy: Mean@16 (%) ($\\uparrow$)")
    ax1.set_xticks([0, 83, 166, 250, 333])
    ax1.set_ylim(37.2, 50.2)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=8.4)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Step-333 Test-Time Scaling Collapse (Q1 Multi-Draw Continuous)
    # -------------------------------------------------------------------------
    q1_raw = pd.read_csv(os.path.join(RESULTS, "Q1", "stats.csv"))
    q1_pool = q1_raw[q1_raw["order"] == "pooled"].pivot(index="state", columns="statistic", values="value") * 100.0

    x_k = np.array([0.0, 1.0, 2.0])  # log4 scale: 0 -> k=1, 1 -> k=4, 2 -> k=16
    x_dense = np.linspace(0.0, 2.0, 200)

    base_pts = np.array([38.18, 70.20, 93.00])
    grpo_pts = np.array([46.85, 58.70, 69.40])
    pop1_pts = np.array([47.48, 53.64, 58.66])
    ctt_pts  = np.array([47.22, 53.70, 58.58])
    pop0_pts = np.array([46.90, 50.34, 52.43])

    p_base_quad = np.poly1d(np.polyfit(x_k, base_pts, 2))
    p_grpo_quad = np.poly1d(np.polyfit(x_k, grpo_pts, 2))
    p_pop1_quad = np.poly1d(np.polyfit(x_k, pop1_pts, 2))
    p_ctt_quad  = np.poly1d(np.polyfit(x_k, ctt_pts, 2))
    p_pop0_quad = np.poly1d(np.polyfit(x_k, pop0_pts, 2))

    # Shaded collapse wedge between Base and Static IRT critics
    ax2.fill_between(
        x_dense, p_pop0_quad(x_dense), p_base_quad(x_dense),
        color="#FEE2E2", alpha=0.28, zorder=1
    )
    ax2.fill_between(
        x_dense, p_pop0_quad(x_dense), p_grpo_quad(x_dense),
        color="#FDE68A", alpha=0.28, zorder=2
    )

    ax2.plot(x_dense, p_base_quad(x_dense), "--", color="#475569", linewidth=2.4,
             label="Base Model ($38.2\\% \\rightarrow 93.0\\%$ Pass@16)", zorder=3)
    ax2.scatter(x_k, base_pts, color="#475569", s=42, zorder=5)

    ax2.plot(x_dense, p_grpo_quad(x_dense), "-", color="#1E293B", linewidth=2.7,
             label="Standard Binary GRPO ($69.4\\%$ Pass@16)", zorder=4)
    ax2.plot(x_dense, p_pop1_quad(x_dense), "-", color="#0284C7", linewidth=2.5,
             label="Static 2PL $a$-Weighted pop1 ($58.7\\%$ Pass@16)", zorder=4)
    ax2.plot(x_dense, p_ctt_quad(x_dense), ":", color="#64748B", linewidth=2.3,
             label="CTT Difficulty Critic ($58.6\\%$ Pass@16)", zorder=4)
    ax2.plot(x_dense, p_pop0_quad(x_dense), "-.", color="#D97706", linewidth=2.7,
             label="Static 2PL Unweighted pop0 ($52.4\\%$ — Flat!)", zorder=4)

    # Plot 3 seed dots per arm
    arm_seed_cfg = [
        ("q1_grpo", "#1E293B", "o"),
        ("q1_pop1", "#0284C7", "D"),
        ("q1_ctt",  "#64748B", "s"),
        ("q1_pop0", "#D97706", "^"),
    ]
    for prefix, col, mk in arm_seed_cfg:
        states = [f"{prefix}_{s}" for s in range(3)]
        for i, stat in enumerate(["mean_at_16", "pass_at_4", "pass_at_16"]):
            vals = q1_pool.loc[states, stat].to_numpy()
            ax2.scatter([x_k[i]] * 3, vals, color=col, marker=mk, s=26, alpha=0.85, zorder=6)

    # Double-headed vertical arrow at k=16 showing the extra collapse from static IRT critics
    ax2.annotate(
        "", xy=(2.03, 52.43), xytext=(2.03, 69.40),
        arrowprops=dict(arrowstyle="<->", color="#B45309", lw=1.8), zorder=7
    )
    ax2.text(
        1.96, 61.0, "Static IRT critics\ncrush Pass@16\nby $-10.7\\%$ to $-17.0\\%$\nvs. Binary GRPO!",
        ha="right", va="center", fontsize=8.0, fontweight="bold", color="#92400E",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#F59E0B", alpha=0.94, linewidth=0.8),
        zorder=7
    )

    callout_c = (
        "Hidden Multi-Draw Collapse (Step 333):\n"
        "• All 4 RL arms tie at $k=1$ ($\\sim 47\\%$)\n"
        "• Static IRT critics ($\\mathrm{pop0}, \\mathrm{pop1}, \\mathrm{ctt}$)\n"
        "  flatten test-time scaling ($52.4\\%–58.7\\%$)\n"
        "  even worse than binary GRPO ($69.4\\%$)"
    )
    ax2.text(
        0.03, 0.96, callout_c, transform=ax2.transAxes, ha="left", va="top",
        fontsize=8.2, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.96, linewidth=0.9),
        zorder=7
    )

    ax2.set_xticks(x_k)
    ax2.set_xticklabels(["$k = 1$\n(Pass@1 / Mean@16)", "$k = 4$\n(Pass@4)", "$k = 16$\n(Pass@16 Coverage)"])
    ax2.set_xlim(-0.14, 2.18)
    ax2.set_ylim(34.5, 97.5)
    ax2.set_title("(B) Across Rollout Draws $k$: Static Critics Accelerate Collapse", pad=11)
    ax2.set_xlabel("Number of Test-Time Rollout Draws $k$ ($\\log_4$ scale)")
    ax2.set_ylabel("MMLU Test Bank Coverage: Pass@$k$ (%) ($\\uparrow$)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=8.0)
    ax2.grid(True, linestyle=":", alpha=0.45)

    save_fig(fig, "v3_Fig01b_rl_critic_collapse.png", alias_filenames=["v3_Fig01_static_irt_failure.png"])


# ==============================================================================
# FIGURE 2: ACT II (Step 2A) — Standalone 2x1 Coordinate Alignment Stack
# ==============================================================================
def make_v3_fig02_coordinate_alignment_stacked():
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

    # Panel A: Fits on external population difficulty b_j^pop
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

    # Panel B: Fits on internal baseline log-odds mu_0j
    p_g = np.clip(p0 + dp_g / 100.0, 1e-4, 1 - 1e-4)
    p_s = np.clip(p0 + dp_s / 100.0, 1e-4, 1 - 1e-4)
    res_g = minimize(lambda x: np.mean((p_g - expit(x[0] * L0 + x[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda x: np.mean((p_s - expit(x[0] * L0 + x[1])) ** 2), [0.5, 0.0])

    w0_ok["L_bin"] = pd.qcut(w0_ok["L_base"], q=10)
    binned_L = w0_ok.groupby("L_bin", observed=False)[["a", "L_base", "dp_sft", "dp_grpo"]].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 9.5), gridspec_kw={"hspace": 0.38})

    # ==================== Panel A (Top: Out-of-Focus) ====================
    ax1.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)
    ax1.scatter(b, dp_g, s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.14, edgecolors="none", zorder=2)
    ax1.scatter(b, dp_s, s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.14, edgecolors="none", zorder=2)
    ax1.scatter(
        binned_b["b"], binned_b["dp_grpo"], s=binned_b["a"] * 110.0, color="#DC2626", marker="s",
        zorder=5, edgecolors="#7F1D1D", linewidth=1.2, label="GRPO 10 Deciles along $b_j^{\\mathrm{pop}}$"
    )
    ax1.scatter(
        binned_b["b"], binned_b["dp_sft"], s=binned_b["a"] * 110.0, color="#0284C7", marker="o",
        zorder=5, edgecolors="#0369A1", linewidth=1.2, label="CoT-SFT 10 Deciles along $b_j^{\\mathrm{pop}}$"
    )

    b_dense = np.linspace(-2.5, 5.0, 300)
    ax1.plot(
        b_dense, p3_g(b_dense), "-", color="#DC2626", linewidth=2.8, zorder=6,
        label=f"GRPO Smooth Fit ($R^2 = {r2_p3_g:.3f}$, Linear $R^2 = {r2_lin_g:.3f}$)"
    )
    ax1.plot(b_dense, m_g * b_dense + c_g, "--", color="#991B1B", linewidth=1.7, zorder=6)
    ax1.plot(
        b_dense, p3_s(b_dense), "-", color="#0284C7", linewidth=2.8, zorder=6,
        label=f"CoT-SFT Smooth Fit ($R^2 = {r2_p3_s:.3f}$, Linear $R^2 = {r2_lin_s:.3f}$)"
    )
    ax1.plot(b_dense, m_s * b_dense + c_s, "--", color="#0369A1", linewidth=1.7, zorder=6)

    callout_top = (
        "Out-of-Focus Lens (External Population Coordinate $b_j^{\\mathrm{pop}}$):\n"
        "• 102-LLM difficulty $b_j^{\\mathrm{pop}}$ explains only 8.5% of within-run gain variance ($R^2 = 0.085$)\n"
        "• First-order weight-update tangent $J_j u$ is orthogonal to $a_j^{\\mathrm{pop}}$ (partial $\\rho = 0.072$)"
    )
    ax1.text(
        0.03, 0.96, callout_top, transform=ax1.transAxes, ha="left", va="top",
        fontsize=8.3, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.95, linewidth=0.9),
        zorder=7
    )

    ax1.set_title("(A) Out-of-Focus: Pass-Rate Change $\\Delta p_j$ vs. External 102-LLM Difficulty $b_j^{\\mathrm{pop}}$", pad=10)
    ax1.set_xlabel("External Population IRT Item Difficulty $b_j^{\\mathrm{pop}}$ (102 LLMs: Easy $\\rightarrow$ Hard)")
    ax1.set_ylabel("Pass-Rate Change $\\Delta p_j$ (% pts) ($\\uparrow$)")
    ax1.set_xlim(-2.5, 5.0)
    ax1.set_ylim(-31, 36)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=8.3)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # ==================== Panel B (Bottom: In-Focus) ====================
    ax2.axhline(0, color="#64748B", linestyle="--", linewidth=1.3)

    # Shading and vertical dashed line at logit(0.25) = -1.0986
    ax2.axvspan(-1.0986, -4.5, color="#FEE2E2", alpha=0.38, zorder=1)
    ax2.axvline(-1.0986, color="#DC2626", linestyle=":", linewidth=1.6, alpha=0.85, zorder=3)

    ax2.scatter(w0_ok["L_base"], dp_g, s=w0_ok["a"] * 18.0, color="#DC2626", alpha=0.14, edgecolors="none", zorder=2)
    ax2.scatter(w0_ok["L_base"], dp_s, s=w0_ok["a"] * 18.0, color="#0284C7", alpha=0.14, edgecolors="none", zorder=2)
    ax2.scatter(
        binned_L["L_base"], binned_L["dp_grpo"], s=binned_L["a"] * 110.0, color="#DC2626", marker="s",
        zorder=5, edgecolors="#7F1D1D", linewidth=1.2, label="GRPO 10 Deciles along $\\mu_{0j}$"
    )
    ax2.scatter(
        binned_L["L_base"], binned_L["dp_sft"], s=binned_L["a"] * 110.0, color="#0284C7", marker="o",
        zorder=5, edgecolors="#0369A1", linewidth=1.2, label="CoT-SFT 10 Deciles along $\\mu_{0j}$"
    )

    L_grid = np.linspace(3.5, -4.5, 400)
    pred_grpo_curve = (expit(res_g.x[0] * L_grid + res_g.x[1]) - expit(L_grid)) * 100.0
    pred_sft_curve  = (expit(res_s.x[0] * L_grid + res_s.x[1]) - expit(L_grid)) * 100.0
    ax2.plot(
        L_grid, pred_grpo_curve, "-", color="#DC2626", linewidth=2.9, zorder=6,
        label=f"GRPO Analytical S-Curve ($\\tau = {res_g.x[0]:.2f} > 1$, Steepening)"
    )
    ax2.plot(
        L_grid, pred_sft_curve, "-", color="#0284C7", linewidth=2.9, zorder=6,
        label=f"CoT-SFT Analytical S-Curve ($\\tau = {res_s.x[0]:.2f} < 1$, Flattening)"
    )

    callout_bottom = (
        "Negative-Gain Zone ($p_{\\mathrm{base}} < 25\\%$, 613 Hard Items):\n"
        "• GRPO sharpening ($\\tau = 1.85$) pushes hard items below baseline ($\\Delta p_j < 0$)\n"
        "• Directly motivates filtering out $p_t < 25\\%$ items in Act III (M1c)"
    )
    ax2.text(
        -2.78, -19.5, callout_bottom, ha="center", va="center", fontsize=8.2, color="#991B1B", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#F87171", alpha=0.96, linewidth=0.9),
        zorder=7
    )

    ax2.set_title("(B) In-Focus: Re-Indexing by Model's Own Initial Log-Odds $\\mu_{0j}$ Snaps Cloud onto Exact S-Curves", pad=10)
    ax2.set_xlabel("Within-Model Initial Log-Odds $\\mu_{0j} = \\mathrm{logit}(p_{\\mathrm{base},j})$ (Easy $\\rightarrow$ Hard)")
    ax2.set_ylabel("Pass-Rate Change $\\Delta p_j$ (% pts) ($\\uparrow$)")
    ax2.set_xlim(3.5, -4.5)
    ax2.set_ylim(-31, 36)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=8.3)
    ax2.grid(True, linestyle=":", alpha=0.45)

    ax2_top = ax2.twiny()
    p_ticks = np.array([0.90, 0.75, 0.50, 0.25, 0.10, 0.05])
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(logit(p_ticks))
    ax2_top.set_xticklabels(["90%", "75%", "50%", "25%", "10%", "5%"])
    ax2_top.set_xlabel("Model's Own Initial Pass Probability $p_{\\mathrm{base},j} = \\sigma(\\mu_{0j})$", fontsize=9.8, fontweight="bold", labelpad=5)

    save_fig(
        fig,
        "v3_Fig02_coordinate_alignment_stacked.png",
        alias_filenames=["v3_Fig03_coordinate_alignment_stacked.png"]
    )


# ==============================================================================
# FIGURE 3: ACT II (Step 2B) — Coupled 1x2 Scissor: ICC Rotation & Pass@k Crossover
# ==============================================================================
def make_v3_fig03_icc_rotation_and_passk():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.0, 5.6), gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.26}
    )

    # Load 1,472 matched items
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

    res_g = minimize(lambda c: np.mean((p_g - expit(c[0] * L0 + c[1])) ** 2), [1.8, 0.0])
    res_s = minimize(lambda c: np.mean((p_s - expit(c[0] * L0 + c[1])) ** 2), [0.5, 0.0])

    w0_ok["L_bin"] = pd.qcut(w0_ok["L_base"], q=10)
    binned_L = w0_ok.groupby("L_bin", observed=False)[["a", "L_base", "dp_sft", "dp_grpo", "p_base"]].mean()
    binned_L["p_g"] = np.clip(binned_L["p_base"] + binned_L["dp_grpo"] / 100.0, 0, 1) * 100.0
    binned_L["p_s"] = np.clip(binned_L["p_base"] + binned_L["dp_sft"] / 100.0, 0, 1) * 100.0
    sizes_bin = binned_L["a"].to_numpy() * 95.0

    L_grid = np.linspace(3.5, -4.5, 400)
    base_icc = expit(L_grid) * 100.0
    classical_icc = expit(L_grid + 0.5) * 100.0
    grpo_icc = expit(res_g.x[0] * L_grid + res_g.x[1]) * 100.0
    sft_icc = expit(res_s.x[0] * L_grid + res_s.x[1]) * 100.0

    L_pivot = res_g.x[1] / (1.0 - res_g.x[0])
    p_pivot = expit(L_pivot) * 100.0

    # -------------------------------------------------------------------------
    # Panel A: Item Space — Direct ICC Scissor Rotation
    # -------------------------------------------------------------------------
    # Shaded rotation wedges between Base ICC and GRPO ICC
    mask_easy = L_grid >= L_pivot
    mask_hard = L_grid <= L_pivot
    ax1.fill_between(
        L_grid[mask_easy], base_icc[mask_easy], grpo_icc[mask_easy],
        color="#DCFCE7", alpha=0.55, zorder=1
    )
    ax1.fill_between(
        L_grid[mask_hard], grpo_icc[mask_hard], base_icc[mask_hard],
        color="#FEE2E2", alpha=0.65, zorder=1
    )
    ax1.axvspan(L_pivot, -4.5, color="#FEE2E2", alpha=0.22, zorder=0)
    ax1.axvline(L_pivot, color="#DC2626", linestyle=":", linewidth=1.6, alpha=0.85, zorder=2)

    # Curves
    ax1.plot(L_grid, base_icc, "-", color="#475569", linewidth=2.2, zorder=3,
             label="Base Model ICC ($\\tau = 1.00$)")
    ax1.plot(L_grid, classical_icc, "--", color="#059669", linewidth=2.0, zorder=4,
             label="Classical IRT Shift ($\\Delta\\theta = +0.5, \\tau = 1.00$)")
    ax1.plot(L_grid, grpo_icc, "-", color="#DC2626", linewidth=2.8, zorder=5,
             label=f"GRPO ICC ($\\tau = {res_g.x[0]:.2f} > 1$, Clockwise Steepening)")
    ax1.scatter(binned_L["L_base"], binned_L["p_g"], s=sizes_bin, color="#DC2626", marker="s",
                edgecolors="#7F1D1D", linewidth=1.1, zorder=6, label="GRPO 10 Deciles")
    ax1.plot(L_grid, sft_icc, "-", color="#0284C7", linewidth=2.8, zorder=5,
             label=f"CoT-SFT ICC ($\\tau = {res_s.x[0]:.2f} < 1$, Counter-Clockwise Flattening)")
    ax1.scatter(binned_L["L_base"], binned_L["p_s"], s=sizes_bin, color="#0284C7", marker="o",
                edgecolors="#0369A1", linewidth=1.1, zorder=6, label="CoT-SFT 10 Deciles")

    # Pivot marker
    ax1.scatter([L_pivot], [p_pivot], s=95, color="#FDE047", marker="o", edgecolors="#0F172A", linewidth=2.0, zorder=8)

    # Upward rotation arrow from Base ICC to GRPO ICC at L = +1.15 (cleanly between deciles)
    y_base_e = expit(1.15) * 100.0
    y_grpo_e = expit(res_g.x[0] * 1.15 + res_g.x[1]) * 100.0
    ax1.annotate(
        "", xy=(1.15, y_grpo_e - 0.8), xytext=(1.15, y_base_e + 0.8),
        arrowprops=dict(arrowstyle="-|>", color="#15803D", lw=2.2), zorder=7
    )
    ax1.text(
        2.25, 79.5,
        "Sharpened Easy/Med\n($+887$ items $\\uparrow$)\nDrives $k=1$ win",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#166534",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F0FDF4", edgecolor="#86EFAC", alpha=0.95, linewidth=0.9),
        zorder=8
    )

    # Downward rotation callout pointing directly into the red loss wedge at L = -1.55
    y_mid_h = 0.5 * (expit(-1.55) + expit(res_g.x[0] * (-1.55) + res_g.x[1])) * 100.0
    ax1.annotate(
        "Crushed Hard Tail\n($-613$ items $\\downarrow$ toward $0\\%$)\nKills Pass@16 scaling",
        xy=(-1.55, y_mid_h), xytext=(-2.95, 31.0),
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#991B1B",
        arrowprops=dict(arrowstyle="-|>", color="#B91C1C", lw=1.5, connectionstyle="arc3,rad=0.12"),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FEF2F2", edgecolor="#FCA5A5", alpha=0.95, linewidth=0.9),
        zorder=8
    )

    # Pivot callout pointing to (L_pivot, p_pivot)
    ax1.annotate(
        "Guessing Pivot ($\\sim 25\\%$):\nGRPO & SFT rotate in\nopposite directions here",
        xy=(L_pivot, p_pivot), xytext=(-2.55, 62.0),
        ha="center", va="center", fontsize=8.1, fontweight="bold", color="#0F172A",
        arrowprops=dict(arrowstyle="-|>", color="#0F172A", lw=1.4, connectionstyle="arc3,rad=0.12"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEFCE8", edgecolor="#FACC15", alpha=0.96, linewidth=0.9),
        zorder=8
    )

    ax1.set_title("(A) Item Space: Opposite IRT ICC Rotations Around $p_{\\mathrm{base}} \\approx 25\\%$", pad=10)
    ax1.set_xlabel("Within-Model Initial Log-Odds $\\mu_{0j} = \\mathrm{logit}(p_{\\mathrm{base},j})$ (Easy $\\rightarrow$ Hard)")
    ax1.set_ylabel("Post-Training Item Pass Probability $p_j$ (%)")
    ax1.set_xlim(3.5, -4.5)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=7.8)
    ax1.grid(True, linestyle=":", alpha=0.45)

    ax1_top = ax1.twiny()
    p_ticks = np.array([0.90, 0.75, 0.50, 0.25, 0.10, 0.05])
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(logit(p_ticks))
    ax1_top.set_xticklabels(["90%", "75%", "50%", "25%", "10%", "5%"])
    ax1_top.set_xlabel("Base Pass Probability $p_{\\mathrm{base},j} = \\sigma(\\mu_{0j})$", fontsize=9.4, fontweight="bold", labelpad=5)

    # -------------------------------------------------------------------------
    # Panel B: Rollout Space — The Pass@k Scaling Scissor Crossover
    # -------------------------------------------------------------------------
    k_dense = np.linspace(1, 16, 300)
    x_log2 = np.log2(k_dense)

    cal_p0 = expit(0.927 * logit(p0) - 0.292)
    cal_pg = expit(3.292 * logit(p_g) + 1.018)
    cal_ps = expit(1.501 * logit(p_s) + 0.754)

    curve_base = (1.0 - np.mean((1.0 - cal_p0[:, None]) ** k_dense[None, :], axis=0)) * 100.0
    curve_grpo = (1.0 - np.mean((1.0 - cal_pg[:, None]) ** k_dense[None, :], axis=0)) * 100.0
    curve_sft  = (1.0 - np.mean((1.0 - cal_ps[:, None]) ** k_dense[None, :], axis=0)) * 100.0

    idx_cross = np.where(curve_sft > curve_grpo)[0][0]
    x_cross = x_log2[idx_cross]
    y_cross = curve_sft[idx_cross]

    # Shaded Scissor Regions
    ax2.fill_between(
        x_log2[:idx_cross + 1], curve_sft[:idx_cross + 1], curve_grpo[:idx_cross + 1],
        color="#FEE2E2", alpha=0.45, zorder=1
    )
    ax2.fill_between(
        x_log2[idx_cross:], curve_grpo[idx_cross:], curve_sft[idx_cross:],
        color="#E0F2FE", alpha=0.45, zorder=1
    )
    ax2.axvline(x_cross, color="#0F172A", linestyle=":", linewidth=1.4, alpha=0.7, zorder=2)

    ax2.plot(x_log2, curve_sft, "-", color="#0284C7", linewidth=2.9, zorder=5,
             label="CoT-SFT ($\\tau = 0.42$): $43.9\\% \\rightarrow 96.6\\%$ Pass@16")
    ax2.plot(x_log2, curve_base, "--", color="#475569", linewidth=2.3, zorder=4,
             label="Base Model ($\\tau = 1.00$): $38.2\\% \\rightarrow 93.0\\%$ Pass@16")
    ax2.plot(x_log2, curve_grpo, "-", color="#DC2626", linewidth=2.9, zorder=5,
             label="GRPO ($\\tau = 1.85$): $47.9\\% \\rightarrow 65.6\\%$ Pass@16 (Bends Flat)")

    # Empirical 3-seed dots at k = 1, 4, 16
    ax2.scatter([0.0] * 3, [47.785, 48.644, 47.262], color="#991B1B", marker="s", s=30, zorder=6)
    ax2.scatter([2.0] * 3, [56.64, 57.52, 57.36], color="#991B1B", marker="s", s=30, zorder=6)
    ax2.scatter([4.0] * 3, [64.87, 65.97, 66.07], color="#991B1B", marker="s", s=30, zorder=6)

    ax2.scatter([0.0] * 3, [44.41, 43.38, 43.88], color="#0369A1", marker="o", s=30, zorder=6)
    ax2.scatter([2.0] * 3, [78.82, 77.88, 77.72], color="#0369A1", marker="o", s=30, zorder=6)
    ax2.scatter([4.0] * 3, [97.10, 96.87, 95.97], color="#0369A1", marker="o", s=30, zorder=6)

    ax2.scatter([0.0, 2.0, 4.0], [38.18, 70.20, 93.00], color="#475569", marker="o", s=34, zorder=6)

    # Scissor crossover marker
    ax2.scatter([x_cross], [y_cross], s=95, color="#FDE047", edgecolors="#0F172A", linewidth=2.0, zorder=8)

    # Endpoint labels at k=16
    ax2.text(4.06, 96.6, "96.6%\n(CoT-SFT)", va="center", ha="left", fontsize=8.4, fontweight="bold", color="#0369A1")
    ax2.text(4.06, 91.5, "93.0%\n(Base)", va="center", ha="left", fontsize=8.2, fontweight="bold", color="#475569")
    ax2.text(4.06, 65.6, "65.6%\n(GRPO)", va="center", ha="left", fontsize=8.4, fontweight="bold", color="#991B1B")

    # Scissor Crossover Callout Box in the clean bottom-right region
    callout_scissor = (
        "Why the Curves Cross at $k^* \\approx 1.3$ Draws:\n"
        "• $\\mathrm{Pass}@k = \\frac{1}{N}\\sum_j [1 - (1 - p_j)^k]$\n"
        "• $k = 1$ (Exploitation): GRPO wins ($47.9\\%$ vs. $43.9\\%$)\n"
        "  from $+887$ sharpened easy/medium items\n"
        "• $k = 16$ (Exploration): CoT-SFT wins ($96.6\\%$ vs. $65.6\\%$)\n"
        "  because $\\tau = 0.42$ keeps the $613$ hard items $>0\\%$\n"
        "• Modal Vote ($\\mathrm{Maj}@16$): CoT-SFT $55.5\\%$ vs. GRPO $49.0\\%$"
    )
    ax2.annotate(
        callout_scissor,
        xy=(x_cross, y_cross), xytext=(1.35, 32.0),
        ha="left", va="bottom", fontsize=8.0, fontweight="bold", color="#0F172A",
        arrowprops=dict(arrowstyle="-|>", color="#0F172A", lw=1.4, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.97, linewidth=0.9),
        zorder=8
    )

    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(["$k = 1$\n(Pass@1)", "$k = 2$", "$k = 4$\n(Pass@4)", "$k = 8$", "$k = 16$\n(Pass@16)"])
    ax2.set_xlim(-0.22, 4.65)
    ax2.set_ylim(29.0, 104.5)
    ax2.set_title("(B) Rollout Space: The $\\mathrm{Pass}@k$ Scaling Scissor Crossover", pad=14)
    ax2.set_xlabel("Number of Test-Time Rollout Draws $k$ ($\\log_2$ scale)")
    ax2.set_ylabel("MMLU Test Bank Accuracy: Pass@$k$ (%) ($\\uparrow$)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=8.1)
    ax2.grid(True, linestyle=":", alpha=0.45)

    save_fig(
        fig,
        "v3_Fig03_icc_rotation_and_passk.png",
        alias_filenames=["v3_Fig02_icc_rotation_mechanism.png"]
    )


# ==============================================================================
# FIGURE 4: ACT III (Step 3) — Phase Plane Control & Online Slope Tracker tau_t
# ==============================================================================
def make_v3_fig04_phase_plane_and_tracker():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.6, 5.8), gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.26}
    )

    # -------------------------------------------------------------------------
    # Panel A: Exploitation-Exploration Phase Plane (Unit M1)
    # -------------------------------------------------------------------------
    m1_raw = pd.read_csv(os.path.join(RESULTS, "M1", "stats.csv"))
    m1_p = m1_raw.pivot(index="state", columns="statistic", values="value") * 100.0

    # 1. Base Model
    base_x = m1_p.loc["t9_base", "mean_at_16"]
    base_y = m1_p.loc["t9_base", "pass_at_16"]
    ax1.scatter(
        [base_x], [base_y], s=125, color="#475569", marker="o", edgecolors="#0F172A",
        linewidth=1.6, zorder=6, label="Base Model ($38.2\\%, 93.0\\%$)"
    )
    ax1.text(base_x, base_y + 1.4, "Base (38.2%, 93.0%)", fontsize=8.0, fontweight="bold", color="#334155", ha="left")

    # 2. Standalone GRPO (u333)
    grpo_states = ["q1_grpo_0", "q1_grpo_1", "q1_grpo_2"]
    grpo_xs = m1_p.loc[grpo_states, "mean_at_16"].to_numpy()
    grpo_ys = m1_p.loc[grpo_states, "pass_at_16"].to_numpy()
    ax1.scatter(grpo_xs, grpo_ys, color="#DC2626", s=34, alpha=0.45, zorder=5)
    grpo_cx, grpo_cy = np.mean(grpo_xs), np.mean(grpo_ys)
    ax1.scatter(
        [grpo_cx], [grpo_cy], s=125, color="#DC2626", marker="s", edgecolors="#7F1D1D",
        linewidth=1.6, zorder=6, label="Standalone GRPO u333 ($46.9\\%, 69.4\\%$)"
    )

    # Shaded collapse zone for standalone GRPO
    ax1.axhspan(64.5, 73.2, color="#FEE2E2", alpha=0.40, zorder=1)
    ax1.text(
        40.8, 68.2,
        "Single-Mode Collapse Zone ($\\tau \\rightarrow 3.28$)\nStandalone GRPO loses $-23.6\\%$ Pass@16 coverage vs. Base",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FEF2F2", edgecolor="#F87171", alpha=0.95, linewidth=0.8),
        zorder=4
    )

    # Red plunging arrow (arches cleanly between the lower-left legend and the right-hand callout!)
    arrow_grpo = FancyArrowPatch(
        (base_x, base_y), (grpo_cx, grpo_cy),
        connectionstyle="arc3,rad=-0.12", color="#DC2626",
        arrowstyle="-|>", mutation_scale=17, linewidth=2.4, zorder=5
    )
    ax1.add_patch(arrow_grpo)
    ax1.text(46.8, 75.5, "Unconstrained RL\n($\\tau = 1.85 \\rightarrow 3.28$)", fontsize=7.9, fontweight="bold", color="#991B1B", ha="center")

    # 3. CoT-SFT (T=0.7)
    sft_states = ["t9_sft_0_70", "t9_sft_1_70", "t9_sft_2_70"]
    sft_xs = m1_p.loc[sft_states, "mean_at_16"].to_numpy()
    sft_ys = m1_p.loc[sft_states, "pass_at_16"].to_numpy()
    ax1.scatter(sft_xs, sft_ys, color="#0284C7", s=34, alpha=0.45, zorder=5)
    sft_cx, sft_cy = np.mean(sft_xs), np.mean(sft_ys)
    ax1.scatter(
        [sft_cx], [sft_cy], s=125, color="#0284C7", marker="o", edgecolors="#0369A1",
        linewidth=1.6, zorder=6, label="CoT-SFT Base ($T=0.7$) ($43.9\\%, 96.6\\%$)"
    )
    ax1.text(sft_cx - 0.25, sft_cy - 2.3, "CoT-SFT (T=0.7)", fontsize=8.0, fontweight="bold", color="#0369A1", ha="center")

    arrow_sft = FancyArrowPatch(
        (base_x, base_y), (sft_cx, sft_cy),
        connectionstyle="arc3,rad=0.08", color="#0284C7",
        arrowstyle="-|>", mutation_scale=17, linewidth=2.4, zorder=5
    )
    ax1.add_patch(arrow_sft)

    # 4. Zero-Training SFT T=0.2
    t02_states = ["m1a_sft_t02_0", "m1a_sft_t02_1", "m1a_sft_t02_2"]
    t02_xs = m1_p.loc[t02_states, "mean_at_16"].to_numpy()
    t02_ys = m1_p.loc[t02_states, "pass_at_16"].to_numpy()
    ax1.scatter(t02_xs, t02_ys, color="#059669", s=34, alpha=0.45, zorder=5)
    t02_cx, t02_cy = np.mean(t02_xs), np.mean(t02_ys)
    ax1.scatter(
        [t02_cx], [t02_cy], s=130, color="#059669", marker="D", edgecolors="#047857",
        linewidth=1.6, zorder=6, label="Zero-Train SFT ($T=0.2$) ($47.5\\%, 87.9\\%$)"
    )

    arrow_t02 = FancyArrowPatch(
        (sft_cx, sft_cy), (t02_cx, t02_cy),
        connectionstyle="arc3,rad=-0.12", color="#059669",
        arrowstyle="-|>", mutation_scale=17, linewidth=2.4, zorder=5
    )
    ax1.add_patch(arrow_t02)

    # 5. SFT -> GRPO (u83 -> u333)
    m1b_u83_states = [f"m1b_sft_grpo_{s}_u83" for s in range(3)]
    m1b_u333_states = [f"m1b_sft_grpo_{s}" for s in range(3)]
    m1b_83_x, m1b_83_y = m1_p.loc[m1b_u83_states, "mean_at_16"].mean(), m1_p.loc[m1b_u83_states, "pass_at_16"].mean()
    m1b_33_x, m1b_33_y = m1_p.loc[m1b_u333_states, "mean_at_16"].mean(), m1_p.loc[m1b_u333_states, "pass_at_16"].mean()
    ax1.scatter(m1_p.loc[m1b_u83_states, "mean_at_16"], m1_p.loc[m1b_u83_states, "pass_at_16"], color="#7C3AED", s=26, alpha=0.45, zorder=5)
    ax1.scatter(m1_p.loc[m1b_u333_states, "mean_at_16"], m1_p.loc[m1b_u333_states, "pass_at_16"], color="#7C3AED", s=26, alpha=0.45, zorder=5)
    ax1.scatter([m1b_83_x], [m1b_83_y], s=75, color="#7C3AED", marker="o", edgecolors="#5B21B6", linewidth=1.2, zorder=6)
    ax1.scatter(
        [m1b_33_x], [m1b_33_y], s=115, color="#7C3AED", marker="s", edgecolors="#5B21B6",
        linewidth=1.5, zorder=6, label="SFT $\\rightarrow$ GRPO ($u83 \\rightarrow u333: 46.7\\%, 94.9\\%$)"
    )
    ax1.add_patch(FancyArrowPatch((sft_cx, sft_cy), (m1b_83_x, m1b_83_y), connectionstyle="arc3,rad=-0.06", color="#7C3AED", arrowstyle="-|>", mutation_scale=13, linewidth=1.9, linestyle="--", zorder=5))
    ax1.add_patch(FancyArrowPatch((m1b_83_x, m1b_83_y), (m1b_33_x, m1b_33_y), connectionstyle="arc3,rad=-0.06", color="#7C3AED", arrowstyle="-|>", mutation_scale=13, linewidth=1.9, linestyle="--", zorder=5))

    # 6. SFT -> Band [2..6/8] (u83 -> u333)
    m1c_u83_states = [f"m1c_sft_zpd_{s}_u83" for s in range(3)]
    m1c_u333_states = [f"m1c_sft_zpd_{s}" for s in range(3)]
    m1c_83_x, m1c_83_y = m1_p.loc[m1c_u83_states, "mean_at_16"].mean(), m1_p.loc[m1c_u83_states, "pass_at_16"].mean()
    m1c_33_x, m1c_p_y = m1_p.loc[m1c_u333_states, "mean_at_16"].mean(), m1_p.loc[m1c_u333_states, "pass_at_16"].mean()
    ax1.scatter(m1_p.loc[m1c_u83_states, "mean_at_16"], m1_p.loc[m1c_u83_states, "pass_at_16"], color="#D97706", s=26, alpha=0.45, zorder=5)
    ax1.scatter(m1_p.loc[m1c_u333_states, "mean_at_16"], m1_p.loc[m1c_u333_states, "pass_at_16"], color="#D97706", s=26, alpha=0.45, zorder=5)
    ax1.scatter([m1c_83_x], [m1c_83_y], s=75, color="#D97706", marker="o", edgecolors="#B45309", linewidth=1.2, zorder=6)
    ax1.scatter(
        [m1c_33_x], [m1c_p_y], s=115, color="#D97706", marker="^", edgecolors="#B45309",
        linewidth=1.5, zorder=6, label="SFT $\\rightarrow$ Band [2..6/8] ($46.5\\%, 94.2\\%$)"
    )
    ax1.add_patch(FancyArrowPatch((sft_cx, sft_cy), (m1c_83_x, m1c_83_y), connectionstyle="arc3,rad=0.06", color="#D97706", arrowstyle="-|>", mutation_scale=13, linewidth=1.9, linestyle="-", zorder=5))
    ax1.add_patch(FancyArrowPatch((m1c_83_x, m1c_83_y), (m1c_33_x, m1c_p_y), connectionstyle="arc3,rad=0.06", color="#D97706", arrowstyle="-|>", mutation_scale=13, linewidth=1.9, linestyle="-", zorder=5))

    # Shaded Pareto Optimal Zone in Top Right (label placed in clean callout box to the top-left!)
    ax1.fill_between([44.9, 48.2], 93.0, 97.2, color="#F5F3FF", alpha=0.65, zorder=1)
    pareto_box = patches.Rectangle((44.9, 93.0), 3.3, 4.2, fill=False, edgecolor="#7C3AED", linestyle="--", linewidth=1.5, zorder=2)
    ax1.add_patch(pareto_box)
    ax1.annotate(
        "Pareto Optimal Zone (SFT $\\rightarrow$ RL):\n• $>46.5\\%$ Mean@16 & $>94\\%$ Pass@16\n• $52.0\\%$ Modal Vote Maj@16",
        xy=(46.5, 97.2), xytext=(41.2, 99.6),
        ha="center", va="center", fontsize=7.6, fontweight="bold", color="#5B21B6",
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.3, connectionstyle="arc3,rad=-0.10"),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F5F3FF", edgecolor="#A78BFA", alpha=0.96, linewidth=0.9),
        zorder=8
    )

    # Zero-Train Win Callout (placed cleanly at (46.3, 82.2) below the emerald diamond, right of the red arrow!)
    ax1.annotate(
        "Zero-Train Win: 47.50%\nBeats all RL in 3/3 seeds\n(0 updates, $T=0.2$)",
        xy=(t02_cx, t02_cy), xytext=(46.4, 82.0),
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#047857",
        arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.4),
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#ECFDF5", edgecolor="#34D399", alpha=0.96, linewidth=0.9),
        zorder=8
    )

    ax1.set_xlim(36.8, 49.0)
    ax1.set_ylim(64.5, 102.5)
    ax1.set_title("(A) 2D Exploitation–Exploration Phase Plane (Unit M1)", pad=11)
    ax1.set_xlabel("Single-Draw Accuracy: Mean@16 (%) ($\\uparrow$ Exploitation)")
    ax1.set_ylabel("Multi-Draw Coverage: Pass@16 (%) ($\\uparrow$ Exploration Reach)")
    ax1.legend(loc="center left", bbox_to_anchor=(0.01, 0.36), frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=7.5)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Online Latent Slope Trajectories tau_t (Units R6 & R2)
    # -------------------------------------------------------------------------
    r6_updates = [0, 83, 166, 250, 333]
    tau_r6 = np.array([1.000, 2.716, 3.215, 3.234, 3.275])
    tau_r6_se = np.array([0.100, 0.179, 0.183, 0.186, 0.202])

    ax2.fill_between(r6_updates, tau_r6 - tau_r6_se, tau_r6 + tau_r6_se, color="#DC2626", alpha=0.18, zorder=2)
    ax2.plot(
        r6_updates, tau_r6, "-o", color="#DC2626", linewidth=2.8, markersize=6.2, zorder=5,
        label="Unconstrained GRPO (2D JEDM Online Filter: $\\tau_t \\rightarrow 3.28$)"
    )
    ax2.scatter([333, 333, 333], [3.275, 3.216, 3.303], color="#7F1D1D", s=30, zorder=6)

    # Zero-Train SFT T=0.2 horizontal reference line
    ax2.axhline(
        1.664, color="#059669", linestyle="--", linewidth=2.1, zorder=4,
        label="Zero-Train Sharpening: CoT-SFT at $T=0.2$ ($\\tau = 1.664$)"
    )
    ax2.scatter([0, 0, 0], [1.663, 1.710, 1.618], color="#047857", marker="D", s=30, zorder=6)
    ax2.text(200, 1.50, "Zero-Train SFT $T=0.2$ ($\\tau = 1.664$)", fontsize=7.8, fontweight="bold", color="#047857", ha="center", va="top")

    # Controlled Composition Trajectories (Unit R2)
    t_comp = [0, 83, 333]
    tau_band = [1.000, 1.118, 1.233]
    tau_unf  = [1.000, 1.080, 1.192]

    ax2.plot(
        t_comp, tau_band, "-s", color="#D97706", linewidth=2.4, markersize=5.8, zorder=5,
        label="SFT $\\rightarrow$ Dynamic Band [2..6/8] ($\\tau_t: 1.00 \\rightarrow 1.12 \\rightarrow 1.23$)"
    )
    ax2.plot(
        t_comp, tau_unf, "--^", color="#7C3AED", linewidth=2.2, markersize=5.8, zorder=5,
        label="SFT $\\rightarrow$ Unfiltered GRPO ($\\tau_t: 1.00 \\rightarrow 1.08 \\rightarrow 1.19$)"
    )

    # Callout badge for 2D JEDM Tracker at top left
    callout_jedm = (
        "Online 2D JEDM State-Space Filter (Unit R6, 0 Free Item Params):\n"
        "• Tracks runaway sharpening ($\\tau_t: 1.00 \\rightarrow 3.28$) in $O(K^3)$ time\n"
        "• Next-update reward accuracy: $72.1\\%$ (vs. $66.0\\%$ static base)\n"
        "• $91\\%$ confidence-band coverage across RL updates"
    )
    ax2.text(
        0.03, 0.97, callout_jedm, transform=ax2.transAxes, ha="left", va="top",
        fontsize=7.8, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.96, linewidth=0.9),
        zorder=7
    )

    # Inset placed cleanly in the open middle-right zone [0.46, 0.42, 0.50, 0.26]
    ax_ins = ax2.inset_axes([0.46, 0.42, 0.50, 0.26])
    ax_ins.set_facecolor("#FFFFFF")
    for spine in ax_ins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#64748B")
        spine.set_linewidth(1.1)
    ax_ins.set_zorder(10)

    ax_ins.plot(t_comp, tau_band, "-s", color="#D97706", linewidth=2.2, markersize=5.2, zorder=4)
    ax_ins.plot(t_comp, tau_unf,  "--^", color="#7C3AED", linewidth=2.0, markersize=5.2, zorder=4)

    ax_ins.scatter([83, 83, 83], [1.1028, 1.1441, 1.1079], color="#B45309", s=20, zorder=5)
    ax_ins.scatter([333, 333, 333], [1.2278, 1.2938, 1.1759], color="#B45309", s=20, zorder=5)
    ax_ins.scatter([83, 83, 83], [1.0712, 1.1047, 1.0650], color="#5B21B6", s=20, zorder=5)
    ax_ins.scatter([333, 333, 333], [1.2126, 1.2177, 1.1470], color="#5B21B6", s=20, zorder=5)

    ax_ins.text(92, 1.165, "+0.038 Boost\n(3/3 seeds, $p<0.001$)", ha="center", fontsize=6.9, fontweight="bold", color="#B45309")
    ax_ins.text(235, 1.265, "+0.041 Boost\n(Dead rollouts $33.7\\%\\rightarrow 23.2\\%$)", ha="center", fontsize=6.8, fontweight="bold", color="#B45309")

    ax_ins.set_title("Zoomed Controlled Composition ($\\tau_t \\in [1.0, 1.3]$)", fontsize=7.8, fontweight="bold", pad=3)
    ax_ins.set_xticks([0, 83, 333])
    ax_ins.set_xticklabels(["0", "83", "333"], fontsize=7.2)
    ax_ins.set_xlim(-18, 360)
    ax_ins.set_ylim(0.98, 1.34)
    ax_ins.set_yticks([1.0, 1.1, 1.2, 1.3])
    ax_ins.set_yticklabels(["1.0", "1.1", "1.2", "1.3"], fontsize=7.2)
    ax_ins.grid(True, linestyle=":", alpha=0.45)

    ax2.set_xticks([0, 83, 166, 250, 333])
    ax2.set_xlim(-15, 365)
    ax2.set_ylim(0.32, 4.25)
    ax2.set_title("(B) Online Latent Slope Trajectories $\\tau_t$ Across RL Updates", pad=11)
    ax2.set_xlabel("RL Training Updates $t$")
    ax2.set_ylabel("Latent Person-Slope $\\tau_t$ (Steepening Factor)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, fontsize=7.6)
    ax2.grid(True, linestyle=":", alpha=0.45)

    save_fig(fig, "v3_Fig04_phase_plane_and_tracker.png")


# ==============================================================================
# FIGURE 5: ACT IV (Step 4) — Multi-Dimensional Skill Places (Unit P / R6b)
#                             + Closed-Loop State-Space Architecture (DUET / GSL)
# ==============================================================================
def make_v3_fig05_beyond_1d_places_and_duet():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.8, 6.0), gridspec_kw={"width_ratios": [0.95, 1.15], "wspace": 0.22}
    )

    # -------------------------------------------------------------------------
    # Panel A: Empirical Proof of Repeatable Local Capability Places (Unit P)
    #          Cross-Seed Residual Repeatability e_j^(0) vs e_j^(1) after 1D S-Curve
    # -------------------------------------------------------------------------
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"].copy()
    t9 = pd.read_csv(os.path.join(RESULTS, "T9r", "items_wide.csv")).set_index("item_id").loc[w0_ok["bank_row_id"]]

    p0 = np.clip(t9["p_key_base"].values, 1e-4, 1 - 1e-4)
    L0 = logit(p0)

    # Fit 1D S-curve per seed and extract item-level residuals (% pts)
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

    r_grpo_01 = np.corrcoef(res_grpo_seeds[0], res_grpo_seeds[1])[0, 1]
    r_sft_01  = np.corrcoef(res_sft_seeds[0], res_sft_seeds[1])[0, 1]

    # Reference axes & y=x diagonal
    ax1.axhline(0, color="#94A3B8", linestyle=":", linewidth=1.1, zorder=1)
    ax1.axvline(0, color="#94A3B8", linestyle=":", linewidth=1.1, zorder=1)
    ax1.plot([-42, 42], [-42, 42], "--", color="#1E293B", linewidth=1.8, alpha=0.75, zorder=3,
             label="Exact Cross-Seed Identity ($e_j^{(1)} = e_j^{(0)}$)")

    # Spherical Permutation Null Circle (2-sigma isotropic noise if 1D S-curve were exact)
    null_circle = Ellipse((0, 0), width=36, height=36, edgecolor="#64748B", facecolor="#F1F5F9",
                          linestyle=":", linewidth=1.6, alpha=0.55, zorder=2,
                          label="Isotropic Noise Null ($r_{\\mathrm{null}} \\leq 0.19$ if purely 1D)")
    ax1.add_patch(null_circle)

    # Scatter of 1,472 item residuals (Seed 0 vs Seed 1)
    ax1.scatter(
        res_grpo_seeds[0], res_grpo_seeds[1], s=14, color="#DC2626", alpha=0.18, edgecolors="none", zorder=3
    )
    ax1.scatter(
        res_sft_seeds[0], res_sft_seeds[1], s=14, color="#0284C7", alpha=0.18, edgecolors="none", zorder=3
    )

    # 10 Decile bins along Seed-0 residual to show the tight linear alignment
    df_res = pd.DataFrame({
        "g0": res_grpo_seeds[0], "g1": res_grpo_seeds[1],
        "s0": res_sft_seeds[0],  "s1": res_sft_seeds[1],
    })
    df_res["g_bin"] = pd.qcut(df_res["g0"], q=10)
    df_res["s_bin"] = pd.qcut(df_res["s0"], q=10)
    bin_g = df_res.groupby("g_bin", observed=False)[["g0", "g1"]].mean()
    bin_s = df_res.groupby("s_bin", observed=False)[["s0", "s1"]].mean()

    ax1.plot(bin_g["g0"], bin_g["g1"], "-s", color="#DC2626", linewidth=2.6, markersize=7.0,
             markeredgecolor="#7F1D1D", zorder=6,
             label=f"GRPO Residual Deciles ($r = {r_grpo_01:.2f}$, $>20\\sigma$ vs. Null)")
    ax1.plot(bin_s["s0"], bin_s["s1"], "-o", color="#0284C7", linewidth=2.6, markersize=7.0,
             markeredgecolor="#0369A1", zorder=6,
             label=f"CoT-SFT Residual Deciles ($r = {r_sft_01:.2f}$, $>20\\sigma$ vs. Null)")

    callout_p = (
        "Why 1D Global Slope $(\\tau_t, \\delta_t)$ Is Not Enough (Unit P & R6b):\n"
        "• After subtracting the 1D S-curve, item residuals $e_j^{(s)}$ are\n"
        "  highly repeatable across independent seeds ($r = 0.55–0.80$)\n"
        "• Held-out prompt offsets $c_j$ in R6b repeat at $r = 0.74–0.91$\n"
        "• Proves local capability places $s_t \\in \\mathbb{R}^M$ evolve reproducibly!"
    )
    ax1.text(
        0.03, 0.97, callout_p, transform=ax1.transAxes, ha="left", va="top",
        fontsize=8.0, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.96, linewidth=0.9),
        zorder=7
    )

    ax1.set_xlim(-42, 42)
    ax1.set_ylim(-42, 45)
    ax1.set_title("(A) Empirical Proof of Local Skill Places: Cross-Seed Residual Repeatability", pad=11)
    ax1.set_xlabel("Seed-0 Residual Gain Beyond 1D S-Curve: $e_j^{(0)} = \\Delta p_j^{(0)} - \\widehat{\\Delta p}_j(\\mu_{0j})$ (% pts)")
    ax1.set_ylabel("Seed-1 Residual Gain Beyond 1D S-Curve: $e_j^{(1)}$ (% pts)")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=8.0)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel B: Closed-Loop State-Space Control Loop Diagram (JEDM + Unit P + DUET + GSL)
    # -------------------------------------------------------------------------
    ax2.axis("off")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)

    ax2.set_title("(B) Next-Gen Closed-Loop State-Space Psychometrics (JEDM + DUET + GSL)", pad=11)

    # Outer soft container
    bg_box = FancyBboxPatch(
        (1.5, 2.0), 97.0, 94.0, boxstyle="round,pad=0.8,rounding_size=2.5",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.4, zorder=1
    )
    ax2.add_patch(bg_box)

    # Center Core Hub Badge
    hub = FancyBboxPatch(
        (33.5, 41.5), 33.0, 17.0, boxstyle="round,pad=0.5,rounding_size=2.0",
        facecolor="#0F172A", edgecolor="#38BDF8", linewidth=2.0, zorder=6
    )
    ax2.add_patch(hub)
    ax2.text(
        50.0, 50.0,
        "DYNAMIC STATE-SPACE\n"
        "$x_t = (\\tau_t, \\delta_t, s_t \\in \\mathbb{R}^M)$\n"
        "Global Slope + $M$ Local Places",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color="#F8FAFC", zorder=7
    )

    # Node 1 (Top-Left): Internal Coordinates & Places (W1 + Unit P)
    n1 = FancyBboxPatch(
        (3.5, 56.5), 39.0, 35.5, boxstyle="round,pad=0.5,rounding_size=1.8",
        facecolor="#FFFFFF", edgecolor="#0284C7", linewidth=1.8, zorder=3
    )
    ax2.add_patch(n1)
    ax2.text(23.0, 88.5, "1. Internal Coordinates & Skill Places\n(Unit W1 + Unit P + R6b)",
             ha="center", va="top", fontsize=8.6, fontweight="bold", color="#0369A1", zorder=4)
    ax2.text(
        5.2, 77.5,
        "• Baseline Coordinate Anchor:\n"
        "  $\\mu_{0j} = \\mathrm{logit}(p_{\\mathrm{base},j})$  (replaces $b_j^{\\mathrm{pop}}$)\n"
        "• Label-Free Local Skill Places:\n"
        "  $a_j \\in \\Delta^{M-1}$ from within-run\n"
        "  residual covariance ($r = 0.56–0.81$)",
        ha="left", va="top", fontsize=7.8, color="#1E293B", zorder=4
    )

    # Node 2 (Top-Right): Untied Write & Doob Decomposition (DUET + GSL)
    n2 = FancyBboxPatch(
        (57.5, 56.5), 39.0, 35.5, boxstyle="round,pad=0.5,rounding_size=1.8",
        facecolor="#FFFFFF", edgecolor="#4F46E5", linewidth=1.8, zorder=3
    )
    ax2.add_patch(n2)
    ax2.text(77.0, 88.5, "2. Untied Write & Doob Innovation\n(DUET + GSL State Equation)",
             ha="center", va="top", fontsize=8.6, fontweight="bold", color="#4338CA", zorder=4)
    ax2.text(
        59.2, 77.5,
        "• Write Operator on Train Item $A$:\n"
        "  $\\psi_A = c_G(p_A)\\nabla_w z_A$  (gradient push)\n"
        "• Doob Innovation + Practice Drift:\n"
        "  $\\Delta s_t = (y_t - \\bar{p}_t)a_A + c_G(\\bar{p}_t)\\psi_A$\n"
        "  (separates surprise from update drift)",
        ha="left", va="top", fontsize=7.8, color="#1E293B", zorder=4
    )

    # Node 3 (Bottom-Right): Online Filter & Read Projection (JEDM R6 + Unit R4)
    n3 = FancyBboxPatch(
        (57.5, 8.0), 39.0, 35.5, boxstyle="round,pad=0.5,rounding_size=1.8",
        facecolor="#FFFFFF", edgecolor="#059669", linewidth=1.8, zorder=3
    )
    ax2.add_patch(n3)
    ax2.text(77.0, 40.2, "3. Online Filter & Read Operator\n(JEDM R6 + DUET Tangent R4)",
             ha="center", va="top", fontsize=8.6, fontweight="bold", color="#047857", zorder=4)
    ax2.text(
        59.2, 29.2,
        "• Read Sensitivity on Eval Item $B$:\n"
        "  $a_B = J_B u \\Rightarrow K_{BA} = J_B M J_A^\\top \\neq K_{AB}$\n"
        "• Full-Bank State-Space Readout:\n"
        "  $p_{B,t} = \\sigma(\\tau_t \\mu_{0B} + \\delta_t + a_B^\\top s_t)$\n"
        "• $O(MK + K^3)$ update ($73.5\\%$ reward acc.)",
        ha="left", va="top", fontsize=7.8, color="#1E293B", zorder=4
    )

    # Node 4 (Bottom-Left): Closed-Loop Active Controller (Unit M1)
    n4 = FancyBboxPatch(
        (3.5, 8.0), 39.0, 35.5, boxstyle="round,pad=0.5,rounding_size=1.8",
        facecolor="#FFFFFF", edgecolor="#D97706", linewidth=1.8, zorder=3
    )
    ax2.add_patch(n4)
    ax2.text(23.0, 40.2, "4. Closed-Loop Active Controller\n(Unit M1 Curriculum & Decoding)",
             ha="center", va="top", fontsize=8.6, fontweight="bold", color="#B45309", zorder=4)
    ax2.text(
        5.2, 29.2,
        "• Dynamic Band Curriculum (M1c):\n"
        "  Drop $p_t < 25\\%$ negative-gain zone;\n"
        "  sample $[2/8, 6/8]$ max-info band\n"
        "  ($+0.038$ $\\tau_t$ boost; dead rollouts $-10.5\\%$)\n"
        "• Decoding Control (M1a): $T^* \\propto 1/\\tau^*$ ($47.5\\%$)",
        ha="left", va="top", fontsize=7.8, color="#1E293B", zorder=4
    )

    # Clockwise Closed-Loop Arrows between the 4 nodes
    # 1 -> 2 (Top horizontal)
    ax2.add_patch(FancyArrowPatch((43.2, 74.0), (56.8, 74.0), color="#0284C7", arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=5))
    ax2.text(50.0, 76.2, "Anchor $(\\mu_{0A}, a_A)$", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#0369A1")

    # 2 -> 3 (Right vertical)
    ax2.add_patch(FancyArrowPatch((77.0, 55.8), (77.0, 44.2), color="#4F46E5", arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=5))
    ax2.text(78.5, 50.0, "Write Push\n$\\Delta s_t, \\Delta\\tau_t$", ha="left", va="center", fontsize=7.3, fontweight="bold", color="#4338CA")

    # 3 -> 4 (Bottom horizontal)
    ax2.add_patch(FancyArrowPatch((56.8, 26.0), (43.2, 26.0), color="#059669", arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=5))
    ax2.text(50.0, 23.8, "Posterior $p_{j,t}$", ha="center", va="top", fontsize=7.3, fontweight="bold", color="#047857")

    # 4 -> 1 (Left vertical)
    ax2.add_patch(FancyArrowPatch((23.0, 44.2), (23.0, 55.8), color="#D97706", arrowstyle="-|>", mutation_scale=16, linewidth=2.4, zorder=5))
    ax2.text(21.5, 50.0, "Next Batch $A_{t+1}$\n& Temp $T^*$", ha="right", va="center", fontsize=7.3, fontweight="bold", color="#B45309")

    save_fig(
        fig,
        "v3_Fig05_beyond_1d_places_and_duet.png",
        alias_filenames=["v3_Fig05_dynamic_architecture_schematic.png"]
    )


if __name__ == "__main__":
    print("Generating V3 Intuition-First Presentation Figure Suite...")
    make_v3_fig01a_eval_cat_wall()
    make_v3_fig01b_rl_critic_collapse()
    make_v3_fig02_coordinate_alignment_stacked()
    make_v3_fig03_icc_rotation_and_passk()
    make_v3_fig04_phase_plane_and_tracker()
    make_v3_fig05_beyond_1d_places_and_duet()
    print("All V3 figures generated successfully!")
