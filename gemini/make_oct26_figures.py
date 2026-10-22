"""
Generate the 5 clean, self-explanatory figures for the Monday, Oct 26, 2026 talk:
"Evaluating and Training Large Language Models: What Can (and Cannot) Item Parameters Tell Us?"

Design rules enforced across all 5 figures:
1. Every figure directly answers the title question by contrasting:
   - What standard LLM evaluation/training sees vs. what IRT item parameters (difficulty b, discrimination a) tell us
   - Both what item parameters CAN tell us and what they CANNOT tell us
2. Zero invented buzzwords ("Declarative Factual", "Procedural Math", "2PL weight", "Prior Scaling" are prohibited).
3. Zero statistical textboxes inside figures; plain-English titles, readable axes (%, percentage points, percentile, question count, IRT difficulty b, IRT discrimination a), direct labels on lines/curves, and individual seed dots.
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
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
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
    os.path.join(REPO_ROOT, "notes", "figures"),
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)
    # Clean old figure files so no stale filenames remain
    for old_f in os.listdir(d):
        if old_f.endswith(".png"):
            os.remove(os.path.join(d, old_f))


def save_both(fig, filename):
    first_path = os.path.join(OUT_DIRS[0], filename)
    fig.savefig(first_path, bbox_inches="tight", facecolor="white")
    for d in OUT_DIRS[1:]:
        shutil.copy2(first_path, os.path.join(d, filename))
    plt.close(fig)
    print(f"Saved: {filename}")


# ==============================================================================
# FIGURE 1: Human vs. LLM Item Difficulty (Unit E1)
# Shows CAN (student item difficulty transfers to held-out students, Spearman 0.85)
# vs. CANNOT (student item difficulty does NOT transfer to 61 LLMs, Spearman 0.09)
# ==============================================================================
def make_fig01():
    df = pd.read_csv(os.path.join(RESULTS, "E1", "difficulty.csv"))
    sub = df.dropna(subset=["b_students", "b_llms"]).copy()  # n = 704 questions

    # Convert to percentiles (0 to 100)
    stu_pct = sub["b_students"].rank(pct=True) * 100.0
    llm_pct = sub["b_llms"].rank(pct=True) * 100.0

    # For the left reference panel: 61 simulated students drawn from the student 2PL
    # (exact E1 synthetic benchmark: median Spearman = 0.854 across 704 items)
    rng = np.random.default_rng(42)
    a_stu = sub["a_students"].values
    b_stu = sub["b_students"].values
    thetas = rng.normal(0.0, 1.0, size=(61, 1))
    probs = expit(a_stu[None, :] * (thetas - b_stu[None, :]))
    Y_sim = rng.binomial(1, probs)
    # Empirical difficulty rank on the 61 simulated students (harder = lower pass rate)
    sim_b_proxy = -np.mean(Y_sim, axis=0)
    sim_pct = pd.Series(sim_b_proxy).rank(pct=True).values * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.4), sharey=True)

    # Left panel: CAN (Student calibration -> 61 Students)
    ax1.scatter(stu_pct, sim_pct, s=16, alpha=0.35, color="#1D4ED8", edgecolors="none")
    ax1.plot([0, 100], [0, 100], color="#1E293B", linestyle="--", linewidth=2.0)
    ax1.text(54, 43, "If difficulty transferred (y = x)", rotation=37,
             fontsize=10, fontweight="bold", color="#1E293B")
    ax1.set_title("CAN: Transfer Across Students\n(4,918 students vs. 61 synthetic students, Spearman = 0.85)",
                  color="#1E3A8A", pad=10)
    ax1.set_xlabel("Question Difficulty Percentile for 4,918 Students (0 = Easiest, 100 = Hardest)")
    ax1.set_ylabel("Question Difficulty Percentile in Target Group (0–100)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Right panel: CANNOT (Student calibration -> 61 LLMs)
    ax2.scatter(stu_pct, llm_pct, s=16, alpha=0.38, color="#DC2626", edgecolors="none")
    ax2.plot([0, 100], [0, 100], color="#64748B", linestyle="--", linewidth=1.8)
    ax2.text(54, 43, "If difficulty transferred (y = x)", rotation=37,
             fontsize=10, fontweight="bold", color="#64748B")
    ax2.axhline(50, color="#DC2626", linestyle="-", linewidth=2.0, alpha=0.85)
    ax2.text(50, 53.5, "Actual LLM trend (flat: Spearman = 0.09)", ha="center",
             fontsize=10, fontweight="bold", color="#991B1B")
    ax2.set_title("CANNOT: Transfer from Students to LLMs\n(4,918 students vs. 61 open-weight LLMs, Spearman = 0.09)",
                  color="#991B1B", pad=10)
    ax2.set_xlabel("Question Difficulty Percentile for 4,918 Students (0 = Easiest, 100 = Hardest)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Figure 1: Human Question Difficulty Transfers Across Students, Not to LLMs (704 Eedi Math Questions)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_both(fig, "Fig01_human_vs_llm_difficulty.png")


# ==============================================================================
# FIGURE 2: 2PL Adaptive Short Testing — CAN on Pretraining Checkpoints (Q2c)
# vs. CANNOT on Post-Trained SFT / GRPO Models (Q2a)
# ==============================================================================
def make_fig02():
    with open(os.path.join(RESULTS, "Q2c", "summary.json")) as f:
        q2c = json.load(f)["outcome"]["cells"]
    q2a_err = pd.read_csv(os.path.join(RESULTS, "Q2a", "errors.csv"))
    q2a_state = q2a_err[q2a_err["target"] == "state"]

    ns = [25, 50, 100]
    # Left panel: Q2c 165 pretraining checkpoints (accuracy > 45%)
    c_rand = [q2c[f"above 0.45 n={n}"]["rmse_classical"] * 100.0 for n in ns]
    c_adap = [q2c[f"above 0.45 n={n}"]["rmse_irt_adaptive"] * 100.0 for n in ns]

    # Right panel: Q2a 7 post-trained states (Base, 3 SFT, 3 GRPO), k=16 draws
    a_sub = q2a_state[q2a_state["k"] == 16].groupby(["n", "method"])["rmse"].mean().unstack() * 100.0
    a_rand = [a_sub.loc[n, "classical"] for n in ns]
    a_adap = [a_sub.loc[n, "irt_adaptive"] for n in ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)

    # Left panel: CAN (Pretraining Checkpoints)
    ax1.plot(ns, c_rand, "-o", color="#475569", linewidth=2.5, markersize=8)
    ax1.plot(ns, c_adap, "-s", color="#16A34A", linewidth=2.8, markersize=8)
    for x, yr, ya in zip(ns, c_rand, c_adap):
        ax1.text(x, yr + 0.45, f"{yr:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#334155")
        offset = -0.75 if x < 100 else 0.45
        ax1.text(x, ya + offset, f"{ya:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#15803D")

    ax1.text(38, 8.8, "Random questions\n(standard eval)", fontsize=10.5, fontweight="bold", color="#334155")
    ax1.text(25, 4.1, "2PL adaptive selection\n(35% lower error at n = 25)", fontsize=10.5, fontweight="bold", color="#15803D")
    ax1.set_title("CAN: Short-Test Eval on Pretraining Checkpoints\n(165 checkpoints across 6 pretraining runs)",
                  color="#15803D", pad=10)
    ax1.set_xlabel("Number of Test Questions Administered (n)")
    ax1.set_ylabel("Accuracy Estimation Error (RMSE, percentage points)")
    ax1.set_xticks(ns)
    ax1.set_xlim(15, 110)
    ax1.set_ylim(2.5, 12.8)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Right panel: CANNOT (Post-Trained SFT & GRPO Models)
    ax2.plot(ns, a_rand, "-o", color="#475569", linewidth=2.5, markersize=8)
    ax2.plot(ns, a_adap, "-s", color="#DC2626", linewidth=2.8, markersize=8)
    for x, yr, ya in zip(ns, a_rand, a_adap):
        ax2.text(x, yr - 0.75, f"{yr:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#334155")
        ax2.text(x, ya + 0.45, f"{ya:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#B91C1C")

    ax2.text(42, 3.3, "Random questions (unbiased)", fontsize=10.5, fontweight="bold", color="#334155")
    ax2.text(45, 9.6, "2PL adaptive selection\n(worse at every n: picks high-a items\nwhere post-trained models misfit)",
             fontsize=10, fontweight="bold", color="#B91C1C")
    ax2.set_title("CANNOT: Same Adaptive Test After SFT / GRPO\n(7 post-trained Llama-3.2-1B states on 1,472 MMLU items)",
                  color="#991B1B", pad=10)
    ax2.set_xlabel("Number of Test Questions Administered (n)")
    ax2.set_xticks(ns)
    ax2.set_xlim(15, 110)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Figure 2: 2PL Adaptive Testing Cuts Error on Pretraining Checkpoints, But Reverses After Post-Training",
                 fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_both(fig, "Fig02_adaptive_testing_can_vs_cannot.png")


# ==============================================================================
# FIGURE 3: What Item Parameters Reveal About SFT vs. GRPO (CAN)
# vs. Why a Single 1D 2PL Ability Score Distorts Credit (CANNOT)
# ==============================================================================
def make_fig03():
    w0 = pd.read_csv(os.path.join(RESULTS, "W0", "items_bank.csv"))
    w0_ok = w0[w0["match_status"] == "matched"]
    it10 = pd.read_csv(os.path.join(SIG_RESULTS, "T10", "runs", "items.csv"))
    flags = pd.read_csv(os.path.join(RESULTS, "Q2g", "s1", "flags.csv"))
    pf = pd.read_csv(os.path.join(RESULTS, "Q2g", "s1", "personfit.csv"))

    pf_all = pf.iloc[:7]
    th_base = pf_all[pf_all["method"] == "base"]["theta"].mean()
    th_sft = pf_all[pf_all["method"] == "sft"]["theta"].mean()
    th_grpo = pf_all[pf_all["method"] == "grpo"]["theta"].mean()

    a3 = flags[(flags["reading"] == "A3_subject_mean_c") & (flags["variant"] == "all") & (flags["state"] == "pooled_states")]
    sub_a = w0_ok.groupby("subject")["a"].mean()
    sub_b = w0_ok.groupby("subject")["b"].mean()
    p_base_sub = it10.groupby("subject")["p_ref"].mean()

    c_sft = a3[a3["method"] == "sft"].set_index("subject")["value"]
    c_grpo = a3[a3["method"] == "grpo"].set_index("subject")["value"]
    dL_sft = c_sft + sub_a * (th_sft - th_base)
    dL_grpo = c_grpo + sub_a * (th_grpo - th_base)

    dp_sft = (expit(logit(p_base_sub) + dL_sft) - p_base_sub) * 100.0
    dp_grpo = (expit(logit(p_base_sub) + dL_grpo) - p_base_sub) * 100.0

    df = pd.DataFrame({
        "b": sub_b,
        "a": sub_a,
        "dp_sft": dp_sft,
        "dp_grpo": dp_grpo,
    }).sort_values("b")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Left panel: CAN — Subject Accuracy Gain vs. Subject IRT Difficulty b
    ax1.axhline(0, color="#64748B", linestyle="--", linewidth=1.4)
    ax1.text(1.50, -3.2, "No change over Base (0%)", fontsize=9.2, color="#64748B", fontweight="bold")

    ax1.scatter(df["b"], df["dp_sft"], s=65, color="#2563EB", alpha=0.9, zorder=4)
    ax1.scatter(df["b"], df["dp_grpo"], s=65, color="#DC2626", marker="s", alpha=0.9, zorder=4)

    # Trend lines across IRT difficulty b
    x_grid = np.linspace(df["b"].min() - 0.1, df["b"].max() + 0.1, 100)
    m_sft, c_sft_line = np.polyfit(df["b"], df["dp_sft"], 1)
    m_grpo, c_grpo_line = np.polyfit(df["b"], df["dp_grpo"], 1)
    ax1.plot(x_grid, m_sft * x_grid + c_sft_line, "-", color="#2563EB", linewidth=2.2, alpha=0.8)
    ax1.plot(x_grid, m_grpo * x_grid + c_grpo_line, "-", color="#DC2626", linewidth=2.2, alpha=0.8)

    # Direct labels for key subjects so the viewer immediately sees what the dots are
    labels_grpo = {
        "marketing": ("Marketing (a=1.45)", (0.06, -0.5)),
        "sociology": ("Sociology (a=1.13)", (0.06, -0.5)),
        "high_school_world_history": ("World History", (0.07, -0.5)),
        "high_school_biology": ("Biology", (0.07, -0.5)),
        "econometrics": ("Econometrics", (-0.65, -0.5)),
    }
    for subj, (txt, (dx, dy)) in labels_grpo.items():
        row = df.loc[subj]
        ax1.text(row["b"] + dx, row["dp_grpo"] + dy, txt, fontsize=8.8, color="#991B1B", fontweight="bold")

    labels_sft = {
        "elementary_mathematics": ("Elementary Math\n(a=0.43)", (-0.68, -1.0)),
        "high_school_mathematics": ("High School Math\n(a=0.28)", (-0.82, 0.5)),
        "high_school_macroeconomics": ("Macroeconomics", (-0.52, 2.2)),
        "high_school_physics": ("Physics (a=0.33)", (-0.35, 2.0)),
    }
    for subj, (txt, (dx, dy)) in labels_sft.items():
        row = df.loc[subj]
        ax1.text(row["b"] + dx, row["dp_sft"] + dy, txt, fontsize=8.8, color="#1E40AF", fontweight="bold")

    ax1.text(-1.05, 34.5, "GRPO gain (red squares)\nwins on low-b, high-a subjects",
             fontsize=10, fontweight="bold", color="#B91C1C")
    ax1.text(0.45, 29.0, "SFT gain (blue circles)\nwins on high-b, low-a subjects",
             fontsize=10, fontweight="bold", color="#1D4ED8")

    ax1.set_title("CAN: IRT Difficulty b Separates Where SFT vs. GRPO Learn\n(15 MMLU subjects; overall accuracy only sees 45.3% vs. 49.3%)",
                  color="#1E3A8A", pad=10)
    ax1.set_xlabel("Subject Mean IRT Difficulty b (102-LLM 2PL: Left = Easier / High-a, Right = Harder / Low-a)")
    ax1.set_ylabel("Accuracy Gain Over Base Model (percentage points)")
    ax1.set_xlim(-1.15, 2.55)
    ax1.set_ylim(-14, 43)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Right panel: CANNOT — 2PL Item Curves & Why a Single 1D Theta Score Flips Credit
    theta_grid = np.linspace(-3.0, 3.0, 300)
    p_mkt = expit(1.45 * (theta_grid - (-0.99)))
    p_math = expit(0.28 * (theta_grid - 2.30))

    ax2.plot(theta_grid, p_mkt * 100.0, color="#DC2626", linewidth=3.0)
    ax2.plot(theta_grid, p_math * 100.0, color="#2563EB", linewidth=3.0)

    ax2.text(-2.7, 74, "Typical Marketing question\n(discrimination a = 1.45, b = -0.99)\nCounts 5x in 1D 2PL ability \u03b8\n(where GRPO gains +21%)",
             fontsize=9.8, fontweight="bold", color="#991B1B")
    ax2.text(-0.4, 14, "Typical High School Math question\n(discrimination a = 0.28, b = +2.30)\nCounts 1x in 1D 2PL ability \u03b8\n(where SFT gains +24%)",
             fontsize=9.8, fontweight="bold", color="#1E40AF")

    # Show the credit reversal directly at the top of the panel
    ax2.set_title("CANNOT: Collapsing to 1D 2PL Ability \u03b8 Distorts Credit\n(Unweighted accuracy: 60% SFT / 40% GRPO -> 2PL \u03b8: 21% SFT / 79% GRPO)",
                  color="#991B1B", pad=10)
    ax2.set_xlabel("Model Ability \u03b8 (102-LLM 2PL Scale)")
    ax2.set_ylabel("Predicted Probability of Correct Answer (%)")
    ax2.set_xlim(-3.0, 3.0)
    ax2.set_ylim(0, 102)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Figure 3: 2D Item Parameters (b, a) Reveal Opposite SFT vs. GRPO Profiles That 1D Scores Hide",
                 fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_both(fig, "Fig03_item_params_reveal_sft_vs_grpo.png")


# ==============================================================================
# FIGURE 4: RL Training With Item Parameters (Unit Q1)
# Left: Training accuracy trajectory (CAN match 5-draw GRPO & 85k-presampled CTT
#       with 0 pre-samples; CANNOT raise the 47% ceiling; saturates by step 83)
# Right: Answer diversity Pass@16 across methods (CANNOT prevent mode collapse;
#        1-draw critics collapse Pass@16 from 69.4% down to 52.4%)
# ==============================================================================
def make_fig04():
    traj = pd.read_csv(os.path.join(RESULTS, "Q1", "trajectory.csv"))
    stats_raw = pd.read_csv(os.path.join(RESULTS, "Q1", "stats.csv"))
    stats = stats_raw[stats_raw["order"] == "pooled"].pivot(index="state", columns="statistic", values="value").reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Left panel: Training curve over updates 0, 83, 166, 250, 333 + all 3 seeds at 333
    palette = {
        "grpo": ("#1E293B", "-o", "GRPO (5 draws/question)"),
        "ctt":  ("#64748B", "--s", "Pass-rate critic (1 draw, needs 85k pre-samples)"),
        "pop0": ("#2563EB", "-^", "2PL IRT critic (1 draw, 0 pre-samples)"),
        "pop1": ("#DC2626", "-D", "a-weighted 2PL critic (1 draw, 0 pre-samples)"),
    }
    for mode, (col, fmt, label) in palette.items():
        sub = traj[traj["mode"] == mode].sort_values("updates")
        ax1.plot(sub["updates"], sub["mean_at_16"] * 100.0, fmt, color=col,
                 linewidth=2.3, markersize=6.5, label=label, alpha=0.9)

    # Plot all 12 final seed dots at step 333 (slightly jittered horizontally so all 12 are visible)
    offsets = {"grpo": -8, "ctt": -3, "pop0": +3, "pop1": +8}
    for mode, (col, _, _) in palette.items():
        s_rows = stats[stats["state"].str.startswith(f"q1_{mode}_")]
        ax1.scatter([333 + offsets[mode]] * len(s_rows), s_rows["mean_at_16"] * 100.0,
                    s=55, color=col, edgecolor="white", linewidth=0.8, zorder=5)

    ax1.axvline(83, color="#D97706", linestyle=":", linewidth=2.0)
    ax1.text(90, 40.2, "Step 83: >85% of gain finished\n(63% of groups have 0 variance\nafter step 83)",
             fontsize=9.5, fontweight="bold", color="#B45309")
    ax1.text(175, 48.7, "All 12 runs (4 methods x 3 seeds)\nfinish within 46.9%–47.5%",
             fontsize=9.8, fontweight="bold", color="#1E293B")

    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.2)
    ax1.set_title("CAN Match Accuracy With 0 Pre-Samples;\nCANNOT Beat the 47% RL Accuracy Ceiling",
                  color="#1E293B", pad=10)
    ax1.set_xlabel("RL Training Updates (32 questions x 5 draws per update)")
    ax1.set_ylabel("MMLU Single-Draw Accuracy (Mean@16, %)")
    ax1.set_xticks([0, 83, 166, 250, 333])
    ax1.set_xlim(-15, 355)
    ax1.set_ylim(37.0, 50.5)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Right panel: What Mean@16 hides — Pass@16 (at least 1 of 16 correct) across methods
    s1_raw = pd.read_csv(os.path.join(RESULTS, "S1", "stats.csv"))
    s1_stats = s1_raw[s1_raw["order"] == "pooled"].pivot(index="state", columns="statistic", values="value")
    base_pass16 = s1_stats.loc["t9_base", "pass_at_16"] * 100.0

    methods_order = ["grpo", "ctt", "pop1", "pop0"]
    x_labels = [
        "GRPO\n(5 draws,\nshutoff gate)",
        "Pass-rate\ncritic\n(1 draw)",
        "a-weighted\n2PL critic\n(1 draw)",
        "Unweighted\n2PL critic\n(1 draw)",
    ]
    colors_right = ["#1E293B", "#64748B", "#DC2626", "#2563EB"]

    ax2.axhline(base_pass16, color="#475569", linestyle="--", linewidth=1.8)
    ax2.text(0.05, base_pass16 - 3.8, f"Base model before RL: {base_pass16:.1f}%",
             fontsize=9.8, fontweight="bold", color="#334155")

    for idx, (mode, col) in enumerate(zip(methods_order, colors_right)):
        s_rows = stats[stats["state"].str.startswith(f"q1_{mode}_")].sort_values("state")
        vals = s_rows["pass_at_16"].values * 100.0
        mean_v = np.mean(vals)
        top_v = np.max(vals)
        # Plot the 3 seed dots
        jitter = [-0.09, 0.0, +0.09]
        ax2.scatter([idx + j for j in jitter], vals, s=70, color=col,
                    edgecolor="white", linewidth=0.9, zorder=4)
        ax2.hlines(mean_v, idx - 0.22, idx + 0.22, color=col, linewidth=3.0, zorder=3)
        ax2.text(idx, max(mean_v + 2.5, top_v + 1.8), f"{mean_v:.1f}%", ha="center",
                 fontsize=10.5, fontweight="bold", color=col)

    ax2.set_title("CAN Reveal Hidden Diversity Collapse:\n1-Draw Critics Collapse Pass@16 Much Harder Than GRPO",
                  color="#991B1B", pad=10)
    ax2.set_xticks(range(len(methods_order)))
    ax2.set_xticklabels(x_labels, fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("Best-of-16 Coverage (Pass@16: >=1 of 16 Correct, %)")
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(45.0, 98.0)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Figure 4: In RL Training (Unit Q1), Static Item Parameters Tie on Accuracy (47%) But Worsen Mode Collapse",
                 fontsize=13.2, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_both(fig, "Fig04_rl_training_can_vs_cannot.png")


# ==============================================================================
# FIGURE 5: Single-Draw vs. Majority-Vote Crossover (S1) & Within-Model
# Pass-Rate Targeting for Unit M1 (Combining SFT + RL)
# ==============================================================================
def make_fig05():
    s1_raw = pd.read_csv(os.path.join(RESULTS, "S1", "stats.csv"))
    s1 = s1_raw[s1_raw["order"] == "pooled"].pivot(index="state", columns="statistic", values="value")
    rw_all = pd.read_csv(os.path.join(RESULTS, "S1", "rewrite1_parsed.csv")).set_index("state")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [1.1, 1.05]})

    # Left panel: Mean@16 vs. Maj@16 crossover on clean 36%–58% y-axis with all 3 seed dots
    x_pos = np.array([0, 1])

    # Base model
    base_mean = s1.loc["t9_base", "mean_at_16"] * 100.0
    base_maj_strict = s1.loc["t9_base", "maj_at_16"] * 100.0
    base_maj_tc = rw_all.loc["t9_base", "maj_tie_correct"] * 100.0
    ax1.plot(x_pos, [base_mean, base_maj_strict], "-o", color="#475569", linewidth=2.5, markersize=8)
    ax1.plot(x_pos, [base_mean, base_maj_tc], ":o", color="#475569", linewidth=1.8, markersize=6, alpha=0.7)
    ax1.text(0 - 0.05, base_mean, f"Base: {base_mean:.1f}%", ha="right", va="center",
             fontsize=10, fontweight="bold", color="#334155")
    ax1.text(1 + 0.05, base_maj_strict - 0.5, f"Base: {base_maj_strict:.1f}% ({base_maj_tc:.1f}%*)",
             ha="left", va="center", fontsize=9.8, fontweight="bold", color="#334155")

    # GRPO (3 seeds)
    grpo_states = ["t9_rl_0_666", "t9_rl_1_666", "t9_rl_2_666"]
    grpo_means = s1.loc[grpo_states, "mean_at_16"].values * 100.0
    grpo_majs = s1.loc[grpo_states, "maj_at_16"].values * 100.0
    grpo_tcs = rw_all.loc[grpo_states, "maj_tie_correct"].values * 100.0
    for m_v, maj_v in zip(grpo_means, grpo_majs):
        ax1.scatter([0, 1], [m_v, maj_v], s=50, color="#DC2626", alpha=0.65, zorder=4)
    ax1.plot(x_pos, [grpo_means.mean(), grpo_majs.mean()], "-s", color="#DC2626", linewidth=3.0, markersize=8, zorder=5)
    ax1.text(0 - 0.05, grpo_means.mean() + 0.6, f"GRPO: {grpo_means.mean():.1f}%\n(wins single draw)",
             ha="right", va="center", fontsize=10, fontweight="bold", color="#B91C1C")
    ax1.text(1 + 0.05, grpo_majs.mean() - 1.2, f"GRPO: {grpo_majs.mean():.1f}% ({grpo_tcs.mean():.1f}%*)\n(flat: 82% answers identical)",
             ha="left", va="center", fontsize=9.8, fontweight="bold", color="#B91C1C")

    # SFT (3 seeds)
    sft_states = ["t9_sft_0_70", "t9_sft_1_70", "t9_sft_2_70"]
    sft_means = s1.loc[sft_states, "mean_at_16"].values * 100.0
    sft_majs = s1.loc[sft_states, "maj_at_16"].values * 100.0
    sft_tcs = rw_all.loc[sft_states, "maj_tie_correct"].values * 100.0
    for m_v, maj_v, tc_v in zip(sft_means, sft_majs, sft_tcs):
        ax1.scatter([0, 1], [m_v, maj_v], s=50, color="#2563EB", alpha=0.65, zorder=4)
        ax1.scatter([1], [tc_v], s=40, color="#2563EB", marker="^", alpha=0.5, zorder=4)
    ax1.plot(x_pos, [sft_means.mean(), sft_majs.mean()], "-o", color="#2563EB", linewidth=3.0, markersize=8, zorder=5)
    ax1.plot(x_pos, [sft_means.mean(), sft_tcs.mean()], "--^", color="#2563EB", linewidth=2.2, markersize=7, alpha=0.85, zorder=4)

    ax1.text(0 - 0.05, sft_means.mean() - 0.5, f"SFT: {sft_means.mean():.1f}%",
             ha="right", va="center", fontsize=10, fontweight="bold", color="#1D4ED8")
    ax1.text(1 + 0.05, sft_majs.mean() + 0.8, f"SFT strict: {sft_majs.mean():.1f}%\n(overtakes GRPO)",
             ha="left", va="center", fontsize=10, fontweight="bold", color="#1D4ED8")
    ax1.text(1 + 0.05, sft_tcs.mean() + 0.3, f"SFT tie-correct*: {sft_tcs.mean():.1f}%",
             ha="left", va="center", fontsize=9.8, fontweight="bold", color="#1E40AF")

    ax1.set_title("Why Base -> RL Hits a Wall at 47.9%,\nWhile SFT Raises the Majority-Vote Ceiling",
                  color="#1E3A8A", pad=10)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Single Draw\n(Mean@16)", "Majority Vote Over 16 Draws\n(Maj@16)"],
                        fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("MMLU Accuracy (%, 3 seed dots per model)")
    ax1.set_xlim(-0.48, 1.68)
    ax1.set_ylim(36.0, 58.5)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Right panel: Probability of mixed rewards in 5 GRPO draws vs. question pass rate p
    p_grid = np.linspace(0, 1, 300)
    mixed_share = (1.0 - p_grid**5 - (1.0 - p_grid)**5) * 100.0

    ax2.fill_between(p_grid * 100.0, 0, mixed_share, where=(p_grid >= 0.2) & (p_grid <= 0.8),
                     color="#DBEAFE", alpha=0.75)
    ax2.plot(p_grid * 100.0, mixed_share, color="#1D4ED8", linewidth=3.0)

    # Mark empirical points on the curve
    pts = [(0.0, "0% correct\n(25.9% of W1:\n0 gradient)", (5, 3), "#991B1B"),
           (0.125, "1/8 (49%)", (4, -4), "#1E3A8A"),
           (0.25, "2/8 (76%)", (4, -3), "#1E3A8A"),
           (0.50, "4/8 (94% mixed:\nmax RL signal)", (-16, -13), "#1E40AF"),
           (1.0, "100% correct\n(9.1% of W1:\n0 gradient)", (-33, 3), "#991B1B")]
    for p_val, lbl, (dx, dy), col in pts:
        y_val = (1.0 - p_val**5 - (1.0 - p_val)**5) * 100.0
        ax2.scatter([p_val * 100.0], [y_val], s=65, color=col, zorder=5)
        ax2.text(p_val * 100.0 + dx, y_val + dy, lbl, fontsize=9.2, fontweight="bold", color=col)

    ax2.text(50, 32, "Unit M1 Target Band\n(question pass rate 20%–80%:\n76%–94% of groups give RL signal)",
             ha="center", fontsize=10, fontweight="bold", color="#1E40AF")

    ax2.set_title("How Model-Own Pass Rates Guide RL (Unit M1):\n35% of Base Groups (63% After Step 83) Give Zero Gradient",
                  color="#1E3A8A", pad=10)
    ax2.set_xlabel("Model's Own Pass Rate on the Question (%)")
    ax2.set_ylabel("Share of 5-Answer Groups With Mixed Rewards (%)")
    ax2.set_xlim(-3, 103)
    ax2.set_ylim(-2, 105)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Figure 5: SFT Raises the Majority-Vote Ceiling (50.4%–55.5%); Model Pass Rates Target RL to Harvest It (Unit M1)",
                 fontsize=13.0, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_both(fig, "Fig05_sampling_crossover_and_m1.png")


if __name__ == "__main__":
    make_fig01()
    make_fig02()
    make_fig03()
    make_fig04()
    make_fig05()
    print("All 5 figures generated successfully.")
