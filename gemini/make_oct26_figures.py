import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# Portable repository resolution
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if (REPO_ROOT / "results" / "W0").exists():
    IPL = str(REPO_ROOT / "results")
else:
    IPL = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\item-params-llm\results"

OUT_DIR = str(SCRIPT_DIR / "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C_BASE = "#555555"
C_SFT = "#1f6feb"     # Royal blue (CoT SFT)
C_GRPO = "#d9381e"    # Vermilion red (GRPO)
C_CTT = "#e69f00"     # Amber
C_POP0 = "#8a3ffc"    # Purple
C_POP1 = "#009e73"    # Teal
C_ZPD = "#118ab2"     # Cyan-blue

def safe_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def set_clean_header(ax, title_text, sub_text, title_size=13.0, sub_size=9.5):
    ax.set_title(title_text, fontsize=title_size, fontweight="bold", pad=22, color="#111111")
    ax.text(0.5, 1.018, sub_text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=sub_size, color="#444444")

def save_fig(fig, name):
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path}")

# ==============================================================================
# PLOT 01: Human Student Difficulty vs. LLM Error Rate (704 Math Questions)
# ==============================================================================
e1_diff = pd.read_csv(os.path.join(IPL, "E1", "difficulty.csv")).dropna(
    subset=["b_students", "b_llms", "share_correct_llms"]
).copy()
n_e1 = len(e1_diff)

b_stu = e1_diff["b_students"].values
err_llm = (1.0 - e1_diff["share_correct_llms"].values) * 100.0
r_cl = float(np.corrcoef(b_stu, err_llm)[0, 1])
rho_cl = float(pd.Series(b_stu).corr(pd.Series(err_llm), method="spearman"))
rho_2pl = float(e1_diff["b_students"].corr(e1_diff["b_llms"], method="spearman"))
m_e1, c_e1 = np.polyfit(b_stu, err_llm, 1)

fig, ax = plt.subplots(figsize=(7.8, 5.8))
ax.scatter(b_stu, err_llm, alpha=0.45, s=28, color="#2b5c8f", edgecolors="none",
           label=f"Diagnostic Math Questions (J = {n_e1})")

xs_b = np.linspace(b_stu.min() - 0.1, b_stu.max() + 0.1, 100)
ax.plot(xs_b, m_e1 * xs_b + c_e1, color=C_GRPO, lw=2.8,
        label=f"OLS Fit: slope = {m_e1:+.2f}%/logit, intercept = {c_e1:.1f}% (r = {r_cl:+.3f}, rho = {rho_cl:+.3f})")

ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3, 103)
ax.set_xlabel("Human Student 2PL Difficulty (4,918 Students, Easy  --->  Hard)", fontsize=11.5, fontweight="bold")
ax.set_ylabel("LLM Error Rate Across 61 Models (% Wrong)", fontsize=11.5, fontweight="bold")
set_clean_header(
    ax,
    "Plot 1. Question Difficulty for Human Students Does Not Transfer to LLMs",
    f"704 Eedi math questions: classical LLM error rate vs. human 2PL difficulty (2PL rank Spearman rho = {rho_2pl:+.3f})"
)
ax.legend(loc="upper left", frameon=True, fontsize=9.8)
save_fig(fig, "Plot01_human_vs_llm_difficulty")

# ==============================================================================
# PLOT 02: 15 MMLU Subjects Sorted by 2PL Weight — Where SFT vs. GRPO Gain
# ==============================================================================
w0_bank = pd.read_csv(os.path.join(IPL, "W0", "items_bank.csv")).dropna(subset=["a", "b"]).copy()
pf_df = pd.read_csv(os.path.join(IPL, "Q2g", "s1", "personfit.csv"))
pf_all = pf_df[pf_df["variant"] == "all"].set_index("state")

th_base = float(pf_all.loc["t9_base", "theta"])
dth_sft = np.mean([float(pf_all.loc[f"t9_sft_{s}_70", "theta"]) - th_base for s in [0, 1, 2]])
dth_grpo = np.mean([float(pf_all.loc[f"t9_rl_{s}_666", "theta"]) - th_base for s in [0, 1, 2]])

fl_s1 = pd.read_csv(os.path.join(IPL, "Q2g", "s1", "flags.csv"))
sub_sft = fl_s1[(fl_s1["variant"] == "all") & (fl_s1["reading"] == "A3_subject_mean_c") & (fl_s1["method"] == "sft") & (fl_s1["state"] == "pooled_states")]
sub_grpo = fl_s1[(fl_s1["variant"] == "all") & (fl_s1["reading"] == "A3_subject_mean_c") & (fl_s1["method"] == "grpo") & (fl_s1["state"] == "pooled_states")]

# Check which subjects exceed the 95th-percentile random-label permutation null in Q2g A3_subject_flag_share
fsh_sft = fl_s1[(fl_s1["variant"] == "all") & (fl_s1["reading"] == "A3_subject_flag_share") & (fl_s1["method"] == "sft") & (fl_s1["state"] == "pooled_states")]
fsh_grpo = fl_s1[(fl_s1["variant"] == "all") & (fl_s1["reading"] == "A3_subject_flag_share") & (fl_s1["method"] == "grpo") & (fl_s1["state"] == "pooled_states")]
sig_sft = set(fsh_sft[fsh_sft["value"].astype(float) > fsh_sft["null_p95_random_labels"].astype(float)]["subject"])
sig_grpo = set(fsh_grpo[fsh_grpo["value"].astype(float) > fsh_grpo["null_p95_random_labels"].astype(float)]["subject"])

c_sft_map = dict(zip(sub_sft["subject"], sub_sft["value"].astype(float)))
c_grpo_map = dict(zip(sub_grpo["subject"], sub_grpo["value"].astype(float)))

sub_grp = w0_bank.groupby("subject").agg(a_mean=("a", "mean"), b_mean=("b", "mean")).reset_index()
sub_grp["c_sft"] = sub_grp["subject"].map(c_sft_map)
sub_grp["c_grpo"] = sub_grp["subject"].map(c_grpo_map)
sub_grp["dL_sft"] = sub_grp["c_sft"] + sub_grp["a_mean"] * dth_sft
sub_grp["dL_grpo"] = sub_grp["c_grpo"] + sub_grp["a_mean"] * dth_grpo
sub_grp["diff_sft_grpo"] = sub_grp["dL_sft"] - sub_grp["dL_grpo"]
sub_grp = sub_grp.sort_values("a_mean").reset_index(drop=True)

SHORT_NAMES = {
    "high_school_mathematics": "High School Math",
    "high_school_physics": "High School Physics",
    "high_school_statistics": "High School Statistics",
    "elementary_mathematics": "Elementary Math",
    "econometrics": "Econometrics",
    "high_school_macroeconomics": "HS Macroeconomics",
    "high_school_microeconomics": "HS Microeconomics",
    "clinical_knowledge": "Clinical Knowledge",
    "philosophy": "Philosophy",
    "high_school_biology": "High School Biology",
    "high_school_world_history": "World History",
    "high_school_geography": "HS Geography",
    "prehistory": "Prehistory",
    "sociology": "Sociology",
    "marketing": "Marketing",
}
sub_grp["short"] = sub_grp["subject"].map(SHORT_NAMES)

diff_vec = sub_grp["diff_sft_grpo"].values
a_vec = sub_grp["a_mean"].values
b_vec = sub_grp["b_mean"].values
r_ab = float(np.corrcoef(a_vec, b_vec)[0, 1])

fig, ax = plt.subplots(figsize=(8.8, 6.2))
y_pos = np.arange(len(sub_grp))
bar_colors = [C_SFT if d > 0.1 else (C_GRPO if d < -0.1 else "#999999") for d in diff_vec]

ax.barh(y_pos, diff_vec, height=0.64, color=bar_colors, edgecolor="#222222", lw=0.8)

for y_i, row in sub_grp.iterrows():
    d_val = row["diff_sft_grpo"]
    star = "*" if (row["subject"] in sig_sft or row["subject"] in sig_grpo) else ""
    if d_val >= 0:
        ax.text(d_val + 0.03, y_i, f"{d_val:+.2f}{star}", va="center", ha="left",
                fontsize=9.2, fontweight="bold", color=C_SFT)
    else:
        col_t = C_GRPO if d_val < -0.1 else "#555555"
        ax.text(d_val - 0.03, y_i, f"{d_val:+.2f}{star}", va="center", ha="right",
                fontsize=9.2, fontweight="bold", color=col_t)

ax.axvline(0, color="#222222", lw=1.3)
ax.axhline(6.5, color="#555555", lw=1.1, ls="--")

ylabels = [f"{r['short']}  [2PL weight = {r['a_mean']:.2f}]" for _, r in sub_grp.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(ylabels, fontsize=9.6, fontweight="bold")
ax.set_xlim(-1.55, 1.78)
ax.set_xlabel("Subject Mean Log-Odds Gain Difference:   <--- GRPO Higher     |     CoT SFT Higher --->",
              fontsize=11.0, fontweight="bold")
set_clean_header(
    ax,
    "Plot 2. Declarative Factual Subjects Favor GRPO; Procedural Math Subjects Favor CoT SFT",
    "15 seen S1 subject means sorted by 2PL weight (a and b share one axis, r = -0.88; partial r = 0.17 given Base log-odds; * = >95th-pct null)",
    title_size=11.6, sub_size=8.6
)
save_fig(fig, "Plot02_subject_weights_and_gains")

# ==============================================================================
# PLOT 03: Regression of Post-Training Log-Odds on Base Model Log-Odds
# ==============================================================================
t9r = pd.read_csv(os.path.join(IPL, "T9r", "items_wide.csv"))
t9r["item_id"] = t9r["item_id"].astype(str)
w0_bank["bank_row_id"] = w0_bank["bank_row_id"].astype(str)
m_t9r = pd.merge(t9r, w0_bank, left_on="item_id", right_on="bank_row_id", suffixes=("_t9", ""))

Lb_t9 = safe_logit(m_t9r["p_key_base"].values, 1e-4)
L_sft_0 = safe_logit(m_t9r["p_key_base"].values + m_t9r["dp_key_sft_0_70"].values, 1e-4)
L_rl_0 = safe_logit(m_t9r["p_key_base"].values + m_t9r["dp_key_rl_0_666"].values, 1e-4)

tau_rl_0, c_rl_0 = np.polyfit(Lb_t9, L_rl_0, 1)
tau_sf_0, c_sf_0 = np.polyfit(Lb_t9, L_sft_0, 1)

fig, ax = plt.subplots(figsize=(8.0, 5.9))
ax.scatter(Lb_t9, L_rl_0, alpha=0.22, s=16, color=C_GRPO, edgecolors="none",
           label=f"GRPO Items (Seed 0, J = {len(m_t9r)})")
ax.scatter(Lb_t9, L_sft_0, alpha=0.22, s=16, color=C_SFT, edgecolors="none",
           label=f"CoT SFT Items (Seed 0, J = {len(m_t9r)})")

xs_lb = np.linspace(-9.2, 9.2, 100)
ax.plot(xs_lb, xs_lb, color="#333333", lw=1.6, ls="--",
        label="Identity Line (y = x, slope = 1.00)")
ax.plot(xs_lb, tau_rl_0 * xs_lb + c_rl_0, color="#8b0000", lw=2.8,
        label=f"GRPO OLS Fit: slope = {tau_rl_0:.2f}, intercept = {c_rl_0:+.2f}")
ax.plot(xs_lb, tau_sf_0 * xs_lb + c_sf_0, color="#003380", lw=2.8,
        label=f"CoT SFT OLS Fit: slope = {tau_sf_0:.2f}, intercept = {c_sf_0:+.2f}")

# Mark the +/- 9.21 clipping boundary cleanly
ax.axhline(9.21, color="#888888", lw=1.0, ls=":", label="Log-Odds Clip Boundary (+/-9.21 at p = 0.0001 / 0.9999)")
ax.axhline(-9.21, color="#888888", lw=1.0, ls=":")

ax.set_xlim(-9.6, 9.6)
ax.set_ylim(-9.8, 10.3)
ax.set_xlabel("Base Model Initial Question Confidence (Log-Odds, Token-0 Readout)", fontsize=11.2, fontweight="bold")
ax.set_ylabel("Post-Training Question Confidence (Log-Odds)", fontsize=11.2, fontweight="bold")
set_clean_header(
    ax,
    "Plot 3. Within-Model Prior Scaling: How SFT and GRPO Regress on Base Confidence",
    "Token-0 readout (Seed 0): GRPO regression slope = 1.87 (> 1, gain amplification); CoT SFT slope = 0.54 (< 1, item reordering)",
    title_size=11.8, sub_size=8.9
)
ax.legend(loc="lower right", frameon=True, fontsize=9.0)
save_fig(fig, "Plot03_prior_scaling_scatter")

# ==============================================================================
# PLOT 04: Single-Draw vs. 16-Draw Majority Vote Crossover (with Tie Range Bar)
# ==============================================================================
s1_stats = pd.read_csv(os.path.join(IPL, "S1", "stats.csv"))
s1_pool = s1_stats[s1_stats["order"] == "pooled"].pivot(index="state", columns="statistic", values="value")
s1_rw1 = pd.read_csv(os.path.join(IPL, "S1", "rewrite1_parsed.csv")).set_index("state")

sft_states = [f"t9_sft_{i}_70" for i in [0, 1, 2]]
grpo_states = [f"t9_rl_{i}_666" for i in [0, 1, 2]]

# Values in %
b_mean16 = float(s1_pool.loc["t9_base", "mean_at_16"]) * 100.0
b_maj_low = float(s1_pool.loc["t9_base", "maj_at_16"]) * 100.0
b_maj_high = float(s1_rw1.loc["t9_base", "maj_tie_correct"]) * 100.0
b_maj_mid = 0.5 * (b_maj_low + b_maj_high)
b_pass16 = float(s1_pool.loc["t9_base", "pass_at_16"]) * 100.0

s_mean16 = float(s1_pool.loc[sft_states, "mean_at_16"].mean()) * 100.0
s_maj_low = float(s1_pool.loc[sft_states, "maj_at_16"].mean()) * 100.0
s_maj_high = float(s1_rw1.loc[sft_states, "maj_tie_correct"].mean()) * 100.0
s_maj_mid = 0.5 * (s_maj_low + s_maj_high)
s_pass16 = float(s1_pool.loc[sft_states, "pass_at_16"].mean()) * 100.0

g_mean16 = float(s1_pool.loc[grpo_states, "mean_at_16"].mean()) * 100.0
g_maj_low = float(s1_pool.loc[grpo_states, "maj_at_16"].mean()) * 100.0
g_maj_high = float(s1_rw1.loc[grpo_states, "maj_tie_correct"].mean()) * 100.0
g_maj_mid = 0.5 * (g_maj_low + g_maj_high)
g_pass16 = float(s1_pool.loc[grpo_states, "pass_at_16"].mean()) * 100.0

fig, ax = plt.subplots(figsize=(8.0, 5.6))
xm = np.array([0, 1, 2])

# Plot lines using the strict Tie=Wrong Maj@16 as the solid anchor, plus vertical tie-range bars at x=1
ax.plot(xm, [b_mean16, b_maj_low, b_pass16], marker="o", ms=8, lw=2.2, ls="--", color=C_BASE,
        label="Base Model (Llama-3.2-1B-Instruct)")
ax.plot(xm, [g_mean16, g_maj_low, g_pass16], marker="s", ms=9, lw=2.8, color=C_GRPO,
        label="After GRPO (3-Seed Mean, Low Entropy = 0.28)")
ax.plot(xm, [s_mean16, s_maj_low, s_pass16], marker="^", ms=9, lw=2.8, color=C_SFT,
        label="After CoT SFT (3-Seed Mean, High Entropy = 0.94)")

# Vertical range bars on Majority Vote (x = 1) showing [Tie=Wrong, Tie=Correct]
ax.errorbar([1.0], [s_maj_mid], yerr=[[s_maj_mid - s_maj_low], [s_maj_high - s_maj_mid]],
            fmt="none", ecolor=C_SFT, elinewidth=4.0, capsize=7, capthick=2.0, zorder=5,
            label="Majority-Vote Tie-Rule Range (Bottom = Tie Wrong, Top = Tie Correct)")
ax.errorbar([1.0], [g_maj_mid], yerr=[[g_maj_mid - g_maj_low], [g_maj_high - g_maj_mid]],
            fmt="none", ecolor=C_GRPO, elinewidth=4.0, capsize=7, capthick=2.0, zorder=5)
ax.errorbar([1.0], [b_maj_mid], yerr=[[b_maj_mid - b_maj_low], [b_maj_high - b_maj_mid]],
            fmt="none", ecolor=C_BASE, elinewidth=3.5, capsize=6, capthick=1.8, zorder=4)

# Direct clean percentage labels (positioned with clear margin inside axes)
ax.text(0.05, g_mean16 + 1.2, f"GRPO: {g_mean16:.1f}%", ha="left", va="bottom", color=C_GRPO, fontweight="bold", fontsize=10)
ax.text(0.05, s_mean16 - 1.2, f"SFT: {s_mean16:.1f}%", ha="left", va="top", color=C_SFT, fontweight="bold", fontsize=10)
ax.text(0.05, b_mean16 - 1.2, f"Base: {b_mean16:.1f}%", ha="left", va="top", color=C_BASE, fontweight="bold", fontsize=9.5)

ax.text(1.06, s_maj_high + 0.6, f"SFT: {s_maj_low:.1f}%–{s_maj_high:.1f}%", ha="left", va="bottom", color=C_SFT, fontweight="bold", fontsize=10)
ax.text(1.06, g_maj_low - 1.2, f"GRPO: {g_maj_low:.1f}%–{g_maj_high:.1f}%", ha="left", va="top", color=C_GRPO, fontweight="bold", fontsize=10)
ax.text(1.06, b_maj_low - 1.0, f"Base: {b_maj_low:.1f}%–{b_maj_high:.1f}%", ha="left", va="top", color=C_BASE, fontweight="bold", fontsize=9.5)

ax.text(2.04, s_pass16 + 0.5, f"SFT: {s_pass16:.1f}%", ha="left", va="center", color=C_SFT, fontweight="bold", fontsize=10)
ax.text(2.04, b_pass16 - 1.2, f"Base: {b_pass16:.1f}%", ha="left", va="center", color=C_BASE, fontweight="bold", fontsize=9.5)
ax.text(2.04, g_pass16, f"GRPO: {g_pass16:.1f}%", ha="left", va="center", color=C_GRPO, fontweight="bold", fontsize=10)

ax.set_xticks(xm)
ax.set_xticklabels([
    "Single Draw\n(Mean@16)",
    "16-Draw Majority Vote\n(Maj@16)",
    "16-Draw Coverage\n(Pass@16)"
], fontsize=11.0, fontweight="bold")
ax.set_xlim(-0.20, 2.40)
ax.set_ylim(33, 103)
ax.set_ylabel("Accuracy on 1,500-Question MMLU Bank (%)", fontsize=11.2, fontweight="bold")
set_clean_header(
    ax,
    "Plot 4. GRPO Wins on a Single Draw, But CoT SFT Crosses Above on Majority Vote",
    "S1 (16 draws x 2 orders): SFT Maj@16 exceeds GRPO in 2 of 3 seeds pooled (0 of 3 on order 1 alone); vertical bars show tie-rule range",
    title_size=12.0, sub_size=9.0
)
ax.legend(loc="upper left", frameon=True, fontsize=9.0)
save_fig(fig, "Plot04_readout_crossover")

# ==============================================================================
# PLOT 05: Unit Q1 RL Training Curves & Early Plateau at Step 83
# ==============================================================================
traj = pd.read_csv(os.path.join(IPL, "Q1", "trajectory.csv"))
q1_stats = pd.read_csv(os.path.join(IPL, "Q1", "stats.csv"))
q1_pool = q1_stats[q1_stats["order"] == "pooled"].pivot(index="state", columns="statistic", values="value")

arm_styles = [
    ("grpo", "Unweighted GRPO (G=5 Rollouts)", C_GRPO, "o", "-"),
    ("ctt",  "Classical Pass-Rate Critic (G=1)", C_CTT, "s", "--"),
    ("pop0", "2PL IRT Unweighted Critic (G=1)", C_POP0, "^", "-."),
    ("pop1", "2PL IRT Weighted Critic (G=1)", C_POP1, "D", ":"),
]

fig, ax = plt.subplots(figsize=(8.2, 5.6))
for mode, lbl, col, mk, ls in arm_styles:
    sub = traj[(traj["mode"] == mode) & (traj["seed"] == 0)].sort_values("updates")
    sts = [f"q1_{mode}_{i}" for i in [0, 1, 2]]
    final_3seed = float(q1_pool.loc[sts, "mean_at_16"].mean()) * 100.0
    ax.plot(sub["updates"], sub["mean_at_16"] * 100.0, marker=mk, ms=8, lw=2.5, ls=ls, color=col,
            label=f"{lbl} (Seed 0 curve | 3-seed final = {final_3seed:.2f}%)")
    ax.scatter([338], [final_3seed], s=95, marker=mk, color=col, edgecolors="black", lw=1.1, zorder=6)

ax.axvline(83, color="#333333", lw=1.4, ls=":")
ax.axvspan(83, 345, color="#f2f2f2", alpha=0.6, label="Post-Step-83 Plateau (84%–95% of Seed 0 gain finished by step 83)")

ax.set_xticks([0, 83, 166, 250, 333])
ax.set_xticklabels(["Step 0\n(Base)", "Step 83\n(Seed 0)", "Step 166\n(Seed 0)", "Step 250\n(Seed 0)", "Step 333\n(Final)"],
                   fontsize=10.2, fontweight="bold")
ax.set_xlim(-10, 350)
ax.set_ylim(37.0, 50.0)
ax.set_xlabel("RL Training Updates (Trajectory Saved for Seed 0; Markers at Right = 3-Seed Finals)", fontsize=11.0, fontweight="bold")
ax.set_ylabel("Single-Draw Accuracy Mean@16 (%)", fontsize=11.2, fontweight="bold")
set_clean_header(
    ax,
    "Plot 5. Population 2PL Critics Do Not Beat Unweighted GRPO or Classical Baseline",
    "Pre-registered Q1 verdict: 2PL critic vs. classical band mean d = -0.32% (SE = 0.39%, fails rule); all 4 arms land within 46.9%–47.5%",
    title_size=11.8, sub_size=9.0
)
ax.legend(loc="lower right", frameon=True, fontsize=8.8)
save_fig(fig, "Plot05_q1_training_curves")

# ==============================================================================
# SCHEMATIC 01: Visual Intuition for 2PL Item Curves — Steep Factuals vs. Flat Math
# ==============================================================================
sub_idx = sub_grp.set_index("subject")
a_mkt, b_mkt = float(sub_idx.loc["marketing", "a_mean"]), float(sub_idx.loc["marketing", "b_mean"])
a_soc, b_soc = float(sub_idx.loc["sociology", "a_mean"]), float(sub_idx.loc["sociology", "b_mean"])
a_phy, b_phy = float(sub_idx.loc["high_school_physics", "a_mean"]), float(sub_idx.loc["high_school_physics", "b_mean"])
a_mth, b_mth = float(sub_idx.loc["high_school_mathematics", "a_mean"]), float(sub_idx.loc["high_school_mathematics", "b_mean"])

theta_grid = np.linspace(-3.2, 3.2, 300)
def icc(th, a, b):
    return 100.0 / (1.0 + np.exp(-a * (th - b)))

fig, ax = plt.subplots(figsize=(8.0, 5.6))
ax.plot(theta_grid, icc(theta_grid, a_mkt, b_mkt), lw=3.0, color=C_GRPO,
        label=f"Marketing (Steep Slope a = {a_mkt:.2f}, Easy b = {b_mkt:+.2f}) ---> 5.2x Score Weight")
ax.plot(theta_grid, icc(theta_grid, a_soc, b_soc), lw=2.4, ls="--", color="#e06666",
        label=f"Sociology (Steep Slope a = {a_soc:.2f}, Easy b = {b_soc:+.2f}) ---> 4.1x Score Weight")
ax.plot(theta_grid, icc(theta_grid, a_phy, b_phy), lw=2.4, ls="--", color="#5a9bd5",
        label=f"High School Physics (Flat Slope a = {a_phy:.2f}, Hard b = {b_phy:+.2f}) ---> 1.2x Weight")
ax.plot(theta_grid, icc(theta_grid, a_mth, b_mth), lw=3.0, color=C_SFT,
        label=f"High School Math (Flat Slope a = {a_mth:.2f}, Hard b = {b_mth:+.2f}) ---> 1.0x Weight")

ax.axhline(50.0, color="#888888", lw=1.0, ls=":")
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(0, 102)
ax.set_xlabel("Population Ability Axis Across 102 Leaderboard LLMs (Low  --->  High)", fontsize=11.2, fontweight="bold")
ax.set_ylabel("2PL Predicted Probability Correct (%)", fontsize=11.2, fontweight="bold")
set_clean_header(
    ax,
    "Schematic 1. Intuition: Why 2PL Ability Weights Factual Recall 5.2x Over Math",
    "In 2PL IRT, curve steepness (discrimination a) is the item's score weight: factual recall scales steeply; math is flat",
    title_size=11.8, sub_size=9.0
)
ax.legend(loc="upper left", frameon=True, fontsize=9.0)
save_fig(fig, "Schematic01_population_vs_posttraining")

# ==============================================================================
# SCHEMATIC 02: Active Learning Zone (ZPD) & Bernoulli Variance Curve (Unit M1)
# ==============================================================================
w1_pr = pd.read_csv(os.path.join(IPL, "W1", "pass_rate.csv"))
counts_k = w1_pr["correct"].value_counts().sort_index()
ks = np.arange(0, 9)
shares_k = [float(counts_k.get(k, 0) / len(w1_pr) * 100.0) for k in ks]
bar_cols = [C_ZPD if (2 <= k <= 6) else "#cccccc" for k in ks]

fig, ax = plt.subplots(figsize=(8.2, 5.5))
bars_w1 = ax.bar(ks, shares_k, width=0.60, color=bar_cols, edgecolor="#222222", lw=0.85)
for b_i, sh in zip(bars_w1, shares_k):
    ax.text(b_i.get_x() + b_i.get_width()/2, sh + 0.5, f"{sh:.1f}%", ha="center", va="bottom",
            fontsize=9.2, fontweight="bold", color="#222222")

ax_twin = ax.twinx()
x_dense = np.linspace(0, 8, 200)
p_grid = x_dense / 8.0
bernoulli_var = p_grid * (1.0 - p_grid)
ax_twin.plot(x_dense, bernoulli_var, color=C_GRPO, lw=2.8, ls="--",
             label="Reward Variance Curve p(1 - p)")
ax_twin.set_ylabel("Bernoulli Reward Variance p(1 - p)", color=C_GRPO, fontsize=11.0, fontweight="bold")
ax_twin.set_ylim(0, 0.35)
ax_twin.grid(False)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[
    Patch(facecolor=C_ZPD, edgecolor="#222222", label="Active ZPD Band (2/8 to 6/8 correct = 51.7% of prompts): High RL Gradient Variance"),
    Patch(facecolor="#cccccc", edgecolor="#222222", label="Exterior Prompts (0/8, 1/8, 7/8, 8/8 = 48.3% of prompts): Near-Zero Variance"),
    Line2D([0], [0], color=C_GRPO, lw=2.8, ls="--", label="Bernoulli Reward Variance p(1 - p) (Peaks at 50% Pass Rate)")
], loc="upper right", frameon=True, fontsize=8.8)

ax.set_xticks(ks)
ax.set_xticklabels([f"{k}/8\n({int(round(k/8*100))}%)" for k in ks], fontsize=9.8, fontweight="bold")
ax.set_ylim(0, 34)
ax.set_xlabel("Model's Empirical Prompt Pass Rate (Correct Rollouts Out of 8, W1 Pool N = 10,682)", fontsize=11.0, fontweight="bold")
ax.set_ylabel("Share of Training Prompts (%)", fontsize=11.2, fontweight="bold")
set_clean_header(
    ax,
    "Schematic 2. Within-Model Psychometrics: Targeting the 20%–80% Pass-Rate Zone (Unit M1)",
    "48.3% of training prompts sit at the extremes (0/8, 1/8, 7/8, 8/8) where Bernoulli reward variance p(1 - p) approaches zero",
    title_size=11.8, sub_size=9.0
)
save_fig(fig, "Schematic02_multidraw_and_m1_zpd")

print("ALL 5 PURE VISUAL PLOTS + 2 SCHEMATICS GENERATED SUCCESSFULLY.")
