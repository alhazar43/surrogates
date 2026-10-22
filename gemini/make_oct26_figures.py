import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.95,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# Portable repository resolution (works in item-params-llm, irt4llm, or surrogates)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if (REPO_ROOT / "results" / "W0").exists():
    IPL = str(REPO_ROOT / "results")
else:
    IPL = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\item-params-llm\results"

OUT_DIR = str(SCRIPT_DIR / "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C_BASE = "#555555"
C_SFT = "#1f6feb"     # Royal blue (SFT)
C_GRPO = "#d9381e"    # Vermilion red (GRPO)
C_CTT = "#e69f00"     # Amber
C_POP0 = "#8a3ffc"    # Purple
C_POP1 = "#009e73"    # Teal
C_ZPD = "#118ab2"     # Cyan-blue

def safe_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def set_title_and_subtitle(ax, title_text, sub_text, title_size=12.2, sub_size=8.9):
    ax.set_title(title_text, fontsize=title_size, fontweight="bold", pad=24)
    ax.text(0.5, 1.02, sub_text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=sub_size, color="#333333")

def save_fig(fig, name):
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path}")

# ==============================================================================
# LOAD CORE DATASETS
# ==============================================================================
w0_bank = pd.read_csv(os.path.join(IPL, "W0", "items_bank.csv"))
w0_valid = w0_bank.dropna(subset=["a", "b"]).copy()
w0_valid["bank_row_id"] = w0_valid["bank_row_id"].astype(str)

t9r = pd.read_csv(os.path.join(IPL, "T9r", "items_wide.csv"))
t9r["item_id"] = t9r["item_id"].astype(str)
m_t9r = pd.merge(t9r, w0_valid, left_on="item_id", right_on="bank_row_id", suffixes=("_t9", ""))

s1_stats = pd.read_csv(os.path.join(IPL, "S1", "stats.csv"))
s1_pool = s1_stats[s1_stats["order"] == "pooled"].pivot(index="state", columns="statistic", values="value")
s1_rw1 = pd.read_csv(os.path.join(IPL, "S1", "rewrite1_parsed.csv")).set_index("state")
q1_stats = pd.read_csv(os.path.join(IPL, "Q1", "stats.csv"))
q1_pool = q1_stats[q1_stats["order"] == "pooled"].pivot(index="state", columns="statistic", values="value")

sft_states = [f"t9_sft_{i}_70" for i in [0, 1, 2]]
grpo_states = [f"t9_rl_{i}_666" for i in [0, 1, 2]]

# ==============================================================================
# FIG 01 (Slide 3): Human vs. LLM Item Difficulty on 704 Eedi Math Items (Unit E1)
# ==============================================================================
e1_diff = pd.read_csv(os.path.join(IPL, "E1", "difficulty.csv")).dropna(subset=["b_students", "b_llms"]).copy()
e1_scores = pd.read_csv(os.path.join(IPL, "E1", "llm_scores.csv"))
n_e1 = len(e1_diff)
n_zh_neg = int((e1_scores["zh_2pl"] < -1.96).sum())
n_llms_e1 = len(e1_scores)

b_stu = e1_diff["b_students"].values
b_llm = e1_diff["b_llms"].values
r_e1 = float(np.corrcoef(b_stu, b_llm)[0, 1])
rho_e1 = float(pd.Series(b_stu).corr(pd.Series(b_llm), method="spearman"))
m_raw, c_raw = np.polyfit(b_stu, b_llm, 1)

# Rank percentiles (0 to 100) since closure.md:29 notes only rank order reads
e1_diff["pct_stu"] = e1_diff["b_students"].rank(pct=True) * 100.0
e1_diff["pct_llm"] = e1_diff["b_llms"].rank(pct=True) * 100.0
p_stu = e1_diff["pct_stu"].values
p_llm = e1_diff["pct_llm"].values
m_pct, c_pct = np.polyfit(p_stu, p_llm, 1)

fig, ax = plt.subplots(figsize=(8.0, 6.2))
ax.scatter(p_stu, p_llm, alpha=0.45, s=26, color="#2b5c8f", edgecolors="none",
           label=f"Eedi Diagnostic Math Items ($J = {n_e1}$)")

xs_p = np.linspace(0, 100, 100)
ax.plot(xs_p, xs_p, color="#555555", lw=1.5, ls="--", label="Identity Line ($y = x$, Perfect Rank Agreement)")
ax.plot(xs_p, m_pct * xs_p + c_pct, color=C_GRPO, lw=2.6,
        label=f"Rank OLS Fit: slope $= {m_pct:+.3f}$, intercept $= {c_pct:.1f}$ (Spearman $\\rho = {rho_e1:+.3f}$)")

ax.set_xlim(-2, 102)
ax.set_ylim(-2, 112)
ax.set_xlabel("Human Student 2PL Difficulty Percentile ($N = 4,918$ Students, Easy $\\rightarrow$ Hard)", fontsize=11)
ax.set_ylabel("61-LLM Difficulty Percentile ($N = 61$ LLMs, Easy $\\rightarrow$ Hard)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 1. What Is Hard for Human Students Is Orthogonal to What Is Hard for LLMs",
    f"Unit E1 ($J = {n_e1}$ items): Spearman $\\rho = {rho_e1:+.3f}$ (raw $b_j$ Pearson $r = {r_e1:+.3f}$, OLS slope $= {m_raw:+.2f}$; {n_zh_neg}/{n_llms_e1} LLMs have $Z_h < -1.96$)"
)
ax.text(2.5, 106.5,
        "Per `closure.md:29`, raw 61-LLM $b_j \\in [-1574, +912]$ rests on the rejected 61-LLM fit, so rank order is plotted.",
        fontsize=8.6, color="#444444", style="italic")
ax.legend(loc="lower right", frameon=True, fontsize=9.2)
save_fig(fig, "Fig01_human_vs_llm_difficulty")

# ==============================================================================
# FIG 02 (Slide 4): The Central Attribution Puzzle — Accuracy vs. 102-LLM 2PL θ
# ==============================================================================
pf_df = pd.read_csv(os.path.join(IPL, "Q2g", "s1", "personfit.csv"))
pf_all = pf_df[pf_df["variant"] == "all"].set_index("state")
pf_par = pf_df[pf_df["variant"] == "parsed"].set_index("state")

acc_base = float(pf_all.loc["t9_base", "accuracy"])
acc_sft = float(pf_all.loc[sft_states, "accuracy"].astype(float).mean())
acc_grpo = float(pf_all.loc[grpo_states, "accuracy"].astype(float).mean())

th_b = float(pf_all.loc["t9_base", "theta"])
th_s = float(pf_all.loc[sft_states, "theta"].astype(float).mean())
th_g = float(pf_all.loc[grpo_states, "theta"].astype(float).mean())

sft_share_acc = (acc_sft - acc_base) / (acc_grpo - acc_base) * 100.0
grpo_share_acc = (acc_grpo - acc_sft) / (acc_grpo - acc_base) * 100.0
sft_share_th = (th_s - th_b) / (th_g - th_b) * 100.0
grpo_share_th = (th_g - th_s) / (th_g - th_b) * 100.0

acc_b_p = float(pf_par.loc["t9_base", "accuracy"])
acc_s_p = float(pf_par.loc[sft_states, "accuracy"].astype(float).mean())
acc_g_p = float(pf_par.loc[grpo_states, "accuracy"].astype(float).mean())
th_b_p = float(pf_par.loc["t9_base", "theta"])
th_s_p = float(pf_par.loc[sft_states, "theta"].astype(float).mean())
th_g_p = float(pf_par.loc[grpo_states, "theta"].astype(float).mean())
sft_acc_p = (acc_s_p - acc_b_p) / (acc_g_p - acc_b_p) * 100.0
sft_th_p = (th_s_p - th_b_p) / (th_g_p - th_b_p) * 100.0

fig, ax = plt.subplots(figsize=(8.8, 4.9))
y_bars = np.array([1.0, 0.0])
bar_h = 0.44

ax.barh(y_bars, [sft_share_acc, sft_share_th], height=bar_h, color=C_SFT, edgecolor="black", lw=0.9,
        label="Share Attributed to Stage 1: Base $\\rightarrow$ CoT SFT (`t9_sft`)")
ax.barh(y_bars, [grpo_share_acc, grpo_share_th], left=[sft_share_acc, sft_share_th], height=bar_h,
        color=C_GRPO, edgecolor="black", lw=0.9,
        label="Share Attributed to Stage 2: SFT $\\rightarrow$ GRPO (`t9_rl`)")

ax.plot([sft_share_acc, sft_share_th], [1.0 - bar_h/2, 0.0 + bar_h/2], color="#222222", lw=2.0, ls="--", zorder=5)

ax.text(sft_share_acc / 2, 1.0,
        f"CoT SFT: {sft_share_acc:.1f}%\n(+{acc_sft - acc_base:.3f} accuracy, 8.9–10.5 SE)",
        ha="center", va="center", color="white", fontweight="bold", fontsize=10.5)
ax.text(sft_share_acc + grpo_share_acc / 2, 1.0,
        f"GRPO: {grpo_share_acc:.1f}%\n(+{acc_grpo - acc_sft:.3f} accuracy, 3.8–5.9 SE)",
        ha="center", va="center", color="white", fontweight="bold", fontsize=10.5)

ax.text(sft_share_th / 2, 0.0,
        f"SFT: {sft_share_th:.1f}%\n(+{th_s - th_b:.3f} θ,\n2.9–5.1 SE)",
        ha="center", va="center", color="white", fontweight="bold", fontsize=10.0)
ax.text(sft_share_th + grpo_share_th / 2, 0.0,
        f"GRPO: {grpo_share_th:.1f}%\n(+{th_g - th_s:.3f} θ, 8.6–10.5 SE)",
        ha="center", va="center", color="white", fontweight="bold", fontsize=10.5)

ax.set_yticks(y_bars)
ax.set_yticklabels([
    "Classical Mean Accuracy\n(Equal Item Weights)",
    "102-LLM 2PL Ability $\\hat{\\theta}$\n(Weighted by $a_j^{\\mathrm{pop}}$)"
], fontsize=11, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(-0.48, 1.58)
ax.set_xlabel("Share of Total Post-Training Gain from Base to GRPO (%)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 2. The Attribution Puzzle: Classical Accuracy and 2PL $\\hat{\\theta}$ Flip Stage Credit",
    f"Seen S1 states (`all` draws; `parsed` gives {sft_acc_p:.0f}/{100-sft_acc_p:.0f} vs {sft_th_p:.0f}/{100-sft_th_p:.0f}; on 12 unseen Q1 finals vs Base, $\\hat{{\\theta}}$ is sharper in 3/12)"
)
ax.legend(loc="upper center", frameon=True, fontsize=9.4, ncol=2)
save_fig(fig, "Fig02_attribution_puzzle")

# ==============================================================================
# PREPARE 15-SUBJECT DATA FOR FIGS 03 & 04
# ==============================================================================
th_base = float(pf_all.loc["t9_base", "theta"])
dth_sft = np.mean([float(pf_all.loc[f"t9_sft_{s}_70", "theta"]) - th_base for s in [0, 1, 2]])
dth_grpo = np.mean([float(pf_all.loc[f"t9_rl_{s}_666", "theta"]) - th_base for s in [0, 1, 2]])

fl_s1 = pd.read_csv(os.path.join(IPL, "Q2g", "s1", "flags.csv"))
sub_rows_sft = fl_s1[(fl_s1["variant"] == "all") & (fl_s1["reading"] == "A3_subject_mean_c") & (fl_s1["method"] == "sft") & (fl_s1["state"] == "pooled_states")]
sub_rows_grpo = fl_s1[(fl_s1["variant"] == "all") & (fl_s1["reading"] == "A3_subject_mean_c") & (fl_s1["method"] == "grpo") & (fl_s1["state"] == "pooled_states")]
c_sft_map = dict(zip(sub_rows_sft["subject"], sub_rows_sft["value"].astype(float)))
c_grpo_map = dict(zip(sub_rows_grpo["subject"], sub_rows_grpo["value"].astype(float)))

sub_grp = w0_valid.groupby("subject").agg(a_mean=("a", "mean"), b_mean=("b", "mean")).reset_index()
sub_grp["c_sft"] = sub_grp["subject"].map(c_sft_map)
sub_grp["c_grpo"] = sub_grp["subject"].map(c_grpo_map)
sub_grp["dL_sft"] = sub_grp["c_sft"] + sub_grp["a_mean"] * dth_sft
sub_grp["dL_grpo"] = sub_grp["c_grpo"] + sub_grp["a_mean"] * dth_grpo
sub_grp["diff_sft_grpo"] = sub_grp["dL_sft"] - sub_grp["dL_grpo"]
sub_grp = sub_grp.sort_values("a_mean").reset_index(drop=True)

SHORT_NAMES = {
    "high_school_mathematics": "HS Math",
    "high_school_physics": "HS Physics",
    "high_school_statistics": "HS Statistics",
    "elementary_mathematics": "Elem Math",
    "econometrics": "Econometrics",
    "high_school_macroeconomics": "HS Macroecon",
    "high_school_microeconomics": "HS Microecon",
    "clinical_knowledge": "Clinical Knowl",
    "philosophy": "Philosophy",
    "high_school_biology": "HS Biology",
    "high_school_world_history": "World History",
    "high_school_geography": "HS Geography",
    "prehistory": "Prehistory",
    "sociology": "Sociology",
    "marketing": "Marketing",
}
sub_grp["short"] = sub_grp["subject"].map(SHORT_NAMES)

# ==============================================================================
# FIG 03 (Slide 5): Clue #1 — The 15 MMLU Subjects in the (a_m^pop, b_m^pop) Plane
# ==============================================================================
a_vec = sub_grp["a_mean"].values
b_vec = sub_grp["b_mean"].values
r_ab_sub = float(np.corrcoef(a_vec, b_vec)[0, 1])
m_ab, c_ab = np.polyfit(a_vec, b_vec, 1)

fig, ax = plt.subplots(figsize=(8.8, 6.2))
pt_cols = [C_SFT if a < 0.56 else (C_GRPO if a > 0.68 else "#777777") for a in a_vec]
ax.scatter(a_vec, b_vec, c=pt_cols, s=110, edgecolors="black", lw=0.9, zorder=4)

xs_a = np.linspace(0.22, 1.50, 100)
ax.plot(xs_a, m_ab * xs_a + c_ab, color="#222222", lw=2.2, ls="--",
        label=f"OLS Fit ($n=15$): slope $= {m_ab:+.2f}$, intercept $= {c_ab:+.2f}$ ($r = {r_ab_sub:+.4f}, R^2 = {r_ab_sub**2*100:.1f}\\%$)")

offsets_03 = {
    "HS Math": (9, -3),
    "HS Physics": (9, -4),
    "Econometrics": (9, 5),
    "Elem Math": (9, -12),
    "HS Macroecon": (9, 4),
    "HS Statistics": (9, -6),
    "HS Microecon": (9, -5),
    "Clinical Knowl": (9, 4),
    "Philosophy": (9, 5),
    "Prehistory": (9, 4),
    "World History": (-110, 8),
    "HS Biology": (-105, -14),
    "HS Geography": (9, -3),
    "Sociology": (9, 5),
    "Marketing": (-108, 6),
}
for _, row in sub_grp.iterrows():
    dx, dy = offsets_03.get(row["short"], (6, 4))
    ax.annotate(f"{row['short']} (a={row['a_mean']:.2f})",
                (row["a_mean"], row["b_mean"]),
                xytext=(dx, dy), textcoords="offset points", fontsize=8.7, fontweight="bold")

ax.axvspan(0.20, 0.56, color=C_SFT, alpha=0.06, label="Low-$\\bar{a}_m^{\\mathrm{pop}}$ Quantitative Subjects ($0.28\\text{–}0.55$)")
ax.axvspan(0.68, 1.55, color=C_GRPO, alpha=0.06, label="High-$\\bar{a}_m^{\\mathrm{pop}}$ Factual/Declarative Subjects ($0.69\\text{–}1.45$)")

ax.set_xlim(0.20, 1.55)
ax.set_ylim(-1.25, 2.65)
ax.set_xlabel("Subject Mean 102-LLM Discrimination $\\bar{a}_m^{\\mathrm{pop}}$ (`W0/items_bank.csv`, Weight in 2PL $\\hat{\\theta}$)", fontsize=11)
ax.set_ylabel("Subject Mean 102-LLM Difficulty $\\bar{b}_m^{\\mathrm{pop}}$ (`W0/items_bank.csv`)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 3. Clue #1: 102-LLM Discrimination $a_j^{\\mathrm{pop}}$ Weights Factual Recall $5.2\\times$ Over Math",
    "Across $n = 15$ MMLU subjects ($J = 1,472$ items), Marketing ($\\bar{a}_m = 1.45$) receives $5.2\\times$ the 2PL score weight of HS Math ($\\bar{a}_m = 0.28$)",
    title_size=12.0
)
ax.legend(loc="upper right", frameon=True, fontsize=8.8)
save_fig(fig, "Fig03_mmlu_subject_irt_map")

# ==============================================================================
# FIG 04 (Slide 6): Clue #2 — Diverging Bar Chart of SFT vs. GRPO Gains Across the 15 Subjects
# ==============================================================================
diff_vec = sub_grp["diff_sft_grpo"].values
r_b = float(np.corrcoef(b_vec, diff_vec)[0, 1])
r_a = float(np.corrcoef(a_vec, diff_vec)[0, 1])
mb, cb = np.polyfit(b_vec, diff_vec, 1)

sp_rows = fl_s1[(fl_s1["variant"] == "all") & (fl_s1["reading"] == "A3_spearman_c_a")].set_index("state")["value"].astype(float)
rho_c_sft = [float(sp_rows[f"t9_sft_{s}_70"]) for s in [0, 1, 2]]
rho_c_grpo = [float(sp_rows[f"t9_rl_{s}_666"]) for s in [0, 1, 2]]

fig, ax = plt.subplots(figsize=(9.2, 6.5))
y_pos = np.arange(len(sub_grp))
bar_colors = [C_SFT if d > 0.1 else (C_GRPO if d < -0.1 else "#888888") for d in diff_vec]

bars = ax.barh(y_pos, diff_vec, height=0.64, color=bar_colors, edgecolor="black", lw=0.8)
for y_i, d_val in zip(y_pos, diff_vec):
    if d_val >= 0:
        ax.text(d_val + 0.03, y_i, f"{d_val:+.2f}", va="center", ha="left", fontsize=8.8, fontweight="bold", color=C_SFT)
    else:
        col_t = C_GRPO if d_val < -0.1 else "#444444"
        ax.text(d_val - 0.03, y_i, f"{d_val:+.2f}", va="center", ha="right", fontsize=8.8, fontweight="bold", color=col_t)

ax.axvline(0, color="#222222", lw=1.2)
ax.axhline(5.5, color="#444444", lw=1.1, ls="--")
ax.axhline(6.5, color="#444444", lw=1.1, ls="--")

ylabels = [f"{r['short']}  (a={r['a_mean']:.2f}, b={r['b_mean']:+.2f})" for _, r in sub_grp.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(ylabels, fontsize=9.2, fontweight="bold")
ax.set_xlim(-1.65, 1.85)
ax.set_xlabel("Subject Mean Log-Odds Advantage $\\overline{\\Delta L}_{\\mathrm{SFT},m} - \\overline{\\Delta L}_{\\mathrm{GRPO},m}$ ($K=32$ Rollouts, Seen S1 States)", fontsize=10.8)
set_title_and_subtitle(
    ax,
    "Fig 4. Clue #2: GRPO Leads All 8 High-$a_m^{\\mathrm{pop}}$ Subjects; SFT Leads 5 of 6 Low-$a_m^{\\mathrm{pop}}$ Subjects",
    f"Sorted by $\\bar{{a}}_m^{{\\mathrm{{pop}}}}$: $r(\\bar{{b}}_m^{{\\mathrm{{pop}}}}, \\Delta) = {r_b:+.4f}$ ($R^2 = {r_b**2*100:.1f}\\%$, OLS slope ${mb:+.2f}$); "
    f"$r(\\bar{{a}}_m^{{\\mathrm{{pop}}}}, \\Delta) = {r_a:+.4f}$ ($R^2 = {r_a**2*100:.1f}\\%$)",
    title_size=11.8
)

# Place red box in the completely empty top-right quadrant (x > 0.12, y in [8..13])
ax.text(0.14, 10.6,
        "Top 8 High-$\\bar{a}_m^{\\mathrm{pop}}$ Subjects (0.69–1.45):\n"
        "• GRPO leads 8/8 (Left / Red bars)\n"
        f"• Item residual $\\rho(c_{{\\mathrm{{GRPO}}}}, a^{{\\mathrm{{pop}}}})$ = {min(rho_c_grpo):+.3f} to {max(rho_c_grpo):+.3f}",
        ha="left", va="center", fontsize=8.8, fontweight="bold", color=C_GRPO,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#fff5f5", edgecolor=C_GRPO))

# Place blue box in the completely empty bottom-left quadrant (x < -0.22, y in [0..5])
ax.text(-1.58, 2.3,
        "Bottom 6 Low-$\\bar{a}_m^{\\mathrm{pop}}$ Subjects (0.28–0.55):\n"
        "• CoT SFT leads 5/6 at $\\pm 0.1$ logit (Right / Blue)\n"
        "• HS Stats ($-0.08$) tie; Microecon ($-0.08$) in middle\n"
        f"• Item residual $\\rho(c_{{\\mathrm{{SFT}}}}, a^{{\\mathrm{{pop}}}})$ = {min(rho_c_sft):+.3f} to {max(rho_c_sft):+.3f}",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=C_SFT,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#eef6ff", edgecolor=C_SFT))

save_fig(fig, "Fig04_subject_sft_vs_grpo_gains")

# ==============================================================================
# FIG 05 (Slide 7): Within-Model Prior Scaling L_{s,j} vs. L_{base,j} (T9r Seed 0)
# ==============================================================================
Lb_t9 = safe_logit(m_t9r["p_key_base"].values, 1e-4)
L_sft_0 = safe_logit(m_t9r["p_key_base"].values + m_t9r["dp_key_sft_0_70"].values, 1e-4)
L_rl_0 = safe_logit(m_t9r["p_key_base"].values + m_t9r["dp_key_rl_0_666"].values, 1e-4)

tau_rl_0, c_rl_0 = np.polyfit(Lb_t9, L_rl_0, 1)
tau_sf_0, c_sf_0 = np.polyfit(Lb_t9, L_sft_0, 1)
r2_dl_rl_0 = float(np.corrcoef(Lb_t9, L_rl_0 - Lb_t9)[0, 1] ** 2 * 100.0)
r2_dl_sf_0 = float(np.corrcoef(Lb_t9, L_sft_0 - Lb_t9)[0, 1] ** 2 * 100.0)

tau_sft_3, tau_rl_3 = [], []
for s in [0, 1, 2]:
    Ls = safe_logit(m_t9r["p_key_base"].values + m_t9r[f"dp_key_sft_{s}_70"].values, 1e-4)
    Lr = safe_logit(m_t9r["p_key_base"].values + m_t9r[f"dp_key_rl_{s}_666"].values, 1e-4)
    tau_sft_3.append(float(np.polyfit(Lb_t9, Ls, 1)[0]))
    tau_rl_3.append(float(np.polyfit(Lb_t9, Lr, 1)[0]))

fig, ax = plt.subplots(figsize=(8.4, 6.1))
ax.scatter(Lb_t9, L_rl_0, alpha=0.24, s=16, color=C_GRPO, edgecolors="none",
           label=f"GRPO Seed 0 (`rl_0_666`, $J={len(m_t9r)}$; upper band at $+9.2$ is $1-10^{{-4}}$ clip)")
ax.scatter(Lb_t9, L_sft_0, alpha=0.24, s=16, color=C_SFT, edgecolors="none",
           label=f"CoT SFT Seed 0 (`sft_0_70`, $J={len(m_t9r)}$)")

xs_lb = np.linspace(-9.2, 9.2, 100)
ax.plot(xs_lb, xs_lb, color="#333333", lw=1.6, ls="--", label="Identity Line ($L_s = L_{\\mathrm{base}}$, slope $\\tau = 1.00$, No Change)")
ax.plot(xs_lb, tau_rl_0 * xs_lb + c_rl_0, color="#8b0000", lw=2.6,
        label=f"GRPO OLS: slope $\\hat{{\\tau}}_0 = {tau_rl_0:.2f}$ (3-seed {min(tau_rl_3):.2f}–{max(tau_rl_3):.2f}), intercept $= {c_rl_0:+.2f}$, $R^2(\\Delta L) = {r2_dl_rl_0:.1f}\\%$")
ax.plot(xs_lb, tau_sf_0 * xs_lb + c_sf_0, color="#003380", lw=2.6,
        label=f"SFT OLS: slope $\\hat{{\\tau}}_0 = {tau_sf_0:.2f}$ (3-seed {min(tau_sft_3):.2f}–{max(tau_sft_3):.2f}), intercept $= {c_sf_0:+.2f}$, $R^2(\\Delta L) = {r2_dl_sf_0:.1f}\\%$")

ax.axhline(0, color="#888888", lw=0.8, ls=":")
ax.axvline(0, color="#888888", lw=0.8, ls=":")
ax.set_xlim(-9.6, 9.6)
ax.set_ylim(-9.8, 10.5)
ax.set_xlabel("Model's Own Base Token-0 Log-Odds $L_{\\mathrm{base},j}$ (`T9r/items_wide.csv`, $\\varepsilon=10^{-4}$)", fontsize=11)
ax.set_ylabel("Post-Training Token-0 Log-Odds $L_{s,j}$ (`T9r`, Seed 0)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 5. Within-Model Prior Scaling: GRPO Stretches $L_{\\mathrm{base},j}$ ($\\hat{\\tau} > 1$) While SFT Lifts Weak Items ($\\hat{\\tau} < 1$)",
    "Confound Check: Subject $\\bar{b}_m^{\\mathrm{pop}}$ correlates $-0.94$ with $\\overline{L}_{\\mathrm{base},m}$ (partial $r = 0.17$ `all` / $0.25$ `parsed`); on $K=32$ draws, $\\hat{\\tau} = 1.43\\text{–}1.46$ (GRPO) vs $0.49\\text{–}0.50$ (SFT)",
    title_size=11.5, sub_size=8.3
)
ax.legend(loc="lower right", frameon=True, fontsize=8.5)
save_fig(fig, "Fig05_within_model_prior_scaling")

# ==============================================================================
# FIG 06 (Slide 8): Consequence for Short-Test CAT — Sociology Bias & RMSE Crossover
# ==============================================================================
q2c_err = pd.read_csv(os.path.join(IPL, "Q2c", "errors.csv"))
q2c_high = q2c_err[(q2c_err["level"] == "band") & (q2c_err["band"] == "above 0.45")].sort_values("n")

ns = [25, 50, 100]
rmse_cl = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "classical")]["rmse"].iloc[0]) for n in ns]
rmse_irtr = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_random")]["rmse"].iloc[0]) for n in ns]
rmse_irt = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_adaptive")]["rmse"].iloc[0]) for n in ns]
bias_irt = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_adaptive")]["bias"].iloc[0]) for n in ns]

fig, ax = plt.subplots(figsize=(8.2, 5.5))
ax.plot(ns, rmse_cl, marker="o", ms=9, lw=2.6, color=C_SFT, label="Classical Random Sampling (`classical`, shrinks as $1/\\sqrt{n}$)")
ax.plot(ns, rmse_irt, marker="s", ms=9, lw=2.6, color=C_GRPO, label="Max-Info 2PL Adaptive CAT (`irt_adaptive`, greedy $(a_j^{\\mathrm{pop}})^2 p_j q_j$)")
ax.plot(ns, rmse_irtr, marker="^", ms=8, lw=2.0, ls="--", color=C_POP1, label="2PL Random Sampling (`irt_random`, random items + 2PL score)")

ax.text(25, rmse_cl[0] + 0.003, f"{rmse_cl[0]:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9.5)
ax.text(25, rmse_irt[0] + 0.003, f"{rmse_irt[0]:.3f}\n(bias {bias_irt[0]:+.3f})", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=8.5)
ax.text(25, rmse_irtr[0] - 0.004, f"{rmse_irtr[0]:.3f}", ha="center", va="top", color=C_POP1, fontweight="bold", fontsize=8.8)

ax.text(50, rmse_cl[1] + 0.004, f"classical: {rmse_cl[1]:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9.0)
ax.text(55, rmse_irt[1] - 0.004, f"adaptive: {rmse_irt[1]:.3f} (bias {bias_irt[1]:+.3f})", ha="left", va="top", color=C_GRPO, fontweight="bold", fontsize=8.5)
ax.text(50, rmse_irtr[1] - 0.004, f"irt_random: {rmse_irtr[1]:.3f}", ha="center", va="top", color=C_POP1, fontweight="bold", fontsize=8.8)

ax.text(100, rmse_cl[2] - 0.004, f"classical: {rmse_cl[2]:.3f}", ha="center", va="top", color=C_SFT, fontweight="bold", fontsize=9.5)
ax.text(100, rmse_irt[2] + 0.003, f"adaptive: {rmse_irt[2]:.3f}\n(bias {bias_irt[2]:+.3f})", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=8.5)
ax.text(92, rmse_irtr[2] - 0.002, f"irt_random: {rmse_irtr[2]:.3f}", ha="right", va="top", color=C_POP1, fontweight="bold", fontsize=8.6)

ax.annotate("Greedy max-info CAT picks 40%–49% Sociology (`K.md:47-51`),\nflatlining at ~0.063 while Classical crosses below at n = 100!",
            xy=(100, rmse_irt[2]), xytext=(48, 0.086),
            arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.5),
            fontsize=8.8, fontweight="bold", color=C_GRPO,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor=C_GRPO))

ax.set_xticks(ns)
ax.set_xticklabels(["n = 25 Items", "n = 50 Items", "n = 100 Items"], fontsize=10.5, fontweight="bold")
ax.set_xlim(18, 110)
ax.set_ylim(0.015, 0.115)
ax.set_xlabel("Short-Test Item Budget $n$ (`Q2c/errors.csv`, Ability Band $> 0.45$)", fontsize=11)
ax.set_ylabel("Level Estimation RMSE against Full Bank", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 6. Consequence for CAT: Greedy High-$a_j^{\\mathrm{pop}}$ Selection Loses to Random Sampling at $n = 100$",
    "Q2c adaptive bias $+0.042\\text{ to }+0.058$ (`Q2c-notes.md:44`); in Q2a ($k=16$), `classical` $\\leq$ `irt_adaptive` in 19 of 21 cells (mean ratio 1.40x–1.65x)",
    title_size=11.6, sub_size=8.5
)
ax.legend(loc="upper right", frameon=True, fontsize=8.8)
save_fig(fig, "Fig06_cat_sociology_bias")

# ==============================================================================
# FIG 07 (Slide 9): Act III — The S1 Multi-Draw Readout Crossover (Mean@16 -> Maj@16 -> Pass@16)
# ==============================================================================
metrics_labels = [
    "Single-Draw\nMean@16",
    "Majority Vote\nMaj@16 (Tie=Wrong)",
    "Majority Vote\nMaj@16* (Tie=Correct)",
    "Best-of-16\nPass@16"
]
base_vals = [
    float(s1_pool.loc["t9_base", "mean_at_16"]),
    float(s1_pool.loc["t9_base", "maj_at_16"]),
    float(s1_rw1.loc["t9_base", "maj_tie_correct"]),
    float(s1_pool.loc["t9_base", "pass_at_16"]),
]
sft_vals = [
    float(s1_pool.loc[sft_states, "mean_at_16"].mean()),
    float(s1_pool.loc[sft_states, "maj_at_16"].mean()),
    float(s1_rw1.loc[sft_states, "maj_tie_correct"].astype(float).mean()),
    float(s1_pool.loc[sft_states, "pass_at_16"].mean()),
]
grpo_vals = [
    float(s1_pool.loc[grpo_states, "mean_at_16"].mean()),
    float(s1_pool.loc[grpo_states, "maj_at_16"].mean()),
    float(s1_rw1.loc[grpo_states, "maj_tie_correct"].astype(float).mean()),
    float(s1_pool.loc[grpo_states, "pass_at_16"].mean()),
]

fig, ax = plt.subplots(figsize=(8.6, 5.7))
xm = np.arange(len(metrics_labels))
ax.plot(xm, base_vals, marker="s", ms=8.5, lw=2.0, ls="--", color=C_BASE,
        label=f"Base (`t9_base`, Entropy $H = {s1_pool.loc['t9_base','entropy']:.3f}$)")
ax.plot(xm, grpo_vals, marker="o", ms=9.5, lw=2.8, color=C_GRPO,
        label=f"S1 GRPO 666u (3-seed mean, Low Entropy $H = {s1_pool.loc[grpo_states,'entropy'].mean():.3f}$)")
ax.plot(xm, sft_vals, marker="^", ms=9.5, lw=2.8, color=C_SFT,
        label=f"S1 CoT SFT 70u (3-seed mean, High Entropy $H = {s1_pool.loc[sft_states,'entropy'].mean():.3f}$)")

# Non-overlapping point labels
ax.text(0 - 0.05, grpo_vals[0] + 0.018, f"{grpo_vals[0]:.3f}", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=9.5)
ax.text(0 + 0.06, sft_vals[0] - 0.018, f"{sft_vals[0]:.3f}", ha="center", va="top", color=C_SFT, fontweight="bold", fontsize=9.5)
ax.text(0, base_vals[0] - 0.022, f"{base_vals[0]:.3f}", ha="center", va="top", color=C_BASE, fontweight="bold", fontsize=8.8)

ax.text(1, sft_vals[1] + 0.018, f"{sft_vals[1]:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9.5)
ax.text(1, grpo_vals[1] - 0.024, f"{grpo_vals[1]:.3f}", ha="center", va="top", color=C_GRPO, fontweight="bold", fontsize=9.5)
ax.text(1, base_vals[1] - 0.022, f"{base_vals[1]:.3f}", ha="center", va="top", color=C_BASE, fontweight="bold", fontsize=8.8)

ax.text(2, sft_vals[2] + 0.018, f"{sft_vals[2]:.3f}*", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9.5)
ax.text(2 + 0.14, grpo_vals[2] + 0.002, f"{grpo_vals[2]:.3f}*", ha="left", va="center", color=C_GRPO, fontweight="bold", fontsize=9.5)
ax.text(2, base_vals[2] - 0.022, f"{base_vals[2]:.3f}*", ha="center", va="top", color=C_BASE, fontweight="bold", fontsize=8.8)

ax.text(3, sft_vals[3] + 0.018, f"{sft_vals[3]:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9.5)
ax.text(3 - 0.10, base_vals[3] - 0.005, f"Base: {base_vals[3]:.3f}", ha="right", va="center", color=C_BASE, fontweight="bold", fontsize=9.0)
ax.text(3, grpo_vals[3] - 0.026, f"{grpo_vals[3]:.3f}", ha="center", va="top", color=C_GRPO, fontweight="bold", fontsize=9.5)

ax.annotate("GRPO leads on 1 draw (0.479 vs 0.439)\nby collapsing entropy ($H = 0.281$),\nleaving +0.002 (+0.011*) voting gain",
            xy=(0, grpo_vals[0]), xytext=(0.18, 0.61),
            arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.5),
            fontsize=8.8, fontweight="bold", color=C_GRPO,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor=C_GRPO))

ax.annotate("SFT overtakes on Maj@16 (0.504 vs 0.481 in 2/3 seeds;\n0.555* vs 0.490* tie-correct), a +0.065 (+0.116*) voting boost!\nPass@16: SFT 0.966 (+0.036 over Base 0.930; GRPO drops to 0.656)",
            xy=(2, sft_vals[2]), xytext=(0.52, 0.78),
            arrowprops=dict(arrowstyle="->", color=C_SFT, lw=1.5),
            fontsize=8.6, fontweight="bold", color=C_SFT,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef6ff", edgecolor=C_SFT))

ax.set_xticks(xm)
ax.set_xticklabels(metrics_labels, fontsize=10, fontweight="bold")
ax.set_ylim(0.32, 1.05)
ax.set_ylabel("Full-Bank Score ($J = 1,500 \\times 32$ Rollouts, `S1`)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 7. The Multi-Draw Crossover: GRPO Wins Single Draw, But SFT Overtakes on Majority Vote",
    "S1 Rule 3 (`SFT Maj@16 > GRPO Maj@16`) fires in 2 of 3 seeds (`ratios -2.77, -0.75, -2.58`; `0/3` on order 1 alone, `S1.md:78`)",
    title_size=11.8
)
ax.legend(loc="upper left", frameon=True, fontsize=8.8)
save_fig(fig, "Fig07_s1_readout_crossover")

# ==============================================================================
# FIG 08 (Slide 10): All 19 States in the (Rollout Entropy H_s, Maj@16) Plane
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.4, 5.7))

ax.scatter(s1_pool.loc["t9_base", "entropy"], s1_pool.loc["t9_base", "maj_at_16"],
           s=125, marker="s", color=C_BASE, edgecolors="black", zorder=6, label="Base (`t9_base`, Maj@16 = 0.414)")
ax.annotate("Base (`t9_base`)\n(H=0.963, Maj@16=0.414)", xy=(0.963, 0.414), xytext=(0.72, 0.424),
            arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.3), fontsize=8.8, fontweight="bold")

ax.scatter(s1_pool.loc[sft_states, "entropy"], s1_pool.loc[sft_states, "maj_at_16"],
           s=115, marker="^", color=C_SFT, edgecolors="black", zorder=5, label="S1 CoT SFT (3 seeds, Maj@16 = 0.504 / 0.555*)")
ax.scatter(s1_pool.loc[grpo_states, "entropy"], s1_pool.loc[grpo_states, "maj_at_16"],
           s=105, marker="o", color=C_GRPO, edgecolors="black", zorder=5, label="S1 GRPO 666u (3 seeds, Maj@16 = 0.481)")

q1_families = [
    ("grpo", "Q1 GRPO (G=5, 3 seeds)", "#ff7f0e", "D"),
    ("ctt", "Q1 CTT Critic (G=1, 3 seeds)", C_CTT, "P"),
    ("pop1", "Q1 2PL Pop1 (a_j-wt, G=1, 3 seeds)", C_POP1, "X"),
    ("pop0", "Q1 2PL Pop0 (unwt, G=1, 3 seeds)", C_POP0, "v"),
]
all_rl_maj = list(s1_pool.loc[grpo_states, "maj_at_16"].values)
for arm, lbl, col, mk in q1_families:
    sts = [f"q1_{arm}_{i}" for i in [0, 1, 2]]
    vals_m = q1_pool.loc[sts, "maj_at_16"].values
    all_rl_maj.extend(vals_m)
    ax.scatter(q1_pool.loc[sts, "entropy"], vals_m,
               s=90, marker=mk, color=col, edgecolors="black", zorder=4, label=lbl)

rl_maj_min, rl_maj_max = float(min(all_rl_maj)), float(max(all_rl_maj))
ax.axhspan(rl_maj_min, rl_maj_max, color=C_GRPO, alpha=0.10,
           label=f"All 15 Base $\\rightarrow$ RL Runs Maj@16 Span ({rl_maj_min:.3f}–{rl_maj_max:.3f})")

ax.annotate("Unit M1 Question (`SFT -> RL`):\nCan starting RL from SFT reduce $H_s$\nwhile keeping SFT's Maj@16 ceiling?",
            xy=(0.35, 0.504), xytext=(0.43, 0.505),
            arrowprops=dict(arrowstyle="->", color=C_ZPD, lw=2.0),
            fontsize=8.8, fontweight="bold", color=C_ZPD,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f8fc", edgecolor=C_ZPD))

ax.set_xlabel("Within-Item Rollout Entropy $H_s$ (nats, `S1` & `Q1`)", fontsize=11)
ax.set_ylabel("Majority-Vote Accuracy $\\mathrm{Maj@16}$ ($J = 1,500$, Tie=Wrong)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 8. All 15 Base $\\rightarrow$ RL Runs Cluster Below Maj@16 $\\leq 0.496$; Only SFT Reaches $> 0.50$",
    "1 Base + 18 post-trained states (`3 S1 SFT`, `3 S1 GRPO`, `12 Q1 RL`; highest RL Maj@16 is `q1_ctt_1` at 0.496)",
    title_size=11.6
)
ax.set_ylim(0.402, 0.525)
ax.set_xlim(0.05, 1.03)
ax.legend(loc="lower left", frameon=True, fontsize=8.3)
save_fig(fig, "Fig08_19states_entropy_vs_majority")

# ==============================================================================
# FIG 09 (Slide 11): Act IV — Unit Q1 RL Trajectories & 12-Run Convergence Band
# ==============================================================================
traj = pd.read_csv(os.path.join(IPL, "Q1", "trajectory.csv"))
arm_styles = [
    ("grpo", "Q1 GRPO (G=5, Unweighted)", C_GRPO, "o", "-"),
    ("ctt",  "Q1 CTT Pass-Band Critic (G=1)", C_CTT, "s", "--"),
    ("pop0", "Q1 Unweighted 2PL Critic pop0 (G=1)", C_POP0, "^", "-."),
    ("pop1", "Q1 $a_j^{\\mathrm{pop}}$-Weighted 2PL Critic pop1 (G=1)", C_POP1, "D", ":"),
]

fig, ax = plt.subplots(figsize=(8.6, 5.7))
shares_u83 = []
arm_means_333 = []
for mode, lbl, col, mk, ls in arm_styles:
    sub = traj[(traj["mode"] == mode) & (traj["seed"] == 0)].sort_values("updates")
    s_map = sub.set_index("updates")["mean_at_16"]
    sh83 = (s_map[83] - s_map[0]) / (s_map[333] - s_map[0]) * 100.0
    shares_u83.append(sh83)
    sts = [f"q1_{mode}_{i}" for i in [0, 1, 2]]
    final_3seed = float(q1_pool.loc[sts, "mean_at_16"].mean())
    arm_means_333.append(final_3seed)
    ax.plot(sub["updates"], sub["mean_at_16"], marker=mk, ms=8, lw=2.4, ls=ls, color=col,
            label=f"{lbl}: Seed 0 ({sh83:.1f}% by u83) | 3-seed final = {final_3seed:.4f}")
    ax.scatter([333], [final_3seed], s=120, marker=mk, color=col, edgecolors="black", lw=1.2, zorder=6)

ax.axhline(base_vals[0], color=C_BASE, lw=1.4, ls="--", label=f"Base Mean@16 = {base_vals[0]:.3f}")
ax.axhline(sft_vals[1], color=C_SFT, lw=1.8, ls="-.", label=f"S1 SFT Maj@16 = {sft_vals[1]:.3f} ({sft_vals[2]:.3f}* tie-correct)")
ax.axvspan(83, 333, color="#eeeeee", alpha=0.55)
ax.axvline(83, color="#333333", lw=1.3, ls=":")

ax.annotate(f"Rapid Early Saturation (Seed 0):\n{min(shares_u83):.1f}%–{max(shares_u83):.1f}% of total gain finishes by update 83!\n"
            f"At u333, all 4 arm means land in {min(arm_means_333):.4f}–{max(arm_means_333):.4f}\n"
            "(width 0.0063 = 0.63 percentage points; 12 runs 0.459–0.480)",
            xy=(83, 0.468), xytext=(110, 0.415),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.5),
            fontsize=8.8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.36", facecolor="white", edgecolor="#444444"))

ax.set_xticks([0, 83, 166, 250, 333])
ax.set_ylim(0.365, 0.520)
ax.set_xlabel("RL Training Updates (`Unit Q1`, Seed 0 Trajectory + 3-Seed Final Means at u333)", fontsize=11)
ax.set_ylabel("Full-Bank Single-Draw Accuracy $\\mathrm{Mean@16}$ ($J = 1,500$)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 9. Static 2PL Critics Do Not Beat Unweighted GRPO: All 4 Arms Land in a 0.63 pp Band",
    "Seed 0 is the only seed with mid-run checkpoint saves (`Q1/trajectory.csv`); S1 GRPO 3-seed mean at 666 updates is 0.4790 (0.473–0.486)",
    title_size=11.6
)
ax.legend(loc="lower right", frameon=True, fontsize=8.0)
save_fig(fig, "Fig09_q1_rl_trajectories")

# ==============================================================================
# FIG 10 (Slide 12): Why GRPO Stalls After Update 83 — Zero-Advantage Starvation
# ==============================================================================
with open(os.path.join(IPL, "W1", "summary.json"), "r", encoding="utf-8") as f:
    w1_sum = json.load(f)
w1_raw_zero = float(w1_sum["wasted_share_g5_raw"]) * 100.0
w1_shrunk_zero = float(w1_sum["wasted_share_g5_shrunk"]) * 100.0

fig, ax = plt.subplots(figsize=(8.4, 5.5))
traces = []
post83_means = []
post83_maxs = []
for s in [0, 1, 2]:
    df_tr = pd.read_csv(os.path.join(IPL, "Q1", "traces", f"grpo_{s}.csv"))
    df_tr["z_smooth"] = df_tr["zero_adv_share"].rolling(15, min_periods=1).mean()
    traces.append(df_tr)
    post_u83 = df_tr[df_tr["update"] > 83]["zero_adv_share"]
    post83_means.append(float(post_u83.mean() * 100.0))
    post83_maxs.append(float(post_u83.max() * 100.0))
    ax.plot(df_tr["update"], df_tr["z_smooth"] * 100.0, lw=1.5, alpha=0.50, color=C_GRPO,
            label="GRPO Per-Seed Traces (15-update moving avg)" if s == 0 else None)

mean_z = np.mean([t["z_smooth"].values for t in traces], axis=0) * 100.0
post83_grand = float(np.mean(post83_means))
ax.plot(traces[0]["update"], mean_z, lw=2.9, color="#8b0000", label="GRPO 3-Seed Mean `zero_adv_share` (%)")
ax.axhline(w1_raw_zero, color=C_BASE, lw=1.5, ls="--",
           label=f"W1 Base $G=5$ Raw Zero-Variance Share ({w1_raw_zero:.1f}%, shrunk {w1_shrunk_zero:.1f}%)")
ax.axhline(post83_grand, color=C_POP0, lw=1.7, ls=":",
           label=f"Post-u83 Mean Zero-Advantage Share ({post83_grand:.1f}%, peak {min(post83_maxs):.0f}%–{max(post83_maxs):.0f}%)")
ax.axvline(83, color="#333333", lw=1.3, ls=":")
ax.axvspan(83, 333, color="#fff0f0", alpha=0.45)

ax.annotate(f"After update 83, {post83_grand:.1f}% of GRPO prompts\nhave 0/5 or 5/5 correct rollouts (zero gradient)!",
            xy=(180, post83_grand), xytext=(115, 78),
            arrowprops=dict(arrowstyle="->", color="#8b0000", lw=1.6),
            fontsize=9.2, fontweight="bold", color="#8b0000",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor=C_GRPO))

ax.set_xlim(0, 333)
ax.set_ylim(25, 92)
ax.set_xlabel("RL Training Update (`Q1` `grpo` Arm, $G = 5$ Rollouts per Prompt)", fontsize=11)
ax.set_ylabel("Prompts with Zero Reward Variance `zero_adv_share` (%)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 10. Why $G = 5$ GRPO Stalls After Update 83: 63.1% of Prompts Produce Zero Gradient",
    "Only $G=5$ `grpo` has `zero_adv_share`; the $G=1$ critics (`ctt, pop0, pop1`) have nonzero $r - \\hat{p}_j$ and higher concentration (`stable_12 = 0.866–0.942`, Table 3)",
    title_size=11.6, sub_size=8.3
)
ax.legend(loc="lower right", frameon=True, fontsize=8.6)
save_fig(fig, "Fig10_grpo_zero_advantage_starvation")

# ==============================================================================
# FIG 11 (Slide 13): Base W1 Pass-Rate Distribution & Interior ZPD Band [0.2, 0.8]
# ==============================================================================
w1_pr = pd.read_csv(os.path.join(IPL, "W1", "pass_rate.csv"))
counts_k = w1_pr["correct"].value_counts().sort_index()
ks = np.arange(0, 9)
shares_k = [float(counts_k.get(k, 0) / len(w1_pr) * 100.0) for k in ks]
zpd_mask = [(2 <= k <= 6) for k in ks]
bar_cols = [C_ZPD if m else "#bbbbbb" for m in zpd_mask]
zpd_total = sum(shares_k[2:7])
boundary_total = 100.0 - zpd_total

fig, ax = plt.subplots(figsize=(8.6, 5.6))
bars_w1 = ax.bar(ks, shares_k, width=0.62, color=bar_cols, edgecolor="black", lw=0.85)
for b_i, sh in zip(bars_w1, shares_k):
    ax.text(b_i.get_x() + b_i.get_width()/2, sh + 0.45, f"{sh:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

ax_twin = ax.twinx()
p_grid = np.linspace(0, 8, 200) / 8.0
fisher_info = p_grid * (1.0 - p_grid)
ax_twin.plot(np.linspace(0, 8, 200), fisher_info, color=C_GRPO, lw=2.6, ls="--",
             label="Bernoulli Variance / 1PL Information $I(p) = p_j(1 - p_j)$")
ax_twin.set_ylabel("Bernoulli Variance / 1PL Information $p_j(1 - p_j)$", color=C_GRPO, fontsize=10.8, fontweight="bold")
ax_twin.set_ylim(0, 0.36)
ax_twin.grid(False)

ax.annotate(f"Interior ZPD Band `2..6 of 8` ($p_j \\in [0.2, 0.8]$): {zpd_total:.1f}% (5,518 Items)\n"
            f"Exterior `0, 1, 7, 8 of 8`: {boundary_total:.1f}% (5,164 Items, near-zero variance)\n"
            "Because SFT reorders item difficulty ($r = 0.29\\text{–}0.60$ vs Base),\n"
            "`M1c` samples $k=8$ on `t9_sft_0..2_70` to filter by $p_{\\mathrm{SFT},j} \\in [0.2, 0.8]$!",
            xy=(4, shares_k[4]), xytext=(1.85, 25.8),
            arrowprops=dict(arrowstyle="->", color=C_ZPD, lw=1.6),
            fontsize=8.3, fontweight="bold", color="#0b556c",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f8fc", edgecolor=C_ZPD))

ax.set_xticks(ks)
ax.set_xticklabels([f"{k}/8\n({k/8:.2f})" for k in ks], fontsize=9.2, fontweight="bold")
ax.set_ylim(0, 34.5)
ax.set_xlabel("Training-Prompt Empirical Pass Rate $p_j = \\mathrm{correct}/8$ (`results/W1/pass_rate.csv`, $N = 10,682$ Prompts)", fontsize=11)
ax.set_ylabel("Share of Training Prompt Pool (%)", fontsize=11)
set_title_and_subtitle(
    ax,
    "Fig 11. Why Within-Model Pass Rate $p_{s,j} \\in [0.2, 0.8]$ Defines the Active RL Learning Zone (`Unit M1`)",
    "Committed Base `W1` pass-rate distribution shown; `Unit M1` (`notes/units/M1.md`, running on cluster) calibrates $p_{\\mathrm{SFT},j}$ on `t9_sft_0..2_70`",
    title_size=11.4, sub_size=8.6
)

save_fig(fig, "Fig11_w1_pass_rate_and_zpd")

print("ALL 11 SINGLE-CONCEPT STORYTELLING FIGURES GENERATED SUCCESSFULLY.")
