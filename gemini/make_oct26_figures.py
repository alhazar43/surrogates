import os
import csv
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

IPL = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\item-params-llm\results"
OUT_DIR = r"D:\surrogates\gemini\figures"
os.makedirs(OUT_DIR, exist_ok=True)

C_BASE = "#555555"
C_SFT = "#1f6feb"     # Deep royal blue (Procedural / SFT)
C_GRPO = "#d9381e"    # Vermilion red (Declarative / GRPO)
C_CTT = "#e69f00"     # Amber
C_POP0 = "#8a3ffc"    # Purple
C_POP1 = "#009e73"    # Teal
C_ZPD = "#118ab2"     # Cyan-blue

def safe_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

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
m_t9r = pd.merge(t9r, w0_valid, left_on="item_id", right_on="bank_row_id")

# ==============================================================================
# FIGURE 1: Molenaar's Non-Ergodicity (Human vs LLM b_j & Pop a_j vs Traj Slope)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), gridspec_kw={"wspace": 0.24})

# Panel A: E1 Human vs LLM difficulty b_j
e1_diff = pd.read_csv(os.path.join(IPL, "E1", "difficulty.csv")).dropna(subset=["b_students", "b_llms"])
b_stu = e1_diff["b_students"].values
b_llm = e1_diff["b_llms"].values
r_e1 = np.corrcoef(b_stu, b_llm)[0, 1]
rho_e1 = pd.Series(b_stu).corr(pd.Series(b_llm), method="spearman")

ax1.scatter(b_stu, b_llm, alpha=0.42, s=22, color="#2b5c8f", edgecolors="none", label="600 Eedi Diagnostic Math Items")
m_fit, c_fit = np.polyfit(b_stu, b_llm, 1)
xs_line = np.linspace(b_stu.min(), b_stu.max(), 100)
ax1.plot(xs_line, m_fit * xs_line + c_fit, color=C_GRPO, lw=2.2, label=f"OLS Fit: r = {r_e1:+.3f}, Spearman ρ = {rho_e1:+.3f}")
ax1.axhline(0, color="#888888", lw=0.8, ls=":")
ax1.axvline(0, color="#888888", lw=0.8, ls=":")
ax1.set_title("A. Human vs. LLM Item Difficulty $b_j$ (Unit E1, $J=600$)", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Human Student 2PL Difficulty $b_j^{\\mathrm{human}}$ ($N=4,918$ Students, Rel $=0.985$)", fontsize=10.5)
ax1.set_ylabel("LLM Crowd 2PL Difficulty $b_j^{\\mathrm{LLM}}$ ($N=61$ LLMs)", fontsize=10.5)
ax1.text(0.04, 0.93, "61 / 61 LLMs reject human 2PL ($l_z < -1.96$)\nBetween-human $b_j \\perp$ Between-LLM $b_j$",
         transform=ax1.transAxes, va="top", ha="left", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f5", edgecolor=C_GRPO, alpha=0.92))
ax1.legend(loc="lower right", frameon=True, fontsize=9.5)

# Panel B: Population discrimination a_j^pop vs Single-Model RL Trajectory Change
Lb_t9 = safe_logit(m_t9r["p_key_base"].values, 1e-4)
L_rl_mean = np.mean([safe_logit(m_t9r["p_key_base"].values + m_t9r[f"dp_key_rl_{s}_666"].values, 1e-4) for s in [0, 1, 2]], axis=0)
dL_rl = L_rl_mean - Lb_t9
a_pop = m_t9r["a"].values
r_apop_dl = np.corrcoef(a_pop, dL_rl)[0, 1]

ax2.scatter(a_pop, dL_rl, alpha=0.35, s=18, color="#5c4d7d", edgecolors="none", label="1,472 MMLU Bank Items (S1/T9r)")
m2, c2 = np.polyfit(a_pop, dL_rl, 1)
xs2 = np.linspace(0, np.percentile(a_pop, 99.5), 100)
ax2.plot(xs2, m2 * xs2 + c2, color=C_GRPO, lw=2.2, label=f"Item-Level Slope: $R^2(a_j^{{\\mathrm{{pop}}}}, \\Delta L_{{\\mathrm{{RL}},j}}) = {r_apop_dl**2*100:.2f}\\%$")
ax2.set_xlim(0, 3.6)
ax2.set_ylim(-6.5, 6.5)
ax2.axhline(0, color="#888888", lw=0.8, ls=":")
ax2.set_title("B. Cross-Model $a_j^{\\mathrm{pop}}$ vs. Single-Model RL Shift ($J=1,472$)", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("102-LLM Population Discrimination $a_j^{\\mathrm{pop}}$ (Fluid Benchmarking)", fontsize=10.5)
ax2.set_ylabel("Within-Model RL Log-Odds Shift $\\Delta L_{\\mathrm{GRPO},j}$ (3-Seed Mean)", fontsize=10.5)
ax2.text(0.04, 0.93,
         "• Signature F5 (MMLU): $\\mathrm{Corr}(a_j^{\\mathrm{pop}}, a_j^{\\mathrm{traj}}) = +0.04$\n"
         "• irt4llm Gate 1 (GSM8K): $\\mathrm{Corr}(a_j^{\\mathrm{pop}}, a_j^{\\mathrm{traj}}) = -0.22$\n"
         "• Cross-sectional 2PL $\\neq$ Intra-model learning trajectory",
         transform=ax2.transAxes, va="top", ha="left", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f0ff", edgecolor=C_POP0, alpha=0.92))
ax2.legend(loc="lower right", frameon=True, fontsize=9.5)

fig.suptitle("CANNOT #1 (Molenaar's Non-Ergodicity): Static Population $(a_j^{\\mathrm{pop}}, b_j^{\\mathrm{pop}})$ Are Orthogonal to Individual Trajectories",
             fontsize=13, fontweight="bold", y=1.02)
save_fig(fig, "F1_non_ergodicity")

# ==============================================================================
# FIGURE 2: Autoregressive Prior-Scaling Law & tau_s Decomposition
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), gridspec_kw={"wspace": 0.24})

L_sft_0 = safe_logit(m_t9r["p_key_base"].values + m_t9r["dp_key_sft_0_70"].values, 1e-4)
L_rl_0 = safe_logit(m_t9r["p_key_base"].values + m_t9r["dp_key_rl_0_666"].values, 1e-4)
dL_sft_0 = L_sft_0 - Lb_t9
dL_rl_0 = L_rl_0 - Lb_t9

ax1.scatter(Lb_t9, dL_rl_0, alpha=0.28, s=15, color=C_GRPO, label="GRPO (Step 666): $\\hat{\\tau}_{\\mathrm{GRPO}} = 1.89 > 1$ (Sharpening)")
ax1.scatter(Lb_t9, dL_sft_0, alpha=0.28, s=15, color=C_SFT, label="CoT SFT (Step 70): $\\hat{\\tau}_{\\mathrm{SFT}} = 0.43 < 1$ (Reordering Attenuation)")

xs_lb = np.linspace(-9.2, 9.2, 100)
m_rl, c_rl = np.polyfit(Lb_t9, dL_rl_0, 1)
m_sf, c_sf = np.polyfit(Lb_t9, dL_sft_0, 1)
ax1.plot(xs_lb, m_rl * xs_lb + c_rl, color="#8b0000", lw=2.4, label=f"GRPO Fit: $\\Delta L = ({m_rl:+.2f})L_{{\\mathrm{{base}}}} {c_rl:+.2f}$ ($R^2 = 50.1\\%$)")
ax1.plot(xs_lb, m_sf * xs_lb + c_sf, color="#003380", lw=2.4, label=f"SFT Fit: $\\Delta L = ({m_sf:+.2f})L_{{\\mathrm{{base}}}} {c_sf:+.2f}$ ($R^2 = 59.2\\%$)")
ax1.axhline(0, color="#666666", lw=0.8, ls=":")
ax1.axvline(0, color="#666666", lw=0.8, ls=":")
ax1.set_title("A. Governing Prior-Scaling Law: $\\Delta L_{s,j} = (\\tau_s - 1)L_{\\mathrm{base},j} + \\delta_s$", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Model's OWN Base Item Log-Odds $L_{\\mathrm{base},j}$", fontsize=10.5)
ax1.set_ylabel("Post-Training Item Shift $\\Delta L_{s,j} = L_{s,j} - L_{\\mathrm{base},j}$", fontsize=10.5)
ax1.text(0.03, 0.95,
         "Adding 102-LLM $z_j^{\\mathrm{pop}} = a_j^{\\mathrm{pop}}(\\theta_s - b_j^{\\mathrm{pop}})$ adds:\n"
         "• $\\Delta R^2 \\leq 1.04\\%$ on SFT   • $\\Delta R^2 \\leq 0.06\\%$ on GRPO\n"
         "Why Q2-A item flags fail: 49%–87% of $\\Delta L_{s,j}$ is own $L_{\\mathrm{base},j}$!",
         transform=ax1.transAxes, va="top", ha="left", fontsize=9.2,
         bbox=dict(boxstyle="round,pad=0.38", facecolor="#fffdf0", edgecolor="#d4a017", alpha=0.94))
ax1.legend(loc="lower left", frameon=True, fontsize=8.8)

# Panel B: Rule 4 Decomposition tau_s = r(L_s, L_base) * (sigma_s / sigma_base)
modalities = [
    "Token-0 Logits\n(T9r SFT)",
    "Token-0 Logits\n(T9r GRPO)",
    "Jeffreys K=32\n(S1 GRPO)",
    "Free-Text CoT\n(T9g/s3 SFT)",
]
r_vals = [0.822, 0.904, 0.908, 0.289]
sig_ratios = [0.523, 2.091, 1.525, 2.351]
tau_vals = [r * s for r, s in zip(r_vals, sig_ratios)]

x = np.arange(len(modalities))
w = 0.25
b1 = ax2.bar(x - w, r_vals, width=w, color="#2b5c8f", label="Preference Rank Preservation $r(L_s, L_{\\mathrm{base}})$")
b2 = ax2.bar(x, sig_ratios, width=w, color="#e69f00", label="Scale Dispersion Ratio $\\sigma(L_s)/\\sigma(L_{\\mathrm{base}})$")
b3 = ax2.bar(x + w, tau_vals, width=w, color="#009e73", label="OLS Slope $\\hat{\\tau}_s = r \\cdot (\\sigma_s / \\sigma_{\\mathrm{base}})$")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax2.axhline(1.0, color="#555555", lw=1.1, ls="--")
ax2.set_ylim(0, 3.55)
ax2.set_xticks(x)
ax2.set_xticklabels(modalities, fontsize=9.5)
ax2.set_ylabel("Component Magnitude", fontsize=10.5)
ax2.set_title("B. Exact Decomposition $\\hat{\\tau}_s = r(L_s, L_{\\mathrm{base}}) \\cdot \\frac{\\sigma(L_s)}{\\sigma(L_{\\mathrm{base}})}$ (Rule 4)", fontsize=12, fontweight="bold", pad=10)
ax2.text(0.02, 0.98,
         "Crucial Insight (Trap 1 Resolved): At the CoT answer token (T9g/s3),\n"
         "SFT dispersion is $2.35\\times$ Base ($\\sigma_s=7.44 > 3.17$), NOT flatter!\n"
         "$\\hat{\\tau}_{\\mathrm{SFT}} = 0.68 < 1$ is pure regression attenuation from reordering ($r=0.289$).",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.33", facecolor="#eef6ff", edgecolor=C_SFT, alpha=0.95))
ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.81), frameon=True, fontsize=7.9)

fig.suptitle("CANNOT #2: Static $(a_j^{\\mathrm{pop}}, b_j^{\\mathrm{pop}})$ Cannot Isolate Item Learning Because Own $L_{\\mathrm{base},j}$ Governs $\\Delta L_{s,j}$",
             fontsize=13, fontweight="bold", y=1.02)
save_fig(fig, "F2_prior_scaling")

# ==============================================================================
# FIGURE 3: HEADLINE CAN #1 — 15-Subject Neuro-Cognitive Double Dissociation (R^2 = 76.9%)
# ==============================================================================
pf_s1 = pd.read_csv(os.path.join(IPL, "Q2g", "s1", "personfit.csv"))
pf_s1_all = pf_s1[pf_s1["variant"] == "all"].set_index("state")["theta"].to_dict()
th_base = pf_s1_all["t9_base"]
dth_sft = np.mean([pf_s1_all[f"t9_sft_{s}_70"] - th_base for s in [0, 1, 2]])
dth_grpo = np.mean([pf_s1_all[f"t9_rl_{s}_666"] - th_base for s in [0, 1, 2]])

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 6.0), gridspec_kw={"wspace": 0.35})

# Panel A: Operator Advantage vs Domain Difficulty b_m^pop (R^2 = 76.9%)
b_vec = sub_grp["b_mean"].values
a_vec = sub_grp["a_mean"].values
diff_vec = sub_grp["diff_sft_grpo"].values
r_b = np.corrcoef(b_vec, diff_vec)[0, 1]
r_a = np.corrcoef(a_vec, diff_vec)[0, 1]

colors_dom = [C_SFT if d > 0.1 else (C_GRPO if d < -0.1 else "#777777") for d in diff_vec]
ax1.scatter(b_vec, diff_vec, c=colors_dom, s=85, zorder=4, edgecolors="black", linewidth=0.8)

mb, cb = np.polyfit(b_vec, diff_vec, 1)
xb_line = np.linspace(b_vec.min() - 0.2, b_vec.max() + 0.2, 100)
ax1.plot(xb_line, mb * xb_line + cb, color="#222222", lw=2.2, ls="--",
         label=f"Linear Fit: $r(\\bar{{b}}_m^{{\\mathrm{{pop}}}}, \\Delta L_{{\\mathrm{{SFT}}}} - \\Delta L_{{\\mathrm{{GRPO}}}}) = {r_b:+.4f}$ ($R^2 = {r_b**2*100:.1f}\\%$)")

offsets_a = {
    "HS Math": (-48, -22),
    "HS Physics": (-22, -22),
    "Elem Math": (8, -6),
    "Econometrics": (8, -12),
    "HS Macroecon": (-55, 6),
    "HS Statistics": (-58, 8),
    "HS Microecon": (8, -10),
    "Clinical Knowl": (8, -6),
    "Philosophy": (8, -12),
    "Prehistory": (-52, 7),
    "HS Geography": (-62, 6),
    "HS Biology": (8, -6),
    "World History": (8, -4),
    "Sociology": (8, -10),
    "Marketing": (-15, 10),
}
for _, row in sub_grp.iterrows():
    dx, dy = offsets_a.get(row["short"], (5, 5))
    ax1.annotate(f"{row['short']} (a={row['a_mean']:.2f})",
                 (row["b_mean"], row["diff_sft_grpo"]),
                 xytext=(dx, dy), textcoords="offset points", fontsize=7.8, fontweight="bold")

ax1.axhline(0, color="#444444", lw=1.2, ls="-")
ax1.axhspan(0, 1.95, color=C_SFT, alpha=0.06)
ax1.axhspan(-1.65, 0, color=C_GRPO, alpha=0.06)
ax1.set_ylim(-1.62, 1.92)
ax1.set_xlim(-1.28, 2.68)
ax1.set_title("A. Item Parameters $(\\bar{a}_m^{\\mathrm{pop}}, \\bar{b}_m^{\\mathrm{pop}})$ Predict SFT vs. RL ($R^2 = 76.9\\%$)", fontsize=11.8, fontweight="bold", pad=10)
ax1.set_xlabel("Domain Mean 2PL Difficulty $\\bar{b}_m^{\\mathrm{pop}}$ ($r = -0.884$ with $\\bar{a}_m^{\\mathrm{pop}}$)", fontsize=10.5)
ax1.set_ylabel("Operator Advantage $\\overline{\\Delta L}_{\\mathrm{SFT},m} - \\overline{\\Delta L}_{\\mathrm{GRPO},m}$ (Logits)", fontsize=10.5)
ax1.text(0.02, 0.97,
         "• Procedural (Low $\\bar{a}_m^{\\mathrm{pop}} \\in [0.28, 0.55]$, High $\\bar{b}_m^{\\mathrm{pop}}$):\n"
         "  CoT SFT wins 5/6 domains (1 tie)\n"
         "• Declarative (High $\\bar{a}_m^{\\mathrm{pop}} \\in [0.69, 1.45]$, Low $\\bar{b}_m^{\\mathrm{pop}}$):\n"
         "  Outcome RL wins 8/8 domains\n"
         f"• $\\bar{{a}}_m^{{\\mathrm{{pop}}}}$: $R^2 = {r_a**2*100:.1f}\\%$;  $\\bar{{b}}_m^{{\\mathrm{{pop}}}}$: $R^2 = {r_b**2*100:.1f}\\%$ ($p = 1.8\\times 10^{{-5}}$)",
         transform=ax1.transAxes, va="top", ha="left", fontsize=8.4,
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#333333", alpha=0.94))
ax1.legend(loc="lower right", frameon=True, fontsize=8.6)

# Panel B: Horizontal Paired Bar Chart Sorted by a_m^pop
y_pos = np.arange(len(sub_grp))
h_bar = 0.38
ax2.barh(y_pos + h_bar/2, sub_grp["dL_sft"], height=h_bar, color=C_SFT, label="CoT SFT Gain $\\overline{\\Delta L}_{\\mathrm{SFT},m}$ (Procedural Schema)")
ax2.barh(y_pos - h_bar/2, sub_grp["dL_grpo"], height=h_bar, color=C_GRPO, label="Outcome RL (GRPO) Gain $\\overline{\\Delta L}_{\\mathrm{GRPO},m}$ (Neural Gain)")
ax2.axvline(0, color="#333333", lw=1.0)
ax2.axhline(6.5, color="#222222", lw=1.3, ls="--")
ax2.text(1.68, 2.3, "Procedural / Quant\n($\\bar{a}_m^{\\mathrm{pop}} = 0.28 - 0.55$)\nSFT wins 5/6,\nRL stalls/regresses!",
         ha="right", va="center", fontsize=8.5, fontweight="bold", color=C_SFT,
         bbox=dict(boxstyle="round,pad=0.32", facecolor="#eef6ff", edgecolor=C_SFT))
ax2.text(1.68, 10.6, "Declarative / Factual\n($\\bar{a}_m^{\\mathrm{pop}} = 0.69 - 1.45$)\nRL wins 8/8\n($r = +0.746$ with $\\bar{a}_m^{\\mathrm{pop}}$)",
         ha="right", va="center", fontsize=8.5, fontweight="bold", color=C_GRPO,
         bbox=dict(boxstyle="round,pad=0.32", facecolor="#fff5f5", edgecolor=C_GRPO))

ylabels = [f"{r['short']} (a={r['a_mean']:.2f})" for _, r in sub_grp.iterrows()]
ax2.set_yticks(y_pos)
ax2.set_yticklabels(ylabels, fontsize=8.8)
ax2.invert_yaxis()
ax2.set_xlim(-0.75, 1.78)
ax2.set_xlabel("Domain Log-Odds Gain Over Base $\\overline{\\Delta L}_{s,m}$ (Logits, $K=32$ Rollouts)", fontsize=10.5)
ax2.set_title("B. 15 Domains Sorted by Item Discrimination $\\bar{a}_m^{\\mathrm{pop}}$ (Table 1.2)", fontsize=11.8, fontweight="bold", pad=10)
ax2.legend(loc="upper center", bbox_to_anchor=(0.48, -0.12), frameon=True, fontsize=8.5, ncol=1)

fig.suptitle("CAN #1 (A Helps B): 2PL Item Parameters $(a_m^{\\mathrm{pop}}, b_m^{\\mathrm{pop}})$ Reveal the Neuro-Cognitive Double Dissociation ($R^2 = 76.9\\%$)",
             fontsize=13, fontweight="bold", y=1.02)
save_fig(fig, "F3_double_dissociation_15domains")

# ==============================================================================
# FIGURE 4: S1 Readout Crossover & The 19-State (Entropy H_s, Maj@16) Plane
# ==============================================================================
s1_stats = pd.read_csv(os.path.join(IPL, "S1", "stats.csv"))
s1_pool = s1_stats[s1_stats["order"] == "pooled"].pivot(index="state", columns="statistic", values="value")
s1_rw1 = pd.read_csv(os.path.join(IPL, "S1", "rewrite1_parsed.csv")).set_index("state")
q1_stats = pd.read_csv(os.path.join(IPL, "Q1", "stats.csv"))
q1_pool = q1_stats[q1_stats["order"] == "pooled"].pivot(index="state", columns="statistic", values="value")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), gridspec_kw={"wspace": 0.25})

# Panel A: Readout Crossover Across Evaluation Functionals
metrics_labels = ["Single-Draw\nMean@16", "Majority Vote\nMaj@16 (Unstarred)", "Majority Vote\nMaj@16* (Tie-Correct)", "Best-of-16\nPass@16"]
base_vals = [
    s1_pool.loc["t9_base", "mean_at_16"],
    s1_pool.loc["t9_base", "maj_at_16"],
    float(s1_rw1.loc["t9_base", "maj_tie_correct"]),
    s1_pool.loc["t9_base", "pass_at_16"],
]
sft_states = [f"t9_sft_{i}_70" for i in [0, 1, 2]]
grpo_states = [f"t9_rl_{i}_666" for i in [0, 1, 2]]
sft_vals = [
    s1_pool.loc[sft_states, "mean_at_16"].mean(),
    s1_pool.loc[sft_states, "maj_at_16"].mean(),
    s1_rw1.loc[sft_states, "maj_tie_correct"].astype(float).mean(),
    s1_pool.loc[sft_states, "pass_at_16"].mean(),
]
grpo_vals = [
    s1_pool.loc[grpo_states, "mean_at_16"].mean(),
    s1_pool.loc[grpo_states, "maj_at_16"].mean(),
    s1_rw1.loc[grpo_states, "maj_tie_correct"].astype(float).mean(),
    s1_pool.loc[grpo_states, "pass_at_16"].mean(),
]

xm = np.arange(len(metrics_labels))
ax1.plot(xm, base_vals, marker="s", ms=8, lw=2.0, ls="--", color=C_BASE, label="Base (t9_base)")
ax1.plot(xm, grpo_vals, marker="o", ms=9, lw=2.6, color=C_GRPO, label="Outcome RL (S1 GRPO, $\\beta_s \\uparrow, H_s=0.281$)")
ax1.plot(xm, sft_vals, marker="^", ms=9, lw=2.6, color=C_SFT, label="CoT SFT (70B Distill, $\\mu_{sj} \\uparrow, H_s=0.943$)")

for idx, (bv, gv, sv) in enumerate(zip(base_vals, grpo_vals, sft_vals)):
    if idx == 0:
        ax1.text(idx - 0.06, gv + 0.018, f"{gv:.3f}", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=9)
        ax1.text(idx + 0.08, sv - 0.022, f"{sv:.3f}", ha="center", va="top", color=C_SFT, fontweight="bold", fontsize=9)
    else:
        ax1.text(idx, gv - 0.032, f"{gv:.3f}", ha="center", va="top", color=C_GRPO, fontweight="bold", fontsize=9)
        ax1.text(idx, sv + 0.020, f"{sv:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9)

ax1.annotate("GRPO wins Mean@16\n(+0.040 over SFT)\nvia concentration $\\beta_s \\uparrow$",
             xy=(0, 0.479), xytext=(0.25, 0.59),
             arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.4),
             fontsize=8.6, fontweight="bold", color=C_GRPO,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", edgecolor=C_GRPO))

ax1.annotate("SFT overtakes on Maj@16 (+0.065 to +0.116* voting boost)\nand reaches Pass@16 = 0.966 (+0.310 over GRPO)!",
             xy=(2, 0.555), xytext=(0.68, 0.77),
             arrowprops=dict(arrowstyle="->", color=C_SFT, lw=1.4),
             fontsize=8.6, fontweight="bold", color=C_SFT,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef6ff", edgecolor=C_SFT))

ax1.set_xticks(xm)
ax1.set_xticklabels(metrics_labels, fontsize=9.5)
ax1.set_ylim(0.32, 1.04)
ax1.set_ylabel("Full-Bank Score ($J=1,500 \\times 32$ Rollouts)", fontsize=10.5)
ax1.set_title("A. Corollary 1: The S1 Evaluation Readout Crossover", fontsize=12, fontweight="bold", pad=10)
ax1.legend(loc="upper left", frameon=True, fontsize=8.8)

# Panel B: All 19 States in the (Rollout Entropy H_s, Majority Vote Maj@16) Plane
ax2.scatter(s1_pool.loc["t9_base", "entropy"], s1_pool.loc["t9_base", "maj_at_16"],
            s=115, marker="s", color=C_BASE, edgecolors="black", zorder=6, label="Base (t9_base, Maj@16 = 0.414)")
ax2.annotate("t9_base\n(0.963, 0.414)", xy=(0.963, 0.414), xytext=(0.78, 0.422),
             arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.2), fontsize=8.5, fontweight="bold")

ax2.scatter(s1_pool.loc[sft_states, "entropy"], s1_pool.loc[sft_states, "maj_at_16"],
            s=105, marker="^", color=C_SFT, edgecolors="black", zorder=5, label="S1 CoT SFT (3 seeds, Maj@16 = 0.504)")
ax2.scatter(s1_pool.loc[grpo_states, "entropy"], s1_pool.loc[grpo_states, "maj_at_16"],
            s=95, marker="o", color=C_GRPO, edgecolors="black", zorder=5, label="S1 GRPO 666u (3 seeds, Maj@16 = 0.481)")

q1_families = [
    ("grpo", "Q1 GRPO (G=5)", "#ff7f0e", "D"),
    ("ctt", "Q1 CTT Critic (G=1)", C_CTT, "P"),
    ("pop1", "Q1 2PL Pop1 (a_j-wt, G=1)", C_POP1, "X"),
    ("pop0", "Q1 2PL Pop0 (unwt, G=1)", C_POP0, "v"),
]
for arm, lbl, col, mk in q1_families:
    sts = [f"q1_{arm}_{i}" for i in [0, 1, 2]]
    ax2.scatter(q1_pool.loc[sts, "entropy"], q1_pool.loc[sts, "maj_at_16"],
                s=85, marker=mk, color=col, edgecolors="black", zorder=4, label=lbl)

ax2.axhspan(0.466, 0.491, color=C_GRPO, alpha=0.10, label="15-Run Base→RL Ceiling Band (0.470–0.489)")
ax2.set_xlabel("Within-Item Rollout Entropy $H_s$ (nats; $\\leftarrow$ Higher Neural Gain $\\beta_s$)", fontsize=10.5)
ax2.set_ylabel("Modal Location Boundary $\\mathrm{Maj@16}$ ($J=1,500$)", fontsize=10.5)
ax2.set_title("B. All 19 Post-Trained States in the $(H_s, \\mathrm{Maj@16})$ Plane (Table 2.1)", fontsize=12, fontweight="bold", pad=10)
ax2.annotate("Unit M1 Target Zone:\nHarvest SFT's Maj@16 ceiling\nvia RL gain modulation ($H_s \\downarrow$)",
             xy=(0.35, 0.505), xytext=(0.43, 0.508),
             arrowprops=dict(arrowstyle="->", color=C_ZPD, lw=1.8),
             fontsize=8.6, fontweight="bold", color=C_ZPD,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f8fc", edgecolor=C_ZPD))
ax2.set_ylim(0.402, 0.526)
ax2.set_xlim(0.05, 1.03)
ax2.legend(loc="lower left", frameon=True, fontsize=7.9, ncol=1)

fig.suptitle("B Helps A (Generative Respondent Duality): Separating Modal Location $\\mu_{sj}$ from Trajectory Concentration $\\beta_s$",
             fontsize=13, fontweight="bold", y=1.02)
save_fig(fig, "F4_readout_crossover_19states")

# ==============================================================================
# FIGURE 5: CAN #2 — The a_j^pop Attribution Reversal Theorem (60%/40% vs 21%/79%)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), gridspec_kw={"wspace": 0.24})

pf_df = pd.read_csv(os.path.join(IPL, "Q2g", "s1", "personfit.csv"))
pf_all = pf_df[pf_df["variant"] == "all"].set_index("state")

acc_base = float(pf_all.loc["t9_base", "accuracy"])
acc_sft = pf_all.loc[sft_states, "accuracy"].astype(float).mean()
acc_grpo = pf_all.loc[grpo_states, "accuracy"].astype(float).mean()

th_b = float(pf_all.loc["t9_base", "theta"])
th_s = pf_all.loc[sft_states, "theta"].astype(float).mean()
th_g = pf_all.loc[grpo_states, "theta"].astype(float).mean()

sft_share_acc = (acc_sft - acc_base) / (acc_grpo - acc_base) * 100
grpo_share_acc = (acc_grpo - acc_sft) / (acc_grpo - acc_base) * 100
sft_share_th = (th_s - th_b) / (th_g - th_b) * 100
grpo_share_th = (th_g - th_s) / (th_g - th_b) * 100

bars_x = ["Classical Accuracy\n($\\Delta \\mathrm{Acc} = \\overline{\\Delta p}$, Unweighted)",
          "Population 2PL Ability $\\hat{\\theta}$\n($\\Delta \\hat{\\theta}$ Weighted by $a_j^{\\mathrm{pop}}$)"]
sft_shares = [sft_share_acc, sft_share_th]
grpo_shares = [grpo_share_acc, grpo_share_th]

ax1.bar(bars_x, sft_shares, width=0.48, color=C_SFT, label="Share Attributed to Base → SFT (Procedural Schema)")
ax1.bar(bars_x, grpo_shares, width=0.48, bottom=sft_shares, color=C_GRPO, label="Share Attributed to SFT → GRPO (Declarative Gain)")

ax1.text(0, sft_shares[0]/2, f"SFT: {sft_shares[0]:.1f}%\n(+{acc_sft-acc_base:.4f} acc)\n8.9–10.5 SE", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
ax1.text(0, sft_shares[0] + grpo_shares[0]/2, f"GRPO: {grpo_shares[0]:.1f}%\n(+{acc_grpo-acc_sft:.4f} acc)\n3.8–5.9 SE", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

ax1.text(1, sft_shares[1]/2, f"SFT: {sft_shares[1]:.1f}%\n(+{th_s-th_b:.3f} θ)\n2.9–5.1 SE", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
ax1.text(1, sft_shares[1] + grpo_shares[1]/2, f"GRPO: {grpo_shares[1]:.1f}%\n(+{th_g-th_s:.3f} θ)\n8.6–10.5 SE", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

ax1.set_ylim(0, 114)
ax1.set_ylabel("Percentage of Total Post-Training Gain (%)", fontsize=10.5)
ax1.set_title("A. The Attribution Reversal: 60% / 40% (Acc) vs. 21% / 79% (2PL $\\hat{\\theta}$)", fontsize=11.8, fontweight="bold", pad=10)
ax1.legend(loc="upper center", frameon=True, fontsize=8.8)

# Panel B: Why? Taylor Expansion Cov(a_j^pop, Delta p_j) across a_j^pop Quintiles
sub_grp["tier"] = pd.qcut(sub_grp["a_mean"], q=5)
tier_df = sub_grp.groupby("tier", observed=False).agg(
    a_mid=("a_mean", "mean"),
    dL_sft=("dL_sft", "mean"),
    dL_grpo=("dL_grpo", "mean"),
).reset_index()

xt = np.arange(len(tier_df))
tier_labels = [f"Q{i+1}\n($\\bar{{a}}={r['a_mid']:.2f}$)" for i, r in tier_df.iterrows()]
ax2.plot(xt, tier_df["dL_sft"], marker="^", ms=9, lw=2.5, color=C_SFT, label="SFT Gain $\\overline{\\Delta L}_{\\mathrm{SFT}}$ ($\\rho(c_{\\mathrm{SFT}}, a^{\\mathrm{pop}}) = -0.169, p < 10^{-9}$)")
ax2.plot(xt, tier_df["dL_grpo"], marker="o", ms=9, lw=2.5, color=C_GRPO, label="GRPO Gain $\\overline{\\Delta L}_{\\mathrm{GRPO}}$ ($\\rho(c_{\\mathrm{GRPO}}, a^{\\mathrm{pop}}) = +0.143$)")
ax2.axhline(0, color="#666666", lw=0.9, ls=":")
ax2.set_xticks(xt)
ax2.set_xticklabels(tier_labels, fontsize=9.2)
ax2.set_ylim(-0.28, 1.18)
ax2.set_xlabel("Item Discrimination $a_j^{\\mathrm{pop}}$ Quintiles Across MMLU Bank ($J=1,472$)", fontsize=10.5)
ax2.set_ylabel("Mean Log-Odds Gain (Logits, $K=32$ Rollouts)", fontsize=10.5)
ax2.set_title("B. Mathematical Mechanism: $\\Delta \\hat{\\theta} \\propto \\overline{\\Delta p} + \\mathrm{Cov}(a_j^{\\mathrm{pop}}, \\Delta p_j) / \\bar{a}^{\\mathrm{pop}}$", fontsize=11.8, fontweight="bold", pad=10)
ax2.text(0.03, 0.96,
         "• 2PL $\\hat{\\theta}$ weights Marketing ($a=1.45$) $5.2\\times$ heavier than HS Math ($a=0.28$)!\n"
         "• Because $\\mathrm{Cov}(a_j^{\\mathrm{pop}}, \\Delta p_{\\mathrm{SFT}}) < 0$, $\\hat{\\theta}$ cuts SFT credit from 60% → 21%.\n"
         "• Because $\\mathrm{Cov}(a_j^{\\mathrm{pop}}, \\Delta p_{\\mathrm{GRPO}}) > 0$, $\\hat{\\theta}$ inflates GRPO credit from 40% → 79%.",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.6,
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffdf0", edgecolor="#d4a017", alpha=0.94))
ax2.legend(loc="lower right", frameon=True, fontsize=8.5)

fig.suptitle("CAN #2 (B Helps A, Theorem 1): How Item Discrimination $a_j^{\\mathrm{pop}}$ Explains the 60%/40% vs. 21%/79% Attribution Reversal",
             fontsize=13, fontweight="bold", y=1.02)
save_fig(fig, "F5_attribution_reversal")

# ==============================================================================
# FIGURE 6: Theorem 2 (l_z Sign Flip) & CANNOT #3 (Why High-a_j CAT Loses by n=100)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6), gridspec_kw={"wspace": 0.25})

# Panel A: Person-Fit Sign Flip l_z^(1) vs l_z^(32)
q2g_q1_pf_df = pd.read_csv(os.path.join(IPL, "Q2g", "q1", "personfit.csv"))
q2g_q1_all = q2g_q1_pf_df[q2g_q1_pf_df["variant"] == "all"].set_index("state")

fams = ["Base\n(t9_base)", "CoT SFT\n(t9_sft)", "S1 GRPO\n(666u, G=5)", "Q1 GRPO\n(333u, G=5)", "Q1 CTT\n(333u, G=1)", "Q1 Pop1\n(a_j-wt)", "Q1 Pop0\n(unwt)"]
lz32_vals = [
    float(pf_all.loc["t9_base", "lz"]),
    pf_all.loc[sft_states, "lz"].astype(float).mean(),
    pf_all.loc[grpo_states, "lz"].astype(float).mean(),
    q2g_q1_all.loc[[f"q1_grpo_{i}" for i in [0,1,2]], "lz"].astype(float).mean(),
    q2g_q1_all.loc[[f"q1_ctt_{i}" for i in [0,1,2]], "lz"].astype(float).mean(),
    q2g_q1_all.loc[[f"q1_pop1_{i}" for i in [0,1,2]], "lz"].astype(float).mean(),
    q2g_q1_all.loc[[f"q1_pop0_{i}" for i in [0,1,2]], "lz"].astype(float).mean(),
]
ssr_med_vals = [
    float(pf_all.loc["t9_base", "ssr_median"]),
    pf_all.loc[sft_states, "ssr_median"].astype(float).mean(),
    pf_all.loc[grpo_states, "ssr_median"].astype(float).mean(),
    q2g_q1_all.loc[[f"q1_grpo_{i}" for i in [0,1,2]], "ssr_median"].astype(float).mean(),
    q2g_q1_all.loc[[f"q1_ctt_{i}" for i in [0,1,2]], "ssr_median"].astype(float).mean(),
    q2g_q1_all.loc[[f"q1_pop1_{i}" for i in [0,1,2]], "ssr_median"].astype(float).mean(),
    q2g_q1_all.loc[[f"q1_pop0_{i}" for i in [0,1,2]], "ssr_median"].astype(float).mean(),
]

xf = np.arange(len(fams))
cols_f = [C_BASE, C_SFT, C_GRPO, "#ff7f0e", C_CTT, C_POP1, C_POP0]
bars_lz = ax1.bar(xf, lz32_vals, width=0.55, color=cols_f, edgecolor="black", lw=0.7)
for b_item, v, sm in zip(bars_lz, lz32_vals, ssr_med_vals):
    ax1.text(b_item.get_x() + b_item.get_width()/2, v - 14, f"{v:.0f}\n(SSR={sm:.1f})",
             ha="center", va="top", fontsize=7.9, fontweight="bold")

ax1.set_xticks(xf)
ax1.set_xticklabels(fams, fontsize=8.2)
ax1.set_ylim(-655, 215)
ax1.axhline(0, color="#333333", lw=1.0)
ax1.set_ylabel("Multi-Draw Standardized Person Fit $l_z^{(32)}$ ($K=32$)", fontsize=10.5)
ax1.set_title("A. Theorem 2: The $l_z^{(1)} = +2.24 \\to l_z^{(32)} = -482$ Person-Fit Sign Flip", fontsize=11.8, fontweight="bold", pad=10)
ax1.text(0.02, 0.97,
         "At $k=1$ draw (Q2a), S1 GRPO has $l_z^{(1)} = +2.24 > +1.645$ (looks Guttman hyper-consistent!).\n"
         "At $K=32$ draws, $\\mathbb{E}[\\mathrm{SSR}_{sj}] = \\frac{p(1-p)}{\\hat{p}(1-\\hat{p})} + K\\frac{(p - \\hat{p}_{2\\mathrm{PL}})^2}{\\hat{p}(1-\\hat{p})}$ multiplies curve divergence\n"
         "($L_{\\mathrm{base},j} \\neq z_j^{\\mathrm{pop}}$) by $K=32$, flipping $l_z^{(32)} \\to -482$ (S1) and $-577$ (Q1 pop0)!",
         transform=ax1.transAxes, va="top", ha="left", fontsize=8.2,
         bbox=dict(boxstyle="round,pad=0.34", facecolor="#fff5f5", edgecolor=C_GRPO, alpha=0.95))

# Panel B: Q2c Adaptive CAT vs Classical Random RMSE as n: 25 -> 50 -> 100
q2c_err = pd.read_csv(os.path.join(IPL, "Q2c", "errors.csv"))
q2c_high = q2c_err[(q2c_err["level"] == "band") & (q2c_err["band"] == "above 0.45")].sort_values("n")

ns = [25, 50, 100]
rmse_cl = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "classical")]["rmse"].iloc[0]) for n in ns]
rmse_irtr = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_random")]["rmse"].iloc[0]) for n in ns]
rmse_irt = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_adaptive")]["rmse"].iloc[0]) for n in ns]
bias_irt = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_adaptive")]["bias"].iloc[0]) for n in ns]

ax2.plot(ns, rmse_cl, marker="o", ms=9, lw=2.6, color=C_SFT, label="Classical Random Sampling ($\\mathrm{RMSE} \\propto 1/\\sqrt{n}$, Unbiased)")
ax2.plot(ns, rmse_irtr, marker="^", ms=8, lw=2.0, ls="--", color=C_POP1, label="2PL Random Sampling (`irt_random`)")
ax2.plot(ns, rmse_irt, marker="s", ms=9, lw=2.6, color=C_GRPO, label="Max-Info 2PL Adaptive CAT (`irt_adaptive`, Picks High-$a_j^{\\mathrm{pop}}$)")

for idx_n, (n_v, rc, ri, bi) in enumerate(zip(ns, rmse_cl, rmse_irt, bias_irt)):
    if idx_n == 1:
        ax2.text(n_v, rc + 0.003, f"{rc:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9)
        ax2.text(n_v, ri - 0.009, f"{ri:.3f}\n(bias {bi:+.3f})", ha="center", va="top", color=C_GRPO, fontweight="bold", fontsize=8.3)
    else:
        ax2.text(n_v, rc - 0.004, f"{rc:.3f}", ha="center", va="top", color=C_SFT, fontweight="bold", fontsize=9)
        ax2.text(n_v, ri + 0.003, f"{ri:.3f}\n(bias {bi:+.3f})", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=8.3)

ax2.set_xticks(ns)
ax2.set_xticklabels(["n = 25 Items", "n = 50 Items", "n = 100 Items"], fontsize=10)
ax2.set_xlim(18, 108)
ax2.set_ylim(0.018, 0.116)
ax2.set_xlabel("Short-Test Item Budget $n$ (Unit Q2c, Pretraining Checkpoints $>0.45$)", fontsize=10.5)
ax2.set_ylabel("Level Estimation RMSE against Full Bank", fontsize=10.5)
ax2.set_title("B. CANNOT #3: Why High-$a_j^{\\mathrm{pop}}$ Adaptive CAT Loses by $n=100$", fontsize=11.8, fontweight="bold", pad=10)
ax2.text(0.02, 0.24,
         "• Max-info CAT greedily selects $I_j = (a_j^{\\mathrm{pop}})^2 \\hat{p}_j(1-\\hat{p}_j) \\Rightarrow$ 40%–49% Sociology!\n"
         "• High-$a_j^{\\mathrm{pop}}$ items have severe prior misfit $\\Rightarrow$ non-vanishing bias (+0.04 to +0.09).\n"
         "• On S1/Q1 post-trained models at $k=16$ (Q2a), CAT RMSE is 1.40x–1.65x worse\n"
         "  than random sampling across ALL cells ($n=25, 50, 100$).",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.2,
         bbox=dict(boxstyle="round,pad=0.34", facecolor="#fffdf0", edgecolor="#d4a017", alpha=0.95))
ax2.legend(loc="upper right", frameon=True, fontsize=8.4)

fig.suptitle("B Helps A & CANNOT #3: Multi-Draw Person-Fit Sign Flip ($l_z$) and High-$a_j^{\\mathrm{pop}}$ CAT Selection Bias",
             fontsize=13, fontweight="bold", y=1.02)
save_fig(fig, "F6_lz_signflip_and_cat_bias")

# ==============================================================================
# FIGURE 7: CANNOT #4 — Q1 RL Training Trajectory Wall (0.479) & Zero-Variance Starvation
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), gridspec_kw={"wspace": 0.25})

traj = pd.read_csv(os.path.join(IPL, "Q1", "trajectory.csv"))
arm_styles = [
    ("grpo", "Q1 GRPO (G=5, Zero-Var Shutoff)", C_GRPO, "o", "-"),
    ("ctt",  "Q1 CTT Pass-Band Critic (G=1)", C_CTT, "s", "--"),
    ("pop0", "Q1 Unweighted 2PL Critic pop0 (G=1)", C_POP0, "^", "-."),
    ("pop1", "Q1 $a_j^{\\mathrm{pop}}$-Weighted 2PL Critic pop1 (G=1)", C_POP1, "D", ":"),
]
for mode, lbl, col, mk, ls in arm_styles:
    sub = traj[(traj["mode"] == mode) & (traj["seed"] == 0)].sort_values("updates")
    ax1.plot(sub["updates"], sub["mean_at_16"], marker=mk, ms=7.5, lw=2.3, ls=ls, color=col, label=f"{lbl} (seed 0 trace)")
    sts = [f"q1_{mode}_{i}" for i in [0, 1, 2]]
    final_3seed = q1_pool.loc[sts, "mean_at_16"].mean()
    ax1.scatter([333], [final_3seed], s=115, marker=mk, color=col, edgecolors="black", lw=1.2, zorder=6)

ax1.axhline(0.414, color=C_BASE, lw=1.5, ls="--", label="Base Modal Boundary Maj@16 = 0.414 (0.459* tie-correct)")
ax1.axhline(0.504, color=C_SFT, lw=1.8, ls="-.", label="SFT Modal Ceiling Maj@16 = 0.504 (0.555* tie-correct)")
ax1.axvspan(83, 333, color="#eeeeee", alpha=0.55)
ax1.axvline(83, color="#333333", lw=1.2, ls=":")
ax1.text(92, 0.426,
         "Update 83 Elbow: 84.4%–100% of total RL gain\n"
         "already finished by Update 83!\n"
         "All 12 Q1 runs + 3 S1 runs hit the 0.469–0.479 wall.",
         fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#555555", alpha=0.95))

ax1.set_xticks([0, 83, 166, 250, 333])
ax1.set_ylim(0.365, 0.522)
ax1.set_xlabel("RL Training Updates (Unit Q1, 333 Updates = 1,665 Steps)", fontsize=10.5)
ax1.set_ylabel("Full-Bank Single-Draw Accuracy Mean@16 ($J=1,500$)", fontsize=10.5)
ax1.set_title("A. All 15 Base → RL Runs Hit the 0.469–0.479 Wall by Update 83", fontsize=11.8, fontweight="bold", pad=10)
ax1.legend(loc="lower right", frameon=True, fontsize=7.8)

# Panel B: Zero-Advantage Share Across Updates in Q1 GRPO (3 seeds)
traces = []
for s in [0, 1, 2]:
    df_tr = pd.read_csv(os.path.join(IPL, "Q1", "traces", f"grpo_{s}.csv"))
    df_tr["z_smooth"] = df_tr["zero_adv_share"].rolling(15, min_periods=1).mean()
    traces.append(df_tr)
    ax2.plot(df_tr["update"], df_tr["z_smooth"] * 100, lw=1.5, alpha=0.55, color=C_GRPO, label=f"Seed {s} (15-step moving avg)" if s==0 else None)

mean_z = np.mean([t["z_smooth"].values for t in traces], axis=0) * 100
ax2.plot(traces[0]["update"], mean_z, lw=2.8, color="#8b0000", label="3-Seed Mean Zero-Advantage Share (%)")
ax2.axhline(35.0, color=C_BASE, lw=1.4, ls="--", label="Initial W1 Base Unbiased $G=5$ Zero-Variance Share (35.0%)")
ax2.axhline(63.1, color=C_POP0, lw=1.5, ls=":", label="Post-Update-83 Mean Zero-Variance Share (63.1%, Peak 84%–91%)")
ax2.axvline(83, color="#333333", lw=1.2, ls=":")

ax2.set_xlim(0, 333)
ax2.set_ylim(20, 95)
ax2.set_xlabel("RL Training Update ($G=5$ Answers per Prompt)", fontsize=10.5)
ax2.set_ylabel("Prompts with Zero Reward Variance `zero_adv_share` (%)", fontsize=10.5)
ax2.set_title("B. Why Static $(a_j^{\\mathrm{pop}}, b_j^{\\mathrm{pop}})$ Multipliers Starve: Variance Collapse", fontsize=11.8, fontweight="bold", pad=10)
ax2.text(0.03, 0.95,
         "• Once $p_{sj} \\to \\{0, 1\\}$, sample variance $\\hat{\\sigma}_{r,j} = 0$ on 63.1% of batches (85.4% in T9r).\n"
         "• Multiplying zero variance by static $a_j^{\\mathrm{pop}}$ yields zero gradient signal!\n"
         "• In $G=1$ critics (pop0/pop1/ctt), no shutoff gate + fixed $L_0=78.98$ norm\n"
         "  causes extreme diversity collapse (stable_12 = 0.866–0.942 vs. 0.757 GRPO).",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.4,
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor=C_GRPO, alpha=0.94))
ax2.legend(loc="lower right", frameon=True, fontsize=8.2)

fig.suptitle("CANNOT #4 (Unit Q1): Why Static 2PL Advantage Weighting Converges to a 0.63% Band (0.4685–0.4748)",
             fontsize=13, fontweight="bold", y=1.02)
save_fig(fig, "F7_q1_wall_and_zero_adv")

# ==============================================================================
# FIGURE 8: CAN #3 / A <-> B — Unit M1 Architecture & Within-Model ZPD Item Parameter p_SFT,j
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.6), gridspec_kw={"wspace": 0.36})

# Panel A: Base W1 Pass-Rate Distribution p_base,j (10,682 Training Items, k=8) & Fisher Information I_j = p_j(1-p_j)
w1_pr = pd.read_csv(os.path.join(IPL, "W1", "pass_rate.csv"))
counts_k = w1_pr["correct"].value_counts().sort_index()
ks = np.arange(0, 9)
shares_k = [counts_k.get(k, 0) / len(w1_pr) * 100 for k in ks]
zpd_mask = [(2 <= k <= 6) for k in ks]
bar_cols = [C_ZPD if m else "#bbbbbb" for m in zpd_mask]

bars_w1 = ax1.bar(ks, shares_k, width=0.62, color=bar_cols, edgecolor="black", lw=0.8)
for b_i, sh, k_v in zip(bars_w1, shares_k, ks):
    ax1.text(b_i.get_x() + b_i.get_width()/2, sh + 0.5, f"{sh:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold")

ax1_twin = ax1.twinx()
p_grid = np.linspace(0, 8, 200) / 8.0
fisher_info = p_grid * (1.0 - p_grid)
ax1_twin.plot(np.linspace(0, 8, 200), fisher_info, color=C_GRPO, lw=2.4, ls="--", label="Within-Model IRT Fisher Info $I_j = p_j(1-p_j)$")
ax1_twin.set_ylabel("Within-Model Rollout Fisher Info $p_j(1-p_j)$", color=C_GRPO, fontsize=10)
ax1_twin.set_ylim(0, 0.33)
ax1_twin.grid(False)

zpd_total = sum(shares_k[2:7])
boundary_total = 100.0 - zpd_total
ax1.set_xticks(ks)
ax1.set_xticklabels([f"{k}/8\n({k/8:.2f})" for k in ks], fontsize=8.6)
ax1.set_ylim(0, 32)
ax1.set_xlabel("Empirical Item Pass Rate $p_j = \\mathrm{correct}/8$ Across 10,682 MMLU Training Prompts", fontsize=10.2)
ax1.set_ylabel("Share of Training Pool (%)", fontsize=10.5)
ax1.set_title("A. Within-Model ZPD Item Parameter $p_j \\in [0.2, 0.8]$ (`2..6 of 8`)", fontsize=11.5, fontweight="bold", pad=10)
ax1.text(0.02, 0.96,
         f"• Exterior ($0..1$ or $7..8$ of 8): {boundary_total:.1f}% of prompts (near-zero RL variance)\n"
         f"• Interior ZPD ($2..6$ of 8, $p_j \\in [0.2, 0.8]$): {zpd_total:.1f}% of prompts\n"
         "• Critical Fix in M1c: Because SFT reorders item difficulty ($r=0.289$),\n"
         "  M1c calibrates $p_{\\mathrm{SFT},j}$ directly on `t9_sft_0..2_70` ($10,682 \\times 8$ draws)!",
         transform=ax1.transAxes, va="top", ha="left", fontsize=8.3,
         bbox=dict(boxstyle="round,pad=0.34", facecolor="#e8f8fc", edgecolor=C_ZPD, alpha=0.95))

# Panel B: The Operator Composition & ZPD Gap Diagram
stages = ["1. Base\n(t9_base)", "2. Base → RL\n(15 Runs: S1+Q1)", "3. Base → SFT\n(t9_sft_0..2_70)", "4. Unit M1 (SFT → RL)\n[M1a / M1b / M1c]"]
mean16_bars = [0.382, 0.474, 0.439, 0.479]
maj16_bars  = [0.414, 0.481, 0.504, 0.504]
maj_tc_bars = [0.459, 0.490, 0.555, 0.555]

xs_st = np.arange(len(stages))
w_s = 0.25
ax2.bar(xs_st - w_s, mean16_bars, width=w_s, color=["#777777", C_GRPO, C_SFT, C_ZPD], edgecolor="black", label="Single-Draw Mean@16")
ax2.bar(xs_st, maj16_bars, width=w_s, color=["#aaaaaa", "#f08080", "#6baed6", "#74c476"], edgecolor="black", label="Majority Vote Maj@16 (Unstarred)")
ax2.bar(xs_st + w_s, maj_tc_bars, width=w_s, color=["#dddddd", "#ffb6c1", "#bdd7e7", "#bae4b3"], edgecolor="black", label="Majority Vote Maj@16* (Tie-Correct)")

ax2.axhline(0.479, color=C_GRPO, lw=1.6, ls="--", label="15-Run Base→RL Immovable Wall (0.479)")
ax2.annotate("Unharvested Condorcet Gap\n(+0.065 to +0.116*)\nin SFT (H = 0.943 nats)",
             xy=(2.0, 0.508), xytext=(0.25, 0.555),
             arrowprops=dict(arrowstyle="->", color=C_SFT, lw=1.5),
             fontsize=8.1, fontweight="bold", color=C_SFT,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="#eef6ff", edgecolor=C_SFT))

ax2.text(2.96, 0.605,
         "Pre-Registered High-Leverage Bet:\n"
         "• +1: M1b/M1c breaks 0.479 wall\n"
         "      (Operator Composition Law)\n"
         "• -1: RL short-path incentive\n"
         "      erodes SFT Maj@16 ceiling\n"
         "• M1a: Tests T=0.2 vs RL gain",
         ha="center", va="bottom", fontsize=7.9, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.30", facecolor="#fffdf0", edgecolor="#d4a017"))

ax2.set_xticks(xs_st)
ax2.set_xticklabels(stages, fontsize=8.8)
ax2.set_ylim(0.32, 0.74)
ax2.set_ylabel("Full-Bank Score ($J=1,500$)", fontsize=10.5)
ax2.set_title("B. Why All 15 Base→RL Runs Stalled at 0.479 & How Unit M1 Tests SFT→RL", fontsize=11.5, fontweight="bold", pad=10)
ax2.legend(loc="upper left", frameon=True, fontsize=7.9)

fig.suptitle("CAN #3 (A <-> B, Unit M1): Using Within-Model Item Difficulty $p_{\\mathrm{SFT},j}$ to Guide SFT → RL Operator Composition",
             fontsize=13, fontweight="bold", y=1.02)
save_fig(fig, "F8_unit_m1_architecture")

print("ALL 8 FIGURES RE-GENERATED CLEANLY.")
