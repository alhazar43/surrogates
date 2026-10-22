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
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.25,
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
C_SFT = "#1f6feb"     # Deep royal blue (SFT)
C_GRPO = "#d9381e"    # Vermilion red (GRPO)
C_CTT = "#e69f00"     # Amber
C_POP0 = "#8a3ffc"    # Purple
C_POP1 = "#009e73"    # Teal
C_ZPD = "#118ab2"     # Cyan-blue

def safe_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def partial_r(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])

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

# ==============================================================================
# FIGURE 1: Cross-Population Difficulty (E1) & Population a_j vs Within-Model Shift
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), gridspec_kw={"wspace": 0.25})

# Panel A: E1 Human vs LLM difficulty b_j (J = 704 items)
e1_diff = pd.read_csv(os.path.join(IPL, "E1", "difficulty.csv")).dropna(subset=["b_students", "b_llms"])
e1_scores = pd.read_csv(os.path.join(IPL, "E1", "llm_scores.csv"))
n_e1 = len(e1_diff)
n_zh_neg = int((e1_scores["zh_2pl"] < -1.96).sum())
n_llms_e1 = len(e1_scores)

b_stu = e1_diff["b_students"].values
b_llm = e1_diff["b_llms"].values
r_e1 = float(np.corrcoef(b_stu, b_llm)[0, 1])
rho_e1 = float(pd.Series(b_stu).corr(pd.Series(b_llm), method="spearman"))
m_fit, c_fit = np.polyfit(b_stu, b_llm, 1)

ax1.scatter(b_stu, b_llm, alpha=0.42, s=22, color="#2b5c8f", edgecolors="none",
            label=f"Eedi Diagnostic Math Items ($J = {n_e1}$)")
xs_line = np.linspace(b_stu.min(), b_stu.max(), 100)
ax1.plot(xs_line, xs_line, color="#444444", lw=1.4, ls="--", label="Identity Line ($y = x$, slope $= 1.00$)")
ax1.plot(xs_line, m_fit * xs_line + c_fit, color=C_GRPO, lw=2.2,
         label=f"OLS Fit: slope $= {m_fit:+.2f}$, $r = {r_e1:+.3f}$, Spearman $\\rho = {rho_e1:+.3f}$")
ax1.axhline(0, color="#888888", lw=0.8, ls=":")
ax1.axvline(0, color="#888888", lw=0.8, ls=":")
ax1.set_title(f"A. Human vs. 61-LLM Item Difficulty $b_j$ (Unit E1, $J={n_e1}$)", fontsize=11.8, fontweight="bold", pad=10)
ax1.set_xlabel("Human Student 2PL Difficulty $b_j^{\\mathrm{students}}$ ($N=4,918$ Students)", fontsize=10.5)
ax1.set_ylabel("61-LLM 2PL Difficulty $b_j^{\\mathrm{LLMs}}$ ($N=61$ LLMs)", fontsize=10.5)
ax1.text(0.03, 0.95,
         f"• {n_zh_neg} / {n_llms_e1} LLMs below $Z_h = -1.96$ under human 2PL (`E1/llm_scores.csv`)\n"
         f"• 61-LLM $b_j$ spans $[{b_llm.min():.0f}, +{b_llm.max():.0f}]$ (rejected 61-LLM fit, `closure.md:29`)\n"
         f"• Rank order only: Spearman $\\rho = {rho_e1:+.3f}$ (raw Pearson $r = {r_e1:+.3f}$)",
         transform=ax1.transAxes, va="top", ha="left", fontsize=8.8,
         bbox=dict(boxstyle="round,pad=0.38", facecolor="#fff5f5", edgecolor=C_GRPO, alpha=0.93))
ax1.legend(loc="lower right", frameon=True, fontsize=8.8)

# Panel B: Population discrimination a_j^pop vs Single-Model Post-Training Log-Odds Shift (T9r, J=1,472)
Lb_t9 = safe_logit(m_t9r["p_key_base"].values, 1e-4)
L_rl_mean = np.mean([safe_logit(m_t9r["p_key_base"].values + m_t9r[f"dp_key_rl_{s}_666"].values, 1e-4) for s in [0, 1, 2]], axis=0)
L_sft_mean = np.mean([safe_logit(m_t9r["p_key_base"].values + m_t9r[f"dp_key_sft_{s}_70"].values, 1e-4) for s in [0, 1, 2]], axis=0)
dL_rl = L_rl_mean - Lb_t9
dL_sft = L_sft_mean - Lb_t9
a_pop = m_t9r["a"].values

r_apop_rl = float(np.corrcoef(a_pop, dL_rl)[0, 1])
r_apop_sft = float(np.corrcoef(a_pop, dL_sft)[0, 1])
m2_rl, c2_rl = np.polyfit(a_pop, dL_rl, 1)
m2_sft, c2_sft = np.polyfit(a_pop, dL_sft, 1)

ax2.scatter(a_pop, dL_rl, alpha=0.28, s=16, color=C_GRPO, edgecolors="none", label=f"GRPO Step 666 Shift $\\Delta L_{{\\mathrm{{GRPO}},j}}$ ($J={len(m_t9r)}$)")
ax2.scatter(a_pop, dL_sft, alpha=0.22, s=15, color=C_SFT, edgecolors="none", label=f"SFT Step 70 Shift $\\Delta L_{{\\mathrm{{SFT}},j}}$ ($J={len(m_t9r)}$)")
xs2 = np.linspace(0, np.percentile(a_pop, 99.5), 100)
ax2.plot(xs2, m2_rl * xs2 + c2_rl, color="#8b0000", lw=2.3,
         label=f"GRPO OLS: slope $= {m2_rl:+.2f}$, $r = {r_apop_rl:+.3f}$ ($R^2 = {r_apop_rl**2*100:.1f}\\%$)")
ax2.plot(xs2, m2_sft * xs2 + c2_sft, color="#003380", lw=2.3, ls="--",
         label=f"SFT OLS: slope $= {m2_sft:+.2f}$, $r = {r_apop_sft:+.3f}$ ($R^2 = {r_apop_sft**2*100:.1f}\\%$)")
ax2.set_xlim(0, 3.6)
ax2.set_ylim(-6.8, 6.8)
ax2.axhline(0, color="#888888", lw=0.8, ls=":")
ax2.set_title(f"B. 102-LLM $a_j^{{\\mathrm{{pop}}}}$ vs. Token-0 Post-Training Shift ($J={len(m_t9r)}$)", fontsize=11.8, fontweight="bold", pad=10)
ax2.set_xlabel("102-LLM Population Discrimination $a_j^{\\mathrm{pop}}$ (`W0/items_bank.csv`)", fontsize=10.5)
ax2.set_ylabel("Token-0 Log-Odds Shift $\\Delta L_{s,j}$ (3-Seed Mean, `T9r`)", fontsize=10.5)
ax2.text(0.03, 0.95,
         f"• In token-0 reads (`T9r`), $a_j^{{\\mathrm{{pop}}}}$ correlates $r = {r_apop_rl:+.3f}$ ($R^2 = {r_apop_rl**2*100:.1f}\\%$)\n"
         f"  with $\\Delta L_{{\\mathrm{{GRPO}},j}}$ and $r = {r_apop_sft:+.3f}$ ($R^2 = {r_apop_sft**2*100:.1f}\\%$) with $\\Delta L_{{\\mathrm{{SFT}},j}}$.\n"
         f"• Why? High-$a_j^{{\\mathrm{{pop}}}}$ items have higher Base log-odds $L_{{\\mathrm{{base}},j}}$ ($r = {np.corrcoef(a_pop, Lb_t9)[0,1]:+.3f}$).",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.8,
         bbox=dict(boxstyle="round,pad=0.38", facecolor="#f4f0ff", edgecolor=C_POP0, alpha=0.93))
ax2.legend(loc="lower right", frameon=True, fontsize=8.4)

fig.suptitle("CANNOT #1: Human vs. LLM Item Difficulty (`E1`) and Cross-Model $a_j^{\\mathrm{pop}}$ vs. Token-0 Shift (`W0` + `T9r`)",
             fontsize=12.8, fontweight="bold", y=1.02)
save_fig(fig, "F1_non_ergodicity")

# ==============================================================================
# FIGURE 2: Within-Model Prior-Scaling Regression & Slope Decomposition
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), gridspec_kw={"wspace": 0.25})

L_sft_0 = safe_logit(m_t9r["p_key_base"].values + m_t9r["dp_key_sft_0_70"].values, 1e-4)
L_rl_0 = safe_logit(m_t9r["p_key_base"].values + m_t9r["dp_key_rl_0_666"].values, 1e-4)
dL_sft_0 = L_sft_0 - Lb_t9
dL_rl_0 = L_rl_0 - Lb_t9

m_rl, c_rl = np.polyfit(Lb_t9, dL_rl_0, 1)
m_sf, c_sf = np.polyfit(Lb_t9, dL_sft_0, 1)
r2_rl_0 = float(np.corrcoef(Lb_t9, dL_rl_0)[0, 1] ** 2 * 100)
r2_sf_0 = float(np.corrcoef(Lb_t9, dL_sft_0)[0, 1] ** 2 * 100)
tau_rl_0 = 1.0 + m_rl
tau_sf_0 = 1.0 + m_sf

# Compute 3-seed ranges from T9r directly
r_sft_t9, sr_sft_t9, tau_sft_t9, r2_sft_t9 = [], [], [], []
r_rl_t9, sr_rl_t9, tau_rl_t9, r2_rl_t9 = [], [], [], []
for s in [0, 1, 2]:
    Ls = safe_logit(m_t9r["p_key_base"].values + m_t9r[f"dp_key_sft_{s}_70"].values, 1e-4)
    Lr = safe_logit(m_t9r["p_key_base"].values + m_t9r[f"dp_key_rl_{s}_666"].values, 1e-4)
    rs = float(np.corrcoef(Lb_t9, Ls)[0, 1])
    srs = float(np.std(Ls, ddof=1) / np.std(Lb_t9, ddof=1))
    rr = float(np.corrcoef(Lb_t9, Lr)[0, 1])
    srr = float(np.std(Lr, ddof=1) / np.std(Lb_t9, ddof=1))
    r_sft_t9.append(rs); sr_sft_t9.append(srs); tau_sft_t9.append(rs * srs)
    r2_sft_t9.append(float(np.corrcoef(Lb_t9, Ls - Lb_t9)[0, 1] ** 2 * 100))
    r_rl_t9.append(rr); sr_rl_t9.append(srr); tau_rl_t9.append(rr * srr)
    r2_rl_t9.append(float(np.corrcoef(Lb_t9, Lr - Lb_t9)[0, 1] ** 2 * 100))

ax1.scatter(Lb_t9, dL_rl_0, alpha=0.28, s=15, color=C_GRPO,
            label=f"GRPO Seed 0 (`rl_0_666`): $\\hat{{\\tau}}_0 = {tau_rl_0:.3f}$ (3-seed {min(tau_rl_t9):.2f}–{max(tau_rl_t9):.2f})")
ax1.scatter(Lb_t9, dL_sft_0, alpha=0.28, s=15, color=C_SFT,
            label=f"SFT Seed 0 (`sft_0_70`): $\\hat{{\\tau}}_0 = {tau_sf_0:.3f}$ (3-seed {min(tau_sft_t9):.2f}–{max(tau_sft_t9):.2f})")

xs_lb = np.linspace(-9.2, 9.2, 100)
ax1.plot(xs_lb, m_rl * xs_lb + c_rl, color="#8b0000", lw=2.4,
         label=f"GRPO Seed 0 Fit: $\\Delta L = ({m_rl:+.2f})L_{{\\mathrm{{base}}}} {c_rl:+.2f}$ ($R^2 = {r2_rl_0:.1f}\\%$)")
ax1.plot(xs_lb, m_sf * xs_lb + c_sf, color="#003380", lw=2.4,
         label=f"SFT Seed 0 Fit: $\\Delta L = ({m_sf:+.2f})L_{{\\mathrm{{base}}}} {c_sf:+.2f}$ ($R^2 = {r2_sf_0:.1f}\\%$)")
ax1.axhline(0, color="#666666", lw=0.8, ls=":")
ax1.axvline(0, color="#666666", lw=0.8, ls=":")
ax1.set_title("A. Token-0 Log-Odds Shift vs. Base $L_{\\mathrm{base},j}$ (`T9r`, Seed 0, $\\varepsilon=10^{-4}$)", fontsize=11.6, fontweight="bold", pad=10)
ax1.set_xlabel("Model's Own Base Token-0 Log-Odds $L_{\\mathrm{base},j}$ (`T9r/items_wide.csv`)", fontsize=10.5)
ax1.set_ylabel("Token-0 Post-Training Shift $\\Delta L_{s,j} = L_{s,j} - L_{\\mathrm{base},j}$", fontsize=10.5)
ax1.text(0.03, 0.95,
         f"• Token-0 ($\\varepsilon=10^{{-4}}$): $R^2(\\Delta L, L_{{\\mathrm{{base}}}})$ = {min(r2_sft_t9):.1f}%–{max(r2_sft_t9):.1f}% (SFT),\n"
         f"  {min(r2_rl_t9):.1f}%–{max(r2_rl_t9):.1f}% (GRPO; upper diagonal band is $1-10^{{-4}}$ clip).\n"
         f"• On $K=32$ draws (`Q2g/s1`), $R^2(\\Delta L, L_{{\\mathrm{{base}}}})$ = 27%–29% (SFT), 9.5%–10.1% (GRPO).\n"
         f"• Q2-A item flags tie classical raw change (`0/6` in S1, `1/12` in Q1).",
         transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.36", facecolor="#fffdf0", edgecolor="#d4a017", alpha=0.94))
ax1.legend(loc="lower left", frameon=True, fontsize=8.3)

# Panel B: Compute T9r and T9g/s3/matched.csv directly (zero hardcoded arrays)
t9g = pd.read_csv(os.path.join(IPL, "T9g", "s3", "matched.csv"))
t9g_free = t9g[t9g["form"] == "free"]
piv_t9g = t9g_free.pivot_table(index="item_id", columns="model", values="p_key_matched", aggfunc="mean")
r_t9g, sr_t9g, tau_t9g, sb_t9g, ss_t9g = [], [], [], [], []
for s in [0, 1, 2]:
    sub_g = piv_t9g[["base", f"sft_{s}_70"]].dropna()
    Lb_g = safe_logit(sub_g["base"].values, 1e-4)
    Ls_g = safe_logit(sub_g[f"sft_{s}_70"].values, 1e-4)
    rg = float(np.corrcoef(Lb_g, Ls_g)[0, 1])
    sbg = float(np.std(Lb_g, ddof=1))
    ssg = float(np.std(Ls_g, ddof=1))
    r_t9g.append(rg); sb_t9g.append(sbg); ss_t9g.append(ssg)
    sr_t9g.append(ssg / sbg); tau_t9g.append(rg * (ssg / sbg))

modalities = [
    "Token-0 Forced\n(T9r SFT, 3-seed)",
    "Token-0 Forced\n(T9r GRPO, 3-seed)",
    "Free-Text CoT\n(T9g/s3 SFT, 3-seed)",
]
r_vals = [float(np.mean(r_sft_t9)), float(np.mean(r_rl_t9)), float(np.mean(r_t9g))]
sig_ratios = [float(np.mean(sr_sft_t9)), float(np.mean(sr_rl_t9)), float(np.mean(sr_t9g))]
tau_vals = [float(np.mean(tau_sft_t9)), float(np.mean(tau_rl_t9)), float(np.mean(tau_t9g))]

x = np.arange(len(modalities))
w = 0.25
b1 = ax2.bar(x - w, r_vals, width=w, color="#2b5c8f", label="Correlation $r(L_s, L_{\\mathrm{base}})$")
b2 = ax2.bar(x, sig_ratios, width=w, color="#e69f00", label="Dispersion Ratio $\\sigma(L_s)/\\sigma(L_{\\mathrm{base}})$")
b3 = ax2.bar(x + w, tau_vals, width=w, color="#009e73", label="OLS Slope $\\hat{\\tau}_s = r \\cdot (\\sigma_s / \\sigma_{\\mathrm{base}})$")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=8.6, fontweight="bold")

ax2.axhline(1.0, color="#555555", lw=1.1, ls="--")
ax2.set_ylim(0, 3.65)
ax2.set_xticks(x)
ax2.set_xticklabels(modalities, fontsize=9.5)
ax2.set_ylabel("Component Magnitude (3-Seed Mean)", fontsize=10.5)
ax2.set_title("B. OLS Slope Identity $\\hat{\\tau}_s = r(L_s, L_{\\mathrm{base}}) \\cdot \\frac{\\sigma(L_s)}{\\sigma(L_{\\mathrm{base}})}$ (`T9r` & `T9g/s3`)", fontsize=11.6, fontweight="bold", pad=10)
ax2.text(0.02, 0.97,
         f"• In `T9g/s3` free-text CoT ($\\varepsilon=10^{{-4}}$), SFT $\\sigma_s = {np.mean(ss_t9g):.2f}$ vs. Base $\\sigma_b = {np.mean(sb_t9g):.2f}$\n"
         f"  (ratio {min(sr_t9g):.2f}–{max(sr_t9g):.2f}), while $r =$ {min(r_t9g):.3f}–{max(r_t9g):.3f} gives $\\hat{{\\tau}} =$ {min(tau_t9g):.2f}–{max(tau_t9g):.2f}.\n"
         f"• On $K=32$ draws (`Q2g/s1/residuals.csv`), SFT $r =$ 0.59–0.61; GRPO\n"
         f"  $r =$ 0.73–0.75, $\\sigma_s/\\sigma_b =$ 1.91–1.99, $\\hat{{\\tau}} =$ 1.43–1.46 per seed.",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.3,
         bbox=dict(boxstyle="round,pad=0.33", facecolor="#eef6ff", edgecolor=C_SFT, alpha=0.95))
ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.75), frameon=True, fontsize=8.2)

fig.suptitle("CANNOT #2: Why Q2-A Item Flags Tie Classical Change — Role of Base Log-Odds $L_{\\mathrm{base},j}$",
             fontsize=12.8, fontweight="bold", y=1.02)
save_fig(fig, "F2_prior_scaling")

# ==============================================================================
# FIGURE 3: 15-Subject SFT vs. GRPO Domain Gains (Seen S1 States, K=32)
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

b_vec = sub_grp["b_mean"].values
a_vec = sub_grp["a_mean"].values
diff_vec = sub_grp["diff_sft_grpo"].values
r_b = float(np.corrcoef(b_vec, diff_vec)[0, 1])
r_a = float(np.corrcoef(a_vec, diff_vec)[0, 1])
r_ab_sub = float(np.corrcoef(a_vec, b_vec)[0, 1])
r_a_grpo = float(np.corrcoef(a_vec, sub_grp["dL_grpo"].values)[0, 1])

colors_dom = [C_SFT if d > 0.1 else (C_GRPO if d < -0.1 else "#777777") for d in diff_vec]
ax1.scatter(b_vec, diff_vec, c=colors_dom, s=85, zorder=4, edgecolors="black", linewidth=0.8,
            label="15 MMLU Subject Means (Seen S1 States)")

mb, cb = np.polyfit(b_vec, diff_vec, 1)
xb_line = np.linspace(b_vec.min() - 0.2, b_vec.max() + 0.2, 100)
ax1.plot(xb_line, mb * xb_line + cb, color="#222222", lw=2.2, ls="--",
         label=f"OLS Fit: slope $= {mb:+.2f}$, intercept $= {cb:+.2f}$, $r = {r_b:+.4f}$ ($R^2 = {r_b**2*100:.1f}\\%$)")

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
    "Prehistory": (-68, -4),
    "HS Geography": (-78, 7),
    "HS Biology": (8, -6),
    "World History": (-82, -4),
    "Sociology": (8, -10),
    "Marketing": (-15, 10),
}
for _, row in sub_grp.iterrows():
    dx, dy = offsets_a.get(row["short"], (5, 5))
    ax1.annotate(f"{row['short']} (a={row['a_mean']:.2f})",
                 (row["b_mean"], row["diff_sft_grpo"]),
                 xytext=(dx, dy), textcoords="offset points", fontsize=7.7, fontweight="bold")

ax1.axhline(0, color="#444444", lw=1.2, ls="-")
ax1.axhspan(0, 2.62, color=C_SFT, alpha=0.06)
ax1.axhspan(-1.72, 0, color=C_GRPO, alpha=0.06)
ax1.set_ylim(-1.72, 2.58)
ax1.set_xlim(-1.28, 2.68)
ax1.set_title("A. 15 Subject Means: $\\bar{b}_m^{\\mathrm{pop}}$ vs. $\\overline{\\Delta L}_{\\mathrm{SFT},m} - \\overline{\\Delta L}_{\\mathrm{GRPO},m}$ (`S1`, Seen)", fontsize=11.4, fontweight="bold", pad=10)
ax1.set_xlabel(f"Subject Mean 2PL Difficulty $\\bar{{b}}_m^{{\\mathrm{{pop}}}}$ ($n=15$ Subjects, $r = {r_ab_sub:+.3f}$ with $\\bar{{a}}_m^{{\\mathrm{{pop}}}}$)", fontsize=10.2)
ax1.set_ylabel("Subject Log-Odds Difference $\\overline{\\Delta L}_{\\mathrm{SFT},m} - \\overline{\\Delta L}_{\\mathrm{GRPO},m}$", fontsize=10.5)
ax1.text(0.02, 0.97,
         f"• Across $n=15$ subject means (seen S1 states): $R^2(\\bar{{b}}_m^{{\\mathrm{{pop}}}}) = {r_b**2*100:.1f}\\%$,\n"
         f"  $R^2(\\bar{{a}}_m^{{\\mathrm{{pop}}}}) = {r_a**2*100:.1f}\\%$ ($r = {r_a:+.4f}$).\n"
         "• Crucial Confound Check: $\\bar{b}_m^{\\mathrm{pop}}$ correlates $-0.94$ with subject mean\n"
         "  Base log-odds $\\overline{L}_{\\mathrm{base},m}$; partial $r(\\bar{b}_m^{\\mathrm{pop}} \\mid \\overline{L}_{\\mathrm{base},m}) = 0.17$ (`all`) / $0.25$ (`parsed`).\n"
         "• Item level ($J=1,472$): $r(b_j^{\\mathrm{pop}}, \\Delta L_{\\mathrm{SFT},j} - \\Delta L_{\\mathrm{GRPO},j}) = 0.42$ ($R^2 = 17.7\\%$).",
         transform=ax1.transAxes, va="top", ha="left", fontsize=7.9,
         bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#333333", alpha=0.94))
ax1.legend(loc="lower right", frameon=True, fontsize=8.2)

# Panel B: Horizontal Paired Bar Chart Sorted by a_m^pop
y_pos = np.arange(len(sub_grp))
h_bar = 0.38
ax2.barh(y_pos + h_bar/2, sub_grp["dL_sft"], height=h_bar, color=C_SFT, label="CoT SFT Subject Mean Gain $\\overline{\\Delta L}_{\\mathrm{SFT},m}$")
ax2.barh(y_pos - h_bar/2, sub_grp["dL_grpo"], height=h_bar, color=C_GRPO, label="GRPO Subject Mean Gain $\\overline{\\Delta L}_{\\mathrm{GRPO},m}$")
ax2.axvline(0, color="#333333", lw=1.0)
ax2.axhline(6.5, color="#222222", lw=1.3, ls="--")
ax2.text(2.12, 2.2,
         "Low-$\\bar{a}_m^{\\mathrm{pop}}$ (0.28–0.55, 6 subjects):\n"
         "SFT leads 5/6 at $\\pm 0.1$ logit\n(HS Stats $-0.077$; Microecon\n$-0.084$ at $\\bar{a}_m=0.61$)",
         ha="right", va="center", fontsize=8.0, fontweight="bold", color=C_SFT,
         bbox=dict(boxstyle="round,pad=0.30", facecolor="#eef6ff", edgecolor=C_SFT))
ax2.text(2.12, 10.6,
         f"High-$\\bar{{a}}_m^{{\\mathrm{{pop}}}}$ (0.69–1.45, 8 subjects):\n"
         f"GRPO leads 8/8\n($r(\\bar{{a}}_m^{{\\mathrm{{pop}}}}, \\overline{{\\Delta L}}_{{\\mathrm{{GRPO}},m}}) = {r_a_grpo:+.3f}$;\n"
         "unseen Q1 arms $r =$ 0.64–0.72)",
         ha="right", va="center", fontsize=8.0, fontweight="bold", color=C_GRPO,
         bbox=dict(boxstyle="round,pad=0.30", facecolor="#fff5f5", edgecolor=C_GRPO))

ylabels = [f"{r['short']} (a={r['a_mean']:.2f})" for _, r in sub_grp.iterrows()]
ax2.set_yticks(y_pos)
ax2.set_yticklabels(ylabels, fontsize=8.8)
ax2.invert_yaxis()
ax2.set_xlim(-0.75, 2.18)
ax2.set_xlabel("Subject Mean Log-Odds Gain Over Base $\\overline{\\Delta L}_{s,m}$ ($K=32$ Rollouts, `Q2g/s1`)", fontsize=10.2)
ax2.set_title("B. 15 MMLU Subjects Sorted by Mean Discrimination $\\bar{a}_m^{\\mathrm{pop}}$", fontsize=11.5, fontweight="bold", pad=10)
ax2.legend(loc="upper center", bbox_to_anchor=(0.48, -0.12), frameon=True, fontsize=8.5, ncol=1)

fig.suptitle("CAN #1 (Descriptive on Seen S1 States): Subject Mean $(\\bar{a}_m^{\\mathrm{pop}}, \\bar{b}_m^{\\mathrm{pop}})$ vs. SFT and GRPO Subject Gains",
             fontsize=12.6, fontweight="bold", y=1.02)
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

metrics_labels = ["Single-Draw\nMean@16", "Majority Vote\nMaj@16 (Unstarred)", "Majority Vote\nMaj@16* (Tie-Correct)", "Best-of-16\nPass@16"]
base_vals = [
    float(s1_pool.loc["t9_base", "mean_at_16"]),
    float(s1_pool.loc["t9_base", "maj_at_16"]),
    float(s1_rw1.loc["t9_base", "maj_tie_correct"]),
    float(s1_pool.loc["t9_base", "pass_at_16"]),
]
sft_states = [f"t9_sft_{i}_70" for i in [0, 1, 2]]
grpo_states = [f"t9_rl_{i}_666" for i in [0, 1, 2]]
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

xm = np.arange(len(metrics_labels))
ax1.plot(xm, base_vals, marker="s", ms=8, lw=2.0, ls="--", color=C_BASE,
         label=f"Base (`t9_base`, $H={s1_pool.loc['t9_base','entropy']:.3f}$, Pass@16 $={base_vals[3]:.3f}$)")
ax1.plot(xm, grpo_vals, marker="o", ms=9, lw=2.6, color=C_GRPO,
         label=f"S1 GRPO 666u (3-seed mean, $H={s1_pool.loc[grpo_states,'entropy'].mean():.3f}$)")
ax1.plot(xm, sft_vals, marker="^", ms=9, lw=2.6, color=C_SFT,
         label=f"S1 CoT SFT 70u (3-seed mean, $H={s1_pool.loc[sft_states,'entropy'].mean():.3f}$)")

for idx, (bv, gv, sv) in enumerate(zip(base_vals, grpo_vals, sft_vals)):
    if idx == 0:
        ax1.text(idx - 0.06, gv + 0.018, f"{gv:.3f}", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=9)
        ax1.text(idx + 0.08, sv - 0.022, f"{sv:.3f}", ha="center", va="top", color=C_SFT, fontweight="bold", fontsize=9)
    else:
        ax1.text(idx, gv - 0.032, f"{gv:.3f}", ha="center", va="top", color=C_GRPO, fontweight="bold", fontsize=9)
        ax1.text(idx, sv + 0.020, f"{sv:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9)
ax1.text(3, base_vals[3] - 0.036, f"Base: {base_vals[3]:.3f}", ha="center", va="top", color=C_BASE, fontweight="bold", fontsize=8.5)

ax1.annotate(f"GRPO leads Mean@16\n({grpo_vals[0]:.3f} vs {sft_vals[0]:.3f})\nwith low entropy ($H=0.281$)",
             xy=(0, grpo_vals[0]), xytext=(0.22, 0.58),
             arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.4),
             fontsize=8.4, fontweight="bold", color=C_GRPO,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", edgecolor=C_GRPO))

ax1.annotate("SFT Maj@16 = 0.504 vs 0.481 (Rule 3 fires in 2/3 seeds;\n0/3 on order 1 alone); voting gap +0.065 (+0.116*).\nPass@16: SFT 0.966 (+0.036 over Base 0.930; GRPO 0.656).",
             xy=(2, sft_vals[2]), xytext=(0.45, 0.75),
             arrowprops=dict(arrowstyle="->", color=C_SFT, lw=1.4),
             fontsize=8.1, fontweight="bold", color=C_SFT,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef6ff", edgecolor=C_SFT))

ax1.set_xticks(xm)
ax1.set_xticklabels(metrics_labels, fontsize=9.5)
ax1.set_ylim(0.32, 1.05)
ax1.set_ylabel("Full-Bank Score ($J=1,500 \\times 32$ Rollouts)", fontsize=10.5)
ax1.set_title("A. S1 Evaluation Readout Across Functionals (`S1/stats.csv`)", fontsize=11.8, fontweight="bold", pad=10)
ax1.legend(loc="upper left", frameon=True, fontsize=8.4)

# Panel B: 1 Base + 18 Post-Trained States in the (Entropy H_s, Maj@16) Plane
ax2.scatter(s1_pool.loc["t9_base", "entropy"], s1_pool.loc["t9_base", "maj_at_16"],
            s=115, marker="s", color=C_BASE, edgecolors="black", zorder=6, label="Base (`t9_base`, Maj@16 = 0.414)")
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
all_rl_maj = list(s1_pool.loc[grpo_states, "maj_at_16"].values)
for arm, lbl, col, mk in q1_families:
    sts = [f"q1_{arm}_{i}" for i in [0, 1, 2]]
    vals_m = q1_pool.loc[sts, "maj_at_16"].values
    all_rl_maj.extend(vals_m)
    ax2.scatter(q1_pool.loc[sts, "entropy"], vals_m,
                s=85, marker=mk, color=col, edgecolors="black", zorder=4, label=lbl)

rl_maj_min, rl_maj_max = float(min(all_rl_maj)), float(max(all_rl_maj))
ax2.axhspan(rl_maj_min, rl_maj_max, color=C_GRPO, alpha=0.10,
            label=f"15 RL Runs Maj@16 Span ({rl_maj_min:.3f}–{rl_maj_max:.3f})")
ax2.set_xlabel("Within-Item Rollout Entropy $H_s$ (nats)", fontsize=10.5)
ax2.set_ylabel("Majority-Vote Accuracy $\\mathrm{Maj@16}$ ($J=1,500$, Unstarred)", fontsize=10.5)
ax2.set_title("B. 1 Base + 18 Post-Trained States in the $(H_s, \\mathrm{Maj@16})$ Plane", fontsize=11.8, fontweight="bold", pad=10)
ax2.annotate("Unit M1 Question:\nDoes SFT → RL reduce $H_s$\nwhile keeping SFT's Maj@16?",
             xy=(0.35, 0.504), xytext=(0.45, 0.507),
             arrowprops=dict(arrowstyle="->", color=C_ZPD, lw=1.8),
             fontsize=8.5, fontweight="bold", color=C_ZPD,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f8fc", edgecolor=C_ZPD))
ax2.set_ylim(0.402, 0.526)
ax2.set_xlim(0.05, 1.03)
ax2.legend(loc="lower left", frameon=True, fontsize=7.9, ncol=1)

fig.suptitle("B Helps A: Comparing Single-Draw Mean@16, Majority Vote Maj@16, and Rollout Entropy $H_s$",
             fontsize=12.8, fontweight="bold", y=1.02)
save_fig(fig, "F4_readout_crossover_19states")

# ==============================================================================
# FIGURE 5: CAN #2 — 2PL Stage Attribution on Seen S1 States & Subject Gain vs a_m
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), gridspec_kw={"wspace": 0.25})

pf_df = pd.read_csv(os.path.join(IPL, "Q2g", "s1", "personfit.csv"))
pf_all = pf_df[pf_df["variant"] == "all"].set_index("state")
pf_par = pf_df[pf_df["variant"] == "parsed"].set_index("state")

acc_base = float(pf_all.loc["t9_base", "accuracy"])
acc_sft = float(pf_all.loc[sft_states, "accuracy"].astype(float).mean())
acc_grpo = float(pf_all.loc[grpo_states, "accuracy"].astype(float).mean())

th_b = float(pf_all.loc["t9_base", "theta"])
th_s = float(pf_all.loc[sft_states, "theta"].astype(float).mean())
th_g = float(pf_all.loc[grpo_states, "theta"].astype(float).mean())

sft_share_acc = (acc_sft - acc_base) / (acc_grpo - acc_base) * 100
grpo_share_acc = (acc_grpo - acc_sft) / (acc_grpo - acc_base) * 100
sft_share_th = (th_s - th_b) / (th_g - th_b) * 100
grpo_share_th = (th_g - th_s) / (th_g - th_b) * 100

# Parsed variant shares
acc_b_p = float(pf_par.loc["t9_base", "accuracy"])
acc_s_p = float(pf_par.loc[sft_states, "accuracy"].astype(float).mean())
acc_g_p = float(pf_par.loc[grpo_states, "accuracy"].astype(float).mean())
th_b_p = float(pf_par.loc["t9_base", "theta"])
th_s_p = float(pf_par.loc[sft_states, "theta"].astype(float).mean())
th_g_p = float(pf_par.loc[grpo_states, "theta"].astype(float).mean())
sft_acc_p = (acc_s_p - acc_b_p) / (acc_g_p - acc_b_p) * 100
sft_th_p = (th_s_p - th_b_p) / (th_g_p - th_b_p) * 100

bars_x = ["Classical Accuracy\n($\\Delta \\mathrm{Acc} = \\overline{\\Delta p}$, Unweighted)",
          "Population 2PL Ability $\\hat{\\theta}$\n(Weighted by $a_j^{\\mathrm{pop}}$)"]
sft_shares = [sft_share_acc, sft_share_th]
grpo_shares = [grpo_share_acc, grpo_share_th]

ax1.bar(bars_x, sft_shares, width=0.48, color=C_SFT, label="Share Attributed to Base → SFT (`t9_sft`)")
ax1.bar(bars_x, grpo_shares, width=0.48, bottom=sft_shares, color=C_GRPO, label="Share Attributed to SFT → GRPO (`t9_rl`)")

ax1.text(0, sft_shares[0]/2, f"SFT: {sft_shares[0]:.1f}%\n(+{acc_sft-acc_base:.3f} acc)\n8.9–10.5 SE", ha="center", va="center", color="white", fontweight="bold", fontsize=9.8)
ax1.text(0, sft_shares[0] + grpo_shares[0]/2, f"GRPO: {grpo_shares[0]:.1f}%\n(+{acc_grpo-acc_sft:.3f} acc)\n3.8–5.9 SE", ha="center", va="center", color="white", fontweight="bold", fontsize=9.8)

ax1.text(1, sft_shares[1]/2, f"SFT: {sft_shares[1]:.1f}%\n(+{th_s-th_b:.3f} θ)\n2.9–5.1 SE", ha="center", va="center", color="white", fontweight="bold", fontsize=9.8)
ax1.text(1, sft_shares[1] + grpo_shares[1]/2, f"GRPO: {grpo_shares[1]:.1f}%\n(+{th_g-th_s:.3f} θ)\n8.6–10.5 SE", ha="center", va="center", color="white", fontweight="bold", fontsize=9.8)

ax1.text(0.5, 102.5,
         f"Seen S1 states (`all` draws; `parsed` gives {sft_acc_p:.0f}/{100-sft_acc_p:.0f} vs {sft_th_p:.0f}/{100-sft_th_p:.0f}).\n"
         "On the 12 unseen Q1 finals vs Base, $\\hat{\\theta}$ is sharper than Accuracy in 3/12 (`report-q1q2.md:79`).",
         ha="center", va="bottom", fontsize=7.6, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffdf0", edgecolor="#d4a017"))

ax1.set_ylim(0, 134)
ax1.set_ylabel("Percentage of Total Post-Training Gain (%)", fontsize=10.5)
ax1.set_title("A. Stage Credit Split on Seen S1 States: 59.5/40.5 vs. 20.9/79.1", fontsize=11.4, fontweight="bold", pad=10)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=True, fontsize=8.2)

# Panel B: Plot all 15 Subject Means directly against Subject Mean a_m^pop (obeys Research Rule 25!)
sp_rows = fl_s1[(fl_s1["variant"] == "all") & (fl_s1["reading"] == "A3_spearman_c_a")].set_index("state")["value"].astype(float)
rho_c_sft = [float(sp_rows[f"t9_sft_{s}_70"]) for s in [0, 1, 2]]
rho_c_grpo = [float(sp_rows[f"t9_rl_{s}_666"]) for s in [0, 1, 2]]

m_sft_a, c_sft_a = np.polyfit(a_vec, sub_grp["dL_sft"].values, 1)
m_grp_a, c_grp_a = np.polyfit(a_vec, sub_grp["dL_grpo"].values, 1)
r_sft_a = float(np.corrcoef(a_vec, sub_grp["dL_sft"].values)[0, 1])
r_grp_a = float(np.corrcoef(a_vec, sub_grp["dL_grpo"].values)[0, 1])

ax2.scatter(a_vec, sub_grp["dL_sft"].values, marker="^", s=70, color=C_SFT, edgecolors="black", lw=0.7,
            label=f"SFT Subject Gain $\\overline{{\\Delta L}}_{{\\mathrm{{SFT}},m}}$ ($n=15$ subjects)")
ax2.scatter(a_vec, sub_grp["dL_grpo"].values, marker="o", s=70, color=C_GRPO, edgecolors="black", lw=0.7,
            label=f"GRPO Subject Gain $\\overline{{\\Delta L}}_{{\\mathrm{{GRPO}},m}}$ ($n=15$ subjects)")
xs_a = np.linspace(a_vec.min() - 0.05, a_vec.max() + 0.05, 100)
ax2.plot(xs_a, m_sft_a * xs_a + c_sft_a, color=C_SFT, lw=2.2, ls="--",
         label=f"SFT OLS: slope $= {m_sft_a:+.2f}$, $r = {r_sft_a:+.3f}$")
ax2.plot(xs_a, m_grp_a * xs_a + c_grp_a, color=C_GRPO, lw=2.2, ls="-",
         label=f"GRPO OLS: slope $= {m_grp_a:+.2f}$, $r = {r_grp_a:+.3f}$")
ax2.axhline(0, color="#666666", lw=0.9, ls=":")
ax2.set_ylim(-0.68, 2.22)
ax2.set_xlabel("Subject Mean 2PL Discrimination $\\bar{a}_m^{\\mathrm{pop}}$ ($n=15$ MMLU Subjects)", fontsize=10.5)
ax2.set_ylabel("Subject Mean Log-Odds Gain $\\overline{\\Delta L}_{s,m}$ ($K=32$)", fontsize=10.5)
ax2.set_title("B. Subject Log-Odds Gain vs. Subject Mean $\\bar{a}_m^{\\mathrm{pop}}$ ($n=15$ Subjects)", fontsize=11.4, fontweight="bold", pad=10)
ax2.text(0.03, 0.96,
         f"• 2PL score equation weights items by $a_j^{{\\mathrm{{pop}}}}$ (Marketing $1.45$ vs HS Math $0.28$, $5.2\\times$).\n"
         f"• Item-level Spearman of IRT residual $c_{{s,j}}$ (after $a_j \\Delta\\theta_s$ removed) with $a_j^{{\\mathrm{{pop}}}}$:\n"
         f"  SFT $\\rho(c_{{\\mathrm{{SFT}}}}, a^{{\\mathrm{{pop}}}})$ = {min(rho_c_sft):+.3f} to {max(rho_c_sft):+.3f}; "
         f"GRPO $\\rho(c_{{\\mathrm{{GRPO}}}}, a^{{\\mathrm{{pop}}}})$ = {min(rho_c_grpo):+.3f} to {max(rho_c_grpo):+.3f}.",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.0,
         bbox=dict(boxstyle="round,pad=0.32", facecolor="#fffdf0", edgecolor="#d4a017", alpha=0.94))
ax2.legend(loc="lower right", frameon=True, fontsize=8.1)

fig.suptitle("CAN #2 (Descriptive on Seen S1 States): How $a_j^{\\mathrm{pop}}$ Weighting Shifts SFT vs. GRPO Stage Attribution",
             fontsize=12.6, fontweight="bold", y=1.02)
save_fig(fig, "F5_attribution_reversal")

# ==============================================================================
# FIGURE 6: Single-Draw vs 32-Trial Person Fit l_z & Short-Test CAT vs Random RMSE
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6), gridspec_kw={"wspace": 0.25})

q2a_pf_df = pd.read_csv(os.path.join(IPL, "Q2a", "personfit.csv")).set_index("state")
q2g_q1_pf_df = pd.read_csv(os.path.join(IPL, "Q2g", "q1", "personfit.csv"))
q2g_q1_all = q2g_q1_pf_df[q2g_q1_pf_df["variant"] == "all"].set_index("state")

fams = ["Base\n(t9_base)", "CoT SFT\n(t9_sft)", "S1 GRPO\n(666u)", "Q1 GRPO\n(333u)", "Q1 CTT\n(333u)", "Q1 Pop1\n(a_j-wt)", "Q1 Pop0\n(unwt)"]
lz1_vals = [
    float(q2a_pf_df.loc["t9_base", "lz"]),
    float(q2a_pf_df.loc[sft_states, "lz"].astype(float).mean()),
    float(q2a_pf_df.loc[grpo_states, "lz"].astype(float).mean()),
    None,
    None,
    None,
    None,
]
lz32_vals = [
    float(pf_all.loc["t9_base", "lz"]),
    float(pf_all.loc[sft_states, "lz"].astype(float).mean()),
    float(pf_all.loc[grpo_states, "lz"].astype(float).mean()),
    float(q2g_q1_all.loc[[f"q1_grpo_{i}" for i in [0,1,2]], "lz"].astype(float).mean()),
    float(q2g_q1_all.loc[[f"q1_ctt_{i}" for i in [0,1,2]], "lz"].astype(float).mean()),
    float(q2g_q1_all.loc[[f"q1_pop1_{i}" for i in [0,1,2]], "lz"].astype(float).mean()),
    float(q2g_q1_all.loc[[f"q1_pop0_{i}" for i in [0,1,2]], "lz"].astype(float).mean()),
]

xf = np.arange(len(fams))
cols_f = [C_BASE, C_SFT, C_GRPO, "#ff7f0e", C_CTT, C_POP1, C_POP0]
bars_lz = ax1.bar(xf, lz32_vals, width=0.55, color=cols_f, edgecolor="black", lw=0.7, label="Q2g 32-Trial $l_z^{(32)}$ (bars, left axis)")
for idx_b, (b_item, v32, v1) in enumerate(zip(bars_lz, lz32_vals, lz1_vals)):
    lbl_txt = f"K=32: {v32:.0f}\n(k=1: {v1:+.2f})" if v1 is not None else f"K=32:\n{v32:.0f}"
    y_off = -10 if idx_b != 1 else -48
    ax1.text(b_item.get_x() + b_item.get_width()/2, v32 + y_off, lbl_txt,
             ha="center", va="top", fontsize=7.3, fontweight="bold")

ax1.set_xticks(xf)
ax1.set_xticklabels(fams, fontsize=8.2)
ax1.set_ylim(-665, 225)
ax1.axhline(0, color="#333333", lw=1.0)
ax1.set_ylabel("Q2g 32-Trial Person Fit $l_z^{(32)}$ (with Q2a $l_z^{(1)}$ annotated)", fontsize=10.2)
ax1.set_title("A. Q2a Single-Draw $l_z^{(1)}$ vs. Q2g 32-Trial Binomial $l_z^{(32)}$", fontsize=11.5, fontweight="bold", pad=10)
ax1.text(0.02, 0.97,
         f"• Two different estimators: Q2a $l_z^{{(1)}}$ (1 order-0 draw at EAP $\\hat{{\\theta}}$) vs. Q2g $l_z^{{(32)}}$ (32 trials at Q2g $\\hat{{\\theta}}$).\n"
         f"• Base ($l_z^{{(1)}}={lz1_vals[0]:+.2f}, l_z^{{(32)}}={lz32_vals[0]:.0f}$) and SFT ($l_z^{{(1)}}={lz1_vals[1]:+.2f}$ mean, $l_z^{{(32)}}={lz32_vals[1]:.0f}$) are negative on BOTH.\n"
         "• S1 GRPO has $l_z^{(1)} =$ +2.03 to +2.52 (mean $+2.24$) at $k=1$, vs $l_z^{(32)} = -482$\n"
         "  (S1) to $-577$ (Q1 pop0) at $K=32$; `report-q1q2.md:46` reads all trained models as misfit.",
         transform=ax1.transAxes, va="top", ha="left", fontsize=7.9,
         bbox=dict(boxstyle="round,pad=0.33", facecolor="#fff5f5", edgecolor=C_GRPO, alpha=0.95))

# Panel B: Q2c Adaptive CAT vs Classical Random RMSE as n: 25 -> 50 -> 100
q2c_err = pd.read_csv(os.path.join(IPL, "Q2c", "errors.csv"))
q2c_high = q2c_err[(q2c_err["level"] == "band") & (q2c_err["band"] == "above 0.45")].sort_values("n")

ns = [25, 50, 100]
rmse_cl = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "classical")]["rmse"].iloc[0]) for n in ns]
rmse_irtr = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_random")]["rmse"].iloc[0]) for n in ns]
rmse_irt = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_adaptive")]["rmse"].iloc[0]) for n in ns]
bias_irt = [float(q2c_high[(q2c_high["n"] == n) & (q2c_high["method"] == "irt_adaptive")]["bias"].iloc[0]) for n in ns]

ax2.plot(ns, rmse_cl, marker="o", ms=9, lw=2.6, color=C_SFT, label="Classical Random Sampling (`classical`)")
ax2.plot(ns, rmse_irtr, marker="^", ms=8, lw=2.0, ls="--", color=C_POP1, label="2PL Random Sampling (`irt_random`)")
ax2.plot(ns, rmse_irt, marker="s", ms=9, lw=2.6, color=C_GRPO, label="Max-Info 2PL Adaptive CAT (`irt_adaptive`)")

# Place labels with zero horizontal overlap
ax2.text(25, rmse_cl[0] + 0.003, f"{rmse_cl[0]:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=9)
ax2.text(25, rmse_irt[0] + 0.004, f"{rmse_irt[0]:.3f}\n(bias {bias_irt[0]:+.3f})", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=8.0)

ax2.text(50, rmse_cl[1] + 0.004, f"classical: {rmse_cl[1]:.3f}", ha="center", va="bottom", color=C_SFT, fontweight="bold", fontsize=8.5)
ax2.text(58, rmse_irt[1] + 0.003, f"adaptive: {rmse_irt[1]:.3f}\n(bias {bias_irt[1]:+.3f})", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=8.0)
ax2.text(50, rmse_irtr[1] - 0.004, f"irt_random: {rmse_irtr[1]:.3f}", ha="center", va="top", color=C_POP1, fontweight="bold", fontsize=8.2)

ax2.text(100, rmse_cl[2] - 0.004, f"{rmse_cl[2]:.3f}", ha="center", va="top", color=C_SFT, fontweight="bold", fontsize=9)
ax2.text(98, rmse_irt[2] + 0.004, f"{rmse_irt[2]:.3f}\n(bias {bias_irt[2]:+.3f})", ha="center", va="bottom", color=C_GRPO, fontweight="bold", fontsize=8.0)

ax2.set_xticks(ns)
ax2.set_xticklabels(["n = 25 Items", "n = 50 Items", "n = 100 Items"], fontsize=10)
ax2.set_xlim(16, 110)
ax2.set_ylim(0.018, 0.118)
ax2.set_xlabel("Short-Test Item Budget $n$ (Unit Q2c, Checkpoints $>0.45$ Band)", fontsize=10.5)
ax2.set_ylabel("Level Estimation RMSE against Full Bank", fontsize=10.5)
ax2.set_title("B. CANNOT #3: Q2c & Q2a Short-Test Adaptive CAT vs. Classical Random", fontsize=11.5, fontweight="bold", pad=10)
ax2.text(0.02, 0.25,
         "• In Q2c ($>0.45$ band), `irt_adaptive` beats `classical` at $n=25, 50$ ($0.064$ vs $0.098, 0.071$),\n"
         "  but `classical` wins at $n=100$ ($0.050$ vs $0.063$; Q2c adaptive bias $+0.042$ to $+0.058$).\n"
         "• At $k=1$ on S1 states (`K.md:47-51, 106`), CAT picks 40%–49% Sociology (selection $+0.049$ to $+0.095$).\n"
         "• In Q2a ($k=16$, S1/Q1), `classical` $\\leq$ `irt_adaptive` in 19/21 cells (mean ratio 1.40x–1.65x).",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.0,
         bbox=dict(boxstyle="round,pad=0.33", facecolor="#fffdf0", edgecolor="#d4a017", alpha=0.95))
ax2.legend(loc="upper right", frameon=True, fontsize=8.4)

fig.suptitle("B Helps A & CANNOT #3: Person-Fit Estimator Comparison ($l_z$) and Short-Test CAT Error (`Q2a/Q2c`)",
             fontsize=12.8, fontweight="bold", y=1.02)
save_fig(fig, "F6_lz_signflip_and_cat_bias")

# ==============================================================================
# FIGURE 7: CANNOT #4 — Q1 Seed-0 Trajectory, 12-Run Band & GRPO Zero-Variance Share
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), gridspec_kw={"wspace": 0.28})

traj = pd.read_csv(os.path.join(IPL, "Q1", "trajectory.csv"))
arm_styles = [
    ("grpo", "Q1 GRPO (G=5)", C_GRPO, "o", "-"),
    ("ctt",  "Q1 CTT Pass-Band Critic (G=1)", C_CTT, "s", "--"),
    ("pop0", "Q1 Fixed-θ 2PL Critic pop0 (G=1)", C_POP0, "^", "-."),
    ("pop1", "Q1 $a_j^{\\mathrm{pop}}$-Weighted 2PL Critic pop1 (G=1)", C_POP1, "D", ":"),
]
shares_u83 = []
arm_means_333 = []
for mode, lbl, col, mk, ls in arm_styles:
    sub = traj[(traj["mode"] == mode) & (traj["seed"] == 0)].sort_values("updates")
    s_map = sub.set_index("updates")["mean_at_16"]
    sh83 = (s_map[83] - s_map[0]) / (s_map[333] - s_map[0]) * 100
    shares_u83.append(sh83)
    ax1.plot(sub["updates"], sub["mean_at_16"], marker=mk, ms=7.5, lw=2.3, ls=ls, color=col,
             label=f"{lbl} (seed 0; {sh83:.1f}% by u83)")
    sts = [f"q1_{mode}_{i}" for i in [0, 1, 2]]
    final_3seed = float(q1_pool.loc[sts, "mean_at_16"].mean())
    arm_means_333.append(final_3seed)
    ax1.scatter([333], [final_3seed], s=115, marker=mk, color=col, edgecolors="black", lw=1.2, zorder=6)

ax1.axhline(base_vals[0], color=C_BASE, lw=1.4, ls="--", label=f"Base Mean@16 = {base_vals[0]:.3f} (Maj@16 = {base_vals[1]:.3f})")
ax1.axhline(sft_vals[1], color=C_SFT, lw=1.8, ls="-.", label=f"S1 SFT Maj@16 = {sft_vals[1]:.3f} ({sft_vals[2]:.3f}* tie-correct)")
ax1.axvspan(83, 333, color="#eeeeee", alpha=0.55)
ax1.axvline(83, color="#333333", lw=1.2, ls=":")
ax1.text(88, 0.432,
         f"• Seed 0 (only seed with mid-run saves): reaches\n"
         f"  {min(shares_u83):.1f}%–{max(shares_u83):.1f}% of its final gain by update 83.\n"
         f"• Final 3-seed arm means at u333: {min(arm_means_333):.4f}–{max(arm_means_333):.4f}\n"
         f"  (0.63 pp band; 12 Q1 runs 0.459–0.480; 3 S1 runs 0.473–0.486).",
         fontsize=7.9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#555555", alpha=0.95))

ax1.set_xticks([0, 83, 166, 250, 333])
ax1.set_ylim(0.365, 0.522)
ax1.set_xlabel("RL Training Updates (Unit Q1, 333 Updates; Seed 0 Trajectory + 3-Seed Final Means)", fontsize=10.0)
ax1.set_ylabel("Full-Bank Single-Draw Accuracy Mean@16 ($J=1,500$)", fontsize=10.5)
ax1.set_title("A. Q1 Seed-0 Trajectory (4 Runs) & 12-Run Final Band", fontsize=11.2, fontweight="bold", pad=10)
ax1.legend(loc="lower right", frameon=True, fontsize=7.4)

# Panel B: Zero-Advantage Share Across Updates in Q1 GRPO (3 seeds) + W1 raw wasted share
with open(os.path.join(IPL, "W1", "summary.json"), "r", encoding="utf-8") as f:
    w1_sum = json.load(f)
w1_raw_zero = float(w1_sum["wasted_share_g5_raw"]) * 100.0
w1_shrunk_zero = float(w1_sum["wasted_share_g5_shrunk"]) * 100.0
l0_tokens = float(w1_sum["mean_tokens_mean"])

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
    ax2.plot(df_tr["update"], df_tr["z_smooth"] * 100, lw=1.5, alpha=0.55, color=C_GRPO,
             label="GRPO Seed Traces (15-update moving avg)" if s == 0 else None)

mean_z = np.mean([t["z_smooth"].values for t in traces], axis=0) * 100
post83_grand = float(np.mean(post83_means))
ax2.plot(traces[0]["update"], mean_z, lw=2.8, color="#8b0000", label="GRPO 3-Seed Mean `zero_adv_share` (%)")
ax2.axhline(w1_raw_zero, color=C_BASE, lw=1.4, ls="--",
            label=f"W1 Base $G=5$ Raw Zero-Variance Share ({w1_raw_zero:.1f}%, shrunk {w1_shrunk_zero:.1f}%)")
ax2.axhline(post83_grand, color=C_POP0, lw=1.5, ls=":",
            label=f"GRPO Post-u83 Mean Zero-Adv Share ({post83_grand:.1f}%, max {min(post83_maxs):.0f}%–{max(post83_maxs):.0f}%)")
ax2.axvline(83, color="#333333", lw=1.2, ls=":")

ax2.set_xlim(0, 333)
ax2.set_ylim(20, 96)
ax2.set_xlabel("RL Training Update (`Q1` `grpo` Arm, $G=5$ Answers per Prompt)", fontsize=10.5)
ax2.set_ylabel("Prompts with Zero Reward Variance `zero_adv_share` (%)", fontsize=10.5)
ax2.set_title("B. $G=5$ `grpo` Zero-Advantage Share vs. $G=1$ Critics", fontsize=11.2, fontweight="bold", pad=10)
ax2.text(0.03, 0.95,
         f"• In $G=5$ `grpo` (no $a_j^{{\\mathrm{{pop}}}}$), `zero_adv_share` averages {post83_grand:.1f}% after u83.\n"
         f"• $G=1$ critics (`ctt, pop0, pop1`) use advantage $r - \\hat{{p}}_j \\neq 0$ (`zero_adv_share` empty);\n"
         f"  Rasch 1PL equals `ctt` in expectation (`M.md:97`; `pop0` uses fixed Fluid $a_j^{{\\mathrm{{pop}}}}$).\n"
         f"• Fixed $L_0 = {l0_tokens:.2f}$ norm in $G=1$ critics yields higher concentration\n"
         "  (`stable_12 = 0.866–0.942` vs `0.757` in `grpo`) within the same 0.63 pp accuracy band.",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.0,
         bbox=dict(boxstyle="round,pad=0.33", facecolor="#fff5f5", edgecolor=C_GRPO, alpha=0.94))
ax2.legend(loc="lower right", frameon=True, fontsize=8.0)

fig.suptitle("CANNOT #4 (Unit Q1): All 4 RL Arms Converge to a 0.63 Percentage-Point Band (0.4685–0.4748)",
             fontsize=12.8, fontweight="bold", y=1.02)
save_fig(fig, "F7_q1_wall_and_zero_adv")

# ==============================================================================
# FIGURE 8: CAN #3 / A <-> B — W1 Pass-Rate Band & Pre-Registered Unit M1 Design
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.6), gridspec_kw={"wspace": 0.34})

# Panel A: Base W1 Pass-Rate Distribution p_base,j (10,682 Training Items, k=8)
w1_pr = pd.read_csv(os.path.join(IPL, "W1", "pass_rate.csv"))
counts_k = w1_pr["correct"].value_counts().sort_index()
ks = np.arange(0, 9)
shares_k = [float(counts_k.get(k, 0) / len(w1_pr) * 100) for k in ks]
zpd_mask = [(2 <= k <= 6) for k in ks]
bar_cols = [C_ZPD if m else "#bbbbbb" for m in zpd_mask]

bars_w1 = ax1.bar(ks, shares_k, width=0.62, color=bar_cols, edgecolor="black", lw=0.8)
for b_i, sh in zip(bars_w1, shares_k):
    ax1.text(b_i.get_x() + b_i.get_width()/2, sh + 0.5, f"{sh:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold")

ax1_twin = ax1.twinx()
p_grid = np.linspace(0, 8, 200) / 8.0
fisher_info = p_grid * (1.0 - p_grid)
ax1_twin.plot(np.linspace(0, 8, 200), fisher_info, color=C_GRPO, lw=2.4, ls="--", label="Bernoulli Variance $p_j(1-p_j)$")
ax1_twin.set_ylabel("Bernoulli Outcome Variance $p_j(1-p_j)$", color=C_GRPO, fontsize=10)
ax1_twin.set_ylim(0, 0.33)
ax1_twin.grid(False)

zpd_total = sum(shares_k[2:7])
boundary_total = 100.0 - zpd_total
ax1.set_xticks(ks)
ax1.set_xticklabels([f"{k}/8\n({k/8:.2f})" for k in ks], fontsize=8.6)
ax1.set_ylim(0, 32)
ax1.set_xlabel("Base Training-Prompt Pass Rate $p_{\\mathrm{base},j} = \\mathrm{correct}/8$ (`results/W1/pass_rate.csv`, $N=10,682$)", fontsize=10.0)
ax1.set_ylabel("Share of Training Pool (%)", fontsize=10.5)
ax1.set_title("A. Base W1 Pass Rates $p_{\\mathrm{base},j}$ & Interior Band `2..6 of 8` (`[0.2, 0.8]`)", fontsize=11.4, fontweight="bold", pad=10)
ax1.text(0.02, 0.96,
         f"• Base `W1`: {zpd_total:.1f}% interior (`2..6 of 8`, $p_j \\in [0.2, 0.8]$), {boundary_total:.1f}% exterior (`0..1` or `7..8`).\n"
         f"• Why `M1c` re-samples on SFT: SFT changes item order ($r = 0.289$ in `T9g/s3`,\n"
         f"  0.59–0.61 in $K=32$ draws), so Base `W1` pass rates are not the SFT model's own.\n"
         f"• `M1c` samples $k=8$ answers per prompt from each `t9_sft_0..2_70` checkpoint first.",
         transform=ax1.transAxes, va="top", ha="left", fontsize=8.1,
         bbox=dict(boxstyle="round,pad=0.33", facecolor="#e8f8fc", edgecolor=C_ZPD, alpha=0.95))

# Panel B: Committed Results ONLY (Base, Q1 12 runs, S1 GRPO 3 runs, S1 SFT 3 runs) + Unit M1 Pre-Registered Box
all_q1_sts = [f"q1_{m}_{s}" for m in ["grpo", "ctt", "pop0", "pop1"] for s in [0, 1, 2]]
q1_12_mean16 = float(q1_pool.loc[all_q1_sts, "mean_at_16"].mean())
q1_12_maj16 = float(q1_pool.loc[all_q1_sts, "maj_at_16"].mean())

stages = [
    "1. Base\n(`t9_base`)",
    "2. Q1 Base → RL\n(12 Runs, 333u)",
    "3. S1 Base → GRPO\n(3 Runs, 666u)",
    "4. S1 Base → SFT\n(3 Runs, 70u)",
]
mean16_bars = [base_vals[0], q1_12_mean16, grpo_vals[0], sft_vals[0]]
maj16_bars  = [base_vals[1], q1_12_maj16,  grpo_vals[1], sft_vals[1]]

xs_st = np.arange(len(stages))
w_s = 0.26
b_m16 = ax2.bar(xs_st - w_s, mean16_bars, width=w_s, color=[C_BASE, "#ff7f0e", C_GRPO, C_SFT], edgecolor="black", label="Single-Draw Mean@16")
b_j16 = ax2.bar(xs_st, maj16_bars, width=w_s, color=["#999999", "#ffbb78", "#f08080", "#6baed6"], edgecolor="black", label="Majority Vote Maj@16 (Unstarred)")

# Only S1 states (indices 0, 2, 3) have rewrite1_parsed.csv maj_tie_correct
s1_tc_x = np.array([0, 2, 3]) + w_s
s1_tc_vals = [base_vals[2], grpo_vals[2], sft_vals[2]]
b_tc = ax2.bar(s1_tc_x, s1_tc_vals, width=w_s, color=["#dddddd", "#ffb6c1", "#bdd7e7"], edgecolor="black", label="Majority Vote Maj@16* (S1 Tie-Correct)")

for idx_bar, bar in enumerate(b_m16):
    h = bar.get_height()
    dy_b = 0.004 if idx_bar != 2 else 0.015
    ax2.text(bar.get_x() + bar.get_width()/2, h + dy_b, f"{h:.3f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
for bar in b_j16:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.004, f"{h:.3f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
for bar in b_tc:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.004, f"{h:.3f}*", ha="center", va="bottom", fontsize=7.4, fontweight="bold")

ax2.axhline(grpo_vals[0], color=C_GRPO, lw=1.4, ls="--", label=f"S1 GRPO Mean@16 = {grpo_vals[0]:.3f} (Q1 12-run mean = {q1_12_mean16:.3f})")
ax2.text(0.02, 0.97,
         "Committed S1 & Q1 Baselines (Left) vs. Running Pre-Registered `Unit M1` (`notes/units/M1.md`):\n"
         "• `M1a`: Token temperature $T=0.2$ on `t9_sft_0..2_70` (0 training).\n"
         "• `M1b`: `t9_sft_0..2_70` → GRPO (Q1 exact settings, 333u + save at u83).\n"
         "• `M1c`: `t9_sft_0..2_70` → ZPD-GRPO on SFT's own $p_{\\mathrm{SFT},j} \\in [0.2, 0.8]$.\n"
         "• Pre-registered Rule 1 (`M1b`): `+1` beats both `t9_sft` & `q1_grpo` on Mean@16 (3/3 seeds,\n"
         "  $>2.92\\,\\mathrm{SE}$); `-1` drops Maj@16 below `t9_sft` (3/3 seeds, $<-2.92\\,\\mathrm{SE}$); `0` otherwise.",
         transform=ax2.transAxes, va="top", ha="left", fontsize=7.8,
         bbox=dict(boxstyle="round,pad=0.32", facecolor="#fffdf0", edgecolor="#d4a017", alpha=0.95))

ax2.set_xticks(xs_st)
ax2.set_xticklabels(stages, fontsize=8.8)
ax2.set_ylim(0.29, 0.78)
ax2.set_ylabel("Full-Bank Score ($J=1,500$, Committed Data Only)", fontsize=10.5)
ax2.set_title("B. Committed Baselines & Pre-Registered `Unit M1` Test (Running)", fontsize=11.4, fontweight="bold", pad=10)
ax2.legend(loc="lower left", frameon=True, fontsize=7.6, ncol=2)

fig.suptitle("CAN #3 (A <-> B, Unit M1): Testing Whether SFT-Own Pass Rates $p_{\\mathrm{SFT},j}$ & SFT → RL Harvest the Voting Gap",
             fontsize=12.6, fontweight="bold", y=1.02)
save_fig(fig, "F8_unit_m1_architecture")

print("ALL 8 FIGURES RE-GENERATED CLEANLY WITH ZERO HARDCODED ARRAYS.")
