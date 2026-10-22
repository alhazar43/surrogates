import os
import csv
import json
import math
import glob

IPL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) if "src" in os.path.abspath(__file__) else r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\item-params-llm"
SIG = r"C:\Users\steph\.gemini\antigravity-cli\brain\d00b65ad-c24f-4025-8dd4-1c0acb407e49\scratch\plms-recent\signature"
IRT4LLM = r"D:\irt4llm"

def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def safe_logit(p, eps=1e-6):
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))

def jeffreys_logit(p, K=32):
    p_tilde = (p * K + 0.5) / (K + 1.0)
    return math.log(p_tilde / (1.0 - p_tilde))

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def sd(xs, ddof=0):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))

def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])

def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den > 0 else 0.0

def rankdata(xs):
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks

def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))

def linreg(xs, ys):
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = (sxy ** 2) / (sxx * syy) if (sxx * syy) > 0 else 0.0
    return slope, intercept, r2

def solve_linear_system(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[max_row] = M[max_row], M[i]
        piv = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= piv
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for j in range(i, n + 1):
                    M[r][j] -= factor * M[i][j]
    return [M[i][n] for i in range(n)]

def ols_multi(X_cols, y):
    k = len(X_cols)
    n = len(y)
    XtX = [[sum(X_cols[i][r] * X_cols[j][r] for r in range(n)) for j in range(k)] for i in range(k)]
    Xty = [sum(X_cols[i][r] * y[r] for r in range(n)) for i in range(k)]
    beta = solve_linear_system(XtX, Xty)
    y_hat = [sum(beta[j] * X_cols[j][r] for j in range(k)) for r in range(n)]
    my = mean(y)
    ss_res = sum((y[r] - y_hat[r]) ** 2 for r in range(n))
    ss_tot = sum((y[r] - my) ** 2 for r in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, y_hat, r2

print("=" * 95)
print("PART 1: AUTOREGRESSIVE PRIOR-SCALING LAW & LOGIT MODALITIES")
print("=" * 95)

w0_rows = read_csv_rows(os.path.join(IPL, "results", "W0", "items_bank.csv"))
t9r_rows = read_csv_rows(os.path.join(IPL, "results", "T9r", "items_wide.csv"))

w0_map = {}
for r in w0_rows:
    if r["a"] != "" and r["b"] != "":
        w0_map[str(r["bank_row_id"])] = r

merged = []
for r in t9r_rows:
    iid = str(r["item_id"])
    if iid in w0_map:
        row = dict(r)
        row.update(w0_map[iid])
        merged.append(row)

ones = [1.0] * len(merged)
a_pop = [float(r["a"]) for r in merged]
b_pop = [float(r["b"]) for r in merged]
ab_pop = [a * b for a, b in zip(a_pop, b_pop)]

print(f"[1.1] W0 + T9r forced-choice 4-order key probabilities (J={len(merged)} matched items):")
print(f"      Corr(a_pop, a_pop*b_pop) = {pearson(a_pop, ab_pop):.4f} (R^2 = {pearson(a_pop, ab_pop)**2*100:.2f}%)")
for eps_v in [1e-6, 1e-4]:
    Lb = [safe_logit(float(r["p_key_base"]), eps_v) for r in merged]
    z_fixed = [a * (-1.209 - b) for a, b in zip(a_pop, b_pop)]
    _, Lb_hat, r2_ols = ols_multi([ones, a_pop, ab_pop], Lb)
    m_zh, sd_zh = mean(Lb_hat), sd(Lb_hat)
    zh_std = [(z - m_zh) / sd_zh for z in Lb_hat]
    sft_r2_dL, sft_tau_ols, sft_tau_orig, sft_gamma = [], [], [], []
    rl_r2_dL, rl_tau_ols, rl_tau_orig, rl_gamma = [], [], [], []
    sft_r2_a, sft_r2_ab, rl_r2_a, rl_r2_ab = [], [], [], []
    for s in ["sft_0_70", "sft_1_70", "sft_2_70"]:
        Ls = [safe_logit(float(r["p_key_base"]) + float(r["dp_key_" + s]), eps_v) for r in merged]
        dL = [ls - lb for ls, lb in zip(Ls, Lb)]
        tau_ols, _, _ = linreg(Lb, Ls)
        tau_orig = sum(ls * lb for ls, lb in zip(Ls, Lb)) / sum(lb * lb for lb in Lb)
        _, _, r2_dL = linreg(Lb, dL)
        _, _, r2_a = linreg(a_pop, dL)
        _, _, r2_ab = ols_multi([ones, a_pop, ab_pop], dL)
        b_full, _, _ = ols_multi([ones, Lb, zh_std], Ls)
        sft_r2_dL.append(round(r2_dL * 100, 2))
        sft_tau_ols.append(round(tau_ols, 4))
        sft_tau_orig.append(round(tau_orig, 4))
        sft_gamma.append(round(b_full[2], 4))
        sft_r2_a.append(round(r2_a * 100, 2))
        sft_r2_ab.append(round(r2_ab * 100, 2))
    for s in ["rl_0_666", "rl_1_666", "rl_2_666"]:
        Ls = [safe_logit(float(r["p_key_base"]) + float(r["dp_key_" + s]), eps_v) for r in merged]
        dL = [ls - lb for ls, lb in zip(Ls, Lb)]
        tau_ols, _, _ = linreg(Lb, Ls)
        tau_orig = sum(ls * lb for ls, lb in zip(Ls, Lb)) / sum(lb * lb for lb in Lb)
        _, _, r2_dL = linreg(Lb, dL)
        _, _, r2_a = linreg(a_pop, dL)
        _, _, r2_ab = ols_multi([ones, a_pop, ab_pop], dL)
        b_full, _, _ = ols_multi([ones, Lb, zh_std], Ls)
        rl_r2_dL.append(round(r2_dL * 100, 2))
        rl_tau_ols.append(round(tau_ols, 4))
        rl_tau_orig.append(round(tau_orig, 4))
        rl_gamma.append(round(b_full[2], 4))
        rl_r2_a.append(round(r2_a * 100, 2))
        rl_r2_ab.append(round(r2_ab * 100, 2))
    print(f"  [eps={eps_v}] R^2(L_base ~ 1 + a + a*b) = {r2_ols*100:.2f}% (2PL formula at th=-1.209: {pearson(Lb, z_fixed)**2*100:.2f}%)")
    print(f"    SFT step 70: tau_OLS={sft_tau_ols} (mean {mean(sft_tau_ols):.4f}), tau_orig={sft_tau_orig} (mean {mean(sft_tau_orig):.4f})")
    print(f"                 R^2(dL~L_b)={sft_r2_dL}%, R^2(dL~a)={sft_r2_a}%, R^2(dL~a,ab)={sft_r2_ab}%, gamma={sft_gamma}")
    print(f"    GRPO st 666: tau_OLS={rl_tau_ols} (mean {mean(rl_tau_ols):.4f}), tau_orig={rl_tau_orig} (mean {mean(rl_tau_orig):.4f})")
    print(f"                 R^2(dL~L_b)={rl_r2_dL}%, R^2(dL~a)={rl_r2_a}%, R^2(dL~a,ab)={rl_r2_ab}%, gamma={rl_gamma}")
    if eps_v == 1e-4:
        print("    [Rule 4 Decomposition: tau_OLS = r(L_s, L_base) * (sigma(L_s) / sigma(L_base))]:")
        for s in ["sft_0_70", "sft_1_70", "sft_2_70", "rl_0_666", "rl_1_666", "rl_2_666"]:
            Ls = [safe_logit(float(r["p_key_base"]) + float(r["dp_key_" + s]), eps_v) for r in merged]
            r_val = pearson(Lb, Ls)
            sig_ratio = sd(Ls, 1) / sd(Lb, 1)
            print(f"      {s:10s}: tau={r_val*sig_ratio:.4f} = r({r_val:.4f}) * (sigma_s/sigma_b)({sig_ratio:.4f}) [sigma_b={sd(Lb,1):.3f}, sigma_s={sd(Ls,1):.3f}]")

# Check how redteam_syn_check.py got 16.17%!
# In redteam_syn_check.py lines 23-39:
# piv_free = s3.pivot_table(index="item_id", columns="model", values="p_key_matched", aggfunc="mean")
# m_s3 = mb.merge(piv_free, left_on="bank_row_id", right_index=True)  (WITHOUT dropna on sft!)
s3_rows = read_csv_rows(os.path.join(IPL, "results", "T9g", "s3", "matched.csv"))
s3_base_free = {}
for r in s3_rows:
    if r["model"] == "base" and r["p_key_matched"] != "":
        s3_base_free.setdefault(r["item_id"], []).append(float(r["p_key_matched"]))
m_s3_base = [r for r in merged if r["item_id"] in s3_base_free]
a_s3b = [float(r["a"]) for r in m_s3_base]
b_s3b = [float(r["b"]) for r in m_s3_base]
ab_s3b = [a * b for a, b in zip(a_s3b, b_s3b)]
Lb_s3b = [safe_logit(mean(s3_base_free[r["item_id"]]), 1e-4) for r in m_s3_base]
_, _, r2_s3b = ols_multi([[1.0]*len(m_s3_base), a_s3b, ab_s3b], Lb_s3b)
r_s3b_fixed = pearson(Lb_s3b, [a*(-1.209 - b) for a, b in zip(a_s3b, b_s3b)])
print(f"\n[1.2] T9g/s3/matched.csv Base free-text (J={len(m_s3_base)} non-NA base items):")
print(f"      OLS R^2(L_base_free ~ 1 + a + a*b) = {r2_s3b*100:.2f}%, R^2(2PL formula at th=-1.209) = {r_s3b_fixed**2*100:.2f}%")

# Check W0_train + W1 pass_rate (J=10,557)
w0_train_rows = read_csv_rows(os.path.join(IPL, "results", "W0", "items_train.csv"))
w1_pr_rows = read_csv_rows(os.path.join(IPL, "results", "W1", "pass_rate.csv"))
w1_map = {str(r["row"]): r for r in w1_pr_rows}
m_train = []
for r in w0_train_rows:
    if r["a"] != "" and r["b"] != "" and str(r["train_row_index"]) in w1_map:
        row = dict(r)
        row.update(w1_map[str(r["train_row_index"])])
        row["L_w1"] = math.log((float(row["correct"]) + 0.5) / (float(row["k"]) - float(row["correct"]) + 0.5))
        m_train.append(row)
a_tr = [float(r["a"]) for r in m_train]
b_tr = [float(r["b"]) for r in m_train]
ab_tr = [a * b for a, b in zip(a_tr, b_tr)]
Lw1_tr = [r["L_w1"] for r in m_train]
_, _, r2_w1 = ols_multi([[1.0]*len(m_train), a_tr, ab_tr], Lw1_tr)
r_w1_2pl = pearson(Lw1_tr, [a*(-1.209 - b) for a, b in zip(a_tr, b_tr)])
print(f"\n[1.3] W0_train + W1 k=8 sampled rollouts (J={len(m_train)}):")
print(f"      R^2(L_base_W1 ~ 1 + a + a*b) = {r2_w1*100:.2f}%, R^2(2PL formula at th=-1.209) = {r_w1_2pl**2*100:.2f}%")

print("\n" + "=" * 95)
print("PART 2: TABLE 2.1, TABLE 2.2, AND ALL 5 UNIFIED COROLLARIES")
print("=" * 95)

s1_stats = read_csv_rows(os.path.join(IPL, "results", "S1", "stats.csv"))
s1_piv = {}
for r in s1_stats:
    if r["order"] == "pooled":
        s1_piv.setdefault(r["state"], {})[r["statistic"]] = float(r["value"])

s1_rw1 = {r["state"]: r for r in read_csv_rows(os.path.join(IPL, "results", "S1", "rewrite1_parsed.csv"))}
q2a_pf = {r["state"]: r for r in read_csv_rows(os.path.join(IPL, "results", "Q2a", "personfit.csv"))}
q2g_s1_pf = {r["state"]: r for r in read_csv_rows(os.path.join(IPL, "results", "Q2g", "s1", "personfit.csv")) if r["variant"] == "all"}
q2g_q1_pf = {r["state"]: r for r in read_csv_rows(os.path.join(IPL, "results", "Q2g", "q1", "personfit.csv")) if r["variant"] == "all"}

print("\n[2.1] Exact S1 State-by-State & 3-Seed Means:")
for st in ["t9_base", "t9_sft_0_70", "t9_sft_1_70", "t9_sft_2_70", "t9_rl_0_666", "t9_rl_1_666", "t9_rl_2_666"]:
    d = s1_piv[st]
    rw = s1_rw1[st]
    pf1 = q2a_pf[st]
    pf32 = q2g_s1_pf[st]
    print(f"  {st:13s} | stable12={d['stable_12']:.4f} | H={d['entropy']:.4f} | mean16={d['mean_at_16']:.4f} (parsed={float(rw['mean_parsed']):.4f})"
          f" | maj16={d['maj_at_16']:.4f} (tie_corr={float(rw['maj_tie_correct']):.4f}) | pass16={d['pass_at_16']:.4f}"
          f" | lz(1)={float(pf1['lz']):+6.2f} | lz(32)={float(pf32['lz']):+7.2f} | MSSR={float(pf32['mssr']):7.2f} | SSR_med={float(pf32['ssr_median']):5.2f}")

for fam_name, st_list in [("S1 SFT (3-seed mean)", ["t9_sft_0_70", "t9_sft_1_70", "t9_sft_2_70"]),
                          ("S1 GRPO (3-seed mean)", ["t9_rl_0_666", "t9_rl_1_666", "t9_rl_2_666"])]:
    print(f"  {fam_name:21s} | stable12={mean([s1_piv[s]['stable_12'] for s in st_list]):.4f}"
          f" | H={mean([s1_piv[s]['entropy'] for s in st_list]):.4f}"
          f" | mean16={mean([s1_piv[s]['mean_at_16'] for s in st_list]):.4f} (parsed={mean([float(s1_rw1[s]['mean_parsed']) for s in st_list]):.4f})"
          f" | maj16={mean([s1_piv[s]['maj_at_16'] for s in st_list]):.4f} (tie_corr={mean([float(s1_rw1[s]['maj_tie_correct']) for s in st_list]):.4f})"
          f" | pass16={mean([s1_piv[s]['pass_at_16'] for s in st_list]):.4f}"
          f" | lz(1)={mean([float(q2a_pf[s]['lz']) for s in st_list]):+6.2f}"
          f" | lz(32)={mean([float(q2g_s1_pf[s]['lz']) for s in st_list]):+7.2f}"
          f" | MSSR={mean([float(q2g_s1_pf[s]['mssr']) for s in st_list]):7.2f}"
          f" | SSR_med={mean([float(q2g_s1_pf[s]['ssr_median']) for s in st_list]):5.2f}")

q1_stats = read_csv_rows(os.path.join(IPL, "results", "Q1", "stats.csv"))
q1_piv = {}
for r in q1_stats:
    if r["order"] == "pooled":
        q1_piv.setdefault(r["state"], {})[r["statistic"]] = float(r["value"])

print("\n[2.2] Q1 Final Arms (3-seed means):")
for arm in ["grpo", "ctt", "pop1", "pop0"]:
    sts = [f"q1_{arm}_{i}" for i in [0, 1, 2]]
    print(f"  Q1 {arm:5s} | stable12={mean([q1_piv[s]['stable_12'] for s in sts]):.4f}"
          f" | H={mean([q1_piv[s]['entropy'] for s in sts]):.4f}"
          f" | mean16={mean([q1_piv[s]['mean_at_16'] for s in sts]):.4f}"
          f" | maj16={mean([q1_piv[s]['maj_at_16'] for s in sts]):.4f}"
          f" | pass16={mean([q1_piv[s]['pass_at_16'] for s in sts]):.4f}"
          f" | lz(32)={mean([float(q2g_q1_pf[s]['lz']) for s in sts]):+7.2f}"
          f" | MSSR={mean([float(q2g_q1_pf[s]['mssr']) for s in sts]):7.2f}"
          f" | SSR_med={mean([float(q2g_q1_pf[s]['ssr_median']) for s in sts]):5.2f}")

print("\n[2.3] Q2c/errors.csv (band level):")
q2c_err = read_csv_rows(os.path.join(IPL, "results", "Q2c", "errors.csv"))
for r in q2c_err:
    if r["level"] == "band":
        print(f"  band={r['band']:11s} | n={r['n']:3s} | method={r['method']:12s} | rmse={float(r['rmse']):.4f} | bias={float(r['bias']):+.4f} | sd={float(r['sd']):.4f}")

print("\n[2.4] Q2g/s1/flags.csv (A3_spearman_c_a):")
for r in read_csv_rows(os.path.join(IPL, "results", "Q2g", "s1", "flags.csv")):
    if r["variant"] == "all" and "spearman" in r["reading"]:
        print(f"  {r['reading']:20s} | {r['state']:14s} | value={float(r['value']):+.4f}")

q2g_s1_prec = read_csv_rows(os.path.join(IPL, "results", "Q2g", "s1", "precision.csv"))
print("\n[2.5] Q2g/s1/precision.csv cols:", list(q2g_s1_prec[0].keys()))
for r in q2g_s1_prec:
    if r["variant"] in ["all", "parsed"]:
        print(f"  S1 ({r['variant']:6s}) | {r['state1']:12s} vs {r['state2']:12s} | z_acc={float(r['abs_z_acc']):6.2f} | z_theta={float(r['abs_z_theta']):6.2f} | sharper={r['theta_sharper']}")

q2g_q1_prec = read_csv_rows(os.path.join(IPL, "results", "Q2g", "q1", "precision.csv"))
print("\n[2.6] Q2g/q1/precision.csv cols:", list(q2g_q1_prec[0].keys()))
for var in ["all", "parsed"]:
    base_finals = [r for r in q2g_q1_prec if r["variant"] == var and r["state1"] == "t9_base" and "_u" not in r["state2"]]
    base_all24 = [r for r in q2g_q1_prec if r["variant"] == var and r["state1"] == "t9_base"]
    n_win_12 = sum(1 for r in base_finals if r["theta_sharper"] == "True")
    n_win_24 = sum(1 for r in base_all24 if r["theta_sharper"] == "True")
    print(f"\n  Q2g/q1/precision.csv (variant={var}):")
    print(f"    t9_base vs 12 Finals: theta_sharper in {n_win_12}/12 | z_acc=[{min(float(r['abs_z_acc']) for r in base_finals):.2f}, {max(float(r['abs_z_acc']) for r in base_finals):.2f}] | z_theta=[{min(float(r['abs_z_theta']) for r in base_finals):.2f}, {max(float(r['abs_z_theta']) for r in base_finals):.2f}]")
    print(f"    t9_base vs All 24 RL: theta_sharper in {n_win_24}/24 | z_acc=[{min(float(r['abs_z_acc']) for r in base_all24):.2f}, {max(float(r['abs_z_acc']) for r in base_all24):.2f}] | z_theta=[{min(float(r['abs_z_theta']) for r in base_all24):.2f}, {max(float(r['abs_z_theta']) for r in base_all24):.2f}]")
    prereg = [r for r in q2g_q1_prec if r["variant"] == var and r.get("group", "") == "preregistered"]
    if not prereg:
        prereg = [r for r in q2g_q1_prec if r["variant"] == var and r["state1"] != "t9_base" and "_u" not in r["state1"] and "_u" not in r["state2"] and r["state1"].split("_")[-1] == r["state2"].split("_")[-1]]
    print(f"    Same-seed cross-arm pairs ({len(prereg)}): theta_sharper in {sum(1 for r in prereg if r['theta_sharper']=='True')}/{len(prereg)}")
    for r in prereg:
        print(f"      {r['group']:15s} | {r['state1']:10s} vs {r['state2']:10s}: z_acc={float(r['abs_z_acc']):5.2f}, z_theta={float(r['abs_z_theta']):5.2f}, sharper={r['theta_sharper']}")

print("\n[2.7] Q2g/s1 & Q2g/q1 reliability.json (B3):")
for tag, p in [("S1", os.path.join(IPL, "results", "Q2g", "s1", "reliability.json")),
               ("Q1", os.path.join(IPL, "results", "Q2g", "q1", "reliability.json"))]:
    with open(p, "r", encoding="utf-8") as f:
        rel_obj = json.load(f)
    print(f"  {tag} reliability.json:", json.dumps(rel_obj, indent=2)[:800])

print("\n" + "=" * 95)
print("PART 3: CROSS-REPO CLAIMS (E1, signature, irt4llm)")
print("=" * 95)
e1_diff = read_csv_rows(os.path.join(IPL, "results", "E1", "difficulty.csv"))
valid_diff = [r for r in e1_diff if r["b_students"] not in ["", "NA"] and r["b_llms"] not in ["", "NA"]]
b_stu = [float(r["b_students"]) for r in valid_diff]
b_llm = [float(r["b_llms"]) for r in valid_diff]
print(f"[3.1] E1/difficulty.csv (J={len(valid_diff)} items): Pearson r(b_stu, b_llm) = {pearson(b_stu, b_llm):.4f}, Spearman rho = {spearman(b_stu, b_llm):.4f}")

with open(os.path.join(IPL, "results", "E1", "fits.json"), "r", encoding="utf-8") as f:
    e1_fits = json.load(f)
print("[3.2] E1/fits.json keys:", list(e1_fits.keys()))
with open(os.path.join(IPL, "results", "E1", "synthetic.json"), "r", encoding="utf-8") as f:
    e1_syn = json.load(f)
print("[3.3] E1/synthetic.json keys:", list(e1_syn.keys()))
print("      E1 synthetic summary:", json.dumps(e1_syn, indent=2)[:800])

print("\n" + "=" * 95)
print("PART 4: NEURO-COGNITIVE DOUBLE DISSOCIATION ACROSS 15 MMLU DOMAINS (W0 + S1 + Q1)")
print("=" * 95)
sub_a, sub_b = {}, {}
for r in w0_rows:
    if r["a"] != "" and r["b"] != "":
        sub_a.setdefault(r["subject"], []).append(float(r["a"]))
        sub_b.setdefault(r["subject"], []).append(float(r["b"]))
mean_a_sub = {k: mean(v) for k, v in sub_a.items()}
mean_b_sub = {k: mean(v) for k, v in sub_b.items()}

pf_s1_map = {r["state"]: float(r["theta"]) for r in read_csv_rows(os.path.join(IPL, "results", "Q2g", "s1", "personfit.csv")) if r["variant"] == "all"}
pf_q1_map = {r["state"]: float(r["theta"]) for r in read_csv_rows(os.path.join(IPL, "results", "Q2g", "q1", "personfit.csv")) if r["variant"] == "all"}
th_base = pf_s1_map["t9_base"]
dth_sft = mean([pf_s1_map[f"t9_sft_{s}_70"] - th_base for s in [0, 1, 2]])
dth_grpo = mean([pf_s1_map[f"t9_rl_{s}_666"] - th_base for s in [0, 1, 2]])

fl_s1_rows = read_csv_rows(os.path.join(IPL, "results", "Q2g", "s1", "flags.csv"))
fl_q1_rows = read_csv_rows(os.path.join(IPL, "results", "Q2g", "q1", "flags.csv"))
c_sft_sub = {r["subject"]: float(r["value"]) for r in fl_s1_rows if r["variant"] == "all" and r["reading"] == "A3_subject_mean_c" and r["method"] == "sft" and r["state"] == "pooled_states"}
c_grpo_sub = {r["subject"]: float(r["value"]) for r in fl_s1_rows if r["variant"] == "all" and r["reading"] == "A3_subject_mean_c" and r["method"] == "grpo" and r["state"] == "pooled_states"}

subs_sorted = sorted(mean_a_sub.keys(), key=lambda k: mean_a_sub[k])
A_vec = [mean_a_sub[s] for s in subs_sorted]
B_vec = [mean_b_sub[s] for s in subs_sorted]
dL_sft_vec = [c_sft_sub[s] + mean_a_sub[s] * dth_sft for s in subs_sorted]
dL_grpo_vec = [c_grpo_sub[s] + mean_a_sub[s] * dth_grpo for s in subs_sorted]
diff_vec = [x - y for x, y in zip(dL_sft_vec, dL_grpo_vec)]

print(f"[4.1] 15-Subject Correlations (S1 K=32 Sampled Rollouts, J=1,472):")
print(f"      r(a_pop, b_pop)             = {pearson(A_vec, B_vec):+.4f}")
print(f"      r(a_pop, dL_SFT)            = {pearson(A_vec, dL_sft_vec):+.4f}")
print(f"      r(a_pop, dL_GRPO)           = {pearson(A_vec, dL_grpo_vec):+.4f}")
print(f"      r(a_pop, dL_SFT - dL_GRPO)  = {pearson(A_vec, diff_vec):+.4f} (R^2 = {pearson(A_vec, diff_vec)**2*100:.2f}%)")
print(f"      r(b_pop, dL_SFT - dL_GRPO)  = {pearson(B_vec, diff_vec):+.4f} (R^2 = {pearson(B_vec, diff_vec)**2*100:.2f}%)")

for m_arm in ["grpo", "ctt", "pop0", "pop1"]:
    dth_m = mean([pf_q1_map[f"q1_{m_arm}_{s}"] - th_base for s in [0, 1, 2]])
    c_m = {r["subject"]: float(r["value"]) for r in fl_q1_rows if r["variant"] == "all" and r["reading"] == "A3_subject_mean_c" and r["method"] == m_arm and r["state"] == "pooled_states"}
    dL_m = [c_m[s] + mean_a_sub[s] * dth_m for s in subs_sorted]
    print(f"      Q1 {m_arm:5s}: r(a_pop, dL) = {pearson(A_vec, dL_m):+.4f} | r(b_pop, dL) = {pearson(B_vec, dL_m):+.4f}")

print("\n[4.2] 15-Subject Double Dissociation Table (Sorted by a_pop):")
for s, a, b, dls, dlg, df in zip(subs_sorted, A_vec, B_vec, dL_sft_vec, dL_grpo_vec, diff_vec):
    winner = "SFT" if df > 0.1 else ("GRPO" if df < -0.1 else "TIE")
    print(f"  {s:30s} | a_pop={a:4.2f} | b_pop={b:+5.2f} | dL_SFT={dls:+6.3f} | dL_GRPO={dlg:+6.3f} | SFT-GRPO={df:+6.3f} ({winner})")

