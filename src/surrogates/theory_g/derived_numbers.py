"""Closed-form numbers of theory/G.md (R1, R4, R9, section 4 reading). No training.

Run: python -I derived_numbers.py [path to results/theory_g_calibration.json]
"""
import json
import math
import sys

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def grpo_shape(p, g=5):
    """E[sqrt(q (1 - q))], q = k / g, k ~ Binomial(g, p): the GRPO expected step per answer over alpha v."""
    k = np.arange(g + 1)
    pk = np.array([math.comb(g, i) for i in k]) * p ** k * (1 - p) ** (g - k)
    q = k / g
    return float((pk * np.sqrt(q * (1 - q))).sum())


def r9_se(k, c, delta, pq=0.2):
    """SE of a_j (and of the logit change over the window, in units of delta) from c evenly spaced checkpoints."""
    t = np.linspace(0.0, delta, c)
    info = k * pq * ((t - t.mean()) ** 2).sum()
    return info, 1.0 / math.sqrt(info)


def k_for_reliability(rel, signal_sd, c, pq=0.2):
    """Answers per item per checkpoint so that the window logit change has reliability rel."""
    se2 = signal_sd ** 2 * (1 - rel) / rel
    t = np.linspace(0.0, 1.0, c)
    return 1.0 / (se2 * pq * ((t - t.mean()) ** 2).sum())


def bank_factor(mu, sd=1.5, n=200_000, seed=0):
    """c'_B = mean_j l_j p_j (1 - p_j) at beta = 1, and bank accuracy, bank logits N(mu, sd^2)."""
    l = np.random.default_rng(seed).normal(mu, sd, n)
    p = sigmoid(l)
    return float((l * p * (1 - p)).mean()), float(p.mean())


def t10_rate(ratio, n=10, crit=1.833113, reps=400_000, seed=1):
    """False positive rate of the one-sided t test over n seeds when a pool shift N(0, (ratio * se)^2) is added."""
    rng = np.random.default_rng(seed)
    shift = ratio / math.sqrt(n) * rng.standard_normal(reps)  # seed SD = 1, SE of the mean = 1/sqrt(n)
    x = rng.standard_normal((reps, n)) + shift[:, None]
    t = x.mean(1) / (x.std(1, ddof=1) / math.sqrt(n))
    return float((t > crit).mean())


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


if __name__ == "__main__":
    out = {}
    out["grpo_shape_rel_p05"] = {str(p): grpo_shape(p) / grpo_shape(0.5) for p in (0.1, 0.2, 0.3)}
    out["pq_rel_p05"] = {str(p): p * (1 - p) / 0.25 for p in (0.1, 0.2, 0.3)}
    out["sqrtpq_rel_p05"] = {str(p): math.sqrt(p * (1 - p)) / 0.5 for p in (0.1, 0.2, 0.3)}
    out["grpo_over_sqrtpq"] = {str(p): grpo_shape(p) / math.sqrt(p * (1 - p)) for p in (0.1, 0.2, 0.5)}
    out["r9"] = {f"k{k}_C{c}_D{d}": r9_se(k, c, d) for k, c, d in ((16, 10, 1.0), (16, 10, 0.5), (64, 20, 1.0))}
    out["logit_diff_se_k16"] = math.sqrt(2 / (16 * 0.2))
    out["k_rel08_sd05_C10"] = k_for_reliability(0.8, 0.5, 10)
    out["k_rel08_sd025_C10"] = k_for_reliability(0.8, 0.25, 10)
    lstar = 1.5434046384182083
    g = 0.25
    out["r4_guess_floor"] = {"zero_at": g + (1 - g) * 0.5, "peak_pos": g + (1 - g) * float(sigmoid(lstar)),
                             "peak_neg": g + (1 - g) * float(sigmoid(-lstar))}
    out["bank_factor"] = {str(mu): bank_factor(mu) for mu in (-0.2, 0.0, 0.2)}
    if len(sys.argv) > 1:
        cal = json.load(open(sys.argv[1]))
        rows = []
        for c in cal["cases"]:
            wse = c["within_pool_se_of_10seed_mean"]
            net = math.sqrt(max(c["pool_effect_sd"] ** 2 - wse ** 2, 0.0))
            seed_sd = math.sqrt((wse * math.sqrt(10)) ** 2 + c["item_paired_se_median"] ** 2)
            seed_se10 = seed_sd / math.sqrt(10)
            row = {"case": c["case"], "run_noise": c["run_noise"], "pool_sd_net": net,
                   "seed_se_of_10seed_read_mean": seed_se10, "ratio": net / seed_se10,
                   "t10_rate_from_ratio": t10_rate(net / seed_se10)}
            if c["true_effect_mean"] > 0.01:  # BP approximation, per-seed and seed-mean parts
                eff, se = c["true_effect_mean"], c["item_paired_se_median"]
                ps = norm_sf((2 * se - eff) / seed_sd)
                row["bp_per_seed_pass"] = ps
                row["bp_two_of_three"] = 3 * ps ** 2 - 2 * ps ** 3
                row["bp_seed_mean_pass"] = norm_sf((2 * se / math.sqrt(3) - eff) / (seed_sd / math.sqrt(3)))
            rows.append(row)
        out["section4"] = rows
    print(json.dumps(out, indent=1))
