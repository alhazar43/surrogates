"""Calibration of the tier 1 reading of surrogate G on a learner whose truth is known.

The learner is a 2PL logistic learner with one ability theta. Item x answers right with
p_x = sigmoid(alpha_x (theta - b_x)). Training draws m items per step, one answer each,
and moves theta by eta * mean_x alpha_x (r - phat_x)(r - p_x), the score-function
(REINFORCE) step with a baseline phat_x from 8 base answers. Its expectation is
alpha_x p_x (1 - p_x), the gradient of p_x.

Two arms share one pass-rate band (base phat in [0.2, 0.8]) and the same cost.
  B: items uniform within the band.
  I: items within the band with probability proportional to a_pop^2, where a_pop is the
     discrimination a population fit would report.
Cases.
  N1: Rasch learner (alpha = 1), a_pop random. True effect zero by construction.
  N2: 2PL learner, a_pop independent of alpha. True effect zero by construction.
  P1, P2: 2PL learner, a_pop = alpha * lognormal noise. Planted effect, size measured
          from exact bank accuracies over all runs.
  A run_noise > 0 adds a per-step Gaussian move of theta, a stand-in for the training noise
  of a many-parameter learner that a one-ability learner lacks.
Readings.
  BP: the peer blueprint's rule (I beats B by 2 paired SE over bank items in at least
      2 of 3 seeds and on the seed mean), bank read with 16 answers per item.
  T3, T10: one-sided t test over seed-level differences, 3 or 10 seeds, alpha 0.05.
Also solves the optimal pass rate when training changes precision, not location.

Run: python -I calibrate_tier1.py   (tiny synthetic records, about 4 min on the PC)
The column pool_effect_sd is the SD over experiments of the 10-seed mean of the exact difference; it
includes seed noise. derived_numbers.py removes it (theory/G.md section 4).
"""
import json
import sys

import numpy as np

N_POOL, N_BANK, M, T, ETA, K0, K_BANK = 2000, 300, 32, 333, 0.015, 8, 16
N_EXP, N_SEED = 400, 10
T_CRIT = {3: 2.919986, 10: 1.833113}  # one-sided 0.05, df = n - 1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_items(rng, n_exp, n, case):
    b = rng.normal(0.0, 1.5, (n_exp, n))
    if case == "N1":
        alpha = np.ones((n_exp, n))
    else:
        alpha = np.exp(rng.normal(0.0, {"N2": 0.5, "P1": 0.3, "P2": 0.5}[case], (n_exp, n)))
    if case in ("N1", "N2"):
        a_pop = np.exp(rng.normal(0.0, 0.5, (n_exp, n)))
    else:
        a_pop = alpha * np.exp(rng.normal(0.0, 0.3, (n_exp, n)))
    return b, alpha, a_pop


def sampler(weights):
    """Return f(rng, size) drawing item indices per experiment, with replacement."""
    n_exp, n = weights.shape
    cdf = np.cumsum(weights, axis=1)
    cdf = cdf / cdf[:, -1:]
    flat = (cdf + np.arange(n_exp)[:, None]).ravel()
    flat[np.arange(1, n_exp + 1) * n - 1] = np.arange(1, n_exp + 1)  # exact upper ends

    def draw(rng, shape_tail):
        u = rng.random((n_exp,) + shape_tail) + np.arange(n_exp).reshape((n_exp,) + (1,) * len(shape_tail))
        idx = np.searchsorted(flat, u, side="right")
        return np.minimum(idx, flat.size - 1) - (np.arange(n_exp) * n).reshape((n_exp,) + (1,) * len(shape_tail))

    return draw


def run_case(case, seed, run_noise=0.0):
    rng = np.random.default_rng(seed)
    b, alpha, a_pop = make_items(rng, N_EXP, N_POOL, case)
    bb, balpha, _ = make_items(rng, N_EXP, N_BANK, case)
    p0 = sigmoid(alpha * (0.0 - b))
    p0hat = rng.binomial(K0, p0) / K0
    band = (p0hat >= 0.2) & (p0hat <= 0.8)
    phat = np.clip(p0hat, 0.05, 0.95)
    draws = {"B": sampler(band * 1.0), "I": sampler(band * a_pop ** 2)}
    theta = {arm: np.zeros((N_EXP, N_SEED)) for arm in draws}
    ex = np.arange(N_EXP)[:, None, None]
    for _ in range(T):
        for arm, draw in draws.items():
            idx = draw(rng, (N_SEED, M))
            al, bi, ph = alpha[ex, idx], b[ex, idx], phat[ex, idx]
            p = sigmoid(al * (theta[arm][:, :, None] - bi))
            r = (rng.random(p.shape) < p) * 1.0
            theta[arm] += ETA * (np.mean(al * (r - ph) * (r - p), axis=2)
                                + run_noise * rng.standard_normal((N_EXP, N_SEED)))
    # exact bank accuracy and the sampled bank read
    acc, read = {}, {}
    for arm in draws:
        pb = sigmoid(balpha[:, None, :] * (theta[arm][:, :, None] - bb[:, None, :]))
        acc[arm] = pb.mean(axis=2)
        read[arm] = rng.binomial(K_BANK, pb) / K_BANK  # (exp, seed, item)
    true_diff = acc["I"] - acc["B"]
    d = read["I"] - read["B"]
    dm = d.mean(axis=2)
    dse = d.std(axis=2, ddof=1) / np.sqrt(N_BANK)
    pass3 = (dm[:, :3] > 2 * dse[:, :3]).sum(axis=1) >= 2
    dbar = d[:, :3, :].mean(axis=1)
    seedmean_ok = dbar.mean(axis=1) > 2 * dbar.std(axis=1, ddof=1) / np.sqrt(N_BANK)
    bp = pass3 & seedmean_ok

    def ttest(n):
        x = dm[:, :n]
        t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n))
        return t > T_CRIT[n]

    return {
        "case": case,
        "run_noise": run_noise,
        "pool_effect_sd": float(true_diff.mean(axis=1).std(ddof=1)),
        "within_pool_se_of_10seed_mean": float(np.median(true_diff.std(axis=1, ddof=1)) / np.sqrt(N_SEED)),
        "true_effect_mean": float(true_diff.mean()),
        "true_effect_se": float(true_diff.std(ddof=1) / np.sqrt(true_diff.size)),
        "run_sd_of_diff": float(true_diff.std(ddof=1)),
        "item_paired_se_median": float(np.median(dse)),
        "bank_acc_B_mean": float(acc["B"].mean()),
        "band_share_mean": float(band.mean()),
        "rate_BP": float(bp.mean()),
        "rate_T3": float(ttest(3).mean()),
        "rate_T10": float(ttest(10).mean()),
        "n_experiments": N_EXP,
    }


def precision_optimum():
    """Pass rate maximizing d p / d beta for p = sigmoid(beta * delta): l tanh(l/2) = 1."""
    lo, hi = 0.5, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * np.tanh(mid / 2) < 1.0:
            lo = mid
        else:
            hi = mid
    return {"logit_star": mid, "pass_rate_star": float(sigmoid(mid))}


if __name__ == "__main__":
    out = {"settings": dict(N_POOL=N_POOL, N_BANK=N_BANK, M=M, T=T, ETA=ETA, K0=K0,
                            K_BANK=K_BANK, N_EXP=N_EXP, N_SEED=N_SEED),
           "precision_optimum": precision_optimum(), "cases": []}
    runs = [("N1", 0.0), ("N2", 0.0), ("P1", 0.0), ("P2", 0.0), ("N2", 0.15), ("P1", 0.15)]
    for i, (case, noise) in enumerate(runs):
        out["cases"].append(run_case(case, seed=1000 + i, run_noise=noise))
        print(json.dumps(out["cases"][-1]), file=sys.stderr)
    print(json.dumps(out, indent=1))
