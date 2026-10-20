"""Calibration of the permutation reading of tier 1 (theory/G.md R7) on the learner of calibrate_tier1.py.

Per experiment one pool and one bank are drawn. Arm I draws within the band with probability
proportional to a_pop^2. Arms P1 to PK draw with the same weights permuted across the band items,
one fixed permutation each. Every weighting is trained for S seeds and its bank read (16 answers per
bank item, the same bank) is averaged over seeds. The p value is (1 + #{k : read_k >= read_I}) / (K + 1);
the reading declares "I helps" at p <= 0.05, which with K = 19 means arm I has the highest read.
Cases N2 (null) and P1 (planted) of calibrate_tier1.py, with and without run noise.

Run: python -I calibrate_permutation.py > ../../../results/theory_g_permutation.json
(tiny synthetic records, about 4 min on the PC)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_tier1 import ETA, K0, K_BANK, M, N_BANK, N_POOL, T, make_items, sampler, sigmoid  # noqa: E402

N_EXP, N_W, N_SEED = 200, 20, 3  # N_W weightings: arm I and K = 19 permutations


def run_case(case, seed, run_noise=0.0):
    rng = np.random.default_rng(seed)
    b, alpha, a_pop = make_items(rng, N_EXP, N_POOL, case)
    bb, balpha, _ = make_items(rng, N_EXP, N_BANK, case)
    p0hat = rng.binomial(K0, sigmoid(alpha * (0.0 - b))) / K0
    band = (p0hat >= 0.2) & (p0hat <= 0.8)
    phat = np.clip(p0hat, 0.05, 0.95)
    # weights per (experiment, weighting): row e * N_W + w; w = 0 is arm I
    w = np.zeros((N_EXP, N_W, N_POOL))
    for e in range(N_EXP):
        inb = np.flatnonzero(band[e])
        w[e, 0, inb] = a_pop[e, inb] ** 2
        for k in range(1, N_W):
            w[e, k, inb] = rng.permutation(a_pop[e, inb]) ** 2
    draw = sampler(w.reshape(N_EXP * N_W, N_POOL))
    row_exp = np.repeat(np.arange(N_EXP), N_W)[:, None, None]
    theta = np.zeros((N_EXP * N_W, N_SEED))
    for _ in range(T):
        idx = draw(rng, (N_SEED, M))
        al, bi, ph = alpha[row_exp, idx], b[row_exp, idx], phat[row_exp, idx]
        p = sigmoid(al * (theta[:, :, None] - bi))
        r = (rng.random(p.shape) < p) * 1.0
        theta += ETA * (np.mean(al * (r - ph) * (r - p), axis=2)
                        + run_noise * rng.standard_normal(theta.shape))
    theta = theta.reshape(N_EXP, N_W, N_SEED)
    pb = sigmoid(balpha[:, None, None, :] * (theta[..., None] - bb[:, None, None, :]))
    acc = pb.mean(axis=3)                                        # exact, (exp, w, seed)
    read = (rng.binomial(K_BANK, pb) / K_BANK).mean(axis=3)      # sampled bank read

    def pvalue(x):  # x: (exp, w)
        return (1 + (x[:, 1:] >= x[:, :1]).sum(axis=1)) / N_W

    out = {"case": case, "run_noise": run_noise, "n_experiments": N_EXP, "K": N_W - 1}
    for s in (1, N_SEED):
        p = pvalue(read[:, :, :s].mean(axis=2))
        out[f"rate_S{s}"] = float((p <= 0.05).mean())
        out[f"mean_p_S{s}"] = float(p.mean())
    eff = acc[:, 0, :].mean(axis=1) - acc[:, 1:, :].mean(axis=(1, 2))
    out["true_effect_I_minus_perm_mean"] = float(eff.mean())
    out["true_effect_se"] = float(eff.std(ddof=1) / np.sqrt(N_EXP))
    out["sd_across_perm_of_read_S1"] = float(np.median(read[:, 1:, 0].std(axis=1, ddof=1)))
    return out


if __name__ == "__main__":
    res = {"settings": dict(N_EXP=N_EXP, K=N_W - 1, N_SEED=N_SEED, N_POOL=N_POOL, N_BANK=N_BANK,
                            M=M, T=T, ETA=ETA, K0=K0, K_BANK=K_BANK), "cases": []}
    for i, (case, noise) in enumerate([("N2", 0.0), ("P1", 0.0), ("N2", 0.15), ("P1", 0.15)]):
        res["cases"].append(run_case(case, seed=2000 + i, run_noise=noise))
        print(json.dumps(res["cases"][-1]), file=sys.stderr)
    print(json.dumps(res, indent=1))
