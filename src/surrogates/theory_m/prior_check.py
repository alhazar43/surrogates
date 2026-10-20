"""Bias of the reorder size sigma_hat under four priors of the null's posterior draws (theory/M.md section 4).

Priors: Beta by the method of moments (attempt 0), a grid prior by nonparametric maximum likelihood (EM),
the same smoothed by a Gaussian kernel of SD 0.5 logit at every EM step, and the logit normal prior of
calibrate.py. Tier 1 design: 1,500 items, k = 32 per state, 16 synthetic learners per cell, the same learners
for every prior. Run: python -I prior_check.py [workers] [out.json]. Synthetic records only.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binom, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I drops the script folder
import calibrate as cal  # noqa: E402

CELLS = [("none", 0), ("none_sharp", 0), ("item_specific", 0.1), ("item_specific", 0.25)]
R, K = 16, 32


def grid_em(y, k, smooth):
    L = binom.pmf(y[:, None], k, cal.P_GRID[None, :])
    S = norm.pdf(cal.Z_GRID[:, None] - cal.Z_GRID[None, :], 0, 0.5)
    S /= S.sum(0, keepdims=True)
    w = np.full(cal.Z_GRID.size, 1 / cal.Z_GRID.size)
    for _ in range(300):
        P = L * w
        P /= P.sum(1, keepdims=True)
        w = P.mean(0)
        if smooth:
            w = S @ w
            w /= w.sum()
    P = L * w
    return P / P.sum(1, keepdims=True)


def make_null(kind):
    def posterior_null(y0, y1, k0, k1, rng, n):
        p0 = cal.draw_grid(grid_em(y0, k0, kind == "grid_smoothed"), rng, n)
        p1 = cal.draw_grid(grid_em(y1, k1, kind == "grid_smoothed"), rng, n)
        rank0 = np.argsort(np.argsort(p0, axis=1), axis=1)
        return p0, np.take_along_axis(np.sort(p1, axis=1), rank0, axis=1)
    return posterior_null


def run(prior):
    if prior in ("beta", "logitnormal"):
        cal.PRIOR = prior
    else:
        cal.PRIOR = "grid"
        cal.posterior_null = make_null(prior)
    out = {}
    for i, cond in enumerate(CELLS):
        rng = np.random.default_rng(500 + i)
        est, rej = [], 0
        for _ in range(R):
            p0, p1 = cal.truth(cond, rng, 1500)
            y0, y1 = rng.binomial(K, p0), rng.binomial(K, p1)
            r = cal.read_pair(y0, y1, K, K, np.random.default_rng(rng.integers(1 << 31)))
            est.append(r["sigma_hat"])
            rej += r["p"] <= cal.ALPHA
        out[f"{cond[0]}({cond[1]})"] = {"sigma_hat_mean": float(np.mean(est)),
                                        "sigma_hat_median": float(np.median(est)), "reject_rate": rej / R}
    return prior, out


if __name__ == "__main__":
    from multiprocessing import Pool
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    path = sys.argv[2] if len(sys.argv) > 2 else "prior_check.json"
    with Pool(workers) as pool:
        res = dict(pool.map(run, ["beta", "grid_npmle", "grid_smoothed", "logitnormal"]))
    res["settings"] = {"R": R, "k": K, "J": 1500, "B": cal.B, "B_size": cal.B_SIZE}
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
