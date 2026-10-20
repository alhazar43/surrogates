"""Calibration of the tier 1 reading of surrogate M on synthetic learners with known truth.

The reading asks whether the after state of one learner keeps the base state's order of items.
Statistic: Spearman rho between the base counts and the after counts over the bank. It is invariant
to every strictly increasing map of either vector, so every order preserving change gives rho = 1
without noise. Null: rho under the co-monotone rearrangement of posterior draws (theory/M.md section 4), under a logit normal
prior and under the Beta prior of attempt 0; the reading rejects only if both priors reject.
Reorder size: the item specific logit SD sigma that, added to the co-monotone null, gives a median
null rho equal to the observed rho; with a 90 percent interval by inversion of the test.

Also computes the tier 2 check: the after state's bank mean accuracy predicted from m anchor items
by an affine relink of the base logits, against the proportion correct on the same m items.

Run: python -I calibrate.py [workers] [out.json]. Designs: a bank of 200 items at k = 16 and 64 (order test
only) and the tier 1 design, 1,500 items, base against after at k = 32 and a split half at k = 8 (order test
and reorder size); then the relink. Synthetic records only.
"""
import json
import sys
import time

import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.stats import binom, norm, rankdata

B = 199            # bootstrap draws per dataset for the order test
B_SIZE = 49        # bootstrap draws per grid point for the reorder size
ALPHA = 0.05       # chosen test level
G_GUESS = 0.25     # fallback success probability of a four option item
PRIOR = "logitnormal"  # prior of the null's posterior draws; "beta" was the prior of attempt 0
BOTH_PRIORS = True     # also run the order test under the Beta prior; the reading rejects only if both do
DZ = 0.1
Z_GRID = np.arange(-9.0, 9.0 + DZ / 2, DZ)
P_GRID = expit(Z_GRID)
SIGMA_GRID = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0])


def spearman_rows(a, b):
    ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def beta_mom(y, k):
    """Method of moments Beta prior for item success probabilities from binomial counts.
    E Var(y/k) over items = Var(p) (1 - 1/k) + m (1 - m) / k, solved for Var(p)."""
    p = y / k
    m, v = p.mean(), p.var()
    v_true = max((v - m * (1 - m) / k) / (1 - 1 / k), 1e-4)
    s = max(m * (1 - m) / v_true - 1, 0.5)
    return m * s, (1 - m) * s


def posterior_grid(y, k):
    """Item posteriors on Z_GRID under a logit normal prior N(mu, sd^2) on eta = logit p, with (mu, sd)
    chosen by marginal maximum likelihood over the bank. Returns a J by G matrix of posterior weights."""
    L = binom.pmf(y[:, None], k, P_GRID[None, :])
    def nll(t):
        w = norm.pdf(Z_GRID, t[0], np.exp(t[1]))
        return -np.log(L @ (w / w.sum()) + 1e-300).sum()
    p = np.clip((y + 0.5) / (k + 1.0), 1e-3, 1 - 1e-3)
    t0 = np.array([logit(p).mean(), np.log(max(logit(p).std(), 0.3))])
    t = minimize(nll, t0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3}).x
    w = norm.pdf(Z_GRID, t[0], np.exp(t[1]))
    post = L * (w / w.sum())
    return post / post.sum(1, keepdims=True)


def draw_grid(post, rng, n):
    """n draws of p per item from the grid posterior, jittered uniformly within the grid cell."""
    c = post.cumsum(1)
    u = rng.random((post.shape[0], n))
    idx = np.minimum(np.stack([np.searchsorted(c[j], u[j]) for j in range(post.shape[0])], 1), Z_GRID.size - 1)
    return expit(Z_GRID[idx] + rng.uniform(-DZ / 2, DZ / 2, idx.shape))


def posterior_null_beta(y0, y1, k0, k1, rng, n):
    """posterior_null under the Beta prior by the method of moments."""
    J = y0.size
    a0, b0 = beta_mom(y0, k0)
    a1, b1 = beta_mom(y1, k1)
    p0 = rng.beta(a0 + y0, b0 + k0 - y0, size=(n, J))
    p1 = rng.beta(a1 + y1, b1 + k1 - y1, size=(n, J))
    rank0 = np.argsort(np.argsort(p0, axis=1), axis=1)
    return p0, np.take_along_axis(np.sort(p1, axis=1), rank0, axis=1)


def posterior_null(y0, y1, k0, k1, rng, n):
    """n posterior draws of (p0*, p1*), with p1* rearranged to the order of p0* (co-monotone null)."""
    if PRIOR == "beta":
        return posterior_null_beta(y0, y1, k0, k1, rng, n)
    else:
        p0 = draw_grid(posterior_grid(y0, k0), rng, n)
        p1 = draw_grid(posterior_grid(y1, k1), rng, n)
    rank0 = np.argsort(np.argsort(p0, axis=1), axis=1)
    return p0, np.take_along_axis(np.sort(p1, axis=1), rank0, axis=1)


def read_pair(y0, y1, k0, k1, rng, size=True):
    """The tier 1 reading of one pair of states.
    Order test: p value of rho_obs against the co-monotone null (B draws).
    Reorder size: sigma_hat, the item specific logit SD that, added to the co-monotone null, gives a median
    null rho equal to rho_obs, and its 90 percent interval by test inversion over SIGMA_GRID (B_SIZE draws)."""
    rho_obs = spearman_rows(y0[None], y1[None])[0]
    p0, p1_null = posterior_null(y0, y1, k0, k1, rng, B)
    rho_null = spearman_rows(rng.binomial(k0, p0), rng.binomial(k1, p1_null))
    out = {"p": (1 + (rho_null <= rho_obs).sum()) / (B + 1), "rho_obs": rho_obs,
           "rho_null_median": np.median(rho_null)}
    if BOTH_PRIORS and PRIOR != "beta":     # the same test under the Beta prior of attempt 0
        b0, b1 = posterior_null_beta(y0, y1, k0, k1, rng, B)
        rb = spearman_rows(rng.binomial(k0, b0), rng.binomial(k1, b1))
        out["p_beta"] = (1 + (rb <= rho_obs).sum()) / (B + 1)
    if not size:
        return out
    p0, p1_null = p0[:B_SIZE], p1_null[:B_SIZE]
    z1 = logit(np.clip(p1_null, 1e-9, 1 - 1e-9))
    eps = rng.standard_normal(z1.shape)
    y0s = rng.binomial(k0, p0)
    med, lo, hi = [], [], []
    for s in SIGMA_GRID:
        r = spearman_rows(y0s, rng.binomial(k1, expit(z1 + s * eps)))
        med.append(np.median(r))
        lo.append(np.quantile(r, 0.05))
        hi.append(np.quantile(r, 0.95))
    med = np.minimum.accumulate(np.array(med))          # enforce a decreasing curve
    if rho_obs >= med[0]:
        est = 0.0
    elif rho_obs <= med[-1]:
        est = float(SIGMA_GRID[-1])
    else:
        est = float(np.interp(-rho_obs, -med, SIGMA_GRID))
    inside = (np.array(lo) <= rho_obs) & (rho_obs <= np.array(hi))
    out["sigma_hat"] = est
    out["ci90"] = (float(SIGMA_GRID[inside].min()), float(SIGMA_GRID[inside].max())) if inside.any() else (est, est)
    return out


def bank(rng, J):
    return rng.lognormal(0.0, 0.35, J), rng.normal(0.0, 1.2, J)


def four_option_items(rng, J):
    """Option logits. Each item, by a Bernoulli(0.5) draw, holds its wrong mass on one distractor or on
    three equal distractors, with equal wrong mass at temperature 1, so the kinds match in p at tau = 1."""
    zc = rng.normal(0.0, 1.5, J)
    w = rng.normal(0.0, 1.0, J)
    one = rng.random(J) < 0.5
    three_equal = np.stack([w, w, w], 1)
    one_heavy = np.stack([w + np.log(3.0), np.full(J, -8.0), np.full(J, -8.0)], 1)
    return zc, np.where(one[:, None], one_heavy, three_equal)


def p_correct(zc, zw, tau):
    num = np.exp(zc / tau)
    return num / (num + np.exp(zw / tau).sum(1))


def truth(cond, rng, J):
    """Return (p0, p1), the base and after success probabilities of one synthetic learner."""
    a, b = bank(rng, J)
    eta0 = a * (0.3 - b)
    name, par = cond
    if name == "none":
        return expit(eta0), expit(eta0)
    if name == "none_sharp":   # no change, base with a U shaped distribution of p
        return expit(3.0 * eta0), expit(3.0 * eta0)
    if name == "shift":
        return expit(eta0), expit(eta0 + par)
    if name == "stretch":
        return expit(eta0), expit(par * eta0 + 0.3)
    if name == "wrapper":   # p = u q + (1 - u) g, the use weight u changes from 0.6 to par
        q = expit(eta0)
        return 0.6 * q + 0.4 * G_GUESS, par * q + (1 - par) * G_GUESS
    if name == "population":  # move along the population ability axis with 2PL slopes a
        return expit(eta0), expit(eta0 + a * par)
    if name == "temperature4":  # four options, decoding temperature par
        zc, zw = four_option_items(rng, J)
        return p_correct(zc, zw, 1.0), p_correct(zc, zw, par)
    if name == "item_specific":  # common shift 1 plus an item specific change of SD par
        return expit(eta0), expit(eta0 + 1.0 + rng.normal(0, par, J))
    if name == "block":  # J/10 items (one subject) raised by par on top of a common shift 0.5
        d = np.full(J, 0.5)
        d[: J // 10] += par
        return expit(eta0), expit(eta0 + d)
    raise ValueError(name)


def run_order(cond, J, R, k0, k1, seed, size=False):
    """Rejection rate of the order test, and the reorder size, over R synthetic learners of one condition."""
    rng = np.random.default_rng(seed)
    rej, rej_beta, rej_both, rhos, nulls, rho_true, est, cover = 0, 0, 0, [], [], [], [], []
    for _ in range(R):
        p0, p1 = truth(cond, rng, J)
        rho_true.append(spearman_rows(p0[None], p1[None])[0])
        y0, y1 = rng.binomial(k0, p0), rng.binomial(k1, p1)
        r = read_pair(y0, y1, k0, k1, rng, size)
        rej += r["p"] <= ALPHA
        if "p_beta" in r:
            rej_beta += r["p_beta"] <= ALPHA
            rej_both += max(r["p"], r["p_beta"]) <= ALPHA
        rhos.append(r["rho_obs"])
        nulls.append(r["rho_null_median"])
        if size:
            est.append(r["sigma_hat"])
            cover.append(r["ci90"])
    rate = rej / R
    row = {"R": R, "reject_rate": rate, "se": float(np.sqrt(max(rate * (1 - rate), 1e-12) / R)),
           "rho_obs_median": float(np.median(rhos)), "rho_null_median": float(np.median(nulls)),
           "rho_noise_free_median": float(np.median(rho_true))}
    if BOTH_PRIORS and PRIOR != "beta":
        row.update({"reject_rate_beta": rej_beta / R, "reject_rate_both": rej_both / R})
    if size:
        est, cover = np.array(est), np.array(cover)
        row.update({"sigma_hat_mean": float(est.mean()), "sigma_hat_median": float(np.median(est)),
                    "sigma_hat_q05": float(np.quantile(est, 0.05)), "sigma_hat_q95": float(np.quantile(est, 0.95)),
                    "ci90_upper_median": float(np.median(cover[:, 1])),
                    "share_ci90_upper_at_most_0.25": float((cover[:, 1] <= 0.25).mean()),
                    "share_ci90_lower_0": float((cover[:, 0] == 0).mean())})
        if cond[0] == "item_specific":
            row["ci90_covers_truth"] = float(((cover[:, 0] <= cond[1]) & (cond[1] <= cover[:, 1])).mean())
    return f"{cond[0]}({cond[1]})", row


def _job(args):
    return args[0], run_order(*args[1:])


def relink_ml(y1a, k1, eta0a):
    """ML fit of p1 = expit(alpha + c * eta0) on anchor items, BFGS with the exact gradient."""
    def nll(t):
        z = t[0] + t[1] * eta0a
        resid = y1a - k1 * expit(z)
        return -(y1a * z - k1 * np.logaddexp(0, z)).sum(), -np.array([resid.sum(), (resid * eta0a).sum()])
    return minimize(nll, np.array([0.0, 1.0]), jac=True, method="BFGS").x


def run_linking(conds, ms, J, R, k0, k1, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for cond in conds:
        res = {}
        for m in ms:
            err_link, err_cls = [], []
            for _ in range(R):
                p0, p1 = truth(cond, rng, J)
                y0 = rng.binomial(k0, p0)
                eta0_hat = logit((y0 + 0.5) / (k0 + 1.0))
                target = p1.mean()
                A = rng.choice(J, m, replace=False)
                y1a = rng.binomial(k1, p1[A])
                al, c = relink_ml(y1a, k1, eta0_hat[A])
                err_link.append(expit(al + c * eta0_hat).mean() - target)
                err_cls.append(y1a.mean() / k1 - target)
            el, ec = np.array(err_link), np.array(err_cls)
            res[f"m={m}"] = {"rmse_relink": float(np.sqrt((el ** 2).mean())),
                             "bias_relink": float(el.mean()),
                             "rmse_classical": float(np.sqrt((ec ** 2).mean()))}
        out[f"{cond[0]}({cond[1]})"] = res
    return out


def order_jobs():
    """(table, cond, J, R, k0, k1, seed, size) for every cell of both designs."""
    jobs = []
    nulls = [("none", 0), ("shift", 1.0), ("stretch", 1.5), ("wrapper", 0.95)]
    confounds = [("population", 1.0), ("temperature4", 0.5)]
    effects = [("item_specific", 0.25), ("item_specific", 0.5), ("item_specific", 1.0),
               ("block", 1.0), ("block", 2.0)]
    for k in (16, 64):
        for i, c in enumerate(nulls + confounds + effects):
            jobs.append((f"j200_order_k{k}", c, 200, 200, k, k, 1000 * k + i, False))
    for i, c in enumerate([("none", 0), ("none_sharp", 0)]):       # split half, 8 draws per half
        jobs.append(("j1500_split_half_k8", c, 1500, 200, 8, 8, 8000 + i, True))
    cells = [("none", 0), ("none_sharp", 0), ("shift", 1.0), ("stretch", 1.378), ("stretch", 3.0),
             ("wrapper", 0.95), ("population", 0.1), ("population", 0.3), ("population", 1.0),
             ("temperature4", 0.9), ("temperature4", 0.726), ("item_specific", 0.05), ("item_specific", 0.1),
             ("item_specific", 0.25), ("item_specific", 0.5), ("block", 0.5), ("block", 1.0)]
    for i, c in enumerate(cells):
        jobs.append(("j1500_base_vs_after_k32", c, 1500, 200 if c[0] in ("none", "none_sharp") else 100,
                     32, 32, 32000 + i, True))
    return jobs


if __name__ == "__main__":
    from multiprocessing import Pool
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    path = sys.argv[2] if len(sys.argv) > 2 else "calibration.json"
    t0 = time.time()
    result = {"settings": {"B": B, "B_size": B_SIZE, "alpha": ALPHA, "prior": PRIOR,
                           "sigma_grid": SIGMA_GRID.tolist()}}
    jobs = sorted(order_jobs(), key=lambda j: -j[2] * j[3] * (2 if j[7] else 1))
    with Pool(workers) as pool:
        for table, (name, row) in pool.imap_unordered(_job, jobs):
            result.setdefault(table, {})[name] = row
            print(table, name, json.dumps(row), flush=True)
    result["linking_k0_64_k1_16"] = run_linking(
        [("shift", 1.0), ("stretch", 1.5), ("wrapper", 0.95), ("item_specific", 0.5), ("block", 2.0)],
        [10, 25, 50], 200, 200, 64, 16, seed=7)
    result["settings"]["seconds"] = round(time.time() - t0, 1)
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
