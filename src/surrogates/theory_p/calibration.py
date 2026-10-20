"""Calibration of the readings of surrogate P on synthetic learners whose truth is known by construction.

Three parts, each printed as a table.
  A. The outcome distribution per prefix o_t. A martingale check under on-policy sampling, and the
     false positive rate and power of fork detection at S continuations per prefix.
  B. The forced answer at the last cut before the key is written (cut k-1). Reference learners (forced
     answer independent of the prefix; forced answer reading the prefix with probability w), the
     difference in differences against items whose answer is given in the question, its false
     positive rate and power, and the cases where the scale or unequal drift breaks the test.
  C. The prediction of the sequence-level tilt. Within-outcome rates under the exact tilt, under plain
     policy-gradient flow on per-method logits, and under a planted within-right change whose size in
     the within-right rate is printed beside it.

Each part has its own seed (A 20261019, B 20261020, C 20261021); part A's stream equals the first run's.
Part A2 takes most of the time (above 10 minutes on one PC core); B and C take a few minutes.
Usage: python -I calibration.py [A1] [A2] [B] [C]   (no argument runs all parts)
"""
import sys

import numpy as np
from scipy import stats

RNG = np.random.default_rng(20261019)
REPS = 2000


def logit(p):
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- part A
def martingale_check(T=12, q=0.55, paths=20000):
    """Outcome = 1 if at least T/2 of T Bernoulli(q) tokens are 1. o_t is exact by the binomial."""
    need = int(np.ceil(T / 2))

    def o(t, s):  # P(final = 1 | s ones among the first t tokens)
        return stats.binom.sf(need - s - 1, T - t, q)

    tok = RNG.random((paths, T)) < q
    s = np.concatenate([np.zeros((paths, 1), int), np.cumsum(tok, 1)], 1)
    ot = np.stack([o(t, s[:, t]) for t in range(T + 1)], 1)
    d = np.diff(ot, axis=1)
    o0 = ot[0, 0]
    # greedy path: every token is the mode (1, since q > 0.5)
    g = np.array([o(t, t) for t in range(T + 1)])
    # two outcomes: the vector form ||o_{t+1} - o_t||^2 equals twice the squared change of o(1)
    return dict(o0=o0, mean_increment=d.mean(), max_abs_mean_increment=np.abs(d.mean(0)).max(),
                sum_sq_one_coordinate=np.mean((d ** 2).sum(1)), predicted_one_coordinate=o0 * (1 - o0),
                sum_sq_vector=np.mean((2 * d ** 2).sum(1)), predicted_vector=1 - o0 ** 2 - (1 - o0) ** 2,
                greedy_drift=g[-1] - g[0])


def tvd(a, b):
    return 0.5 * np.abs(a - b).sum(-1)


def chi2_two(a, b):
    tot = a + b
    ea = tot * (a.sum(-1, keepdims=True) / (a.sum(-1, keepdims=True) + b.sum(-1, keepdims=True)))
    eb = tot - ea
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.where(ea > 0, (a - ea) ** 2 / ea, 0) + np.where(eb > 0, (b - eb) ** 2 / eb, 0)).sum(-1)


def fork_detection(S, eps=0.10, T=100, delta=0.0, reps=400, alpha=0.05, W=10):
    """T positions, K=5 outcomes. A fork of TVD size delta at position T//2 (delta=0 is no fork).
    Raw rule (Forking Fast's forking point definition): TVD(o_t, o_t+1) > eps.
    Tested rule: chi-square test of homogeneity of the two count vectors, Bonferroni over T-1 pairs."""
    p0 = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    p1 = p0.copy()
    p1[0] -= delta
    p1[1] += delta
    P = np.where(np.arange(T)[:, None] < T // 2, p0, p1)
    raw_false, raw_any, raw_hit, tst_false, tst_any, tst_hit = [], [], [], [], [], []
    win_any, win_hit = [], []
    crit = stats.chi2.isf(alpha / (T - 1), df=4)
    for _ in range(reps):
        c = np.stack([RNG.multinomial(S, P[t]) for t in range(T)])
        f = c / S
        raw = tvd(f[:-1], f[1:]) > eps
        tot = c[:-1] + c[1:]
        exp_a = tot * (c[:-1].sum(1, keepdims=True) / (2 * S))
        with np.errstate(divide="ignore", invalid="ignore"):
            x2 = np.where(exp_a > 0, (c[:-1] - exp_a) ** 2 / exp_a + (c[1:] - exp_a) ** 2 / exp_a, 0).sum(1)
        tst = x2 > crit
        cs = np.concatenate([np.zeros((1, 5)), np.cumsum(c, 0)])
        win = np.zeros(T - 1, bool)
        for t in range(W - 1, T - W):   # left window t-W+1..t, right window t+1..t+W
            win[t] = chi2_two(cs[t + 1] - cs[t + 1 - W], cs[t + 1 + W] - cs[t + 1]) > crit
        fork_pair = T // 2 - 1
        mask = np.ones(T - 1, bool)
        mask[fork_pair] = False
        raw_false.append(raw[mask].mean()); raw_any.append(raw[mask].any()); raw_hit.append(raw[fork_pair])
        tst_false.append(tst[mask].mean()); tst_any.append(tst[mask].any()); tst_hit.append(tst[fork_pair])
        near = np.abs(np.arange(T - 1) - fork_pair) <= (W - 1 if delta > 0 else -1)
        win_any.append(win[~near & mask].any()); win_hit.append(win[fork_pair])
    return dict(S=S, delta=delta, raw_fpr_per_pair=np.mean(raw_false), raw_any_false_in_chain=np.mean(raw_any),
                raw_power=np.mean(raw_hit), test_fpr_per_pair=np.mean(tst_false),
                test_any_false_in_chain=np.mean(tst_any), test_power=np.mean(tst_hit),
                window_any_false_in_chain=np.mean(win_any), window_power=np.mean(win_hit))


# ---------------------------------------------------------------- part B
def cut_reading(n, w, k=8, c=2, h=0.9, sigma=0.5, drift_needed=0.5, drift_given=0.5, scale="logit",
                reps=REPS, alpha=0.05):
    """n items whose answer needs k steps (one-pass capacity c < k) and n items whose answer is given
    in the question. Forced key probability at cut j = sigmoid(logit(base_i) + drift * j/k + sigma*eps).
    On a needed item the forced answer reads the prefix with probability w; then the key probability is
    h at every cut with at most c steps left. The cut read is j = k-1, the last one before the key is
    written. d_i = (cut k-1) - (no chain) on the stated scale.
    Statistic: mean d over needed items minus mean d over given items, one-sided Welch t-test."""
    j = k - 1
    assert k - j <= c
    rej, eff = 0, []
    for _ in range(reps):
        bn = RNG.beta(2, 8, n)          # one-pass accuracy on needed items, mean 0.2 (chance 0.1)
        bg = RNG.beta(9, 1, n)          # one-pass accuracy on items with the answer given, mean 0.9
        bn, bg = np.clip(bn, 1e-3, 1 - 1e-3), np.clip(bg, 1e-3, 1 - 1e-3)
        z0n = logit(bn) + sigma * RNG.standard_normal(n)
        zkn_idle = logit(bn) + drift_needed * j / k + sigma * RNG.standard_normal(n)
        z0g = logit(bg) + sigma * RNG.standard_normal(n)
        zkg = logit(bg) + drift_given * j / k + sigma * RNG.standard_normal(n)
        used = RNG.random(n) < w
        zkn = np.where(used, logit(h) + sigma * RNG.standard_normal(n), zkn_idle)
        if scale == "logit":
            dn, dg = zkn - z0n, zkg - z0g
        else:
            dn, dg = sigmoid(zkn) - sigmoid(z0n), sigmoid(zkg) - sigmoid(z0g)
        t, p = stats.ttest_ind(dn, dg, equal_var=False, alternative="greater")
        rej += p < alpha
        eff.append(np.mean(sigmoid(zkn) - sigmoid(zkn_idle)))
    return dict(n=n, w=w, scale=scale, drift=(drift_needed, drift_given), reject_rate=rej / reps,
                planted_rise_in_key_prob=np.mean(eff))


# ---------------------------------------------------------------- part C
def make_item(M=6):
    while True:
        pi0 = RNG.dirichlet(np.ones(M))
        r = RNG.random(M) < 0.5
        if 0 < r.sum() < M:
            break
    phi = RNG.random(M) < 0.5
    return pi0, r.astype(float), phi.astype(float)


def within_right(pi, r, phi):
    return (pi * r * phi).sum() / (pi * r).sum()


def tilt(pi0, r, gain):
    """Exact tilt pi0 * exp(c r) with c chosen so that the success probability rises by `gain` of the gap."""
    p0 = (pi0 * r).sum()
    target = p0 + gain * (1 - p0)
    odds = target / (1 - target) / (p0 / (1 - p0))
    pi = pi0 * np.where(r > 0, odds, 1.0)
    return pi / pi.sum()


def pg_flow(pi0, r, gain, eta=0.05, max_steps=200000):
    """Plain policy gradient on one logit per method: dl_m = eta * pi_m * (r_m - p)."""
    l = np.log(pi0)
    p0 = (pi0 * r).sum()
    target = p0 + gain * (1 - p0)
    for _ in range(max_steps):
        pi = np.exp(l - l.max()); pi /= pi.sum()
        p = (pi * r).sum()
        if p >= target:
            break
        l += eta * pi * (r - p)
    return pi


def tilt_checks(N=100, runs=32, gain=0.5, reps=500, alpha=0.05):
    out = {}
    # exact values over items
    d_tilt, d_pg, pooled_before, pooled_after = [], [], [], []
    items = [make_item() for _ in range(2000)]
    cache_tilt = [tilt(pi0, r, gain) for pi0, r, phi in items]
    cache_pg = [pg_flow(pi0, r, gain) for pi0, r, phi in items]
    for (pi0, r, phi), pt, pg in zip(items, cache_tilt, cache_pg):
        d_tilt.append(within_right(pt, r, phi) - within_right(pi0, r, phi))
        d_pg.append(within_right(pg, r, phi) - within_right(pi0, r, phi))
    out["exact_tilt_max_abs_change"] = np.max(np.abs(d_tilt))
    # property held by the base's most probable right method and by no other right method
    modal = []
    for (pi0, r, phi) in items:
        m = np.zeros_like(phi); m[np.argmax(pi0 * r)] = 1.0
        modal.append(np.where(r > 0, m, phi))
    d_pgm = np.array([within_right(pg, r, ph) - within_right(pi0, r, ph)
                      for (pi0, r, _), pg, ph in zip(items, cache_pg, modal)])
    out["pg_flow_modal_property_change_mean"] = d_pgm.mean()
    out["pg_flow_modal_property_change_median"] = np.median(d_pgm)
    out["pg_flow_modal_property_change_min"] = d_pgm.min()
    d_pg = np.array(d_pg)
    out["pg_flow_change_mean_signed"] = d_pg.mean()
    out["pg_flow_change_mean_abs"] = np.mean(np.abs(d_pg))
    out["pg_flow_change_median_abs"] = np.median(np.abs(d_pg))
    out["pg_flow_change_p90_abs"] = np.quantile(np.abs(d_pg), 0.9)

    # pooled within-right rate across heterogeneous items under the exact tilt (aggregation check)
    # items: half with p0=0.2 and a1=0.8, half with p0=0.8 and a1=0.2; tilt raises p by half the gap
    def pooled(p_lo, p_hi):
        return (p_lo * 0.8 + p_hi * 0.2) / (p_lo + p_hi)
    out["pooled_within_right_before"] = pooled(0.2, 0.8)
    out["pooled_within_right_after_tilt"] = pooled(0.2 + 0.5 * 0.8, 0.8 + 0.5 * 0.2)

    # sampled test: per item, within-right rate after minus before, t-test over items with >= 2 right runs
    def sampled(kind, delta=0.0):
        """Rejection rate; for kind "planted" also the mean expected change of the within-right rate,
        delta (1 - a_1(x)), over the items drawn (the tilt keeps a_1, so this is the planted size)."""
        rej, size = 0, []
        for _ in range(reps):
            diffs = []
            for i in range(N):
                j = RNG.integers(len(items))
                pi0, r, phi = items[j]
                if kind == "pg_modal":
                    phi = modal[j]
                # planted: the tilt, then within right runs a share delta gains the property
                pa = cache_pg[j] if kind in ("pg", "pg_modal") else cache_tilt[j]
                phia = phi
                mb = RNG.choice(len(pi0), runs, p=pi0)
                ma = RNG.choice(len(pa), runs, p=pa)
                rb, ra = r[mb] > 0, r[ma] > 0
                if rb.sum() < 2 or ra.sum() < 2:
                    continue
                fb = phi[mb][rb]
                fa = phia[ma][ra]
                if kind == "planted":
                    fa = np.where(RNG.random(fa.size) < delta, 1.0, fa)
                    size.append(delta * (1 - within_right(pi0, r, phi)))
                diffs.append(fa.mean() - fb.mean())
            t, p = stats.ttest_1samp(diffs, 0.0)
            rej += p < alpha
        return (rej / reps, np.mean(size)) if kind == "planted" else rej / reps
    out["sampled_fpr_exact_tilt"] = sampled("tilt")
    out["sampled_flag_rate_pg_flow"] = sampled("pg")
    out["sampled_flag_rate_pg_flow_modal_property"] = sampled("pg_modal")
    for dl in (0.05, 0.10, 0.20, 0.40):
        rate, size = sampled("planted", dl)
        out[f"sampled_power_planted_share_{dl:.2f}"] = rate
        out[f"planted_share_{dl:.2f}_within_right_change"] = size
    return out


def main(parts):
    global RNG
    if "A1" in parts or "A2" in parts:
        RNG = np.random.default_rng(20261019)
        res = martingale_check()          # drawn first, so A2 sees the first run's stream
        if "A1" in parts:
            print("A1 martingale check")
            for kk, vv in res.items():
                print(f"  {kk:28s} {vv: .5f}")
    if "A2" in parts:
        print("A2 fork detection, T=100 positions, K=5, eps=0.10, Bonferroni alpha=0.05")
        print("  S   delta raw_fpr/pair raw_any_false raw_power test_fpr/pair test_any_false test_power"
              "  win10_any_false win10_power")
        for S in (5, 10, 20, 30, 50, 100, 200):
            for delta in (0.0, 0.1, 0.2, 0.3):
                r = fork_detection(S, delta=delta)
                print(f"  {S:3d} {delta:.1f}   {r['raw_fpr_per_pair']:.3f}        {r['raw_any_false_in_chain']:.3f}"
                      f"         {r['raw_power']:.3f}     {r['test_fpr_per_pair']:.4f}        "
                      f"{r['test_any_false_in_chain']:.3f}          {r['test_power']:.3f}       "
                      f"{r['window_any_false_in_chain']:.3f}          {r['window_power']:.3f}")
    if "B" in parts:
        RNG = np.random.default_rng(20261020)
        print("B cut reading at cut k-1 = 7, k=8 steps, one-pass capacity 2, one-sided Welch t at alpha=0.05")
        print("  n    w     scale  drift(needed,given) reject_rate planted_rise_key_prob")
        for scale, dr in (("logit", (0.5, 0.5)), ("prob", (0.5, 0.5)), ("logit", (1.0, 0.5))):
            for n in (25, 50, 100, 200):
                for w in (0.0, 0.05, 0.1, 0.2, 0.4):
                    if (scale != "logit" or dr != (0.5, 0.5)) and w > 0:
                        continue
                    r = cut_reading(n, w, scale=scale, drift_needed=dr[0], drift_given=dr[1])
                    print(f"  {n:3d} {w:.2f}  {scale:5s}  {dr}          {r['reject_rate']:.3f}       "
                          f"{r['planted_rise_in_key_prob']:.3f}")
    if "C" in parts:
        RNG = np.random.default_rng(20261021)
        print("C tilt checks, 6 methods per item, gain = half the gap to 1, N=100 items, 32 runs before and after")
        for kk, vv in tilt_checks().items():
            print(f"  {kk:40s} {vv: .4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["A1", "A2", "B", "C"])
