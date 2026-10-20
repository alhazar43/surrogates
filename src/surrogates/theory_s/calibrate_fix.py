"""Calibration of the corrected readings of surrogate S (theory/S.md, fix of attempt 1). Synthetic records only.

Part F2. Steady-state smoother variance of the local level model (Proposition 2, offline floor).
Part P.  Pooled Cochran-Mantel-Haenszel test of linked options over K pairs: population residue, per-pair
         standard deviation, false positive rate against the pooled count (simulated and normal approximation).
Part P2. Finite-sample mean of the pooled log OR_MH at K = 50.
Part L.  Pooled marginal-likelihood score test of local dependence (the corrected tier 1 reading), with the MH
         test on the same draws: false positive rate and power against K, rest length, delta, and a departure
         in which the linked items measure a second level correlated rho with the rest level.
Part L2. MLS with the level from items of the linked skill, under the departure of Part L.
Part Q.  Population log OR_MH against prevalence pi (Proposition 4d).
Part T.  Tier 2 with the posterior misconception probability and 5-fold cross-fitting by student: false positive
         rate, power, and the map from c4 to eta_1.
Part D.  Closed-form calibration of the decision rule of T1a (D3).

Run: python -I calibrate_fix.py <part>    (output calibrate_fix_<part>.txt beside this file, written by the caller)
"""
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script directory
import calibrate as cal  # noqa: E402

ALPHA = 0.05  # chosen test level
Z = stats.norm.ppf(1 - ALPHA / 2)
GRID = np.linspace(-4, 4, 41)
LOGPRIOR = -0.5 * GRID ** 2


# ---------------------------------------------------------------- Part F2
def part_f2():
    print("Part F2. Local level model, steady-state filter and Rauch-Tung-Striebel smoother")
    for v, info in ((0.01, 0.25), (0.0025, 0.25)):
        pf = (-info * v + np.sqrt((info * v) ** 2 + 4 * info * v)) / (2 * info)
        pp = pf + v
        j = pf / pp
        ps = (pf - j * j * pp) / (1 - j * j)  # fixed point of Ps = Pf + J^2 (Ps - Pp)
        print(f"  v={v} I={info}: filter var {pf:.5f} sd {np.sqrt(pf):.4f}; smoother var {ps:.5f} sd {np.sqrt(ps):.4f}")


# ---------------------------------------------------------------- shared simulator for tier 1
def sim_pair(rng, n, gamma, delta, n_rest, b_rest, rho=1.0, pi=0.3, zeta=0.5, rest_on_link=False):
    """One linked pair (b = 0 for both items) on n students. Returns, for students wrong on both,
    the rest score, A, B. With rho < 1 the linked items depend on theta_link = rho theta + sqrt(1-rho^2) e."""
    th = rng.standard_normal(n)
    tl = th if rho >= 1.0 else rho * th + np.sqrt(1 - rho * rho) * rng.standard_normal(n)
    z = (rng.random(n) < pi) & (gamma > 0)
    lev = tl if rest_on_link else th  # rest_on_link: the rest items are items of the linked skill
    rest = (rng.random((n, n_rest)) < cal.sigmoid(lev[:, None] - b_rest[None, :])).sum(1)
    ch, wr = [], []
    for _ in range(2):
        wr.append(rng.random(n) >= cal.sigmoid(tl - zeta * z))
        lg = np.stack([-delta * tl + gamma * z, np.zeros(n), np.full(n, -0.5)], 1)
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        ch.append(rng.random(n) < p[:, 0])
    both = wr[0] & wr[1]
    return rest[both], ch[0][both], ch[1][both]


def strata_of(rest, n_strata, n_rest):
    """Strata of the rest score cut at its sample quantiles among the students given (n_strata 0: each score value)."""
    return np.digitize(rest, np.quantile(rest, np.linspace(0, 1, n_strata + 1)[1:-1])) if n_strata else rest


# ---------------------------------------------------------------- Part P
def population_residue(rng, n_rest, n_strata, delta, n_total=400000, chunk=50000):
    b_rest = rng.standard_normal(n_rest)
    tabs = {}
    edges = None
    for _ in range(n_total // chunk):
        r, a, b = sim_pair(rng, chunk, 0.0, delta, n_rest, b_rest)
        if n_strata:
            if edges is None:
                edges = np.quantile(r, np.linspace(0, 1, n_strata + 1)[1:-1])
            st = np.digitize(r, edges)
        else:
            st = r
        for s in np.unique(st):
            m = st == s
            t = tabs.setdefault(int(s), np.zeros(4))
            t += [np.sum(a[m] & b[m]), np.sum(a[m] & ~b[m]), np.sum(~a[m] & b[m]), np.sum(~a[m] & ~b[m])]
    num = sum(t[0] * t[3] / t.sum() for t in tabs.values() if t.sum() > 1)
    den = sum(t[1] * t[2] / t.sum() for t in tabs.values() if t.sum() > 1)
    return np.log(num / den)


def fpr_normal(resid, sd1, k):
    mu = resid * np.sqrt(k) / sd1
    return stats.norm.sf(Z - mu) + stats.norm.cdf(-Z - mu)


def part_p():
    rng = np.random.default_rng(31)
    print("Part P. MH test pooled over K linked pairs, 1000 students per pair, rest 250 items, 10 strata, delta 1.0")
    print("  population residue log OR_MH (400000 students):")
    res = {}
    for nr, ns, dl in ((250, 10, 1.0), (250, 0, 1.0), (250, 10, 0.5), (250, 10, 2.0), (50, 10, 1.0)):
        res[(nr, ns, dl)] = population_residue(rng, nr, ns, dl)
        lab = "each score value" if ns == 0 else f"{ns} strata"
        print(f"    rest {nr}, {lab}, delta {dl}: {res[(nr, ns, dl)]:.4f}")
    b_rest = rng.standard_normal(250)
    lors, nb = [], []
    for _ in range(400):
        r, a, b = sim_pair(rng, 1000, 0.0, 1.0, 250, b_rest)
        lor, _ = cal.mh_test(a, b, strata_of(r, 10, 250))
        lors.append(lor); nb.append(len(r))
    sd1 = np.nanstd(lors)
    print(f"  one pair: median students wrong on both {np.median(nb):.0f}, SD of log OR_MH {sd1:.4f} (400 logs)")
    print("  simulated rejection rate of the pooled MH test under the hard null:")
    for k, reps in ((1, 400), (20, 300), (50, 100)):
        rej = 0
        for _ in range(reps):
            aa, bb, ss = [], [], []
            for j in range(k):
                r, a, b = sim_pair(rng, 1000, 0.0, 1.0, 250, b_rest)
                aa.append(a); bb.append(b); ss.append(strata_of(r, 10, 250) + 100 * j)
            rej += cal.mh_test(np.concatenate(aa), np.concatenate(bb), np.concatenate(ss))[1] < ALPHA
        print(f"    K={k}: {rej / reps:.3f} ({reps} logs, MC SE {np.sqrt(0.05 * 0.95 / reps):.3f}); "
              f"normal approximation {fpr_normal(res[(250, 10, 1.0)], sd1, k):.3f}")
    print("  normal approximation, FPR(K) = Phi(-1.96 + r sqrt(K)/s1) + Phi(-1.96 - r sqrt(K)/s1):")
    print("    K | pooled wrong on both | 10 strata d1.0 | each value d1.0 | 10 strata d0.5 | 10 strata d2.0")
    for k in (1, 5, 10, 20, 50, 100, 200):
        vals = [fpr_normal(res[key], sd1, k) for key in ((250, 10, 1.0), (250, 0, 1.0), (250, 10, 0.5), (250, 10, 2.0))]
        print(f"    {k} | {k * np.median(nb):.0f} | " + " | ".join(f"{v:.3f}" for v in vals))


def part_p2():
    """Finite-sample mean of the pooled log OR_MH at K = 50, against the population residue of Part P."""
    rng = np.random.default_rng(6)
    b = rng.standard_normal(250)
    lo, rej = [], 0
    for _ in range(200):
        aa, bb, ss = [], [], []
        for j in range(50):
            r, a, b2 = sim_pair(rng, 1000, 0.0, 1.0, 250, b)
            aa.append(a); bb.append(b2); ss.append(strata_of(r, 10, 250) + 100 * j)
        lor, p = cal.mh_test(np.concatenate(aa), np.concatenate(bb), np.concatenate(ss))
        lo.append(lor); rej += p < ALPHA
    lo = np.array(lo)
    print("Part P2. Pooled MH, K = 50 pairs of 1000 students, rest 250, 10 strata per pair, delta 1.0, 200 logs")
    print(f"  pooled log OR_MH mean {lo.mean():.4f} (MC SE {lo.std() / np.sqrt(len(lo)):.4f}), median {np.median(lo):.4f}, "
          f"SD {lo.std():.4f}, 5 to 95 percent {np.quantile(lo, 0.05):.4f} to {np.quantile(lo, 0.95):.4f}; rejection {rej / 200:.3f}")
    for k in (1, 5, 10, 20, 50, 100, 200):
        print(f"  normal approximation with r = pooled mean, s1 = 0.3042: K={k}: {fpr_normal(lo.mean(), 0.3042, k):.3f}")


# ---------------------------------------------------------------- Part L
def mls_pair(rest, a, b, b_rest, b_link=0.0):
    """Marginal-likelihood score statistic of local dependence psi = 0 for one pair.
    Level model: theta ~ N(0,1), Rasch rest items with known b_rest, both linked items wrong (b_link known).
    Option model: P(A=1 | theta, wrong) = sigmoid(a_j + d_j theta), same for B; nuisances (a_j, d_j, a_k, d_k) by ML.
    Returns the efficient score sum and its outer-product variance."""
    lw = (rest[:, None] * GRID[None, :] - np.logaddexp(0.0, GRID[:, None] - b_rest[None, :]).sum(1)[None, :]
          + 2 * np.log(1 - cal.sigmoid(GRID - b_link))[None, :] + LOGPRIOR[None, :])
    w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    A = a.astype(float); B = b.astype(float)

    def post(par):
        pa = cal.sigmoid(par[0] + par[1] * GRID); pb = cal.sigmoid(par[2] + par[3] * GRID)
        f = (np.where(A[:, None] > 0, pa, 1 - pa) * np.where(B[:, None] > 0, pb, 1 - pb))
        g = w * f
        s = g.sum(1, keepdims=True)
        return pa, pb, g / s, s[:, 0]

    def scores(par):
        pa, pb, p, s = post(par)
        ra = A[:, None] - pa[None, :]; rb = B[:, None] - pb[None, :]
        S = np.column_stack([(p * ra).sum(1), (p * ra * GRID).sum(1), (p * rb).sum(1), (p * rb * GRID).sum(1)])
        return S, np.log(s).sum(), pa, pb, p

    par = np.zeros(4)
    S, ll, *_ = scores(par)
    for _ in range(100):  # BHHH ascent with step halving
        step = np.linalg.solve(S.T @ S + 1e-9 * np.eye(4), S.sum(0))
        t = 1.0
        while t > 1e-4:
            S2, ll2, *_ = scores(par + t * step)
            if ll2 >= ll - 1e-12:
                break
            t /= 2
        par = par + t * step
        conv = abs(ll2 - ll) < 1e-10
        S, ll = S2, ll2
        if conv:
            break
    S, _, pa, pb, p = scores(par)

    U = A * B - (p * (pa * pb)[None, :]).sum(1)
    beta = np.linalg.lstsq(S, U, rcond=None)[0]
    ue = U - S @ beta
    return ue.sum(), (ue ** 2).sum(), par[1]


def part_l():
    rng = np.random.default_rng(37)
    print("Part L. Pooled marginal-likelihood score test (MLS) and pooled MH test (10 strata) on the same draws")
    print("  1000 students per pair; rejection at alpha 0.05, two-sided")
    print("  case | K | logs | MLS rejection | MH rejection | mean fitted d_j")
    cases = (("null, delta 1.0, rest 250", 0.0, 1.0, 250, 1.0, 1, 300),
             ("null, delta 1.0, rest 250", 0.0, 1.0, 250, 1.0, 20, 200),
             ("null, delta 1.0, rest 250", 0.0, 1.0, 250, 1.0, 50, 100),
             ("null, delta 1.0, rest 50", 0.0, 1.0, 50, 1.0, 20, 200),
             ("null, delta 2.0, rest 50", 0.0, 2.0, 50, 1.0, 20, 200),
             ("null, delta 0, rest 250", 0.0, 0.0, 250, 1.0, 20, 200),
             ("planted gamma 0.7, delta 1.0, rest 250", 0.7, 1.0, 250, 1.0, 1, 300),
             ("planted gamma 0.7, delta 1.0, rest 250", 0.7, 1.0, 250, 1.0, 20, 200),
             ("planted gamma 0.7, delta 1.0, rest 50", 0.7, 1.0, 50, 1.0, 20, 200),
             ("departure rho 0.8, null, delta 1.0, rest 250", 0.0, 1.0, 250, 0.8, 1, 300),
             ("departure rho 0.8, null, delta 1.0, rest 250", 0.0, 1.0, 250, 0.8, 20, 200),
             ("departure rho 0.95, null, delta 1.0, rest 250", 0.0, 1.0, 250, 0.95, 20, 200))
    for label, gamma, delta, nr, rho, k, reps in cases:
        b_rest = rng.standard_normal(nr)
        rej_l = rej_m = 0
        dj = []
        for _ in range(reps):
            us = vs = 0.0
            aa, bb, ss = [], [], []
            for j in range(k):
                r, a, b = sim_pair(rng, 1000, gamma, delta, nr, b_rest, rho=rho)
                u, v, d = mls_pair(r, a, b, b_rest)
                us += u; vs += v; dj.append(d)
                aa.append(a); bb.append(b); ss.append(strata_of(r, 10, nr) + 100 * j)
            rej_l += abs(us / np.sqrt(vs)) > Z
            rej_m += cal.mh_test(np.concatenate(aa), np.concatenate(bb), np.concatenate(ss))[1] < ALPHA
        print(f"  {label} | {k} | {reps} | {rej_l / reps:.3f} | {rej_m / reps:.3f} | {np.mean(dj):.3f}", flush=True)


def part_l2():
    rng = np.random.default_rng(47)
    print("Part L2. MLS with the level taken from n_skill items of the linked skill (the departure rho 0.8 holds for")
    print("  the general level, which the test no longer uses). 1000 students per pair, alpha 0.05, two-sided")
    print("  case | K | logs | MLS rejection | MH rejection")
    for label, gamma, ns, k, reps in (("null, delta 1.0, 10 skill items", 0.0, 10, 20, 200),
                                      ("planted gamma 0.7, delta 1.0, 10 skill items", 0.7, 10, 20, 200),
                                      ("null, delta 1.0, 10 skill items", 0.0, 10, 50, 100)):
        b_rest = rng.standard_normal(ns)
        rej_l = rej_m = 0
        for _ in range(reps):
            us = vs = 0.0
            aa, bb, ss = [], [], []
            for j in range(k):
                r, a, b = sim_pair(rng, 1000, gamma, 1.0, ns, b_rest, rho=0.8, rest_on_link=True)
                u, v, _ = mls_pair(r, a, b, b_rest)
                us += u; vs += v
                aa.append(a); bb.append(b); ss.append(r + 100 * j)
            rej_l += abs(us / np.sqrt(vs)) > Z
            rej_m += cal.mh_test(np.concatenate(aa), np.concatenate(bb), np.concatenate(ss))[1] < ALPHA
        print(f"  {label} | {k} | {reps} | {rej_l / reps:.3f} | {rej_m / reps:.3f}", flush=True)


# ---------------------------------------------------------------- Part Q
def part_q():
    rng = np.random.default_rng(41)
    print("Part Q. Population log OR_MH, gamma 1.4, delta 0, stratified on the true theta (50 strata), 400000 students")
    for pi in (0.1, 0.3, 0.5, 0.7, 0.9):
        tabs = {}
        for _ in range(8):
            n = 50000
            th = rng.standard_normal(n)
            z = rng.random(n) < pi
            ch, wr = [], []
            for _k in range(2):
                wr.append(rng.random(n) >= cal.sigmoid(th - 0.5 * z))
                lg = np.stack([1.4 * z, np.zeros(n), np.full(n, -0.5)], 1)
                p = np.exp(lg); p /= p.sum(1, keepdims=True)
                ch.append(rng.random(n) < p[:, 0])
            both = wr[0] & wr[1]
            st = np.digitize(th[both], np.quantile(th, np.linspace(0, 1, 51)[1:-1]))
            a, b, zb = ch[0][both], ch[1][both], z[both]
            for s in np.unique(st):
                m = st == s
                t = tabs.setdefault(int(s), np.zeros(5))
                t += [np.sum(a[m] & b[m]), np.sum(a[m] & ~b[m]), np.sum(~a[m] & b[m]), np.sum(~a[m] & ~b[m]), zb[m].sum()]
        num = sum(t[0] * t[3] / t[:4].sum() for t in tabs.values())
        den = sum(t[1] * t[2] / t[:4].sum() for t in tabs.values())
        nz = sum(t[4] for t in tabs.values()); nt = sum(t[:4].sum() for t in tabs.values())
        q = nz / nt
        print(f"  pi {pi}: log OR_MH {np.log(num / den):.4f}; share of z among wrong on both {q:.3f}, variance {q * (1 - q):.3f}")


# ---------------------------------------------------------------- Part T
def sim_tracker(rng, n, eta1, n_linked=10, eta0=0.05, pi=0.3, zeta=0.5, gamma=1.4, n_early=30, max_practice=10, n_test=5):
    theta = rng.standard_normal(n)
    z = rng.random(n) < pi
    b_e = rng.standard_normal((n, n_early))
    pen = np.zeros((n, n_early)); pen[:, :n_linked] = zeta * z[:, None]
    y_e = rng.random((n, n_early)) < cal.sigmoid(theta[:, None] - b_e - pen)
    p_m = cal.sigmoid(-0.5 + gamma * z)
    chose = (rng.random((n, n_linked)) < p_m[:, None]) & ~y_e[:, :n_linked]
    x = chose.sum(1); m = (~y_e[:, :n_linked]).sum(1)
    n_prac = rng.integers(0, max_practice + 1, n)
    b_t = rng.standard_normal((n, n_test))
    y_t = rng.random((n, n_test)) < cal.sigmoid((theta + (eta0 + eta1 * z) * n_prac)[:, None] - b_t)
    return dict(z=z, x=x, m=m, lvl=cal.eap(y_e, b_e), z_hat=(x + 0.5) / (m + 1.0), n=n_prac, b_t=b_t, y_t=y_t)


def em_two_class(x, m, iters=200):
    """Two-class binomial mixture for the linked option count x out of m linked wrong answers."""
    pi, p0, p1 = 0.3, 0.3, 0.7
    last = -1.0
    for _ in range(iters):
        l1 = np.log(pi) + x * np.log(p1) + (m - x) * np.log(1 - p1)
        l0 = np.log(1 - pi) + x * np.log(p0) + (m - x) * np.log(1 - p0)
        r = 1 / (1 + np.exp(l0 - l1))
        pi = r.mean()
        p1 = (r * x).sum() / max((r * m).sum(), 1e-9); p0 = ((1 - r) * x).sum() / max(((1 - r) * m).sum(), 1e-9)
        p1, p0 = np.clip(p1, 1e-4, 1 - 1e-4), np.clip(p0, 1e-4, 1 - 1e-4)
        if abs(pi - last) < 1e-7:
            break
        last = pi
    return pi, p0, p1


def z_post(x, m, par):
    pi, p0, p1 = par
    l1 = np.log(pi) + x * np.log(p1) + (m - x) * np.log(1 - p1)
    l0 = np.log(1 - pi) + x * np.log(p0) + (m - x) * np.log(1 - p0)
    return 1 / (1 + np.exp(l0 - l1))


def c4_fit(d, idx, zp):
    nt = d["y_t"].shape[1]
    rep = lambda v: np.repeat(v[idx], nt)
    zc = zp - zp.mean()
    zc_r = np.repeat(zc, nt)
    x = np.column_stack([np.ones(len(idx) * nt), rep(d["lvl"]), zc_r, rep(d["n"]), zc_r * rep(d["n"])])
    beta, cov = cal.logit_fit(x, d["y_t"][idx].ravel().astype(float), -d["b_t"][idx].ravel(),
                              np.repeat(np.arange(len(idx)), nt))
    return beta[4], cov[4, 4]


def tier2_stat(rng, d, design):
    n = len(d["z"])
    perm = rng.permutation(n)
    if design == "crossfit5":
        folds = np.array_split(perm, 5)
        est = []
        for f in range(5):
            test = folds[f]; train = np.concatenate([folds[g] for g in range(5) if g != f])
            par = em_two_class(d["x"][train], d["m"][train])
            est.append(c4_fit(d, test, z_post(d["x"][test], d["m"][test], par)))
        c = np.array([e[0] for e in est]); v = np.array([e[1] for e in est])
        return (c / v).sum() / (1 / v).sum(), 1 / (1 / v).sum()
    test, train = perm[: n // 5], perm[n // 5:]
    par = em_two_class(d["x"][train], d["m"][train])
    return c4_fit(d, test, z_post(d["x"][test], d["m"][test], par))


def part_t():
    rng = np.random.default_rng(43)
    print("Part T. Tier 2, c4 on the posterior probability z_post from a two-class model fitted on train students")
    big = sim_tracker(rng, 200000, 0.0)
    for name, zz in (("z_hat", big["z_hat"]), ("z_post (EM on these students)", z_post(big["x"], big["m"], em_two_class(big["x"], big["m"])))):
        x = np.column_stack([np.ones(len(zz)), big["lvl"], zz])
        print(f"  slope of z on {name} given theta_hat: {np.linalg.lstsq(x, big['z'].astype(float), rcond=None)[0][2]:.3f} (200000 students, 10 linked)")
    big4 = sim_tracker(rng, 200000, 0.0, n_linked=4)
    zz = z_post(big4["x"], big4["m"], em_two_class(big4["x"], big4["m"]))
    x = np.column_stack([np.ones(len(zz)), big4["lvl"], zz])
    print(f"  slope of z on z_post given theta_hat, 4 linked: {np.linalg.lstsq(x, big4['z'].astype(float), rcond=None)[0][2]:.3f}; "
          f"mean linked wrong answers {big4['m'].mean():.2f} (10 linked: {big['m'].mean():.2f})")
    print("  students | linked | design | eta_1 | logs | rejection | mean c4 | median c4 | SD c4 | mean SE")
    for n, nl, design, eta1, reps in ((4918, 10, "crossfit5", 0.0, 200), (4918, 10, "crossfit5", -0.04, 200),
                                      (4918, 10, "crossfit5", -0.08, 100),
                                      (4918, 10, "split20", 0.0, 150), (4918, 10, "split20", -0.04, 150),
                                      (4918, 4, "crossfit5", 0.0, 100), (4918, 4, "crossfit5", -0.04, 100)):
        cs, ses, rej = [], [], 0
        for _ in range(reps):
            d = sim_tracker(rng, n, eta1, n_linked=nl)
            c, v = tier2_stat(rng, d, design)
            cs.append(c); ses.append(np.sqrt(v)); rej += abs(c / np.sqrt(v)) > Z
        cs = np.array(cs)
        print(f"  {n} | {nl} | {design} | {eta1} | {reps} | {rej / reps:.3f} | {cs.mean():.4f} | {np.median(cs):.4f} | "
              f"{cs.std():.4f} | {np.mean(ses):.4f}", flush=True)


# ---------------------------------------------------------------- Part D
def part_d():
    print("Part D. T1a rule: a subject pair passes when its minority order proportion is at least 0.1 (chosen),")
    print("  among 100 students (the minimum, chosen). Probability that one pair passes, binomial:")
    for p in (0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3):
        print(f"    true minority proportion {p}: {stats.binom.sf(9, 100, p):.3f}")
    for lock, mix in ((0.05, 0.0), (0.05, 0.3)):
        f = (1 - mix) * stats.binom.sf(9, 100, lock) + mix * stats.binom.sf(9, 100, 0.2)
        print(f"  locked pairs at {lock}, {mix:.0%} of pairs at 0.2: expected share passing {f:.3f} against the 0.2 rule (chosen)")


if __name__ == "__main__":
    {"F2": part_f2, "P": part_p, "L": part_l, "L2": part_l2, "P2": part_p2, "Q": part_q, "T": part_t, "D": part_d}[sys.argv[1]]()
