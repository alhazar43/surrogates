"""Calibration of the readings of surrogate S on synthetic learners with known truth.

Part F. The steady-state floor of a scalar random-walk filter (theory/S.md, Proposition 2).
Part A. Tier 1 reading: the Mantel-Haenszel association of two content-linked wrong options,
        stratified on a rest score (theory/S.md, section 4.1).
Part B. Tier 2 reading: does a tracked misconception state predict the response to later
        practice beyond the level. Naive gain regression against the nested interaction test
        (theory/S.md, section 4.2).

Run: python -I calibrate.py        (Parts F, A, B; output calibrate_output.txt)
     python -I calibrate.py extra  (Parts A2, B2; output calibrate_output_extra.txt)
"""
import sys
import numpy as np
from scipy import stats

SEED = 20261019
ALPHA = 0.05  # chosen test level


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- Part F
def filter_floor(q, info):
    """Fixed point of P = 1 / (1/(P+q) + I) and the iterated recursion."""
    p_closed = (-info * q + np.sqrt((info * q) ** 2 + 4 * info * q)) / (2 * info)
    p = 1.0
    for _ in range(10000):
        p = 1.0 / (1.0 / (p + q) + info)
    window = 1.0 / np.sqrt(q * info)
    return p_closed, p, np.sqrt(p), window


# ---------------------------------------------------------------- Part A
def mh_test(a, b, strata):
    """Cochran-Mantel-Haenszel test of A independent of B given strata. Returns (log OR_MH, p)."""
    num = den = 0.0
    s_obs = s_exp = s_var = 0.0
    for s in np.unique(strata):
        m = strata == s
        n = float(m.sum())
        if n < 2:
            continue
        a1, b1 = a[m], b[m]
        n11 = float(np.sum(a1 & b1)); n10 = float(np.sum(a1 & ~b1))
        n01 = float(np.sum(~a1 & b1)); n00 = float(np.sum(~a1 & ~b1))
        num += n11 * n00 / n
        den += n10 * n01 / n
        r1 = n11 + n10; c1 = n11 + n01
        s_obs += n11
        s_exp += r1 * c1 / n
        s_var += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1))
    if s_var <= 0 or num == 0 or den == 0:
        return np.nan, 1.0
    chi2 = (s_obs - s_exp) ** 2 / s_var
    return np.log(num / den), stats.chi2.sf(chi2, 1)


def simulate_linked(rng, n_students, gamma, delta, pi=0.3, zeta=0.5, n_rest=20, n_strata=5):
    """One synthetic log for two linked items j, k and n_rest unlinked items.

    theta ~ N(0,1). z ~ Bernoulli(pi) is the planted misconception (z = 0 for all when gamma = 0).
    P(correct on j or k) = sigmoid(theta - zeta * z).  Given wrong, the linked option m has
    logit delta * (-theta) + gamma * z against two other wrong options with logits 0 and -0.5.
    delta > 0 makes weaker students choose m more often with no misconception: the hard null.
    """
    theta = rng.standard_normal(n_students)
    z = (rng.random(n_students) < pi) & (gamma > 0)
    b_rest = rng.standard_normal(n_rest)
    rest = (rng.random((n_students, n_rest)) < sigmoid(theta[:, None] - b_rest[None, :])).sum(1)
    chose_m = []
    wrong = []
    for _ in range(2):
        w = rng.random(n_students) >= sigmoid(theta - zeta * z)
        logits = np.stack([-delta * theta + gamma * z, np.zeros(n_students), np.full(n_students, -0.5)], 1)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        chose_m.append(rng.random(n_students) < p[:, 0])
        wrong.append(w)
    both = wrong[0] & wrong[1]
    a, b = chose_m[0][both], chose_m[1][both]
    edges = np.quantile(rest, np.linspace(0, 1, n_strata + 1)[1:-1])
    strata = np.digitize(rest[both], edges)
    return a, b, strata, both.sum()


def part_a(rng, reps=500):
    rows = []
    for n_students in (1000, 3000):
        for label, gamma, delta in (("null, no ability dependence", 0.0, 0.0),
                                    ("null, ability dependence 1.0", 0.0, 1.0),
                                    ("planted gamma 0.7, dependence 1.0", 0.7, 1.0),
                                    ("planted gamma 1.4, dependence 1.0", 1.4, 1.0)):
            rej_mh = rej_crude = 0
            lors, ns = [], []
            for _ in range(reps):
                a, b, strata, nb = simulate_linked(rng, n_students, gamma, delta)
                lor, p = mh_test(a, b, strata)
                _, pc = mh_test(a, b, np.zeros_like(strata))
                rej_mh += p < ALPHA
                rej_crude += pc < ALPHA
                lors.append(lor); ns.append(nb)
            rows.append((n_students, label, np.median(ns), np.nanmean(lors), rej_mh / reps, rej_crude / reps))
    return rows


# ---------------------------------------------------------------- Part B
GRID = np.linspace(-4, 4, 41)
PRIOR = np.exp(-0.5 * GRID ** 2)


def eap(y, b):
    """Posterior mean of theta on a grid, N(0,1) prior, Rasch likelihood. y, b: (n, items)."""
    eta = GRID[None, None, :] - b[:, :, None]
    ll = (y[:, :, None] * eta).sum(1) - np.logaddexp(0.0, eta).sum(1)
    w = np.exp(ll - ll.max(1, keepdims=True)) * PRIOR[None, :]
    return (w * GRID).sum(1) / w.sum(1)


def logit_fit(x, y, offset, cluster):
    """Newton logistic regression with a cluster-robust (sandwich) covariance."""
    beta = np.zeros(x.shape[1])
    for _ in range(30):
        mu = sigmoid(x @ beta + offset)
        h = (x * (mu * (1 - mu))[:, None]).T @ x
        step = np.linalg.solve(h, x.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = sigmoid(x @ beta + offset)
    score = x * (y - mu)[:, None]
    g = np.zeros((cluster.max() + 1, x.shape[1]))
    np.add.at(g, cluster, score)
    hinv = np.linalg.inv((x * (mu * (1 - mu))[:, None]).T @ x)
    cov = hinv @ (g.T @ g) @ hinv
    return beta, cov


def auc(score, y):
    r = stats.rankdata(score)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def simulate_tracker(rng, n_students, eta1, eta0=0.05, pi=0.3, zeta=0.5, gamma=1.4,
                     n_early=30, n_linked=10, max_practice=10, n_test=5):
    """Early phase: n_early items, the first n_linked carry the misconception option.
    Later phase: n ~ Uniform{0..max_practice} practice attempts, assigned at random,
    then n_test unlinked test items. Learning rate eta = eta0 + eta1 * z."""
    theta = rng.standard_normal(n_students)
    z = rng.random(n_students) < pi
    b_e = rng.standard_normal((n_students, n_early))
    pen = np.zeros((n_students, n_early)); pen[:, :n_linked] = zeta * z[:, None]
    y_e = rng.random((n_students, n_early)) < sigmoid(theta[:, None] - b_e - pen)
    # wrong-option choice on linked items
    p_m = sigmoid(-0.5 + gamma * z)
    chose_m = (rng.random((n_students, n_linked)) < p_m[:, None]) & ~y_e[:, :n_linked]
    n_wrong_l = (~y_e[:, :n_linked]).sum(1)
    z_hat = (chose_m.sum(1) + 0.5) / (n_wrong_l + 1.0)  # share of linked wrong answers on option m
    lvl = eap(y_e, b_e)
    score = y_e.mean(1)
    n_prac = rng.integers(0, max_practice + 1, n_students)
    eta = eta0 + eta1 * z
    b_t = rng.standard_normal((n_students, n_test))
    y_t = rng.random((n_students, n_test)) < sigmoid((theta + eta * n_prac)[:, None] - b_t)
    return dict(z_hat=z_hat, lvl=lvl, score=score, n=n_prac, b_t=b_t, y_t=y_t)


def nested_design(d):
    ns, nt = d["y_t"].shape
    rep = lambda v: np.repeat(v, nt)
    zc = d["z_hat"] - d["z_hat"].mean()
    x_full = np.column_stack([np.ones(ns * nt), rep(d["lvl"]), rep(zc), rep(d["n"]), rep(zc * d["n"])])
    return x_full, d["y_t"].ravel().astype(float), -d["b_t"].ravel(), np.repeat(np.arange(ns), nt)


def part_b(rng, reps=200, n_students=2000):
    rows = []
    for label, eta1 in (("null, eta homogeneous", 0.0), ("planted eta1 -0.04 per attempt", -0.04)):
        rej_naive = rej_nested = 0
        coefs, dauc = [], []
        for _ in range(reps):
            d = simulate_tracker(rng, n_students, eta1)
            # naive: gain on score regressed on score and z_hat (OLS t-test on z_hat)
            gain = d["y_t"].mean(1) - d["score"]
            x = np.column_stack([np.ones(n_students), d["score"], d["z_hat"]])
            beta, res, *_ = np.linalg.lstsq(x, gain, rcond=None)
            r = gain - x @ beta
            s2 = r @ r / (n_students - 3)
            se = np.sqrt(s2 * np.linalg.inv(x.T @ x)[2, 2])
            rej_naive += 2 * stats.t.sf(abs(beta[2] / se), n_students - 3) < ALPHA
            # nested: interaction of z_hat with practice, level and z_hat main effects in the model
            xf, y, off, cl = nested_design(d)
            bf, cov = logit_fit(xf, y, off, cl)
            zstat = bf[4] / np.sqrt(cov[4, 4])
            rej_nested += 2 * stats.norm.sf(abs(zstat)) < ALPHA
            coefs.append(bf[4])
            # Delta AUC on separate test students: fit on this log, score a fresh log
            dt = simulate_tracker(rng, n_students, eta1)
            xt, yt, offt, _ = nested_design(dt)
            br, _ = logit_fit(xf[:, [0, 1, 3]], y, off, cl)
            dauc.append(auc(xt @ bf + offt, yt) - auc(xt[:, [0, 1, 3]] @ br + offt, yt))
        dauc = np.array(dauc)
        rows.append((label, rej_naive / reps, rej_nested / reps, np.mean(coefs),
                     dauc.mean(), np.median(dauc), np.quantile(dauc, 0.05), np.quantile(dauc, 0.95)))
    return rows


def part_a_rest(rng, reps=300, n_students=3000):
    """Hard null and planted case with longer rest scores and 10 strata.
    Eedi tasks 3 and 4 hold 1,382,727 / 4,918 = 281 answers per student on average."""
    rows = []
    for n_rest in (20, 100, 250):
        for label, gamma in (("null, ability dependence 1.0", 0.0), ("planted gamma 0.7, dependence 1.0", 0.7)):
            rej = 0
            for _ in range(reps):
                a, b, strata, _ = simulate_linked(rng, n_students, gamma, 1.0, n_rest=n_rest, n_strata=10)
                rej += mh_test(a, b, strata)[1] < ALPHA
            rows.append((n_rest, label, rej / reps))
    return rows


def extra():
    rng = np.random.default_rng(SEED + 1)
    print(f"seed {SEED + 1}, alpha {ALPHA}")
    print("\nPart A2. MH test, 3000 students, 10 strata, 300 replicates per row")
    print("  rest items | case | rejection MH")
    for row in part_a_rest(rng):
        print("  {} | {} | {:.3f}".format(*row))
    print("\nPart B2. Tracker reading, 4918 students per log, 100 replicates per row")
    print("  case | rejection naive gain | rejection nested interaction | mean interaction coef | dAUC mean | median | 5% | 95%")
    for row in part_b(rng, reps=100, n_students=4918):
        print("  {} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(*row))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "extra":
        return extra()
    rng = np.random.default_rng(SEED)
    print(f"seed {SEED}, alpha {ALPHA}, python {sys.version.split()[0]}, numpy {np.__version__}")
    print("\nPart F. Filter floor, random walk variance q per attempt, Fisher information I per answer")
    for q, info in ((0.01, 0.25), (0.0025, 0.25), (0.01, 0.1)):
        pc, pi_, sd, win = filter_floor(q, info)
        print(f"  q={q} I={info}: P closed {pc:.5f}, P iterated {pi_:.5f}, posterior SD {sd:.4f} logit, window {win:.1f} attempts")
    print("\nPart A. Linked-option MH test, 500 replicates per row")
    print("  students | case | median n both wrong | mean log OR_MH | rejection MH | rejection crude")
    for row in part_a(rng):
        print("  {} | {} | {:.0f} | {:.3f} | {:.3f} | {:.3f}".format(*row))
    print("\nPart B. Tracker reading, 2000 students per log, 200 replicates per row")
    print("  case | rejection naive gain | rejection nested interaction | mean interaction coef | dAUC mean | median | 5% | 95%")
    for row in part_b(rng):
        print("  {} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(*row))


if __name__ == "__main__":
    main()
