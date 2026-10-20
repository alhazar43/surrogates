"""Checks for the fix attempt of theory/R.md (unit TR, attempt 1). Synthetic records only.

Part D. Answers of a fixed feature learner on the conflict cells of the R5 design, trained by full batch gradient
  descent from zero under squared loss and under cross entropy. Symmetric design (one kappa_s for all surfaces)
  and an asymmetric design (kappa_s drawn per surface).
Part E. Power of the seed level one sided t test by the noncentral t distribution, with the T_s spread of
  calibrate_output.txt (sigma_delta = 0.3), and the seed count for power 0.8.
Part F. Expectation of the K option statistic w as a function of lam, eps and eta, and the item level power
  at lam = 0.1 for three slip rates.
Part G. Calibration of the Lambda_c reading (binary and K option designs), the roles of c and s exchanged.
Part H. Part A of calibrate.py on the probe grid of Hermann et al. 2023 (plus and minus 3 mu, mu = 1.2816).

Run: python -I fix_checks.py   (numpy and scipy only; output in fix_checks_output.txt)
"""
import time
import numpy as np
from scipy import stats

ALPHA = 0.05
N_SEEDS = 10


# ---------------------------------------------------------------- Part D
def design(K, m, ks_per_surface, kc, kcs):
    cs = np.repeat(np.arange(K), m)                 # structure of each training cell, key r(c) = c
    ss = np.arange(K * m)                           # one surface per training cell
    n = K * m
    Phi = np.hstack([np.sqrt(ks_per_surface)[:, None] * np.eye(n), np.sqrt(kc) * np.eye(K)[cs],
                     np.sqrt(kcs) * np.eye(n)])
    # conflict cells: structure c with every trained surface s' whose structure c' != c
    probes, keys, cues = [], [], []
    for c in range(K):
        for s in range(n):
            if cs[s] != c:
                probes.append(np.concatenate([np.sqrt(ks_per_surface[s]) * np.eye(n)[s], np.sqrt(kc) * np.eye(K)[c],
                                              np.zeros(n)]))
                keys.append(c)
                cues.append(cs[s])
    return Phi, np.eye(K)[cs], np.array(probes), np.array(keys), np.array(cues)


def train_ce(Phi, Y, steps):
    """Full batch gradient descent from zero on the summed cross entropy, step 2 / lambda_max(Phi Phi^T)."""
    W = np.zeros((Phi.shape[1], Y.shape[1]))
    lr = 2.0 / np.linalg.eigvalsh(Phi.T @ Phi).max()
    out = {}
    for t in range(1, max(steps) + 1):
        F = Phi @ W
        P = np.exp(F - F.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        W -= lr * Phi.T @ (P - Y)
        if t in steps:
            out[t] = W.copy()
    return out


def read_answers(W, probes, keys, cues):
    F = probes @ W
    ans = F.argmax(1)
    P = np.exp(F - F.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    idx = np.arange(len(keys))
    rule = np.mean(ans == keys)
    cue = np.mean(ans == cues)
    p_rule = np.mean(P[idx, keys])
    share2 = np.mean(P[idx, keys] / (P[idx, keys] + P[idx, cues]))
    tie = np.mean(np.isclose(F[idx, keys], F[idx, cues], rtol=1e-9, atol=1e-12))
    return rule, cue, tie, p_rule, share2


def part_d():
    K, kc, kcs = 4, 1.0, 0.5
    steps = (200, 5000)
    print("Part D. Fixed features, full batch gradient descent from zero, K = %d, kappa_c = 1, kappa_cs = 0.5" % K)
    print("design loss m steps | argmax rule share, argmax cue share, tie share | softmax p(rule), p(rule)/(p(rule)+p(cue))")
    for name, ks_fn in (("symmetric ks=4", lambda rng, n: 4.0 * np.ones(n)),
                        ("asymmetric ks~4*LogN(0,0.5)", lambda rng, n: 4.0 * np.exp(rng.normal(0, 0.5, n)))):
        rng = np.random.default_rng(7)
        for m in (2, 3, 4, 5, 8, 16):
            ks = ks_fn(rng, K * m)
            Phi, Y, probes, keys, cues = design(K, m, ks, kc, kcs)
            # squared loss minimum norm solution by kernel regression (same as gradient descent from zero)
            G = Phi @ Phi.T
            A = np.linalg.solve(G, Y)
            F = probes @ Phi.T @ A
            idx = np.arange(len(keys))
            print("%s sq-minnorm m=%2d       | %.3f %.3f %.3f | score ratio %.4f"
                  % (name, m, np.mean(F.argmax(1) == keys), np.mean(F.argmax(1) == cues),
                     np.mean(np.isclose(F[idx, keys], F[idx, cues])),
                     np.mean(F[idx, keys] / (F[idx, keys] + F[idx, cues]))))
            Ws = train_ce(Phi, Y, steps)
            for t in steps:
                r = read_answers(Ws[t], probes, keys, cues)
                print("%s ce        m=%2d t=%4d | %.3f %.3f %.3f | %.4f %.4f" % ((name, m, t) + r))


# ---------------------------------------------------------------- Part E
def power_t(lam, sd, n):
    df = n - 1
    crit = stats.t.ppf(1 - ALPHA, df)
    return stats.nct.sf(crit, df, lam / (sd / np.sqrt(n)))


def part_e():
    sds = {0.02: 0.0524, 0.05: 0.0506, 0.10: 0.0491}     # calibrate_output.txt, sigma_delta = 0.3
    print("\nPart E. Seed level one sided t test, alpha 0.05, noncentral t, T_s sd from calibrate_output.txt (sigma_delta 0.3)")
    print("lam sd | power at n = 5, 10, 20, 40 | n for power 0.8")
    for lam, sd in sds.items():
        pw = [power_t(lam, sd, n) for n in (5, 10, 20, 40)]
        n80 = next(n for n in range(3, 1000) if power_t(lam, sd, n) >= 0.8)
        print("%.2f %.4f | %.3f %.3f %.3f %.3f | %d" % ((lam, sd) + tuple(pw) + (n80,)))


# ---------------------------------------------------------------- Part F
def ew(lam, eps, eta, K=4):
    num = lam + (1 - lam) * (eta / K + (1 - eta) * eps / (K - 1))
    den = lam + (1 - lam) * (eta * (K - 1) / K + (1 - eta) * eps)
    return num, den, num / den


def k_option_answers(rng, n_learn, lam, eps, eta, K=4, C=8, R=5, swap=False):
    """Part B of calibrate.py, vectorized over learners. With swap the roles of key and cue are exchanged:
    the learner reads the surface (right on h with prob 1 - eps), and lam plants use of the structure."""
    n_probe = C * (K - 1) * R
    base = np.repeat(np.arange(C) % K, (K - 1) * R)                       # the option the learner's base reads
    others = np.array([[o for o in range(K) if o != b] for b in np.arange(C) % K])
    other = np.concatenate([np.repeat(others[c], R) for c in range(C)])   # the balanced other option
    grp = np.repeat(np.arange(C), (K - 1) * R)
    q = rng.dirichlet(0.3 * np.ones(K - 1), size=(n_learn, C))            # biased slips per group
    cum = np.cumsum(q, axis=2)[:, grp, :]                                 # n_learn x n_probe x K-1
    err_idx = (rng.random((n_learn, n_probe, 1)) > cum).sum(2)
    err_idx = np.minimum(err_idx, K - 2)
    err = others[grp][np.arange(n_probe)[None, :], err_idx]
    ans = np.where(rng.random((n_learn, n_probe)) < 1 - eps, base[None, :], err)
    u = rng.integers(0, K, size=(n_learn, n_probe))
    ans = np.where(rng.random((n_learn, n_probe)) < eta, u, ans)
    ans = np.where(rng.random((n_learn, n_probe)) < lam, other[None, :], ans)
    off = ans != base[None, :]                    # off the base answer (wrong answers, or non-cue answers if swap)
    hits = np.sum(off & (ans == other[None, :]), 1)
    n_off = np.sum(off, 1)
    share = np.where(n_off > 0, hits / np.maximum(n_off, 1), 1.0 / (K - 1))
    p_item = stats.binom.sf(hits - 1, n_off, 1.0 / (K - 1))
    return share, p_item


def part_f(rng):
    print("\nPart F. K option statistic w, E[w] by the ratio of expectations, K = 4, eta = 0.1; item level binomial test, one learner")
    print("eps lam | E[num] E[den] E[w] | simulated mean w | item level reject (4000 learners)")
    for eps in (0.1, 0.2, 0.4):
        for lam in (0.0, 0.1):
            num, den, r = ew(lam, eps, 0.1)
            share, p = k_option_answers(rng, 4000, lam, eps, 0.1)
            print("%.1f %.2f | %.4f %.4f %.4f | %.4f | %.3f" % (eps, lam, num, den, r, share.mean(), np.mean(p < ALPHA)))


# ---------------------------------------------------------------- Part G and H
def binary_block(rng, zc, zs, lam, sigma_delta, n_exp, swap=False, a=3.0):
    """n_exp experiments of N_SEEDS learners. Without swap: Part A of calibrate.py (statistic T_s).
    With swap: the base reads z_s, lam plants the answer sign(z_c), statistic T_c = mean y sign(z_c)."""
    x_read, x_other = (zs, zc) if swap else (zc, zs)
    n_l = n_exp * N_SEEDS
    delta = rng.normal(0.0, sigma_delta, size=(n_l, 1))
    p_pos = 1.0 / (1.0 + np.exp(-(a * x_read[None, :] + delta * x_other[None, :])))
    y = np.where(rng.random((n_l, x_read.size)) < p_pos, 1, -1)
    y = np.where(rng.random((n_l, x_read.size)) < lam, np.sign(x_other)[None, :].astype(int), y)
    T = np.mean(y * np.sign(x_other)[None, :], 1).reshape(n_exp, N_SEEDS)
    # item level sign test on the first learner of each experiment, pairs (+|x_other|, -|x_other|)
    pos = x_other > 0
    o_pos = np.lexsort((np.abs(x_other[pos]), x_read[pos]))
    o_neg = np.lexsort((np.abs(x_other[~pos]), x_read[~pos]))
    y1 = y.reshape(n_exp, N_SEEDS, -1)[:, 0, :]
    d = y1[:, pos][:, o_pos] - y1[:, ~pos][:, o_neg]
    n_plus, n_nz = np.sum(d > 0, 1), np.sum(d != 0, 1)
    rej_item = np.mean(stats.binom.sf(n_plus - 1, n_nz, 0.5) < ALPHA)
    rej_t = np.mean(stats.ttest_1samp(T, 0.0, axis=1, alternative="greater").pvalue < ALPHA)
    return T.ravel(), rej_item, rej_t


def part_g(rng):
    v = np.linspace(-3.0, 3.0, 30)
    zc, zs = [g.ravel() for g in np.meshgrid(v, v, indexing="ij")]
    print("\nPart G. Lambda_c calibration. Binary grid of calibrate.py, base reads z_s with slope 3 and a seed level dependence")
    print("on z_c of sd sigma_delta; with prob lam_c the answer is sign(z_c). Statistic T_c. 2000 experiments of 10 learners.")
    print("sigma_delta lam_c | T_c mean sd | reject: item sign test, seed t test")
    for sd in (0.0, 0.3):
        for lam in (0.0, 0.02, 0.05, 0.1):
            T, ri, rt = binary_block(rng, zc, zs, lam, sd, 2000, swap=True)
            print("%.1f %.2f | %.4f %.4f | %.3f %.3f" % (sd, lam, T.mean(), T.std(), ri, rt))
    print("K option. Base reads the surface: answers h(s) with prob 0.8, biased slips Dirichlet(0.3) per surface group,")
    print("eta 0.1 cue blind noise; with prob lam_c the answer is r(c). w_c = share of non-cue answers equal to r(c), null 1/3.")
    print("lam_c | w_c mean sd | reject: item binomial, seed t test (2000 experiments of 10 learners)")
    for lam in (0.0, 0.05, 0.1, 0.2):
        share, p = k_option_answers(rng, 2000 * N_SEEDS, lam, 0.2, 0.1, swap=True)
        S = share.reshape(2000, N_SEEDS)
        rej_item = np.mean(p.reshape(2000, N_SEEDS)[:, 0] < ALPHA)
        rej_t = np.mean(stats.ttest_1samp(S, 1.0 / 3, axis=1, alternative="greater").pvalue < ALPHA)
        print("%.2f | %.4f %.4f | %.3f %.3f" % (lam, share.mean(), share.std(), rej_item, rej_t))


def part_h(rng):
    v = np.linspace(-3.0 * 1.2816, 3.0 * 1.2816, 30)
    zc, zs = [g.ravel() for g in np.meshgrid(v, v, indexing="ij")]
    print("\nPart H. Part A on the grid of Hermann et al., 30 values over plus and minus 3 mu, mu = 1.2816 on both axes.")
    print("sigma_delta lam | T_s mean sd | reject: item sign test, seed t test (2000 experiments of 10 learners)")
    for lam in (0.0, 0.05, 0.1):
        T, ri, rt = binary_block(rng, zc, zs, lam, 0.3, 2000)
        print("0.3 %.2f | %.4f %.4f | %.3f %.3f" % (lam, T.mean(), T.std(), ri, rt))


if __name__ == "__main__":
    t0 = time.time()
    part_d()
    part_e()
    rng = np.random.default_rng(20261019)
    part_f(rng)
    part_g(rng)
    part_h(rng)
    print("\nRun time %.1f s" % (time.time() - t0))
