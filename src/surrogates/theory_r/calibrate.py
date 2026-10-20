"""Calibration of the tier 1 reading of surrogate R on learners whose truth is known by construction.

Part A. Binary product probe grid (the probe grid of Hermann et al. 2023, 30 x 30 over z_c and z_s).
  Statistic T_s = mean over probes of y_hat * sign(z_s). A learner invariant to z_s gives E[T_s] = 0
  by the product design; a learner that answers by the cue with probability lam gives E[T_s] = lam.
  Two tests: the item level sign test on the pairs (z_c, +|z_s|), (z_c, -|z_s|) within one learner, and
  the seed level one sample t test of T_s across learners.
Part B. K option lookup grid (each probe a distinct surface; the cue names one wrong option).
  Statistic: share of wrong answers on conflict probes that equal the option the cue names.
  Null share 1/(K-1) holds by the balanced design when the learner ignores the cue map.
Part C. The threshold in surfaces per structure for a kernel learner. Closed form rule share on conflict
  probes m kc / (m kc + ks) against kernel regression and gradient descent on the same design.

Run: python -I calibrate.py   (numpy and scipy only, about 3.5 minutes on a laptop CPU; output in calibrate_output.txt)
"""
import numpy as np
from scipy import stats

ALPHA = 0.05
N_REP = 2000          # replicates of one experiment (one experiment = N_SEEDS learners)
N_SEEDS = 10


# ---------------------------------------------------------------- Part A
def grid_binary():
    v = np.linspace(-3.0, 3.0, 30)          # 30 values, symmetric, none at zero
    zc, zs = np.meshgrid(v, v, indexing="ij")
    return zc.ravel(), zs.ravel()


def answers_binary(rng, zc, zs, lam, delta, a=3.0):
    """One learner. With prob lam the answer is the cue sign(z_s); else Bernoulli(sigmoid(a z_c + delta z_s))."""
    p_pos = 1.0 / (1.0 + np.exp(-(a * zc + delta * zs)))
    rule = np.where(rng.random(zc.size) < p_pos, 1, -1)
    cue = np.sign(zs).astype(int)
    use_cue = rng.random(zc.size) < lam
    return np.where(use_cue, cue, rule)


def part_a(rng):
    zc, zs = grid_binary()
    # pair index: the probe (zc, zs) with zs > 0 is paired with (zc, -zs)
    pos = zs > 0
    order_pos = np.lexsort((np.abs(zs[pos]), zc[pos]))
    order_neg = np.lexsort((np.abs(zs[~pos]), zc[~pos]))
    rows = []
    for sigma_delta in (0.0, 0.3):
        for lam in (0.0, 0.02, 0.05, 0.1, 0.2):
            rej_item, rej_seed_t, rej_seed_sign, ts_all = 0, 0, 0, []
            for _ in range(N_REP):
                ts = np.empty(N_SEEDS)
                for k in range(N_SEEDS):
                    delta = rng.normal(0.0, sigma_delta)
                    y = answers_binary(rng, zc, zs, lam, delta)
                    ts[k] = np.mean(y * np.sign(zs))
                    if k == 0:   # item level test on the first learner only
                        d = y[pos][order_pos] - y[~pos][order_neg]
                        n_plus, n_nz = np.sum(d > 0), np.sum(d != 0)
                        if n_nz > 0 and stats.binomtest(n_plus, n_nz, 0.5, alternative="greater").pvalue < ALPHA:
                            rej_item += 1
                ts_all.append(ts)
                if np.std(ts) > 0 and stats.ttest_1samp(ts, 0.0, alternative="greater").pvalue < ALPHA:
                    rej_seed_t += 1
                if stats.binomtest(int(np.sum(ts > 0)), N_SEEDS, 0.5, alternative="greater").pvalue < ALPHA:
                    rej_seed_sign += 1
            ts_all = np.concatenate(ts_all)
            rows.append((sigma_delta, lam, ts_all.mean(), np.median(ts_all), ts_all.std(),
                         np.quantile(ts_all, 0.025), np.quantile(ts_all, 0.975),
                         rej_item / N_REP, rej_seed_t / N_REP, rej_seed_sign / N_REP))
    print("Part A. Binary product grid, 900 probes per learner, %d learners per experiment, %d experiments" % (N_SEEDS, N_REP))
    print("sigma_delta lam | T_s mean median sd q2.5 q97.5 | reject: item sign test, seed t test, seed sign test")
    for r in rows:
        print("%.1f %.2f | %.4f %.4f %.4f %.4f %.4f | %.3f %.3f %.3f" % r)


# ---------------------------------------------------------------- Part B
def part_b(rng, K=4, C=8, R=5, eps=0.2, eta=0.1):
    """C structures, each with R conflict probes per wrong option (distinct surfaces): C*(K-1)*R probes.
    Rule learner: right with prob 1-eps; errors drawn from a structured bias q_c (Dirichlet(0.3)) over wrong options.
    Surface sensitivity unrelated to the cue: with prob eta the answer is u(s), an option drawn per surface
    independently of the cue map. Planted cue use: with prob lam the answer is h(s), the option the cue names."""
    n_probe = C * (K - 1) * R
    rows = []
    for lam in (0.0, 0.05, 0.1, 0.2):
        rej_item, rej_seed, shares = 0, 0, []
        for _ in range(N_REP):
            sh = np.empty(N_SEEDS)
            for k in range(N_SEEDS):
                right = np.repeat(np.arange(C) % K, (K - 1) * R)               # r(c)
                wrong_opts = np.array([[o for o in range(K) if o != r] for r in np.arange(C) % K])
                named = np.concatenate([np.repeat(wrong_opts[c], R) for c in range(C)])   # h(s), balanced
                q = rng.dirichlet(0.3 * np.ones(K - 1), size=C)
                err_idx = np.concatenate([rng.choice(K - 1, size=(K - 1) * R, p=q[c]) for c in range(C)])
                err_ans = wrong_opts[np.repeat(np.arange(C), (K - 1) * R), err_idx]
                ans = np.where(rng.random(n_probe) < 1 - eps, right, err_ans)
                u = rng.integers(0, K, size=n_probe)                         # surface specific, cue blind
                ans = np.where(rng.random(n_probe) < eta, u, ans)
                ans = np.where(rng.random(n_probe) < lam, named, ans)
                wrong = ans != right
                hits = np.sum(ans[wrong] == named[wrong])
                n_w = np.sum(wrong)
                sh[k] = hits / n_w if n_w else 1.0 / (K - 1)
                if k == 0 and n_w and stats.binomtest(hits, n_w, 1.0 / (K - 1), alternative="greater").pvalue < ALPHA:
                    rej_item += 1
            shares.append(sh)
            if stats.ttest_1samp(sh, 1.0 / (K - 1), alternative="greater").pvalue < ALPHA:
                rej_seed += 1
        shares = np.concatenate(shares)
        rows.append((lam, shares.mean(), np.median(shares), shares.std(), np.quantile(shares, 0.025),
                     np.quantile(shares, 0.975), rej_item / N_REP, rej_seed / N_REP))
    print("\nPart B. K=%d options, %d structures, %d conflict probes per learner, eps=%.2f, eta=%.2f; null share %.3f"
          % (K, C, n_probe, eps, eta, 1.0 / (K - 1)))
    print("lam | hit share among wrong: mean median sd q2.5 q97.5 | reject: item binomial, seed t test")
    for r in rows:
        print("%.2f | %.4f %.4f %.4f %.4f %.4f | %.3f %.3f" % r)


# ---------------------------------------------------------------- Part C
def part_c(K=4, ks=4.0, kc=1.0, kcs=0.5):
    """Training cells: K structures x m surfaces, each surface paired with one structure, key r(c) = h(s) = c.
    Kernel k(x,x') = ks 1[s=s'] + kc 1[c=c'] + kcs 1[s=s' and c=c']. Conflict probe: structure c with a
    surface trained on structure c' != c. Rule share = score(r(c)) / (score(r(c)) + score(r(c')))."""
    print("\nPart C. Kernel learner, ks/kc = %.1f, kcs = %.1f; closed form share m kc/(m kc + ks)" % (ks / kc, kcs))
    print("m | closed form | kernel regression | gradient descent on features (from zero)")
    for m in (1, 2, 4, 8, 16):
        cs = np.repeat(np.arange(K), m)
        ss = np.arange(K * m)
        Y = np.eye(K)[cs]
        G = ks * (ss[:, None] == ss[None, :]) + kc * (cs[:, None] == cs[None, :]) \
            + kcs * ((ss[:, None] == ss[None, :]) & (cs[:, None] == cs[None, :]))
        A = np.linalg.solve(G, Y)
        c_new, s_probe = 0, m      # surface index m belongs to structure 1
        kvec = ks * (ss == s_probe) + kc * (cs == c_new) + kcs * ((ss == s_probe) & (cs == c_new))
        f = kvec @ A
        share_kr = f[0] / (f[0] + f[1])
        # explicit features: sqrt(ks) onehot(s), sqrt(kc) onehot(c), sqrt(kcs) onehot(cell); squared loss
        n_cell = K * m
        Phi = np.hstack([np.sqrt(ks) * np.eye(K * m), np.sqrt(kc) * np.eye(K)[cs], np.sqrt(kcs) * np.eye(n_cell)])
        W = np.zeros((Phi.shape[1], K))
        lr = 0.5 / np.linalg.eigvalsh(Phi.T @ Phi).max()
        for _ in range(20000):
            W -= lr * Phi.T @ (Phi @ W - Y)
        phi_probe = np.concatenate([np.sqrt(ks) * np.eye(K * m)[s_probe], np.sqrt(kc) * np.eye(K)[c_new], np.zeros(n_cell)])
        g = phi_probe @ W
        share_gd = g[0] / (g[0] + g[1])
        print("%2d | %.4f | %.4f | %.4f" % (m, m * kc / (m * kc + ks), share_kr, share_gd))


if __name__ == "__main__":
    rng = np.random.default_rng(20261019)
    part_a(rng)
    part_b(rng)
    part_c()
