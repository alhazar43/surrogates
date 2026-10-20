"""Calibration of the tier 1 reading of surrogate E on tiny synthetic learners.

The reading is the response ratio rho of theory/E.md section 1: the mean move toward
the key on practiced probes over the mean move on conflict siblings, after one plain
gradient step on one conflict case with its key shown. Every learner here is built by
hand, so its true rho is known from its weights.

Run with: python -I calibrate.py   (3 min 41 s on the PC in the RevE rerun)
"""
import numpy as np
from scipy import stats

D, H = 16, 6            # input dimension, hidden width of the two-layer learner
SIGMA = 0.3             # probe noise, standard deviation per input coordinate
ETA = 0.01              # step size of the probe step
N_CASE, N_PROBE = 10, 20  # conflict cases per learner, probes per group per case
S_SEED = 8              # learners per arm (the unit of inference)
R_REP = 400             # replicates per condition for false positive rate and power
ALPHA = 0.05            # chosen test level
E1, E2, E3 = np.eye(D)[0], np.eye(D)[1], np.eye(D)[2]  # cue, rule, unrelated directions


def sig(t):
    return 1.0 / (1.0 + np.exp(-t))


# ---------- learners: z(X) is the log odds of answer +1 ----------
class Linear:
    def __init__(self, w):
        self.w = w.copy()

    def z(self, X):
        return X @ self.w

    def step(self, x, y, eta):
        g = -y * (1.0 - sig(y * self.z(x[None])[0]))   # dL/dz for L = softplus(-y z)
        new = Linear(self.w)
        new.w -= eta * g * x
        return new

    def kernel(self, xb, xe):
        return xb @ xe


class TwoLayer:
    def __init__(self, a, W):
        self.a, self.W = a.copy(), W.copy()

    def z(self, X):
        return (X @ self.W.T) @ self.a

    def step(self, x, y, eta):
        g = -y * (1.0 - sig(y * self.z(x[None])[0]))
        new = TwoLayer(self.a, self.W)
        new.a -= eta * g * (self.W @ x)
        new.W -= eta * g * np.outer(self.a, x)
        return new

    def kernel(self, xb, xe):  # hold part + use part (B32 split)
        return (self.W @ xb) @ (self.W @ xe) + (self.a @ self.a) * (xb @ xe)


def two_layer(rng, s_u, a_u, s_v, a_v, tau=0.5, tau_a=0.3):
    W = rng.normal(0, tau / np.sqrt(D), (H, D))
    a = rng.normal(0, tau_a, H)
    W[0], W[1] = s_u * E1, s_v * E2
    a[0], a[1] = a_u, a_v
    return TwoLayer(a, W)


def linear(rng, w_u, w_v, tau=0.1):
    return Linear(w_u * E1 + w_v * E2 + rng.normal(0, tau, D))


# ---------- probes ----------
def probes(rng, n):
    """Practiced (cue = rule), conflict siblings (cue = -rule, key follows rule), unrelated."""
    y = rng.choice([-1.0, 1.0], (3, n))
    noise = rng.normal(0, SIGMA, (3, n, D))
    pr = y[0][:, None] * (E1 + E2) + noise[0]
    cs = y[1][:, None] * (-E1 + E2) + noise[1]
    ne = y[2][:, None] * E3 + noise[2]
    return (pr, y[0]), (cs, y[1]), (ne, y[2])


def response(learner, rng, feedback=True, normalize=True):
    """Per group (pr, cs, ne), one case. normalize=True returns the kernel column estimate
    y_b y_e dz_b / (eta g_e), with g_e = -dL/dz_e the signed scalar of the delivery;
    normalize=False returns the raw move toward each probe's key, y_b dz_b / eta."""
    ye = rng.choice([-1.0, 1.0])
    xe = ye * (-E1 + E2) + rng.normal(0, SIGMA, D)
    ze = learner.z(xe[None])[0]
    label = ye if feedback else np.sign(ze)   # own answer when no feedback
    g = label * (1.0 - sig(label * ze))
    after = learner.step(xe, label, ETA)
    out = []
    for X, y in probes(rng, N_PROBE):
        dz = after.z(X) - learner.z(X)
        out.append(np.mean(y * ye * dz) / (ETA * g) if normalize else np.mean(y * dz) / ETA)
    return np.array(out)


def raw_cs_move(learner, rng, feedback):
    return np.mean([response(learner, rng, feedback, normalize=False)[1] for _ in range(N_CASE)])


def rho_of(learner, rng, feedback=True, per_case=False, normalize=True):
    R = np.array([response(learner, rng, feedback, normalize) for _ in range(N_CASE)])
    num, den = R[:, 0] - R[:, 2], R[:, 1] - R[:, 2]
    return num / den if per_case else num.mean() / den.mean()


def rho_true(learner):
    e = -E1 + E2
    return learner.kernel(E1 + E2, e) / learner.kernel(e, e)


# ---------- tests ----------
def perm_p(x, y, rng, n_perm=2000):
    pooled = np.concatenate([x, y])
    obs = abs(x.mean() - y.mean())
    idx = np.argsort(rng.random((n_perm, pooled.size)), axis=1)
    perm = pooled[idx]
    diff = np.abs(perm[:, :x.size].mean(1) - perm[:, x.size:].mean(1))
    return (1 + np.sum(diff >= obs - 1e-12)) / (n_perm + 1)


def seed_level(make_a, make_b, rng, normalize=True):
    ra = np.array([rho_of(make_a(rng), rng, normalize=normalize) for _ in range(S_SEED)])
    rb = np.array([rho_of(make_b(rng), rng, normalize=normalize) for _ in range(S_SEED)])
    return perm_p(ra, rb, rng) < ALPHA, rb.mean() - ra.mean()


def case_level(make_a, make_b, rng):
    """The wrong unit: one learner per arm, cases as replicates, paired t-test."""
    la, lb = make_a(rng), make_b(rng)
    state = rng.bit_generator.state
    ra = rho_of(la, rng, per_case=True)
    rng.bit_generator.state = state      # same cases and probes for both learners
    rb = rho_of(lb, rng, per_case=True)
    return stats.ttest_rel(ra, rb).pvalue < ALPHA


def rate(fn, make_a, make_b, seed):
    rng = np.random.default_rng(seed)
    hits, diffs = [], []
    for _ in range(R_REP):
        r = fn(make_a, make_b, rng)
        hits.append(r[0] if isinstance(r, tuple) else r)
        if isinstance(r, tuple):
            diffs.append(r[1])
    k = int(np.sum(hits))
    lo, hi = stats.binomtest(k, R_REP).proportion_ci(0.95)
    return k / R_REP, lo, hi, (np.mean(diffs) if diffs else np.nan)


def main():
    rng = np.random.default_rng(0)
    r2 = np.sqrt(2.0)
    absent = lambda g: two_layer(g, r2, r2, 0.0, 0.0)
    held = lambda k: (lambda g: two_layer(g, r2, r2, k, 0.0))
    both = lambda g: two_layer(g, 1.0, 1.0, 1.0, 1.0)
    lin_cue = lambda g: linear(g, 2.0, 0.0)
    lin_mix = lambda g: linear(g, 1.0, 1.0)

    print("1. True rho from the weights, and rho read by one step (mean of 20 learners)")
    for name, mk in [("linear, cue weights", lin_cue), ("linear, mixed weights", lin_mix),
                     ("two-layer, rule absent", absent), ("two-layer, rule held unused k=1", held(1.0)),
                     ("two-layer, uses both", both)]:
        L = [mk(rng) for _ in range(20)]
        t = np.mean([rho_true(l) for l in L])
        fb = np.mean([rho_of(l, rng) for l in L])
        nf = np.mean([rho_of(l, rng, feedback=False) for l in L])
        X = (E1 + E2)[None]; C = (-E1 + E2)[None]
        zp = np.mean([l.z(X)[0] for l in L]); zc = np.mean([l.z(C)[0] for l in L])
        m1 = np.mean([raw_cs_move(l, rng, True) for l in L])
        m0 = np.mean([raw_cs_move(l, rng, False) for l in L])
        print(f"   {name:32s} true {t:+.3f} read {fb:+.3f} read no feedback {nf:+.3f}"
              f" | z prac {zp:+.2f} z conf {zc:+.2f} | sibling move fb {m1:+.2f} no fb {m0:+.2f}")

    print("2. False positive rate and power of the seed-level permutation test, alpha 0.05")
    conds = [("null N0, linear cue vs linear mixed", lin_cue, lin_mix),
             ("null N1, two-layer absent vs absent", absent, absent)]
    conds += [(f"planted, absent vs held k={k}", absent, held(k)) for k in (0.25, 0.5, 0.75, 1.0)]
    for i, (name, ma, mb) in enumerate(conds):
        f, lo, hi, d = rate(seed_level, ma, mb, 100 + i)
        L = [mb(rng) for _ in range(50)]; L0 = [ma(rng) for _ in range(50)]
        dt = np.mean([rho_true(l) for l in L]) - np.mean([rho_true(l) for l in L0])
        print(f"   {name:38s} rejection {f:.3f} [{lo:.3f}, {hi:.3f}]  mean read diff {d:+.3f}  true diff {dt:+.3f}")

    print("3. The wrong unit: case-level paired t-test, one learner per arm")
    for i, (name, ma, mb) in enumerate(conds[:2]):
        f, lo, hi, _ = rate(case_level, ma, mb, 200 + i)
        print(f"   {name:38s} rejection {f:.3f} [{lo:.3f}, {hi:.3f}]")

    print("3b. Raw ratio of means without dividing each case by its delivery scalar, seed-level test")
    f, lo, hi, d = rate(lambda a, b, g: seed_level(a, b, g, normalize=False), lin_cue, lin_mix, 300)
    print(f"   {conds[0][0]:38s} rejection {f:.3f} [{lo:.3f}, {hi:.3f}]  mean read diff {d:+.3f}")

    print("4. Entrenchment in the diagonal two-feature learner (B25, B26 point 3)")
    for lam in (0.0, 0.05):
        a, wu, b, wv = 0.1, 0.1, 0.09, 0.09     # balanced small start, cue slightly ahead
        Pu0, Pv0 = a * wu, b * wv
        for _ in range(200000):
            r = 1.0 - (1.5 * a * wu + 1.0 * b * wv)       # practiced input u=1.5, v=1, target 1
            da, dwu = 1.5 * wu * r - lam * a, 1.5 * a * r - lam * wu
            db, dwv = 1.0 * wv * r - lam * b, 1.0 * b * r - lam * wv
            a, wu, b, wv = a + 1e-3 * da, wu + 1e-3 * dwu, b + 1e-3 * db, wv + 1e-3 * dwv
        al, be = a * a + wu * wu, b * b + wv * wv
        rho0 = (2 * Pv0 - 2 * Pu0) / (2 * Pv0 + 2 * Pu0)
        expo = np.log(a * wu / Pu0) / np.log(b * wv / Pv0) if lam == 0 else np.nan
        print(f"   weight penalty {lam}: practiced z {1.5*a*wu + b*wv:.3f}, P_u {a*wu:.3f}, P_v {b*wv:.4f}, "
              f"rho start {rho0:+.3f} end {(be - al) / (be + al):+.3f}, log-ratio exponent {expo:.3f}")

    print("5. Bayesian mixture over two rules, slip 0.1 over 4 options, one conflict case with key")
    lr = np.log(0.9 / (0.1 / 3))
    for w in (0.01, 0.1, 0.3):
        o = w / (1 - w) * np.exp(lr)
        print(f"   prior rule weight {w}: posterior {o / (1 + o):.3f}; practiced probes move 0, so rho_ctx = 0")


if __name__ == "__main__":
    main()
