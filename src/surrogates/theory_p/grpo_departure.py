"""Fixed points of an outcome-only reward under three KL placements, on random token trees with known truth.

seq    sequence-level reverse KL penalty (the tilt of theory/P.md D5)
tokKL  exact per-token reverse KL added to the loss, no gradient through the prefix distribution
k3     the per-token k3 estimator of Shao et al. 2024 eq. 4 added to the loss (GRPO), same gradient rule

Stationary rules at prefix h, with q(h, a) the success probability after token a under the trained policy:
  tokKL  pi(a | h) = pi_ref(a | h) exp(c q(h, a) / beta) / Z(h)
  k3     pi(a | h) = beta pi_ref(a | h) / (beta - c (q(h, a) - q(h)))
Checks the identity  log pi(y) / pi_ref(y) = (c / beta)(R(y) - q(root)) + sum_t KL(pi(. | h_t) || pi_ref(. | h_t))
for tokKL, and reports the signed change of two within-right rates.
  repair  the run passes a prefix h_t (0 < t < D) with q_ref(h_t) < q_ref(root) / 2 and ends right
  random  a property drawn at random per run
The k3 rule is defined only where beta > c (q differences lie in (-1, 1)), so it runs only then.
Equal run lengths, so the 1/|o| weight of GRPO plays no part here.
Tiny synthetic trees, numpy only, seconds on one core. Usage: python -I grpo_departure.py [trees] [beta]
"""
import itertools
import sys

import numpy as np

V, D, C = 3, 3, 1.0


def prefixes():
    return [h for t in range(D) for h in itertools.product(range(V), repeat=t)]


def runs():
    return list(itertools.product(range(V), repeat=D))


def run_probs(pol):
    return np.array([np.prod([pol[y[:t]][y[t]] for t in range(D)]) for y in runs()])


def q_table(pol, R):
    q = {y: r for y, r in zip(runs(), R)}
    for t in range(D - 1, -1, -1):
        for h in itertools.product(range(V), repeat=t):
            q[h] = sum(pol[h][a] * q[h + (a,)] for a in range(V))
    return q


def fixed_point(ref, R, beta, kind, iters=20000, tol=1e-12):
    pol = {h: v.copy() for h, v in ref.items()}
    for _ in range(iters):
        q = q_table(pol, R)
        new = {}
        for h in pol:
            qa = np.array([q[h + (a,)] for a in range(V)])
            if kind == "tokKL":
                v = ref[h] * np.exp(C * qa / beta)
            else:
                v = beta * ref[h] / np.maximum(beta - C * (qa - pol[h] @ qa), 1e-9)
            new[h] = 0.5 * pol[h] + 0.5 * v / v.sum()
        resid = max(np.abs(new[h] - pol[h]).max() for h in pol)
        pol = new
        if resid < tol:
            break
    return pol, resid


def main(trees=200, beta=0.5, seed=20261019):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in (("seq", "tokKL", "k3") if beta > C else ("seq", "tokKL"))}
    ident, resid_max = [], 0.0
    for _ in range(trees):
        ref = {h: rng.dirichlet(np.ones(V)) for h in prefixes()}
        R = (rng.random(V ** D) < 0.5).astype(float)
        if R.sum() in (0, V ** D):
            continue
        qr = q_table(ref, R)
        rep = np.array([float(r > 0 and min(qr[y[:t]] for t in range(1, D)) < qr[()] / 2)
                        for y, r in zip(runs(), R)])
        rnd = (rng.random(V ** D) < 0.5).astype(float)
        p_ref = run_probs(ref)
        wr = lambda p: p * R / (p * R).sum()
        pols = {"seq": None}
        p_seq = p_ref * np.exp(C * R / beta)
        probs = {"seq": p_seq / p_seq.sum()}
        for kind in ("tokKL", "k3") if beta > C else ("tokKL",):
            pol, res = fixed_point(ref, R, beta, kind)
            resid_max = max(resid_max, res)
            probs[kind] = run_probs(pol)
            pols[kind] = pol
        pol, q = pols["tokKL"], q_table(pols["tokKL"], R)
        kl = {h: float(pol[h] @ np.log(pol[h] / ref[h])) for h in pol}
        lhs = np.log(probs["tokKL"] / p_ref)
        rhs = np.array([C / beta * (r - q[()]) + sum(kl[y[:t]] for t in range(D)) for y, r in zip(runs(), R)])
        ident.append(np.abs(lhs - rhs).max())
        for kind, p in probs.items():
            out[kind].append((0.5 * np.abs(wr(p) - wr(p_ref)).sum(),
                              wr(p) @ rep - wr(p_ref) @ rep, wr(p) @ rnd - wr(p_ref) @ rnd,
                              (p_ref * R).sum(), (p * R).sum(), float(rep.any())))
    print(f"trees {len(ident)}, beta {beta}, c {C}, largest fixed-point residual {resid_max:.1e}")
    print(f"tokKL identity, largest absolute error over runs and trees {max(ident):.1e}")
    print("kind   wr_TVD_mean wr_TVD_max | repair_change mean median min max share_positive (trees with a repair run) | "
          "random_change mean_abs max_abs signed_mean | p_ref p_after")
    for kind, v in out.items():
        v = np.array(v)
        m = v[:, 5] > 0
        rc = v[m, 1]
        print(f"{kind:6s} {v[:, 0].mean():.4f} {v[:, 0].max():.4f} | {rc.mean():+.4f} {np.median(rc):+.4f} "
              f"{rc.min():+.4f} {rc.max():+.4f} {np.mean(rc > 0):.3f} ({m.sum()}) | {np.abs(v[:, 2]).mean():.4f} "
              f"{np.abs(v[:, 2]).max():.4f} {v[:, 2].mean():+.4f} | {v[:, 3].mean():.3f} {v[:, 4].mean():.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 200, float(a[1]) if len(a) > 1 else 0.5)
