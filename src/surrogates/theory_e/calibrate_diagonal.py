"""Calibration of the tier 1 reading on the diagonal learner of theory/E.md E4.

calibrate.py calibrates the reading on a dense two-layer learner z = a . (W x), whose rows
of W move in every input direction. E4 and E11 (L2, L7) use the diagonal learner
z = sum_k a_k s_k phi_k(x) with fixed feature directions phi_k(x) = f_k . x. This script
builds the diagonal learner from the same random draws as calibrate.two_layer (s_k f_k = W_k),
so both learners compute the same function and differ only in which weights the test step moves.

Blocks
1. True and read rho of the diagonal learners; read rho at step size 0.01 and 0.005 (A1 check)
   for the dense and the diagonal learners on the same learners, cases and probes.
2. False positive rate and power of the seed-level permutation test on the diagonal learner.
3. True differences in rho from 20000 learners per arm, dense and diagonal.

Run with: python -I calibrate_diagonal.py   (about 4 minutes on the PC)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate as c  # noqa: E402


class Diagonal:
    """z(x) = sum_k a_k s_k (f_k . x); a and s are the trained weights, f fixed unit rows."""

    def __init__(self, a, s, F):
        self.a, self.s, self.F = a.copy(), s.copy(), F

    def z(self, X):
        return (X @ self.F.T) @ (self.a * self.s)

    def step(self, x, y, eta):
        g = -y * (1.0 - c.sig(y * self.z(x[None])[0]))   # dL/dz for L = softplus(-y z)
        phi = self.F @ x
        new = Diagonal(self.a, self.s, self.F)
        new.a -= eta * g * self.s * phi
        new.s -= eta * g * self.a * phi
        return new

    def kernel(self, xb, xe):
        return np.sum((self.a ** 2 + self.s ** 2) * (self.F @ xb) * (self.F @ xe))


def diagonal(rng, s_u, a_u, s_v, a_v):
    """Same draws as calibrate.two_layer; W_k = s_k f_k with s_k = norm(W_k)."""
    dense = c.two_layer(rng, s_u, a_u, s_v, a_v)
    W, a = dense.W, dense.a
    s = np.linalg.norm(W, axis=1)
    F = np.zeros_like(W)
    F[0], F[1] = c.E1, c.E2
    s[0], s[1] = s_u, s_v
    for k in range(2, W.shape[0]):
        F[k] = W[k] / s[k]
    return Diagonal(a, s, F)


def read_at(learners, eta, seed):
    old = c.ETA
    c.ETA = eta
    rng = np.random.default_rng(seed)
    out = np.array([c.rho_of(l, rng) for l in learners])
    c.ETA = old
    return out


def main():
    rng = np.random.default_rng(1)
    r2 = np.sqrt(2.0)
    d_absent = lambda g: diagonal(g, r2, r2, 0.0, 0.0)
    d_held = lambda k: (lambda g: diagonal(g, r2, r2, k, 0.0))
    d_both = lambda g: diagonal(g, 1.0, 1.0, 1.0, 1.0)
    t_absent = lambda g: c.two_layer(g, r2, r2, 0.0, 0.0)
    t_held = lambda g: c.two_layer(g, r2, r2, 1.0, 0.0)
    t_both = lambda g: c.two_layer(g, 1.0, 1.0, 1.0, 1.0)

    print("1. Diagonal learner, true and read rho (mean of 20 learners); read at eta 0.01 and 0.005")
    for name, mk in [("linear, cue weights", lambda g: c.linear(g, 2.0, 0.0)),
                     ("linear, mixed weights", lambda g: c.linear(g, 1.0, 1.0)),
                     ("dense, rule absent", t_absent), ("dense, rule held unused k=1", t_held),
                     ("dense, uses both", t_both),
                     ("diagonal, rule absent", d_absent), ("diagonal, rule held unused k=1", d_held(1.0)),
                     ("diagonal, uses both", d_both)]:
        L = [mk(rng) for _ in range(20)]
        t = np.mean([c.rho_true(l) for l in L])
        r1, r05 = read_at(L, 0.01, 11), read_at(L, 0.005, 11)
        nf = np.mean([c.rho_of(l, rng, feedback=False) for l in L])
        print(f"   {name:32s} true {t:+.3f} read eta .01 {r1.mean():+.3f} eta .005 {r05.mean():+.3f}"
              f" largest change per learner {np.max(np.abs(r1 - r05)):.4f} | read no feedback {nf:+.3f}")

    print("2. Diagonal learner, seed-level permutation test, alpha 0.05, 400 replicates")
    conds = [("null N1d, diagonal absent vs absent", d_absent, d_absent)]
    conds += [(f"planted, diagonal absent vs held k={k}", d_absent, d_held(k))
              for k in (0.25, 0.375, 0.5, 0.75, 1.0)]
    for i, (name, ma, mb) in enumerate(conds):
        f, lo, hi, d = c.rate(c.seed_level, ma, mb, 400 + i)
        print(f"   {name:40s} rejection {f:.3f} [{lo:.3f}, {hi:.3f}]  mean read diff {d:+.3f}")

    print("3. True difference in rho, 20000 learners per arm (mean, sd per learner)")
    g = np.random.default_rng(7)
    for label, mk_a, mk_h in [("dense", t_absent, lambda k: (lambda q: c.two_layer(q, r2, r2, k, 0.0))),
                              ("diagonal", d_absent, d_held)]:
        A = np.array([c.rho_true(mk_a(g)) for _ in range(20000)])
        for k in (0.0, 0.25, 0.375, 0.5, 0.75, 1.0):
            B = np.array([c.rho_true(mk_h(k)(g)) for _ in range(20000)])
            print(f"   {label:8s} k={k:<5} rho absent {A.mean():+.4f} held {B.mean():+.4f} "
                  f"diff {B.mean() - A.mean():+.4f}  sd absent {A.std():.4f} sd held {B.std():.4f}")


if __name__ == "__main__":
    main()
