"""Revival time against the initial kernel in the toy of theory/E.md E10, the tier 2 threshold,
and the GRPO factor of panel B11 and B31 point 1.

Toy. One pair (a, s) on a feature phi_v with unit input, target log odds t, squared loss
(1/2)(t - a s)^2, gradient flow. P = a s, delta = s^2 - a^2 is conserved, and
dP/dt = sqrt(delta^2 + 4 P^2) (t - P). The initial kernel on phi_v is K0 = a^2 + s^2.
Held: a = 0, s = kappa, so K0 = kappa^2, P0 = 0. Absent: a = s = epsilon, so K0 = 2 epsilon^2, P0 = K0 / 2.

Blocks
1. Time to P = t/2 for held and absent at equal K0, by ODE and by quadrature, beside the closed forms.
2. The window of rank inversion, held K_h against absent K_a.
3. Spearman r_s threshold 0.6 over 19 conditions, false positive rate and power.
4. w_G(p) at G = 5 and the largest gap of the GRPO end point shift.

Run with: python -I revival_toy.py   (4 min 18 s on the PC, mostly block 3)
"""
import numpy as np
from scipy import integrate, stats
from scipy.special import comb


def t_half_ode(a, s, t):
    rhs = lambda _, y: [y[1] * (t - y[0] * y[1]), y[0] * (t - y[0] * y[1])]
    ev = lambda _, y: y[0] * y[1] - t / 2
    ev.terminal, ev.direction = True, 1
    sol = integrate.solve_ivp(rhs, (0, 1e5), [a, s], events=ev, rtol=1e-10, atol=1e-13)
    return sol.t_events[0][0]


def t_half_quad(K0, P0, t):
    delta = np.sqrt(max(K0 ** 2 - 4 * P0 ** 2, 0.0))
    f = lambda P: 1.0 / (np.sqrt(delta ** 2 + 4 * P ** 2) * (t - P))
    return integrate.quad(f, P0, t / 2, limit=200)[0]


def held_time(K, t):
    return t_half_quad(K, 0.0, t)


def absent_time(K, t):
    return t_half_quad(K, K / 2, t)


def w_G(p, G):
    k = np.arange(G + 1)
    pk = comb(G, k) * p ** k * (1 - p) ** (G - k)
    rbar = k / G
    s = np.sum(pk * np.sqrt(rbar * (1 - rbar)))   # population std of the group's rewards
    return s / (p * (1 - p))


def main():
    print("1. Time to P = t/2, held against absent at equal initial kernel K0")
    for t in (1.0, 4.0):
        for K in (8.0, 2.0, 1.0, 0.5, 0.1, 0.02, 0.002, 2e-4):
            th_o = t_half_ode(0.0, np.sqrt(K), t)
            th_q = held_time(K, t)
            line = (f"   t={t:<4} K0={K:<7} held ode {th_o:8.4f} quad {th_q:8.4f}"
                    f" | ln(4t/K)/(2t) {np.log(4 * t / K) / (2 * t):8.4f}  ln2/K {np.log(2) / K:8.4f}")
            if K / 2 < t / 2:
                ta_o = t_half_ode(np.sqrt(K / 2), np.sqrt(K / 2), t)
                line += (f" | absent ode {ta_o:8.4f} closed {np.log(2 * t / K - 1) / (2 * t):8.4f}"
                         f" gap {th_o - ta_o:+.4f}")
            else:
                line += " | absent starts above t/2"
            print(line)

    print("2. Rank inversion: held K_h revives slower than absent K_a iff K_h < about 2 K_a (K << t)")
    t = 1.0
    for Ka in (0.02, 0.002):
        ta = absent_time(Ka, t)
        for ratio in (1.0, 1.5, 1.9, 2.0, 2.1, 3.0):
            th = held_time(ratio * Ka, t)
            print(f"   K_a={Ka:<6} K_h/K_a={ratio:<4} held {th:.4f} absent {ta:.4f} held slower: {th > ta}")

    print("3. Spearman r_s over 19 conditions, threshold 0.6 (chosen)")
    rng = np.random.default_rng(3)
    n, R = 19, 200000
    base = np.arange(n)
    perm = np.argsort(rng.random((R, n)), axis=1)
    rs = 1 - 6 * np.sum((perm - base) ** 2, axis=1) / (n * (n * n - 1))
    print(f"   null (independent ranks): P(r_s >= 0.6) = {np.mean(rs >= 0.6):.4f},"
          f" one-sided 0.05 critical value {np.quantile(rs, 0.95):.3f}")
    for r in (0.5, 0.6, 0.7, 0.8, 0.9):
        cov = [[1, r], [r, 1]]
        X = rng.multivariate_normal([0, 0], cov, size=(20000, n))
        hits = [stats.spearmanr(x[:, 0], x[:, 1])[0] >= 0.6 for x in X]
        print(f"   bivariate normal, Pearson {r}: power P(r_s >= 0.6) = {np.mean(hits):.3f}")

    print("4. GRPO factor w_G(p) at G = 5, population std (panel B11 point 2, B31 point 1)")
    ps = np.concatenate([np.linspace(1e-4, 0.5, 2001)])
    w = np.array([w_G(p, 5) for p in ps])
    print(f"   w_5(1/2) = {w_G(0.5, 5):.4f}, w_5(1e-4) = {w_G(1e-4, 5):.4f}, "
          f"min over p {w.min():.4f} at p {ps[w.argmin()]:.3f}, monotone on (0, 1/2]: {bool(np.all(np.diff(w) <= 1e-12))}")
    print(f"   sup gap of the end point shift times beta = 2 - w_5(1/2) = {2 - w_G(0.5, 5):.4f}")
    print(f"   over base p from 0.05 to 0.95: {w_G(0.05, 5):.4f} to {w_G(0.5, 5):.4f}")


if __name__ == "__main__":
    main()
