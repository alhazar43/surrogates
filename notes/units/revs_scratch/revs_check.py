"""RevS checks of theory/S.md on synthetic records only. Run: python -I revs_check.py <part>

Part H. Population residue of the hard null (delta 1.0) by source: rest score length against stratum width.
Part F2. Steady-state filter and smoother variance of the local level model of Proposition 2.
Part C4. Scale of c4 in the tier 2 learner: linear projection of z on z_hat.
Part S. FPR by simulation at 3000 students for strata choices the source script did not try.
"""
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, "C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_s")
# the module path is inserted by hand since python -I ignores PYTHONPATH
import calibrate as cal  # noqa: E402


def population_tables(rng, n_total, chunk, n_rest, n_strata, gamma=0.0, delta=1.0, true_theta=False,
                      full_res=False, pi=0.3, zeta=0.5):
    """Accumulate per-stratum 2x2 tables over n_total students. Edges from the first chunk."""
    b_rest = rng.standard_normal(n_rest)
    tabs = {}
    edges = None
    for _ in range(n_total // chunk):
        th = rng.standard_normal(chunk)
        z = (rng.random(chunk) < pi) & (gamma > 0)
        if true_theta:
            lvl = th
        else:
            lvl = np.zeros(chunk)
            for i in range(n_rest):
                lvl += rng.random(chunk) < cal.sigmoid(th - b_rest[i])
        ch, wr = [], []
        for _k in range(2):
            w = rng.random(chunk) >= cal.sigmoid(th - zeta * z)
            lg = np.stack([-delta * th + gamma * z, np.zeros(chunk), np.full(chunk, -0.5)], 1)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            ch.append(rng.random(chunk) < p[:, 0]); wr.append(w)
        both = wr[0] & wr[1]
        if full_res:
            st = lvl[both].astype(int)
        else:
            if edges is None:
                edges = np.quantile(lvl, np.linspace(0, 1, n_strata + 1)[1:-1])
            st = np.digitize(lvl[both], edges)
        a, b = ch[0][both], ch[1][both]
        for s in np.unique(st):
            m = st == s
            t = tabs.setdefault(int(s), np.zeros(4))
            t += [np.sum(a[m] & b[m]), np.sum(a[m] & ~b[m]), np.sum(~a[m] & b[m]), np.sum(~a[m] & ~b[m])]
    num = den = 0.0
    nb = 0.0
    for t in tabs.values():
        n = t.sum()
        if n < 2:
            continue
        num += t[0] * t[3] / n; den += t[1] * t[2] / n; nb += n
    return np.log(num / den), nb


def part_h():
    rng = np.random.default_rng(7)
    print("Part H. Population log OR_MH under the hard null, delta 1.0, gamma 0, 400000 students")
    rows = [("true theta", 0, 5, True, False), ("true theta", 0, 10, True, False),
            ("true theta", 0, 50, True, False),
            ("rest 20", 20, 5, False, False), ("rest 20", 20, 10, False, False), ("rest 20", 20, 0, False, True),
            ("rest 250", 250, 10, False, False), ("rest 250", 250, 50, False, False),
            ("rest 250", 250, 0, False, True)]
    for name, nr, ns, tt, fr in rows:
        lor, nb = population_tables(rng, 400000, 50000, max(nr, 1), ns, true_theta=tt, full_res=fr)
        lab = "each score value" if fr else f"{ns} strata"
        print(f"  {name:10s} | {lab:16s} | log OR_MH {lor:.4f} | wrong on both {nb/400000:.3f} of students")
    print("  planted gamma 0.7, rest 250, 10 strata:")
    lor, _ = population_tables(rng, 400000, 50000, 250, 10, gamma=0.7)
    print(f"    log OR_MH {lor:.4f}")
    lor, _ = population_tables(rng, 400000, 50000, 250, 10, gamma=0.7, delta=0.0)
    print(f"    same with delta 0: log OR_MH {lor:.4f}")


def part_f2():
    print("Part F2. Local level model, steady state")
    for v, info in ((0.01, 0.25), (0.0025, 0.25)):
        pf = (-info * v + np.sqrt((info * v) ** 2 + 4 * info * v)) / (2 * info)
        pp = pf + v
        j = pf / pp
        ps = (pf - j * j * pp) / (1 - j * j)
        print(f"  v={v} I={info}: filter var {pf:.5f} sd {np.sqrt(pf):.4f}; smoother var {ps:.5f} sd {np.sqrt(ps):.4f}")


def part_c4():
    print("Part C4. Scale of c4: projection of z on z_hat in the learner of section 4.2")
    rng = np.random.default_rng(11)
    # reproduce z with the generator of calibrate.simulate_tracker by drawing in the same order
    n = 20000
    theta = rng.standard_normal(n)
    z = rng.random(n) < 0.3
    b_e = rng.standard_normal((n, 30))
    pen = np.zeros((n, 30)); pen[:, :10] = 0.5 * z[:, None]
    y_e = rng.random((n, 30)) < cal.sigmoid(theta[:, None] - b_e - pen)
    p_m = cal.sigmoid(-0.5 + 1.4 * z)
    chose = (rng.random((n, 10)) < p_m[:, None]) & ~y_e[:, :10]
    nw = (~y_e[:, :10]).sum(1)
    zh = (chose.sum(1) + 0.5) / (nw + 1.0)
    lvl = cal.eap(y_e, b_e)
    x = np.column_stack([np.ones(n), lvl, zh])
    coef = np.linalg.lstsq(x, z.astype(float), rcond=None)[0]
    print(f"  slope of z on z_hat given theta_hat {coef[2]:.3f}; implied c4 at eta1 -0.04: {-0.04*coef[2]:.4f}")
    print(f"  corr(z, z_hat) {np.corrcoef(z, zh)[0,1]:.3f}; mean linked wrong answers {nw.mean():.2f}")
    # power scaling of the nested test from the source's 0.305 at 2000 students
    mu = stats.norm.ppf(1 - 0.305) * -1 + 1.96  # mean z giving power 0.305, one tail dominant
    for ns in (984, 2000, 3934, 4918):
        m = mu * np.sqrt(ns / 2000)
        pw = stats.norm.sf(1.96 - m) + stats.norm.cdf(-1.96 - m)
        print(f"  students {ns}: predicted power {pw:.3f}")


def part_s():
    print("Part S. FPR under the hard null at 3000 students, 300 replicates")
    rng = np.random.default_rng(13)
    for label, nr, ns, fr in (("rest 250, 10 strata", 250, 10, False), ("rest 250, each score value", 250, 0, True),
                              ("rest 20, each score value", 20, 0, True)):
        rej = 0
        for _ in range(300):
            th = rng.standard_normal(3000)
            b_rest = rng.standard_normal(nr)
            rest = (rng.random((3000, nr)) < cal.sigmoid(th[:, None] - b_rest[None, :])).sum(1)
            ch, wr = [], []
            for _k in range(2):
                w = rng.random(3000) >= cal.sigmoid(th)
                lg = np.stack([-th, np.zeros(3000), np.full(3000, -0.5)], 1)
                p = np.exp(lg); p /= p.sum(1, keepdims=True)
                ch.append(rng.random(3000) < p[:, 0]); wr.append(w)
            both = wr[0] & wr[1]
            if fr:
                st = rest[both]
            else:
                st = np.digitize(rest[both], np.quantile(rest, np.linspace(0, 1, ns + 1)[1:-1]))
            rej += cal.mh_test(ch[0][both], ch[1][both], st)[1] < 0.05
        print(f"  {label}: rejection {rej/300:.3f}")


if __name__ == "__main__":
    {"H": part_h, "F2": part_f2, "C4": part_c4, "S": part_s, "P": lambda: None}[sys.argv[1]]()


def part_p():
    """Pooled MH over K independent linked pairs, each on its own 1000 students, rest 250, 10 strata per pair."""
    print("Part P. Pooled MH over K linked pairs, 1000 students per pair, rest 250 items, 10 strata, 150 replicates")
    rng = np.random.default_rng(17)
    for k in (1, 20):
        for gamma in (0.0, 0.7):
            rej = 0
            for _ in range(150):
                aa, bb, ss = [], [], []
                for j in range(k):
                    a, b, s, _ = cal.simulate_linked(rng, 1000, gamma, 1.0, n_rest=250, n_strata=10)
                    aa.append(a); bb.append(b); ss.append(s + 100 * j)
                rej += cal.mh_test(np.concatenate(aa), np.concatenate(bb), np.concatenate(ss))[1] < 0.05
            print(f"  K={k} gamma={gamma}: rejection {rej/150:.3f}")


if __name__ == "__main__" and sys.argv[1] == "P":
    part_p()
