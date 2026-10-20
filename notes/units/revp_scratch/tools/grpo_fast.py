"""Fixed points of an outcome-only reward with three KL placements on a depth-D token tree with V tokens per step.
seq:  sequence-level reverse KL (the tilt of P.md D5).
tokKL: exact per-token reverse KL added to the loss (no gradient through the prefix distribution).
k3:   GRPO's per-token k3 estimator added to the loss (Shao et al. 2024 eq. 3), gradient coefficient A + beta(pi_ref/pi - 1).
Reports the largest change of the within-right distribution over runs, and of a within-right property rate."""
import numpy as np, itertools
rng = np.random.default_rng(7)
V, D, c = 3, 3, 1.0; beta = float(__import__("sys").argv[2])
def leaves(): return list(itertools.product(range(V), repeat=D))
def run_probs(pol):  # pol: dict prefix -> prob vector
    return np.array([np.prod([pol[l[:t]][l[t]] for t in range(D)]) for l in leaves()])
def q_values(pol, R):  # success prob from prefix h after token a
    L = leaves(); P = {}
    for l, r in zip(L, R): P[l] = r
    def q(h):
        if len(h) == D: return P[h]
        return sum(pol[h][a]*q(h+(a,)) for a in range(V))
    return q
worst = {}
for trial in range(int(__import__("sys").argv[1])):
    ref = {h: rng.dirichlet(np.ones(V)) for t in range(D) for h in itertools.product(range(V), repeat=t)}
    R = (rng.random(V**D) < 0.5).astype(float)
    if R.sum() in (0, V**D): continue
    phi = (rng.random(V**D) < 0.5).astype(float)
    p_ref = run_probs(ref)
    def within(p): return p*R/(p*R).sum()
    w_ref = within(p_ref)
    out = {}
    # seq tilt
    p = p_ref*np.exp(c*R/beta); p /= p.sum(); out["seq"] = p
    for kind in ("tokKL", "k3"):
        pol = {h: v.copy() for h, v in ref.items()}
        for it in range(4000):
            q = q_values(pol, R); new = {}
            for h in pol:
                qa = np.array([q(h+(a,)) for a in range(V)]); qh = pol[h] @ qa
                if kind == "tokKL":
                    v = ref[h]*np.exp(c*qa/beta); v /= v.sum()
                else:  # solve pi = beta*ref/(beta - c(qa-qh)) with qh self-consistent: fixed-point step
                    den = beta - c*(qa - qh)
                    v = beta*ref[h]/np.maximum(den, 1e-9); v /= v.sum()
                new[h] = 0.5*pol[h] + 0.5*v
            delta = max(np.abs(new[h]-pol[h]).max() for h in pol); pol = new
            if delta < 1e-12: break
        out[kind] = run_probs(pol); worst.setdefault("resid_"+kind, []).append((delta,0,0))
    for k, p in out.items():
        dw = np.abs(within(p) - w_ref).sum()/2
        dphi = abs(within(p) @ phi - w_ref @ phi)
        worst.setdefault(k, []).append((dw, dphi, (p*R).sum()))
for k, v in worst.items():
    v = np.array(v)
    print(f"{k:6s} within-right TVD mean {v[:,0].mean():.4f} max {v[:,0].max():.4f} | within-right phi change mean {v[:,1].mean():.4f} max {v[:,1].max():.4f} | p after mean {v[:,2].mean():.3f}")
