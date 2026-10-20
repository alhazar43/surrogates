import importlib.util, numpy as np, sys
from scipy import stats
spec = importlib.util.spec_from_file_location("cal", "C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_p/calibration.py")
cal = importlib.util.module_from_spec(spec); spec.loader.exec_module(cal)
# D2
o0 = stats.binom.sf(5, 12, 0.55)
print("o0", o0, "scalar o0(1-o0)", o0*(1-o0), "vector 1-sum o0^2", 1-o0**2-(1-o0)**2)
# C: items with fixed seed
cal.RNG = np.random.default_rng(1)
items = [cal.make_item() for _ in range(2000)]
a1 = np.array([cal.within_right(pi0, r, phi) for pi0, r, phi in items])
print("mean a1 (within-right base rate)", a1.mean(), "-> effective planted change per delta = delta*(1-a1):", {d: d*(1-a1).mean() for d in (0.05,0.1,0.2)})
pg = [cal.pg_flow(pi0, r, 0.5) for pi0, r, phi in items]
d = np.array([cal.within_right(p, r, phi) - cal.within_right(pi0, r, phi) for (pi0, r, phi), p in zip(items, pg)])
print("D6 random phi: signed mean", d.mean(), "sd", d.std(), "mean abs", np.abs(d).mean(), "t over 2000 items", d.mean()/d.std()*np.sqrt(len(d)))
