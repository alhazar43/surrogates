"""RevS check of Proposition 4d: population log OR_MH against prevalence pi, gamma 1.4, delta 0, true theta, 50 strata."""
import sys
import numpy as np
sys.path.insert(0, "C:/Users/YuanW/PLMS/surrogates/notes/units/revs_scratch")
import revs_check as rc  # noqa: E402
rng = np.random.default_rng(19)
for pi in (0.1, 0.3, 0.5, 0.7, 0.9):
    lor, _ = rc.population_tables(rng, 400000, 50000, 1, 50, gamma=1.4, delta=0.0, true_theta=True, pi=pi)
    print(f"pi {pi}: log OR_MH {lor:.4f}")
