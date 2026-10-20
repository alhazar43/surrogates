"""Checks that part A2 of calibration.py draws the first run's stream: the S = 5 rows must match the result file."""
import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location(
    "cal", "C:/Users/YuanW/PLMS/surrogates/src/surrogates/theory_p/calibration.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
cal.RNG = np.random.default_rng(20261019)
cal.martingale_check()
for delta in (0.0, 0.1):
    r = cal.fork_detection(5, delta=delta)
    print(delta, *(f"{r[k]:.3f}" for k in ("raw_fpr_per_pair", "raw_power", "window_any_false_in_chain", "window_power")))
