import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import calibrate as c
rng=np.random.default_rng(7); r2=np.sqrt(2)
for k in (0.0,0.25,0.5,0.75,1.0):
    A=[c.rho_true(c.two_layer(rng,r2,r2,0,0)) for _ in range(20000)]
    B=[c.rho_true(c.two_layer(rng,r2,r2,k,0)) for _ in range(20000)]
    print(k, 'mean diff %.4f'%(np.mean(B)-np.mean(A)), 'sd per learner %.4f'%np.std(A), 'MC se at n=50: %.4f'%(np.sqrt(np.var(A)/50+np.var(B)/50)))
# E4 diagonal learner values
for k in (0.25,0.5,0.75,1.0):
    print('diag', k, (k**2-4)/(k**2+4) - (-1))
