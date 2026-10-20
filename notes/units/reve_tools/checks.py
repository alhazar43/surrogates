import numpy as np
from scipy import stats, integrate
# CP intervals
for k in (20,23,74,237,383,276,48):
    lo,hi=stats.binomtest(k,400).proportion_ci(0.95)
    print('CP',k,k/400,round(lo,3),round(hi,3))
# power interpolation
print('interp 0.8 at', 0.099+(0.8-0.593)/(0.958-0.593)*(0.163-0.099))
# normal approx
for d,p in ((0.099,0.593),(0.163,0.958)):
    z=stats.norm.ppf(p)+1.96; print('se',d/z)
# E8
lr=np.log(0.9*3/0.1); print('LR',np.exp(lr), 'ceiling', np.log(9))
# E10: time to reach P=t/2 from held (a=0,s=k) and absent (a=s=e), gradient flow, loss .5(t-as)^2
def T(a,s,t,frac=0.5,dt=1e-3,tmax=2e4):
    tt=0.0
    while a*s<frac*t and tt<tmax:
        r=t-a*s; a,s=a+dt*s*r, s+dt*a*r; tt+=dt
    return tt
t=1.0
print('E10 rows: K, T_held(K), T_absent(K)')
for K in (2.0,1.0,0.5,0.1,0.02,0.002,2e-4):
    k=np.sqrt(K); e=np.sqrt(K/2)
    print(f'K={K:<7} held {T(0.0,k,t):8.3f}  absent {T(e,e,t):8.3f}  kappa^-2*ln2 {np.log(2)/K:8.3f}  logistic {(1/(2*t))*np.log(t/(K/2)-1):8.3f}')
# t larger
t=4.0
for K in (8.0,1.0,0.1):
    k=np.sqrt(K); e=np.sqrt(K/2)
    print(f't=4 K={K:<5} held {T(0.0,k,t,dt=2e-4):8.3f} absent {T(e,e,t,dt=2e-4):8.3f}')
