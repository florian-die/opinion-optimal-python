import opinion as op
import numpy as np
from scipy.optimize import root

trace = False

Y0 = np.array([0.0,4.0,5.0,6.0,10.0,11.0,12.0])
P0 = np.array([-0.44,1.82]) # from previous run, without saturation

tf = np.array([20.0])
Z0 = np.concatenate((P0,tf))

eta = 0.5
sat = 2.0

mu_ini = 1.0
mu_end = 0.19
mu_step = 0.01

''' initial problem '''
sol = root(op.F_zero,Z0,args=(Y0,eta,mu_ini,sat))
Z0 = sol.x

if trace:
    print(mu_ini)
    print(sol)  
    op.trace(Z0,Y0,mu_ini,sat)

''' mu continuation'''
mu = mu_ini
while mu > mu_end:
    mu = mu-mu_step
    print(mu)
    sol = root(op.F_zero,Z0,args=(Y0,eta,mu,sat))
    
    Z0 = sol.x
    
    if trace:    
        print(sol)
        op.trace(Z0,Y0,mu,sat)

''' plot final solution '''
print(mu)
print(sol)
op.trace(Z0,Y0,mu,sat)
