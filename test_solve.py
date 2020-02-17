import opinion as op
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import odeint

Y0 = np.array([0.0,4.0,5.0,6.0,10.0,11.0,12.0])
P0 = np.array([0.01,0.01])

tf = np.array([20.0])
Z0 = np.concatenate((P0,tf))

eta = 0.5

mu = 1.0

#F = op.F_zero(Z0,Y0,eta)

sol = root(op.F_zero,Z0,args=(Y0,eta,mu))
print(sol.message)

Z0 = sol.x
tf = Z0[-1]
P0 = Z0[0:-1]

X0 = np.concatenate((Y0,P0)) 

t = np.linspace(0.0,tf,100)
X = odeint(op.dynX, X0, t)

x0 = X[:,0].reshape(-1,1)
xi = X[:,1:-2]+x0

plt.plot(t,x0,'r')
plt.plot(t,xi,'b')
plt.show()

