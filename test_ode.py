import opinion as op
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

Y0 = np.array([0.0,4.0,5.0,6.0,10.0,11.0,12.0])
P0 = np.array([0.01,0.01])
X0 = np.concatenate((Y0,P0))

t = np.linspace(0.0,30.0,100)

X = odeint(op.dynX, X0, t)

x0 = X[:,0].reshape(-1,1)
xi = X[:,1:-2]+x0

plt.plot(t,x0,'r')
plt.plot(t,xi,'b')
plt.show()