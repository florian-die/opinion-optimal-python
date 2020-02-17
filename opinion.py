import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def f(x):
    return 1.0/(1.0+x**2);

def df(x):
    return -2.0*x*f(x)**2;

def h(x):
    return x*f(x);

def dh(x):
    return f(x)+x*df(x);

def saturate(x,sat=np.infty):
    if x > sat:
        return sat
    if x < -sat:
        return -sat
    return x;

def control(X,mu=1.0,sat=np.infty):
    u = (X[-2]+X[-1])/mu;
    return saturate(u,sat);

def dynX(X,t=0.0,mu=1.0,sat=np.infty):
    dX = np.zeros(X.shape)
    u = control(X,mu,sat)
    dX[0] = u
    dX[1:-2] = -h(X[1:-2])-u
    dX[-2] = X[-2]*dh(X[1])
    dX[-1] = X[-1]*dh(X[-3])
    return dX;

def Hamil(X,mu=1.0,sat=np.infty):
    u = control(X,mu,sat)    
    return 1 + 0.5*mu*u**2 - (X[-2]+X[-1])*u - X[-2]*h(X[1]) - X[-1]*h(X[-3]);

def F_zero(Z,Y0,eta,mu=1.0,sat=np.infty):
    tf = Z[-1]
    P0 = Z[0:-1]
    
    X0 = np.concatenate((Y0,P0))
    
    X = odeint(dynX, X0, (0.0,tf), args=(mu,sat))
    Xf = X[-1,:]
    
    F = np.zeros(3)
    
    F[0] = Hamil(Xf,mu,sat)
    F[1] = Xf[1]+eta
    F[2] = Xf[-3]-eta
    
    return F;

def trace(Z0,Y0,mu,sat):
    
    tf = Z0[-1]
    P0 = Z0[0:-1]
    X0 = np.concatenate((Y0,P0)) 
    
    t = np.linspace(0.0,tf,100)
    X = odeint(dynX, X0, t, args=(mu,sat))
    
    x0 = X[:,0].reshape(-1,1)
    xi = X[:,1:-2]+x0
    
    u = np.zeros(t.shape)
    for i in range(t.shape[0]):
        u[i] = control(X[i,:],mu,sat)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,x0,'r')
    ax[0].plot(t,xi,'b')
    
    ax[1].plot(t,u,'r')
    plt.show()