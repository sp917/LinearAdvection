#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:09:08 2017

@author: sp917
"""

import numpy as np
import matplotlib.pyplot as plt

def v(dx,N):
    w = np.zeros(N)
    for n in range(0,N):
        w[n] = n*dx
    return w

class data: #This class provides the necessary data for solving the equation
    def __init__(self,X,T,a,dx,dt):
        self.X = X
        self.T = T
        self.a = a
        self.dx = dx
        self.dt = dt
        self.c = a*dt/dx
        self.N = int(X/dx) + 1
        self.M = int(T/dt) + 1
        self.x = v(dx,self.N)
        self.t = v(dt,self.M)
    def f(self,x):
        x = x%(self.X)
        return np.exp(-(x-0.1)**2/0.001)
    

def exactsolution(y):
    y.N = int(y.X/y.dx) + 1
    y.M = int(y.T/y.dt) + 1
    exact = np.zeros(shape=(y.M,y.N))
    for n in range(0,y.N):
        for m in range(0,y.M):
            exact[m][n] = y.f(y.x[n] - y.a*y.t[m])
    return exact


def C(n,N,c): #C will be the matrix such that u^{n+1} = Cu^{n}
    if n==1:
        A = np.zeros(shape=(N,N))
        B = np.zeros(shape = (N,N)) 
        
        for n in range(0,N):
            for k in range(0,N):        
                A[n][k] = (n==k) + (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
                B[n][k] = (n==k) - (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
                
        return np.dot(np.linalg.inv(A),B)


#Function solving the linear advection equation using the selected method
    
def CNCS(y):
    
    D = C(1,y.N,y.c)
    
    #u is our solution
    
    u = np.zeros(shape = (y.M,y.N))
    
    #Set initial condition
    
    u[0] = y.f(y.x)
    
    #Now we may iterate:

    for m in range(1,y.M):
        u[m] = np.dot(D,u[m-1])
    
    return u


#Graph the solution at specified time
def plot(y,tt):
    m = int((y.M-1)*tt/y.T)
    u = CNCS(y)
    exact = exactsolution(y)
    plt.axis([0,y.X,-1.3,1.4])
    plt.plot(y.x,u[m], label = 'CNCS')
    plt.plot(y.x,exact[m], label = 'Exact Solution')
    plt.legend()
    plt.title('t = %.3f' % y.t[m])
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()

y = data(1,1,0.7,0.001,0.001)
plot(y,0.999)


    
   