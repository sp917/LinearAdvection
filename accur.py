#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:07:09 2017

@author: sp917
"""

import numpy as np
import matplotlib.pyplot as plt

#Define intial condition function
def f(x,X):
        x = x%X
        return    np.exp(-(x-0.1)**2/0.001) #1 - (x<0.1) - (x>0.2) 

def main(X,T,a,dx,dt):
    
    method = 'CNCS' #Solve equation using specified method
    
    #Initialize variables:

    a = 0.7
    X = 1
    T = 1
    dx = 0.001
    N = int(X/dx) + 1
    
    A = np.zeros(shape = (N,N)) #CNCS
    B = np.zeros(shape = (N,N)) #CNCS
    F = np.zeros(shape = (N,N)) #FTBS

    DT = [0.0001,0.00025,0.0005,0.00075,0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1]

    x = np.zeros(N)
    for n in range(0,N):
        x[n] = n*dx
    
    AVERR = []
   
    for dt in DT:
        c = a*dt/dx
        M = int(T/dt) + 1
        u = np.zeros(shape = (M,N))
        t = np.zeros(M)
        
    for m in range(0,M):
        t[m] = m*dt
        
    exact = np.zeros(shape=(M,N))
        
    #Initial Condition:
        
    u[0] = f(x,X)
    
    #Iteration:
    
    if method == 'CNCS':
        for n in range(0,N):
            for k in range(0,N):        
                A[n][k] = (n==k) + (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
                B[n][k] = (n==k) - (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
            
        AinvB = np.dot(np.linalg.inv(A),B)
    
        for m in range(1,M):
            u[m] = np.dot(AinvB,u[m-1])
            
    else:
        for n in range(0,N):
            for k in range(0,N):  
                    F[n][k] = (1-c)*((n==k%N)) + c*(n==((k+1)%N))
        for m in range(1,M):
            u[m] = np.dot(F,u[m-1])
    
    #Exact solution:

    for n in range(0,N):
        for m in range(0,M):
            exact[m][n] = f(x[n] - a*t[m],X)
    
    #Calculation of errors:
    Tot = sum(sum(u))
    err = np.abs(u - exact)
    averr = sum(sum(err))/Tot #this is the l1 definition of error
    AVERR = AVERR + [averr]
    


    DT = np.array(DT)
    grad = np.zeros(shape = DT.size)
    grad[0] = 0
    for n in range(1,grad.size):
        change = np.log(AVERR[n]) - np.log(AVERR[n-1])
        interval = np.log(DT[n]) - np.log(DT[n-1])
        grad[n] = change/interval
        for n in range(0,DT.size):
            print("""\hline
          %f & %.6e & %.3f %f \\\ """ % (DT[n], AVERR[n], grad[n], a*DT[n]/dx))

    #Plot graph    
    plt.rc('text', usetex = True)
    plt.xlabel(r'$\log(\Delta t)$')
    plt.ylabel(r'$\log(\text{Error})$')
    plt.plot(np.log(DT),np.log(AVERR),'ro')
    plt.show()
    
main(X=1,T=1,a=0.7)