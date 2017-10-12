#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:07:09 2017

@author: sp917
"""

#Solve equation using specified method
import numpy as np
import matplotlib.pyplot as plt

a = 0.7
X = 1
T = 1

DT = [0.0001,0.00025,0.0005,0.00075,0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1]


dx = 0.001
N = int(X/dx) + 1

x = np.zeros(N)
for n in range(0,N):
    x[n] = n*dx
    
A = np.zeros(shape = (N,N)) #CNCS
B = np.zeros(shape = (N,N)) #CNCS

def f(b):
    b = b%X
    return    np.exp(-(b-0.1)**2/0.001) #1 - (b<0.1) - (b>0.2) 

AVERR = []
for dt in DT:
    
    c = a*dt/dx
    M = int(T/dt) + 1
    u = np.zeros(shape = (M,N))
    t = np.zeros(M)
    
    for m in range(0,M):
        t[m] = m*dt
        
    u[0] = f(x)
    
    for n in range(0,N):
        for k in range(0,N):        
            A[n][k] = (n==k) + (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
            B[n][k] = (n==k) - (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
            
    AinvB = np.dot(np.linalg.inv(A),B)
    
    for m in range(1,M):
        u[m] = np.dot(AinvB,u[m-1])
    
    #Exact solution:
    
    exact = np.zeros(shape=(M,N))

    for n in range(0,N):
        for m in range(0,M):
            exact[m][n] = f(x[n] - a*t[m])
            
    err = np.abs(u - exact)
    averr = sum(sum(err))/(M*N)
    AVERR = AVERR + [averr]
    


DT = np.array(DT)
grad = np.zeros(shape = DT.size)
grad[0] = 0
for n in range(1,grad.size):
    grad[n] = (np.log(AVERR[n]) - np.log(AVERR[n-1]))/(np.log(DT[n]) - np.log(DT[n-1]))

for n in range(0,DT.size):
    print("""\hline
				%f	& %.9f & %.9f \\\ """ % (DT[n], AVERR[n], grad[n]))

plt.plot(np.log(DT),np.log(AVERR),'ro')