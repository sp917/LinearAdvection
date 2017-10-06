#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:19:42 2017

@author: sp917
"""

# Solves linear advection equation du/dt + a du/dx = 0
# On domain [0,infty)x[0,X]
# With initial condition u(x,0) = f(x)
# And periodic boundary conditions

# Using the FTBS method u^{m+1}_{n} = u^{m}_{n} - c(u^{m}_{n} - u^{m}_{n-1})

import numpy as np
import matplotlib.pyplot as plt

a = 0.7

# Set ranges

X = 2*np.pi
T = 5

#increments

dx = 0.01
dt = 0.01

#Courant number

c = a*dt/dx

#u will be an MxN matrix

N = int(X/dx) + 1
M = int(T/dt) + 1

#boundary function

def f(b):
    return 1 - (b%X>0.5) - (b%X<0.25)

u1 = np.zeros(shape = (M,N))
u2 = np.zeros(shape=(M,N))
x = np.zeros(N)
t = np.zeros(M)

for n in range(0,N):
    x[n] = n*dx
    
for m in range(0,M):
    t[m] = m*dt

#initial condition:

u1[0] = f(x)
u2[0] = f(x)

#Define iteration matrices A,B such that u^{n+1} = Au^{n}

A = np.zeros(shape = (N,N)) #FTBS
B = np.zeros(shape = (N,N)) #CTCS
C = np.zeros(shape = (N,N)) #FTCS


for n in range(0,N):
    for k in range(0,N):
        A[n][k] = (1-c)*(n==k) + c*(n==k+1)
        B[n][k] = c*(n==k-1) - c*(n==k+1)
        C[n][k] = (n==k) - 0.5*c*(n==k-1) + 0.5*c*(n==k+1)
    
    
#FTBS:

for m in range(1,M):
    u1[m] = np.dot(A,u1[m-1])
    
#CTCS:
#Since CTCS is a two-step method we need a one-step method for the first iteration 
#We use FTCS

u2[1] = np.dot(C,u2[0])

for m in range(2,M):
    u2[m] = u2[m-2] - np.dot(B,u2[m-1])

    
#Exact solution:
    
exact = np.zeros(shape=(M,N))

for n in range(0,N):
    for m in range(0,M):
        exact[m][n] = f(x[n] - a*t[m])

t = 4.9
m = int(M*t/T)

plt.axis([0,X,0,1.3])
plt.plot(x,u1[m])
plt.plot(x,u2[m])
plt.plot(x,exact[m])
plt.show()