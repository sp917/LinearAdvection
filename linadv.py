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
# Using the FTBS and CTCS methods

import numpy as np
import matplotlib.pyplot as plt

a = 0.7

# Set ranges

X = 1
T = 1

#increments

dx = 0.001
dt = 0.001

#Courant number

c = a*dt/dx

#u will be an MxN matrix

N = int(X/dx) + 1
M = int(T/dt) + 1

#boundary function

def f(b):
    b = b%X
    return     np.exp(-(b-0.1)**2/0.001) #1 - (b<0.1) - (b>0.2)


CNCS = np.zeros(shape = (M,N))
#BTBS = np.zeros(shape=(M,N))
#CTCS = np.zeros(shape=(M,N))
FTBS = np.zeros(shape=(M,N))

x = np.zeros(N)
t = np.zeros(M)

for n in range(0,N):
    x[n] = n*dx
    
for m in range(0,M):
    t[m] = m*dt

#initial condition:

CNCS[0] = f(x)
#BTBS[0] = f(x)
#CTCS[0] = f(x)
FTBS[0] = f(x)


#Define iteration matrices e.g. A such that u^{n+1} = Au^{n}

A = np.zeros(shape = (N,N)) #CNCS
B = np.zeros(shape = (N,N)) #CNCS
#C = np.zeros(shape = (N,N)) #BTBS
#D = np.zeros(shape = (N,N)) #FTCS
#E = np.zeros(shape = (N,N)) #CTCS
F = np.zeros(shape = (N,N)) #FTBS


for n in range(0,N):
    for k in range(0,N):        
        A[n][k] = (n==k) + (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
        B[n][k] = (n==k) - (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
       #C[n][k] = (n==k)*(1+c) - c*( n == (k+1)%N)
       #D[n][k] = (n==k%N) - 0.5*c*(n==((k-1)%N)) + 0.5*c*(n==((k+1)%N))
       #E[n][k] = c*(n==((k-1)%N)) - c*(n==((k+1)%N))
        F[n][k] = (1-c)*((n==k%N)) + c*(n==((k+1)%N))


#CNCS:

AinvB = np.dot(np.linalg.inv(A),B)

for m in range(1,M):
    CNCS[m] = np.dot(AinvB,CNCS[m-1])
    
#BTBS:
    
#invC = np.linalg.inv(C)

#for m in range(1,M):
#    BTBS[m] = np.dot(invC,BTBS[m-1])

#CTCS:
#Since CTCS is a two-step method we need a one-step method for the first iteration 
#We use FTCS

#CTCS[1] = np.dot(D,CTCS[0])

#for m in range(2,M):
 #   CTCS[m] = CTCS[m-2] - np.dot(E,CTCS[m-1])

#FTBS:

for m in range(1,M):
    FTBS[m] = np.dot(F,FTBS[m-1])
    
#Exact solution:
    
exact = np.zeros(shape=(M,N))

for n in range(0,N):
    for m in range(0,M):
        exact[m][n] = f(x[n] - a*t[m])

#Errors:
def err(W):
    return np.abs(exact - W)

P = M*N

def averr(W):
    return sum(sum(err(W)))/P
    

tt = T
m = int((M-1)*tt/T)

plt.axis([0,X,-1.3,1.4])
plt.plot(x,CNCS[m], label = 'CNCS')
#plt.plot(x,BTBS[m], label = 'BTBS')
#plt.plot(x,CTCS[m], label = 'CTCS')
plt.plot(x,FTBS[m], label = 'FTBS')
plt.plot(x,exact[m], label = 'Exact Solution')
plt.legend()
plt.title('t = %.3f' % t[m])
plt.xlabel('x')
plt.ylabel('u')
plt.show()

print("CNCS: %f" % averr(CNCS))
#print("BTBS: %f" % averr(BTBS))
#print("CTCS: %f" % averr(CTCS))
print("FTBS: %f" % averr(FTBS))
