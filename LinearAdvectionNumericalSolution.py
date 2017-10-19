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
        self.c = a*dt/dx #Courant number
        self.N = int(X/dx) + 1 #number of grid points in x-direction
        self.M = int(T/dt) + 1 #number of grid points in t-direction
        self.x = v(dx,self.N) #vector of all possible values of x
        self.t = v(dt,self.M) #vector of all possible values of t
    def f(self,x): #Initial condition
        x = x%(self.X)
        return 1 - (x<0.1) - (x>0.2) #np.exp(-(x-0.1)**2/0.001)# 
    def exactsolution(self):
        exact = np.zeros(shape=(self.M,self.N))
        for n in range(0,self.N):
            for m in range(0,self.M):
                exact[m][n] = self.f(self.x[n] - self.a*self.t[m])
        return exact
    def changedt(self,dtnew):
        self.dt = dtnew
        self.c = self.a*self.dt/self.dx
        self.M = int(self.T/self.dt) + 1
        self.t = v(self.dt,self.M)
    def changedx(self,dxnew):
        self.dx = dxnew
        self.c = self.a*self.dt/self.dx
        self.N = int(self.X/self.dx) + 1
        self.x = v(self.dx,self.N)
    

def C(i,N,c): #C will be the matrix such that u^{n+1} = Cu^{n}
    
    if i=='CNCS': 
        A = np.zeros(shape=(N,N))
        B = np.zeros(shape = (N,N)) 
        for n in range(0,N):
            for k in range(0,N):        
                A[n][k] = (n==k) + (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
                B[n][k] = (n==k) - (c/4)*((n == (k-1)%N) - (n == (k+1)%N))
        return np.dot(np.linalg.inv(A),B)
    
    if i=='FTBS': 
        A = np.zeros(shape = (N,N))
        for n in range(0,N):
                for k in range(0,N):
                    A[n][k] = (1-c)*((n==k%N)) + c*(n==((k+1)%N))
        return A

#Function solving the linear advection equation using the selected method
    
def solve(y,i):
        
    D = C(i,y.N,y.c)
    u = np.zeros(shape = (y.M,y.N))
    
    #Set initial condition
    
    u[0] = y.f(y.x)
    
    #Now we may iterate:

    for m in range(1,y.M):
        u[m] = np.dot(D,u[m-1])
        
    return u

#Define a function to calculate the l1 normed error
    
def error(y,u):
    ex = y.exactsolution()
    er = np.abs(ex - u)
    Tot = sum(sum(u))
    ertot = sum(sum(er))/Tot
    return ertot

#Plot the graph of the solution at specified time and print the total error

def plotsol(y,i,u,tt):
    m = int((y.M-1)*tt/y.T)
    plt.axis([0,y.X,-1.3,1.4])
    plt.plot(y.x,u[m],label = i)
    plt.legend()
    plt.title('t = %.3f' % y.t[m])
    plt.xlabel('x')
    plt.ylabel('u')
    
#Solve for various values of dx and calculate the errors
    
def ploterr(y,i):
    
    DX = [0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1]
    Errors = []
    
    for j in range(0,len(DX)):
        y.changedx(DX[j])
        u = solve(y,i)
        er = error(y,u)
        Errors += [er]
 
    #We want to plot log(error) graphs
    
    Errors = np.array(Errors)
    Errors = np.log(Errors)
    DX = np.array(DX)
    DX = np.log(DX)

    plt.xlabel(r'$\log(\Delta x)$')
    plt.ylabel(r'$\log(Error)$')
    plt.plot(DX,Errors,'o',label = i)
    plt.legend()
    
    

def main():
    
    y = data(X=1,T=1,a=0.7,dx=0.001,dt=0.001)
    
    print("Solving with CNCS")
    
    CNCS = solve(y,'CNCS')
    
    print("Solved with CNCS")
    
    print("Solving with FTBS")
    
    FTBS = solve(y,'FTBS')
    
    print("Solved with FTBS")
    
    print("Calculating exact solution")
    
    Exact = y.exactsolution()
    
    print("Plotting solutions")
    
    plotsol(y, 'CNCS', CNCS, 1)
    plotsol(y,'FTBS', FTBS, 1)
    plotsol(y, 'Exact', Exact,1)
    plt.show()
    
    print("Plotting errors")
    
    ploterr(y,'FTBS')
    ploterr(y,'CNCS')
    
    plt.show()
    
main()




    
    
   