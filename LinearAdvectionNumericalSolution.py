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

def gauss(x):
    return np.exp(-(x-0.1)**2/0.001)

def step(x):
    return 1 - (x<0.1) - (x>0.2)

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
        return gauss(x)
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
    

#Function solving the linear advection equation using the selected method
def solve(y,i):
    D = C(i,y.N,y.c)
    u = np.zeros(shape = (y.M,y.N))
    
    #Set initial condition
    u[0] = y.f(y.x)
    
    #Iterate
    for m in range(1,y.M):
        u[m] = np.dot(D,u[m-1])
    return u


#Function to calculate the l1 normed error
def error(y,u):
    ex = y.exactsolution()
    er = np.abs(ex - u)
    Tot = sum(sum(u))
    ertot = sum(sum(er))/Tot
    return ertot


def dxvalues(L, h):
    DX = []
    for k in range(1,L+1):
        DX += [k*h]
    DX = np.array(DX)
    logDX = np.log(DX)
    return DX, logDX


def errorlogerror(y, i, DX):
    Errors = []
    
    for j in range(0,DX.size):
        y.changedx(DX[j])
        y.changedt(DX[j]) # this keeps the courant number the same
        u = solve(y,i)
        er = error(y,u)
        Errors += [er]    

    #We want to plot log(error) graphs
    Errors = np.array(Errors)
    logErrors = np.log(Errors)
    return Errors, logErrors


#Plot the graph of the solution at specified time and print the total error
def plotsol(y,i,u,tt):
    m = int((y.M-1)*tt/y.T)
    plt.axis([0,y.X,-1.3,1.4])
    plt.plot(y.x,u[m],label = i)
    plt.legend()
    plt.rc('text', usetex = True)
    plt.title(r'$t = %.3f$' % y.t[m])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    
    
#Output a table of error values
def table(DX, Errors, logDX, logErrors, d2, d1, d0):
    if (DX.size != Errors.size):
        print("Mismatched array size")
    else:
        print('%20s %20s %20s' % ('dx', 'Error', 'Gradient'))
        grad = 2*d2*logDX + d1 #we want the gradient of the log-log graph
        for j in range(0,DX.size):
            s = '\hline\n%20.3f & %20f & %20f' % (DX[j], Errors[j], grad[j])
            print(s)#prints in LaTeX format
     
        
#Solve for various values of dx and calculate the errors
def ploterr(logDX, logErrors, i, d2, d1, d0):
    plt.rc('text', usetex = True)
    plt.xlabel(r'$\log(\Delta x)$')
    plt.ylabel(r'$\log(Error)$')
    plt.plot(logDX,logErrors,'o',label = i)
    plt.plot(logDX, d2*(logDX**2) + d1*(logDX) + d0,label = i)
    plt.legend()
    
    
#Solve then plot at the specified time:
def SOLVING(y, methods): 
     for i in methods:
        print("\nRunning %s with dx = %.3f, dt = %.3f, a = %.2f" 
              % (i,y.dx,y.dt,y.c))
        u = solve(y,i)
        print("\nSolved with %s" % i)
        plotsol(y, i, u, 1)


#Solves for various values of dx then plots log(dx) against log(error)
def ERRORPLOTTING(y, methods, DX, logDX):
    for i in methods:
        print("\nTable of errors for %s:" % i)
        Errors, logErrors = errorlogerror(y,i,DX)
        d2,d1,d0 = np.polyfit(logDX, logErrors, 2) #quadratic of best fit
        table(DX, Errors, logDX, logErrors, d2, d1, d0)
        ploterr(logDX, logErrors, i, d2, d1, d0)


def main():
    
    y = data(X=1,T=1,a=0.7,dx=0.001,dt=0.001)
    methods = ['CNCS', 'FTBS']
    DX,logDX = dxvalues(100,0.001)
    
    print("\nCalculating exact solution")
    
    Exact = y.exactsolution()
    
    print("\nPlotting solutions")
    
    SOLVING(y,methods)
        
    plotsol(y, 'Exact', Exact,1)
    
    plt.show()
    
    print("\nCalculating errors")

    ERRORPLOTTING(y, methods, DX, logDX)
    
    print("\nPlotting errors for both methods:")
    
    plt.show()
    
    print("\a")
    
    
main()




    
    
   