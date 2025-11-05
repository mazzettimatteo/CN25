import numpy as np
import matplotlib.pyplot as plt


def backtrackingFun(f,df,x,alpha=1,rho=0.5,c=1.e-4):
    while f(x-alpha*df(x))>f(x)-c*alpha*(np.linalg.norm(df(x))**2):
        alpha*=rho
    return alpha

def GD(f,df,x0,alpha,maxIt,fToll,xToll,alphaConst):    #f:R^n->R
    cont=0
    n=len(x0)
    dfNorm=np.linalg.norm(df(x0))
    while cont<maxIt and dfNorm>fToll:
        p=-df(x0)
        alpha=alpha if alphaConst else backtrackingFun(f,df,x0)#backtracking o costante
        xNew=x0+alpha*p
        if(np.linalg.norm(xNew-x0)<xToll):
            break
        x0=xNew
        dfNorm=np.linalg.norm(df(x0))
        cont+=1
    return x0,f(x0),cont    #p.to di minimo, val di minimo, iterazioni fatte
