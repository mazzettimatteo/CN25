import numpy as np
import matplotlib.pyplot as plt
#import funs as myFun


def graph3Diters(x, y, z, traj,f):
    traj=np.array(traj)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=True)

    zTraj=[f(i) for i in traj]
    ax.plot3D(traj[:,0], traj[:,1], zTraj, 'r.-', label='GD path')
    ax.plot3D(traj[-1,0], traj[-1,1], 'bo', label='Minimum') # final point

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
def graph2Diters(x, y, z,traj):
    traj=np.array(traj)
    fig, ax = plt.subplots(figsize=(6, 6))
    contours = ax.contour(x, y, z, levels=20, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)

    ax.plot(traj[:,0], traj[:,1], 'r.-', label='GD path')  # red line with dots
    ax.plot(traj[-1,0], traj[-1,1], 'bo', label='Minimum') # final point

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.axis('equal')
    plt.show()
def backtrackingFun(f,df,x,alpha=1,rho=0.5,c=1.e-4):
    while f(x-alpha*df(x))>f(x)-c*alpha*(np.linalg.norm(df(x))**2):
        alpha*=rho
    return alpha
def graphValsF(funVals, iters):
    funVals = np.array(funVals)
    
    plt.figure(figsize=(6,4))
    plt.plot(range(iters), funVals[:iters], '.', color='blue', label='f(x)',ms=2)

    plt.xlabel("Iterazioni")
    plt.ylabel("Valore")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def graphValsDF(gradVals, iters):
    
    gradNorms = np.array([np.linalg.norm(g) for g in gradVals])  # norma dei gradienti

    plt.figure(figsize=(6,4))
    plt.plot(range(iters), gradNorms[:iters], '.', color='green', label='||∇f(x)||',ms=2)

    plt.xlabel("Iterazioni")
    plt.ylabel("Valore")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def GD(f,df,x0,alpha,maxIt,fToll,xToll,alphaConst):    #f:R^n->R
    cont=0
    n=len(x0)
    dfNorm=np.linalg.norm(df(x0))

    allIters=[np.array(x0)]
    fVals=[np.array(f(x0))]
    dfVals=[np.array(df(x0))]

    while cont<maxIt and dfNorm>fToll:
        p=-df(x0)
        alpha=alpha if alphaConst else backtrackingFun(f,df,x0)#backtracking o costante
        xNew=x0+alpha*p
        if(np.linalg.norm(xNew-x0)<xToll):
            break
        x0=xNew

        allIters.append(x0.copy())
        fVals.append(f(x0).copy())
        dfVals.append(df(x0).copy())

        dfNorm=np.linalg.norm(df(x0))
        cont+=1
    return x0,f(x0),cont,alphaConst,alpha,allIters,fVals,dfVals    
#p.to di minimo, val di minimo, iterazioni fatte, alfaConst,alfa,traj[],fVals[],dfVals[]

maxIt=10000
fT=1.e-6
xT=1.e-5

<<<<<<< HEAD
=======

#---------------------------------------------funz. B
>>>>>>> 96bb6bdcca32b573ec3e9b2607c09ef58d858e2e
xB = np.linspace(-2, 4, 200)
yB = np.linspace(-2, 4, 200)
XB, YB = np.meshgrid(xB, yB)
ZB=(1-XB)**2 +100*((YB-(XB**2))**2)
def funB(vector):
    x,y=vector
    return ((1-x)**2) +100*((y-(x**2))**2)
<<<<<<< HEAD
=======

def gradB(vector):
    x,y=vector
    grad=np.array([-2*(1-x)-400*x*(y-x**2) , 200*(y-x**2)])
    return grad
>>>>>>> 96bb6bdcca32b573ec3e9b2607c09ef58d858e2e

def gradB(vector):
    x,y=vector
    grad=np.array([-2*(1-x)-400*x*(y-x**2) , 200*(y-x**2)])
    return grad

<<<<<<< HEAD
x0=(0.0,0.0)

(xMin, val, iters,alphaWasConst,a,allIt,fArray,dfArray)=GD(funB,gradB, x0, 0.0010, maxIt, fT, xT, True)
=======
(xMin, val, iters,alphaWasConst,a,allIt,fArray,dfArray)=GD(funB, gradB, (0,2), 0.001, maxIt, fT, xT, True) 
>>>>>>> 96bb6bdcca32b573ec3e9b2607c09ef58d858e2e
res=(xMin, val, iters,alphaWasConst,a)
print(f"{res}")

graphValsF(fArray,iters)
graphValsDF(dfArray,iters)

#graph2Diters(myFun.XA, myFun.YA, myFun.ZA, allIt)
#graph3Diters(myFun.XA, myFun.YA, myFun.ZA, allIt, myFun.funA)

