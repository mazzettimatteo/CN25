import numpy as np
import matplotlib.pyplot as plt
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

def GD(f,df,x0,alpha,maxIt,fToll,xToll,alphaConst):    #f:R^n->R
    cont=0
    n=len(x0)
    dfNorm=np.linalg.norm(df(x0))
    allIters=[np.array(x0)]
    while cont<maxIt and dfNorm>fToll:
        p=-df(x0)
        alpha=alpha if alphaConst else backtrackingFun(f,df,x0)#backtracking o costante
        xNew=x0+alpha*p
        if(np.linalg.norm(xNew-x0)<xToll):
            break
        x0=xNew
        allIters.append(x0.copy())
        dfNorm=np.linalg.norm(df(x0))
        cont+=1
    return x0,f(x0),cont,alphaConst,alpha,allIters    #p.to di minimo, val di minimo, iterazioni fatte




maxIt=10000
fT=1.e-6
xT=1.e-5

(xMin, val, iters,alphaWasConst,a,allIt)=GD(funB, gradB, (0,2), 0.001, maxIt, fT, xT, False)
res=(xMin, val, iters,alphaWasConst,a)
print(f"{res}")

graph2Diters(XB, YB, ZB, allIt)
graph3Diters(XB, YB, ZB, allIt, funB)