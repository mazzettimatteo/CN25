import numpy as np
import matplotlib.pyplot as plt
import funs as myFun


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
    plt.plot(range(iters), funVals[:iters], '.', color='blue', label='f(x)',markersize=2)

    plt.xlabel("Iterazioni")
    plt.ylabel("Valore")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def graphValsDF(gradVals, iters):
    
    gradNorms = np.array([np.linalg.norm(g) for g in gradVals])  # norma dei gradienti

    plt.figure(figsize=(6,4))
    plt.plot(range(iters), gradNorms[:iters], '.', color='green', label='||∇f(x)||',markersize=2)

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

#funzione D
def make_D(A):
    def D(wantF, x):
        n=len(x)
        oneVect=np.ones(n)
        b=A @ oneVect

        normContent=(A @ x)-b

        funz=0.5*(np.linalg.norm(normContent)**2)
        grad=A.T @ ((A @ x) - b)

        return funz if wantF else grad
    return D

n = 5
A = np.random.rand(n,n)
Dfun = make_D(A)

fD  = lambda x: Dfun(True,  x)
dfD = lambda x: Dfun(False, x)

x0rand = np.random.randn(n)
print(f"matrice A:{A}")
print(f"x0rand: {x0rand}")

print("-----------------------------------------------")

(xMin, val, iters,alphaWasConst,a,allIt,fArray,dfArray)=GD(fD,dfD, x0rand, 0.001, maxIt, fT, xT, False)
res=(xMin, val, iters,alphaWasConst,a)
print(f"{res}")

graphValsF(fArray,iters)
graphValsDF(dfArray,iters)


print("-----------------------------------------------")
(xMin, val, iters,alphaWasConst,a,allIt,fArray,dfArray)=GD(fD,dfD, x0rand, 0.001, maxIt, fT, xT, True)
res=(xMin, val, iters,alphaWasConst,a)
print(f"{res}")
graphValsF(fArray,iters)
graphValsDF(dfArray,iters)

print("-----------------------------------------------")
(xMin, val, iters,alphaWasConst,a,allIt,fArray,dfArray)=GD(fD,dfD, x0rand, 0.005, maxIt, fT, xT, True)
res=(xMin, val, iters,alphaWasConst,a)
print(f"{res}")
graphValsF(fArray,iters)
graphValsDF(dfArray,iters)

print("-----------------------------------------------")
x0rand = np.random.randn(n)
print(f"nuovo x0rand: {x0rand}")
(xMin, val, iters,alphaWasConst,a,allIt,fArray,dfArray)=GD(fD,dfD, x0rand, 0.001, maxIt, fT, xT, False)
res=(xMin, val, iters,alphaWasConst,a)
print(f"{res}")
graphValsF(fArray,iters)
graphValsDF(dfArray,iters)



#graph2Diters(myFun.XA, myFun.YA, myFun.ZA, allIt)
#graph3Diters(myFun.XA, myFun.YA, myFun.ZA, allIt, myFun.funA)


def SGD (f,df,x0,maxEpoche,fToll,xToll,alpha,k):
    n=len(x0)
    S_k = np.arange(0,n,1)
    cont=0
    batch = S_k[:k]
    dfVals = [np.sum(df(x0, j) for j in batch)]
    dfNorm=np.linalg.norm(dfVals[-1])
    for epoca in range(0,maxEpoche):
        np.random.shuffle(S_k) 
        for i in range(0,n,k):
            batch=S_k[i:i+k]
            p=-np.sum([df(x0, j) for j in batch])
            xNew=x0+alpha*p            
            if (np.linalg.norm(xNew-x0)<xToll): 
                return x0,f(x0),epoca,cont
            x0=xNew
            cont+=1
            dfVals.append(p)
        dfNorm=np.linalg.norm(dfVals[-1])
        if(dfNorm<fToll):
            break   
    return x0,f(x0),epoca,cont
