import numpy as np
import matplotlib.pyplot as plt
def graphValsF(funVals, iters):
    funVals = np.array(funVals)
    
    plt.figure(figsize=(6,4))
    plt.plot(range(iters), funVals[:iters], '.', color='blue', label='f(x)')

    plt.xlabel("Iterazioni")
    plt.ylabel("Valore")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def graphValsDF(gradVals, iters):
    
    gradNorms = np.array([np.linalg.norm(g) for g in gradVals])  # norma dei gradienti

    plt.figure(figsize=(6,4))
    plt.plot(range(iters), gradNorms[:iters], '.', color='green', label='||∇f(x)||')

    plt.xlabel("Iterazioni")
    plt.ylabel("Valore")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def graphErrors(e1,e2,e3,eGD,cont1,cont2,cont3,contGD):
    n=len(x0)

    plt.figure(figsize=(6,4))
    plt.plot(range(cont1), e1[:cont1],label="Test 1")
    plt.plot(range(cont2), e2[:cont2],label="Test 2")
    plt.plot(range(cont3), e3[:cont3],label="Test 3")
    plt.plot(range(contGD), eGD[:contGD],label="GD")#???????????????????????????????????????????????????

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def backtrackingFun(f,df,x,alpha=1,rho=0.5,c=1.e-4):
    while f(x-alpha*df(x))>f(x)-c*alpha*(np.linalg.norm(df(x))**2):
        alpha*=rho
    return alpha
def GD(f,df,x0,alpha,maxIt,fToll,xToll,alphaConst):    #f:R^n->R
    cont=0
    n=len(x0)
    dfNorm=np.linalg.norm(df(x0))

    #-------------------------------------------------------------------------------
    xTrue=0.5*(np.arange(1, n+1)+np.sqrt(np.arange(1, n+1)**2+2))
    firstErr=np.linalg.norm(x0-xTrue)/np.linalg.norm(xTrue)
    err=[np.array(firstErr)]

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

        temp=np.linalg.norm(x0 - xTrue)/np.linalg.norm(xTrue)#-------------------------------
        err.append(temp)

        allIters.append(x0)
        fVals.append(f(x0))
        dfVals.append(df(x0))

        dfNorm=np.linalg.norm(df(x0))
        cont+=1
    return x0,f(x0),cont,alphaConst,alpha,allIters,fVals,dfVals,err    



def SGD (f,df,x0,maxEpoche,maxIt,fToll,xToll,alpha,k):
    n=len(x0)
    S_k = np.arange(0,n,1)
    
    cont=0
    
    xTrue=0.5*(np.arange(1, n+1)+np.sqrt(np.arange(1, n+1)**2+2))
    firstErr=np.linalg.norm(x0-xTrue)/np.linalg.norm(xTrue)
    err=[np.array(firstErr)]

    allIters=[np.array(x0)]
    fVals=[np.array(f(x0))] 
    
    batch = S_k[:k]
    dfVals = [np.sum([df(x0, j) for j in batch], axis=0)]

    dfNorm=np.linalg.norm(dfVals[-1])

    for epoca in range(0,maxEpoche):
        np.random.shuffle(S_k) 
        for i in range(0,n,k):
            batch=S_k[i:i+k]
            
            p=-np.sum([df(x0, j) for j in batch], axis=0)

            xNew=x0+alpha*p            

            if (np.linalg.norm(xNew-x0)<xToll): 
                print("deltaX>xToll !!!!!!!!!!!!!!!!!!!!!!")
                return x0,f(x0),epoca,cont,alpha,allIters,fVals,dfVals,err
            x0=xNew
            cont+=1

            allIters.append(x0)
            fVals.append(f(x0))
            dfVals.append(p)
            
            temp=np.linalg.norm(x0 - xTrue)/np.linalg.norm(xTrue)
            err.append(temp)

        dfNorm=np.linalg.norm(dfVals[-1])
        if(dfNorm<fToll):
            print("dfNorm>fToll !!!!!!!!!!!!!!!!!!!!!!")
            break

    print("epoche finite")   
    return x0,f(x0),epoca,cont,alpha,allIters,fVals,dfVals,err

def funF(x):
    n = len(x)
    idx=[]
    for j in range(1,n+1): idx.append(j)
    idx = np.array(idx)         # 1,2,...,n stessa dim di x

    return sum((x - idx)**2) - sum(np.log(x))


def gradFstoc(x,i):
    n = len(x)
    grad=np.zeros_like(x)
    grad[i]=2*(x[i]-(i+1)) - 1/(x[i])
    return grad

fF  = lambda x: funF(x)
dfF = lambda x,i: gradFstoc(x,i)


def GDfunF(x):
    n = len(x)
    idx=[]
    for j in range(1,n+1): idx.append(j)
    idx = np.array(idx)                     
    return np.sum((x - idx)**2) - np.sum(np.log(x))


def GDgradF(x):
    n = len(x)
    idx=[]
    for j in range(1,n+1): idx.append(j)
    idx = np.array(idx)
    return 2*(x - idx) - 1/(x)

fGD  = lambda x: GDfunF(x)
dfGD = lambda x: GDgradF(x)


maxIt=10000
fT=1.e-6
xT=1.e-5
maxEp=150

#xStar=(1.3660254,2.2247449,3.1583124,4.1213204,5.0980762, ecc)

x0 = np.ones(5) 

batchSize=2
(xMin, val, epoche, iters1,a,allIt,fArray,dfArray,Er1)=SGD(fF,dfF,x0,maxEp,maxIt,fT,xT,0.001,batchSize)
res=(xMin, fGD(xMin),a)
print(f"{res}")
print(f"epoche:{epoche}, itersTot={iters1}")
print("----------------------------------------")
batchSize=3
(xMin, val, epoche, iters2,a,allIt,fArray,dfArray,Er2)=SGD(fF,dfF,x0,maxEp,maxIt,fT,xT,0.001,batchSize)
res=(xMin, fGD(xMin),a)
print(f"{res}")
print(f"epoche:{epoche}, itersTot={iters2}")
print("----------------------------------------")
batchSize=1
(xMin, val, epoche, iters3,a,allIt,fArray,dfArray,Er3)=SGD(fF,dfF,x0,maxEp,maxIt,fT,xT,0.001,batchSize)
res=(xMin, fGD(xMin),a)
print(f"{res}")
print(f"epoche:{epoche}, itersTot={iters3}")
print("----------------------------------------")





print("------------------------------GD------------------------------")
(xMin, val, itersGD,alphaWasConst,a,allIt,fArray,dfArray,errGD)=GD(fGD,dfGD,x0,0.001,maxIt,fT,xT,True)
GDres=(xMin, val, itersGD,alphaWasConst,a)
print(GDres)


graphErrors(Er1,Er2,Er3,errGD,iters1,iters2,iters3,itersGD)
