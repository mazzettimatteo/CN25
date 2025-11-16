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
def graphErrors(erroreRel,cont,n):
    n=len(x0)

    plt.figure(figsize=(6,4))
    plt.plot(range(cont), erroreRel[:cont],color="orange",label="Errore relativo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def SGD (f,df,x0,maxEpoche,maxIt,fToll,xToll,alpha,k):
    n=len(x0)
    S_k = np.arange(0,n,1) # Inizializziamo il dataset
    
    cont=0
    #-------------------------------------------------------------------------------------
    xTrue=0.5*(np.arange(1, n+1)+np.sqrt(np.arange(1, n+1)**2+2))
    firstErr=np.linalg.norm(x0-xTrue)/np.linalg.norm(xTrue)
    err=[np.array(firstErr)]

    allIters=[np.array(x0)]
    fVals=[np.array(f(x0))] 
    
    batch = S_k[:k]#select the first k indexes
    dfVals = [np.sum(df(x0, j) for j in batch)]

    dfNorm=np.linalg.norm(dfVals[-1])

    for epoca in range(0,maxEpoche):
        np.random.shuffle(S_k)  #randomizziamo gli indici del mini-batch
        for i in range(0,n,k):
            batch=S_k[i:i+k]
            
            p=-np.sum([df(x0, j) for j in batch])

            xNew=x0+alpha*p            

            if (np.linalg.norm(xNew-x0)<xToll): # Controllo se sto facendo progressi (tolleranza x)
                print("deltaX>xToll !!!!!!!!!!!!!!!!!!!!!!")
                break
            x0=xNew
            cont+=1

            allIters.append(x0.copy())
            fVals.append(f(x0).copy())
            dfVals.append(p)
            #-------------------------------------------------------------------------------------------------
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
    idx = np.array(idx)         # 1,2,...,n  (same length as x)
    if np.any(x - idx <= 0):
        return np.inf               # outside domain → reject
    return sum((x - idx)**2) - sum(np.log(x))


def gradFstoc(x,i):
    n = len(x)
    grad=np.zeros_like(x)
    grad[i]=2*(x[i]-(i+1)) - 1/(x[i])
    return grad

fF  = lambda x: funF(x)
dfF = lambda x,i: gradFstoc(x,i)

maxIt=10000
fT=1.e-10
xT=1.e-5
maxEp=100

#xStar=(1.3660254,2.2247449,3.1583124,4.1213204,5.0980762)

x0 = np.array([1.1,2.1,3.1,4.1,5.1]) # must satisfy x0[k] > k+1

batchSize=2
(xMin, val, epoche, iters,a,allIt,fArray,dfArray,Er)=SGD(fF,dfF,x0,maxEp,maxIt,fT,xT,0.001,batchSize)
res=(xMin, val,a)
print(f"{res}")
print(f"epoche:{epoche}, itersTot={iters}")
graphErrors(Er,iters,x0)
print("----------------------------------------")
batchSize=3
(xMin, val, epoche, iters,a,allIt,fArray,dfArray,Er)=SGD(fF,dfF,x0,maxEp,maxIt,fT,xT,0.001,batchSize)
res=(xMin, val,a)
print(f"{res}")
print(f"epoche:{epoche}, itersTot={iters}")
graphErrors(Er,iters,x0)
print("----------------------------------------")
batchSize=4
(xMin, val, epoche, iters,a,allIt,fArray,dfArray,Er)=SGD(fF,dfF,x0,maxEp,maxIt,fT,xT,0.001,batchSize)
res=(xMin, val,a)
print(f"{res}")
print(f"epoche:{epoche}, itersTot={iters}")
graphErrors(Er,iters,len(x0))
print("----------------------------------------")