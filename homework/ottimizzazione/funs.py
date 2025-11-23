import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------funz. A
xA = np.linspace(2, 8, 200)
yA = np.linspace(-1, 5, 200)
XA, YA = np.meshgrid(xA, yA)
ZA=(XA-5)**2+(YA-2)**2
#lets convert that int a real function
def funA(vector):
    x,y=vector
    return (x-5)**2+(y-2)**2
def gradA(vector):
    x,y=vector
    return np.array([2*(x-5),2*(y-2)])


#---------------------------------------------funz. B
xB = np.linspace(-2, 4, 200)
yB = np.linspace(-2, 4, 200)
XB, YB = np.meshgrid(xB, yB)
ZB=(1-XB)**2 +100*((YB-(XB**2))**2)
def funB(vector):
    x,y=vector
    return ((1-x)**2) +100*((y-(x**2))**2)

def gradB(vector):
    x,y=vector
    grad=np.array([-2*(1-x)-400*x*(y-x**2) , 200*(y-x**2)])
    return grad


#----------------------------------------------funz D
def D(wantF,x):
    #n must be x lenght
    n=len(x)
    A=np.random.random((n,n))
    oneVect=np.ones(n)
    b=A @ oneVect
    normContent=(A @ x)-b

    funz=0.5*(np.linalg.norm(normContent)**2)
    grad=A.T @ ((A @ x) - b)

    output=funz if wantF else grad
    return output


#-----------------------------------------funzF
def funF(x):
    n = len(x)
    idx=[]
    for j in range(1,n+1): idx.append(j)
    idx = np.array(idx)         # 1,2,...,n  (same length as x)
    if np.any(x - idx <= 0):
        return np.inf               # outside domain → reject
    return np.sum((x - idx)**2) - np.sum(np.log(x))


def gradF(x):
    n = len(x)
    idx=[]
    for j in range(1,n+1): idx.append(j)
    idx = np.array(idx)
    if np.any(x - idx <= 0):
        raise ValueError("Gradient undefined: x_k must be > k")
    return 2*(x - idx) - 1/(x)

fF  = lambda x: funF(x)
dfF = lambda x: gradF(x)



#-------------------------------------------------funzDstoc
def makeD(A):
    b=A@np.ones(A.shape[1])

    def f(x, batch):
        ABatch=A[batch, :]
        bBatch=b[batch]
        residue=ABatch@x - bBatch
        
        return 0.5*(np.linalg.norm(residue))**2
    
    def df(x, batch):
        ABatch=A[batch, :]
        bBatch=b[batch]
        residue=ABatch@x-bBatch

        return ABatch.T@residue
    
    return f,df

Dstoc=makeD(A)

fDstoc,dfDstoc=Dstoc