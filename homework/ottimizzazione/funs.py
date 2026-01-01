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

#-----------------------------------------funzF per GD


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

    return 2*(x - idx) - 1/(x)






#-----------------------------------------funzF per SGD
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