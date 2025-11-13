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
def funAndGradD(x,n):
    #n must be x lenght
    A=np.random.random((n,n))
    oneVect=np.ones(n)
    b=A @ oneVect
    normContent=(A @ x)-b

    funz=0.5*(np.linalg.norm(normContent)**2)
    grad=A.T @ ((A @ x) - b)

    return (funz,grad)



#-----------------------------------------funzF
def funF(x,n):
    #x must contasin only positive nums
    i=np.arange(1,n)
    funz=sum((x-i)**2) - sum(np.log((x-i)))
def gradF(x,n):
    i=np.arange(1,n)
    return 2*(x-i)-(1/x)
"""
def stocGradF(x,n):
    """
