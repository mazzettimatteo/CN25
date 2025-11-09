import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make3Dgraph(x, y, z):
    """Draws a 3D surface plot for z = f(x, y)."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=True)
    #ax.set_title("Superficie 3D: z = f(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
def make2Dgraph(x, y, z):
    """Draws a 2D contour plot (level curves) for z = f(x, y)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    contours = ax.contour(x, y, z, levels=20, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    #ax.set_title("Proiezione 2D (curve di livello)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')
    plt.show()
def graph3Dmin(x, y, z,xMin,value):
    """Draws a 3D surface plot for z = f(x, y)."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.plot3D(xMin[0],xMin[1],value,'o',color='red')

    #ax.set_title("Superficie 3D: z = f(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
def graph2Dmin(x, y, z,xMin,value):
    """Draws a 2D contour plot (level curves) for z = f(x, y)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    contours = ax.contour(x, y, z, levels=20, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    #ax.set_title("Proiezione 2D (curve di livello)")
    ax.plot(xMin[0],xMin[1],'o',color='red')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')

#---------------------------------------------funz. A
"""xA = np.linspace(2, 8, 200)
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
"""
#---------------------------------------------funz. B
"""xB = np.linspace(-2, 4, 200)
yB = np.linspace(-2, 4, 200)
XB, YB = np.meshgrid(xB, yB)
ZB=(1-XB)**2 +100*((YB-(XB**2))**2)
def funB(vector):
    x,y=vector
    return ((1-x)**2) +100*((y-(x**2))**2)

def gradB(vector):
    x,y=vector
    grad=np.array([-2*(1-x)-400*x*(y-x**2) , 200*(y-x**2)])
    return grad"""
#----------------------------------------------funz D
"""def funAndGradD(x,n):
    #n must be x lenght
    A=np.random.random((n,n))
    oneVect=np.ones(n)
    b=A @ oneVect
    normContent=(A @ x)-b

    funz=0.5*(np.linalg.norm(normContent)**2)
    grad=A.T @ ((A @ x) - b)

    return (funz,grad)"""
#-----------------------------------------funzF
"""def funF(x,n):
    #x must contasin only positive nums
    i=np.arange(1,n)
    funz=sum((x-i)**2) - sum(np.log((x-i)))
def gradF(x,n):
    i=np.arange(1,n)
    return 2*(x-i)-(1/x)
def stocGradF(x,n):
    """


#graph2Dmin(XA,YA,fun([XA,YA]),xStar,val)
#graph3Dmin(XA,YA,fun([XA,YA]),xStar,val)
plt.show()