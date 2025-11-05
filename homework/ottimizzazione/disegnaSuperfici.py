import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make3Dgraph(x, y, z):
    """Draws a 3D surface plot for z = f(x, y)."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_title("Superficie 3D: z = f(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def make2Dgraph(x, y, z):
    """Draws a 2D contour plot (level curves) for z = f(x, y)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    contours = ax.contour(x, y, z, levels=20, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_title("Proiezione 2D (curve di livello)")
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

    ax.set_title("Superficie 3D: z = f(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def graph2Dmin(x, y, z,xMin,value):
    """Draws a 2D contour plot (level curves) for z = f(x, y)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    contours = ax.contour(x, y, z, levels=20, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_title("Proiezione 2D (curve di livello)")
    ax.plot(xMin[0],xMin[1],'o',color='red')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')


x = np.linspace(2, 8, 200)
y = np.linspace(-1, 5, 200)
X, Y = np.meshgrid(x, y)
Z=(X-5)**2+(Y-2)**2
#lets convert that int a real function
def fun(vector):
    x,y=vector
    return (x-5)**2+(y-2)**2
def grad(vector):
    x,y=vector
    return np.array([2*(x-5),2*(y-2)])


make3Dgraph(X,Y,fun(X,Y))
make2Dgraph(X,Y,fun(X,Y))
#graph2Dmin(X,Y,fun([X,Y]),xStar,val)
#graph3Dmin(X,Y,fun([X,Y]),xStar,val)
plt.show()