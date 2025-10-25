import numpy as np
import matplotlib.pyplot as plt

iter=50
toll=1.e-6

f1=lambda x: np.log(x+1)-x
f2=lambda x: x**2-np.cos(x)
f3=lambda x: np.sin(x)-(x/2)
f4=lambda x: np.exp(x)-3*x


a1=-0.5; b1=0; 
x1=np.linspace(a1,b1,100)
y1=f1(x1)
xAxis1=np.zeros_like(x1)

a2=-1.5; b2=0; 
x2=np.linspace(a2,b2,100)
y2=f2(x2)
xAxis2=np.zeros_like(x2)

g1=lambda x: np.log(x+1)
g2=lambda x: (np.cos(x))**(1/2)
g3=lambda x: 2*np.sin(x)
g4=lambda x: np.exp(x)/3

df1=lambda x: 1/(x+1)-1
df2=lambda x: 2*x+np.sin(x)
df3=lambda x: np.cos(x)-(1/2)
df4=lambda x: np.exp(x)-3

def newton(f,df,x0,t1,t2,N,a,b):
    x=np.linspace(a,b,1000)
    xAxis=np.zeros_like(x)
    plt.figure()
    plt.plot(x,f(x))
    plt.plot(x,xAxis)
    cont=0
    while(np.abs(f(x0))>t1 and cont<N): 
        plt.plot(x0,np.abs(f(x0)),'.',color='r')    #se togli abs(fx0) e metti solo fx0 viene bene
        xNew=x0-f(x0)/df(x0)
        plt.plot([xNew,x0],[0,f(x0)])
        delta=np.abs(x0-xNew)
        if(delta<t2): 
            break
        x0=xNew
        cont+=1
    plt.show()
    return x0




x1Star=newton(f1,df1,0.5,toll,toll,iter,a1,b1)
print(f"Uno zero di f1 è in x={x1Star}")
f1val=f1(x1Star)
print(f"Il valore di f1 in x1Star è {f1val}")   

x2Star=newton(f2,df2,0.5,toll,toll,iter,a2,b2)
print(f"Uno zero di f2 è in x={x2Star}")
f2val=f2(x2Star)
print(f"Il valore di f2 in x2Star è {f2val}")





