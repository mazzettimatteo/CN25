import numpy as np
import matplotlib.pyplot as plt

f1=lambda x: np.log(x+1)-x
f2=lambda x: x**2-np.cos(x)
f3=lambda x: np.sin(x)-(x/2)
f4=lambda x: np.exp(x)-3*x

df1=lambda x: 1/(x+1)-1
df2=lambda x: 2*x+np.sin(x)
df3=lambda x: np.cos(x)-(1/2)
df4=lambda x: np.exp(x)-3

def newton(f,Df,x0,t1,t2,N,a,b):

    x=np.linspace(a,b,100)
    xAxis=np.zeros_like(x)
    plt.figure()
    plt.plot(x,f(x))
    plt.plot(x,xAxis)

    cont=0
    while (np.abs(f(x0))>t1 and cont<N):
        fx0=f(x0)
        Dfx0=Df(x0)

        plt.plot([x0,x0], [fx0,0],'k')  #da(x0,fx0) a (x0,0)
        xNew = x0 - fx0/Dfx0
        plt.plot([x0,xNew], [fx0,0],'k')
        plt.plot(x0,np.abs(fx0),marker = '.',color = 'r')
        
        delta=np.abs(x0-xNew)
        if(delta<t2): 
            break
        x0 = xNew
        
        cont+=1
    plt.grid()
    plt.show()
    return(x0,fx0,cont)

iter=500
t=1.e-12

(x,val,it)=newton(f4,df4,1,t,t,iter,-2,2)
print(f"x*={x}, f(x*)={val}, iters={it}")