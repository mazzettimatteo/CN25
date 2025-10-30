import numpy as np
import matplotlib.pyplot as plt

f1=lambda x: np.log(x+1)-x
f2=lambda x: x**2-np.cos(x)
f3=lambda x: np.sin(x)-(x/2)
f4=lambda x: np.exp(x)-3*x


g1=lambda x: np.log(x+1)
g2=lambda x: (np.cos(x))**(1/2)
g3=lambda x: 2*np.sin(x)
g4=lambda x: np.exp(x)/3

def puntoFisso(f,g,x0,t1,t2,N,a,b):

    x=np.linspace(a,b,1000)
    plt.figure()
    plt.plot(x,g(x))
    plt.plot(x,x)

    cont=0
    while(np.abs(f(x0))>t1 and cont<N):

        plt.plot([x0,x0],[x0,g(x0)],'k')

        xNew=g(x0)

        #connect
        plt.plot([x0,g(x0)],[xNew,xNew],'k')
        plt.plot(x0,np.abs(g(x0)),'.',color='r')    #se togli abs(fx0) e metti solo fx0 viene bene

        delta=np.abs(x0-xNew)
        if(delta<t2): 
            break
        x0=xNew
        cont+=1
    
    plt.grid()
    plt.show()
    return (x0,f(x0),cont)


iter=10
t=1.e-6

(x,val,it)=puntoFisso(f4,g4,0.2,t,t,iter,-0.5,2)
print(f"x*={x}, f(x*)={val}, iters={it}")


###metodo da ricopiare su pdf
def puntoFisso(f,g,x0,t1,t2,maxIt):
    cont=0
    while(np.abs(f(x0))>t1 and cont<maxIt):
        xNew=g(x0)
        delta=np.abs(x0-xNew)
        if(delta<t2): 
            break
        x0=xNew
        cont+=1
    return (x0,f(x0),cont)