import numpy as np
import matplotlib.pyplot as plt

iter=50
toll=1.e-10

f1=lambda x: np.log(x+1)-x
f2=lambda x: x**2-np.cos(x)
f3=lambda x: np.sin(x)-(x/2)
f4=lambda x: np.exp(x)-3*x



def bisezione(f,a,b,N,t):
    
    x=np.linspace(a,b,100)
    xAxis=np.zeros_like(x)
    plt.figure()
    plt.plot(x,f(x))
    plt.plot(x,xAxis)
    i=0

    if(f(a)*f(b)>0):
        raise ValueError("f(a)*f(b)>0")

    if(np.abs(f(a))<t): return (a,f(a),i)
    elif(np.abs(f(b))<t): return (b,f(b),i)
    
    for i in range(N):
        c=(a+b)/2
        fc=f(c)

        plt.plot(c,np.abs(fc),'.',color='r')    #se togli abs(fc) e metti solo fc viene bene

        if(abs(fc)<t):
            return (c,f(c),i)
        elif(f(a)*fc<0):
            b=c
        else:
            a=c
            
    plt.show()
    return (c,f(c),i)

(x,val,it)=bisezione(f4,0,1.5,iter,toll)
print(f"x*={x}, f(x*)={val}, iters={it}")
plt.grid()
plt.show()
###funzione da ricopiare in pdf


"""
def bisezione(f,a,b,maxIt,t):
    i=0
    if(f(a)*f(b)>0):
        raise ValueError("f(a)*f(b)>0")
    
    if(np.abs(f(a))<t): return (a,f(a),i)
    elif(np.abs(f(b))<t): return (b,f(b),i)

    for i in range(maxIt):
        c=(a+b)/2
        fc=f(c)
        if(abs(fc)<t):
            return (c,fc,i)
        elif(f(a)*fc<0):
            b=c
        else:
            a=c
            
    return (c,fc,i)
"""