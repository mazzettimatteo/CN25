import numpy as np
import matplotlib.pyplot as plt



#0)
f=lambda x: x-np.cos(x)
df=lambda x: 1+np.sin(x)


#2)

def puntoFisso(f,df,x0,t1,t2,M): 
    cont=0
    while(np.abs(f(x0))>t1 and cont<M):
        xNew=x0-(f(x0)/df(x0))
        delta=np.abs(x0-xNew)
        if(delta<t2):
            break
        x0=xNew
        cont+=1
    return(xNew,cont)


x0=0
sol=puntoFisso(f,df,x0,1.e-6,1.e-6,100)
print(f"soluzione newton={sol[0]},numero di iterazioni={sol[1]}")