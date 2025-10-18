
import numpy as np
import matplotlib.pyplot as plt

f1=lambda x: np.log(x+1)-x
g1=lambda x: np.log(x+1)
h1=lambda x: 1/(1/(x+1)-1)

iter=50
toll=1.e-10


def puntoFisso(f,g,x0,t1,t2,N):
    cont=0
    while(np.abs(f(x0))>t1 and cont<N): 
        xNew=g(x0)
        delta=np.abs(x0-xNew)
        if(delta<t2): 
            break
        x0=xNew
        cont+=1
    return x0


x1Star=puntoFisso(f1,h1,0.5,toll,toll,iter)
print(f"Uno zero di f1 è in x={x1Star}")
f1val=f1(x1Star)
print(f"Il valore di f1 in x1Star è {f1val}")   #errato