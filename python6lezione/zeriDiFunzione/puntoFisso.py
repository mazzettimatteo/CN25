import numpy as np
import matplotlib.pyplot as plt

#DA RIVEDERE per i plot dentro il while

"""
0) definire la funzione g(x)=x-f(x)phi(x)
1) fai grafico di g(x) e y=x
2) implementare metoo iterativo(con ciclo while)
"""

#0)
f=lambda x: x-np.cos(x)
g=lambda x: np.cos(x)

#1)
x=np.linspace(-1,1,100)
y1=g(x)
y2=x

plt.figure()

plt.plot(x,y1)
plt.plot(x,y2)

plt.show()

#2)

def puntoFisso(f,g,x0,t1,t2,M): 
    #f(x), g(x)=x-f(x)phi(x), x0 punto iniziale, t1 tolleranza per f, t2 tolleranza per x, M numero massimo iterazioni
    cont=0
    while( np.abs(f(x0))>t1 and cont<M):    
        """
        tre condizioni di terminazione possibili:
          su g abs(x_k-g(x_k))<t 
          su f abs(f(x_k))<t    <--usiamo questa
          su x abs(x_k+1 - x_k)<t   <--e questa
        """
        xNew=g(x0)
        delta=np.abs(x0-xNew)
        if(delta<t2): 
            break
        #plt.plot([x0,g(x0)],[xNew,g(xNew)],'green')
        x0=xNew
        cont=cont+1
        #plt.show()
    return(x0,cont)


x0=0
sol=puntoFisso(f,g,x0,1.e-6,1.e-6,100)
print(f"soluzione punto fisso={sol[0]},numero di iterazioni={sol[1]}")