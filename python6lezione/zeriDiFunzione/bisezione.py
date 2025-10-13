#all'esame codice non commentato


import numpy as np
import matplotlib.pyplot as plt


"""
usiamo la funzione f(x)=x-cos(x) nell'intervallo [-1,1]
0) definiamo la funzione che useremo
1) facciamo il grafico della funzione
2) implementiamo il metodo di bisezione
"""

#0)
f=lambda x: x-np.cos(x)

#1)
a=-1
b=1
x=np.linspace(a,b,100)
y=f(x)

plt.figure()
plt.plot(x,y)

y1=np.zeros(len(x))#disegnamo anche l'asse x

plt.plot(x,y1,'red')
plt.show()

#2)

def bisezione(fun,a,b,N):
    for i in range(N):
        c=(a+b)/2
        if(fun(a)*fun(c)<0):
            b=c
        else:
            a=c
    return c


n=15
sol=bisezione(f,a,b,n)
print(f"soluzione bisezione {sol}")