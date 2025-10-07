import numpy as np
import sys

print(sys.float_info) #informazioni sul sistema floating point F usato

print(np.spacing(1e9))#input=distanza fra 10^9 e il successivo

print(4.9-4.845) #risultato stampato non esatto dovuto agli errori,
#dunque mai confrontare con a==b se si lavora con i relali
#confrontare con la vivinanza: abs(a-b)<t

print((0.1+0.2+0.3)==0.6)

#calcoliamo e
esp=np.logspace(0,16,17)
print(esp)

for i in range(0,17):
    s=(1+1/esp[i])**esp[i]
    print("i",i,"errore",np.abs(s-np.exp(1)))

#-----------------------------FUNZIONI FSOLVE-------------------------------------

from scipy import optimize

f = lambda x: np.cos(x)-x
r=optimize.fsolve(f,-2)#x_0=-2
print("r=",r,"f(r)=",f(r))

def bisezione(fun,a,b,iterazioni):
    for i in range(1,iterazioni):
        c=(a+b)/2
        if(fun(c)*fun(a)<0):
            b=c
        else:
            a=c
    return c

print(bisezione(f,-2,2,100))

#fai esercizio su slide 
g=lambda x: x**2-2

print(bisezione(g,0,2,100))