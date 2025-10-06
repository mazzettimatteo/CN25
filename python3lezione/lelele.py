#libreriee built-in
from random import * 
#con * importo tutte le funzione e non devo piu mettere random.function ma solo function
import math as m

print(m.exp(2))





def cubo(a:float)->float:
    return a**3

#associo alla variabile x la funzione cubo
x=cubo
print(x)
print(type(x))
print(x(3))

#funzione che prende in input una funzione

def funzione(w, v:tuple[float])->list[float]:
    y=[0]*len(v)

    for _i in range(0,len(v)):
        y[_i]=w(v[_i])
    
    return y

t=(1,2,3)
print(funzione(x,t))
print(funzione(cubo,t))

#funzioni lambda
quadrato = lambda x: x**2   #nome = lambda inputVar: returnValue
print(quadrato(6))

def somma(*args):#*args è una tupla
    #dati in input tot argomenti, ne restituisco la somma
    res=0
    for arg in args:
        res=res+arg

    return res

print(somma(1,2,3,4,5,6,7,8,9,0))