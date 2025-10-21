import numpy as np
import matplotlib.pyplot as plt



#alfa_k costante e numero fisso di iterazioni nel METODO DEL GRADIENTE
def gradientDescent(f,df,x0,alpha,xTrue,N=100):
    """
    Dati del problema:
    f:R^n->R^n, n>=1;
    df=gradiente di f;
    Parametri dell'algoritmo:
    x0=iterato iniziale;
    alpha=passo;
    N=# max di iterazioni, default=100;
    """

    errore=np.zeros((N,))#vettore colonna con N rìghe
    funValues=np.zeros((N,))
    cont=0
    errore[cont]=np.abs(xTrue-x0)
    funValues[cont]=f(x0)
    
    
    while(cont<N-1):  #oppure un ciclo for
        xNew=x0-alpha*df(x0)
        cont+=1
        x0=xNew

        errore[cont]=np.abs(xTrue-x0)
        funValues[cont]=f(x0)
    return (xNew,cont,errore,funValues)


f=lambda x: (x-1)**2 + np.exp(x)
df=lambda x: 2*(x-1)+np.exp(x)
a=1.e-2
x0=0
xTrue=0.31492  #di solito non lo si ha
iter=200

(sol,n,errorArray,funVals)=gradientDescent(f,df,x0,a,xTrue,iter)

print(f"soluzione calcolata {sol} con {n} iterazioni")
print(f"errore assoluto {np.abs(xTrue-sol)}")

plt.plot(errorArray)
#plt.legend("k sulle ascisse")
plt.show()
