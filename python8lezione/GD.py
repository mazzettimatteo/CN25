import numpy as np
import matplotlib.pyplot as plt
"""
Implementazione del metodo del gradiente (Gradient Descent).

Parametri:
----------
f : funzione
    La funzione obiettivo da minimizzare.
df : funzione
    La derivata prima (se f: ℝ → ℝ) oppure il gradiente di f (se f: ℝⁿ → ℝ).
x0 : float o np.ndarray
    Punto iniziale da cui parte l'algoritmo (stima iniziale della soluzione).
alpha : float
    Tasso di apprendimento (lunghezza del passo) che determina quanto si avanza
    nella direzione opposta al gradiente a ogni iterazione.
maxit : int, opzionale (default=100)
    Numero massimo di iterazioni consentite per l’algoritmo.

Ritorna:
--------
x : float o np.ndarray
    Ultimo punto calcolato (approssimazione della soluzione).
"""

def backtracking(f, df, x):

    alpha=1
    rho=1/2
    c=1.e-4
    while f(x-alpha*df(x)) > f(x) +  c*alpha*np.linalg.norm(df(x)):#condizioni d'armio
        alpha = rho*alpha

    return (alpha)

def GD(f, df, x0, maxit, x_true, tolf, tolx):#alpha
    k=0 #contatore 

    grad=np.zeros((maxit,))
    errore=np.zeros((maxit,))
    fun=np.zeros((maxit,))

    #inizializzazione vettori
    errore[k]=np.abs(x_true-x0)
    fun[k]=f(x0)
    grad[k]=np.abs(df(x0))


    """
    criteri d'arresto:
    ||df(xk)||<t1   gradiente =0
    ||xk+1 -xk||<t2 i due iterati molto vicini
    k<=maxit    numero di iterazioni massimo(non è detto che abbiamo trovato il punto stazionario,
                abbiamo solo finito le iterazioni) 
                a diff. degli altri è un termine di continuazione e non di arresto
    """
    condizione=True

    while(condizione):
        alpha=backtracking(f,df,x0) # lo calcolo su x_k vecchio che nel nostro caso è x0
        #CALCOLO IL NUOVO ITERATO
        x=x0-alpha*df(x0)
        k=k+1
        
        x0=x

        errore[k]=np.abs(x_true-x0)
        fun[k]=f(x0)
        grad[k]=np.abs(df(x0))


        condizione=(k<maxit) and (np.linalg.norm(df(x0))**2>tolf) and (np.linalg.norm(x-x0)>tolx)


        errore=errore[:k+1] #come scrivere slicing 0:k+1, prendo tutti i num da 0 a k+1
        fun=fun[:k+1]
        grad=grad[:k+1]

    return (x,k,errore,fun,grad)


'''

def f(x):
    2*(x-1)**2+np.exp(x)
'''

f=lambda x: (x-1)**2+np.exp(x)

df=lambda x: 2*(x-1)+np.exp(x)


x0=0

#per controllare convergenza 
x_true=0.31492

maxit=400
tolF=1.e-6
tolX=1.e-5

sol,numit,errore,fun,grad =GD(f,df,x0,maxit,x_true,tolF,tolX)
print(f'Soluzione calclata: {sol} in {numit} iterazioni')

print(f'Errore {np.abs(x_true-sol)}')

'''

plt.plot(errore)
plt.show()

plt.figure()
plt.plot(fun,'r')
plt.show()


plt.figure()
plt.plot(grad,'c')#si vede molto bene che il gradiente tende a zero
plt.show()


plt.figure()
plt.semilogy(grad,'r') 
plt.show()
'''

plt.figure()
plt.loglog(grad,'m') 
plt.show()


#PARTE 1: Modifichiamo: iterazioni,punto iniziale e passo

#passo più piccolo richiede più iterazioni per arrivare al punto cercato
#ATTENZIONE: se il passo è troppo grande salto il mio minimo e vado a qualcosa che non converge al minimo (es 1)


#PARTE 2: Backtrackin e criteri d'arresto
