import matplotlib.pyplot as plt
import numpy as np


"""
1) scelgo un modello predittivo f(z,theta)
2) calcolo theta (vettore) utilizzando i dati a disposizione
3) minimizzo loss function L( min||phi*theta-y||^2)
"""

##Creazione del problema test
"""
1) scelgo le mie x_i da 1..n
2) scelgo la funzione f e calcolo y_i=f(x_i)=theta0+theta1x_1+theta2(x_2)^2+...
3) aggiungiamo una casualità(rumore e) a ogni y
"""
def f(x, alpha): # Definiamo la funzione f(x, alpha) che prende in input il vettore alpha dei coefficienti e un valore x, e ritorna il valore del rispettivo polinomio
    d = alpha.shape[0] - 1 #il grado è il numero di componenti di alpha -1
    #shape ritonra una coppia, per prendere la lunghezza prendo shape[0]
    y = 0 #y scalare in cui memorizzo y+alpha[i]*x^i
    for i in range(d+1):
        y = y + alpha[i] * x ** i
    return y

# Definiamo alpha_true = (1, 1, ..., 1) 
# NOTA: si può scegliere un QUALUNQUE vettore alpha_true. Quello indicato è solo un esempio.
d = 3 # grado del polinomio
alpha_true = np.ones((d+1,))

# Scegliamo un valore di x
x = 3
y = f(x, alpha_true)
print(f"(x = {x}, y = {y})")

# Numero di dati
n = 15

# Definiamo gli x_i
x = np.linspace(0, 1, n)

#creiamo il rumore e
sigma = 0.1 # Definiamo la deviazione standard del rumore
e = np.random.normal(loc=0, scale=sigma, size=(n, )) # Generiamo il rumore

y = np.zeros_like(x) # Inizializziamo y vettore
for i in range(n): #calcolo y_i per ogni x_i
    y[i] = f(x[i], alpha_true) + e[i]
#qui y=f(x)+e dove la f è stata definita scegliendo i theta come vogliamo noi

##ora che ho definito i dati, uso il problea test(il risultato dovrebbero essere i theta da me scelti)

# Definiamo la funzione f(x, alpha) che prende in input il vettore alpha dei coefficienti e un vettore x di lunghezza n, e ritorna il valore del rispettivo polinomio calcolato su x elemento per elemento
def f(x, alpha):#f è  come quella di prima(alpha è theta)
    d = alpha.shape[0] - 1 # abbiamo detto che la shape di alpha è (d+1, )

    y = np.zeros_like(x) # Questa è l'unica riga che dobbiamo cambiare rispetto a prima!
    for i in range(d+1):
        y = y + alpha[i] * x ** i
    return y

# Andiamo a rappresentare la curva *vera* in [0, 1]
xx = np.linspace(0, 1, 100)
yy = f(xx, alpha_true)

plt.plot(xx, yy, 'b')
plt.plot(x, y, 'ro')
plt.grid()
plt.show()

def vandermonde(x, d):
    r"""
    Preso in input un numpy array "x" di lunghezza (n, ) contentente i dati, e un valore intero "d" rappresentante il grado del polinomio, 
    costruisce e ritorna la matrice di vandermonde X di grado d, associata a x.

    Parameters:
    x (ndarray): Il vettore dei dati di input.
    d (int): Il grado massimo del polinomio.

    Returns:
    X (ndarray): La matrice di Vandermonde di grado "d", associata ad x.
    """
    n = x.shape[0]
    
    # Inizializzo la matrice di Vandermonde con shape (n, d+1)
    X = np.zeros((n, d+1))
    
    # Costruisco la matrice di Vandermonde
    for i in range(d+1):
        X[:, i] = x ** i
    return X

# Definiamo la matrice di Vandermonde tramite la funzione appena definita con grado d = 5
d = 5
X = vandermonde(x, d)