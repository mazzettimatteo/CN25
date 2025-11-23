import numpy as np
import matplotlib.pyplot as plt

def f(x, alpha):
    d = alpha.shape[0] - 1 #shape di alpha è (d+1, )
    y = 0
    for i in range(d+1):
        y = y + alpha[i] * x ** i
    return y

#definiamo alpha_true = (1, 1, ..., 1), posso definirlo come voglio
d = 5 #grado del polinomio in x 
alphaTrue = np.zeros((d+1,))
for i in range(d+1):
    if(i==0): alphaTrue[i]=0
    else: alphaTrue[i]=np.sqrt(i)**(1/i) #alpha[i]=radice 2i di =i^(1/2i)

print(f"alphaTrue={alphaTrue}")
# Scegliamo un valore di x
x = 3
y = f(x, alphaTrue)
print(f"(x = {x}, y = {y})")

#definiamo x[] con n x_i equispaziati fra 0 e 1
n = 6 #n è il numero di punti campionati: è il numero di dati usati per determinare i coeff. del polinomio
#se n=d+1 ho un sistema quadrato, se n> dho più eq che incognite, se n<d ho inf sols
x = np.linspace(0, 1, n)

e=np.random.normal(loc=0,scale=0.1, size=(n,)) #rumore, scale=sigma=0.1, size=1 o n??

#calcoliamo y[]
y=np.zeros_like(x)
for i in range(n):
    y[i]=f(x[i],alphaTrue)+e[i]

#ridefiniamo f 
def f(x, alpha):
    d = alpha.shape[0] - 1 
    y = np.zeros_like(x) #unica riga diversa
    for i in range(d+1):
        y = y + alpha[i] * x ** i
    return y

#grafichiamo la "vera" curva
xx = np.linspace(0, 1, 100)
yy = f(xx, alphaTrue)

plt.plot(xx, yy, 'b')
plt.plot(x, y, 'ro')
plt.grid()
plt.show()

#-----------------------APPROX AI MINIMI QUADRATI con SVD ------------------------

#definiamo la M(R)_{n,d} di vandermonde che è associata a x1,..,xn
def vandermonde(x,d):
    n=x.shape[0]
    X=np.zeros((n,d+1))
    for i in range(d+1):
        X[:,i]=x**i #elementi da 0 a posizione i
    return X

X=vandermonde(x,d)

#funzione per calcolo valore del residuo: r(alpha)=||Xalpha-y||^2
def residue(X,y,alpha):
    r=np.linalg.norm(X@alpha - y)
    return r
#testiamo la funzione con un alpha random
alpha = np.random.randn(d+1)
print(f"Residuo: {residue(X, y, alpha)}.")

#calcoliamo la SVD di X

U,s,Vt = np.linalg.svd(X) #s è sigma minuscolo, 

Sigma=np.zeros((n,d+1))
for i in range(d+1):
    Sigma[i,i]=s[i]
print(f"|| X - U Sigma V^T || = {np.linalg.norm(X - U @ Sigma @ Vt)}")

#soluzione alphaSVD
alphaSVD=np.zeros((d+1,))
for i in range(d+1):
    alphaSVD+=(U[:,i].T @ y)/s[i]*Vt[i,:]#????????????????

# Rappresentiamo tutte le soluzioni su grafico
xx = np.linspace(0, 1, 100)
yy_true = f(xx, alphaTrue)
yy_svd = f(xx, alphaSVD)

plt.plot(xx, yy_svd, 'g')
plt.plot(xx, yy_true, 'b')
plt.plot(x, y, 'ro')
plt.legend(["SVD", "True"])
plt.grid()
plt.show()

plt.plot(x, y, 'ro')
plt.plot(xx, yy_svd, 'g')
plt.grid()
plt.show()