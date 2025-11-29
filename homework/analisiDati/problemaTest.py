import numpy as np
import matplotlib.pyplot as plt

def f(x, alpha): #y=a[i]*x^i
    d=alpha.shape[0] - 1 #shape di alpha è (d+1, )
    y=0
    for i in range(d+1):
        y=y + alpha[i] * x ** i
    return y

d=8 #grado del polinomio in x -------------------------------------------------------------------------------------PARAM DA CAMBIARE
alphaTrue=np.zeros((d+1,)) #definiamo alpha true arbitrariamente
for i in range(d+1):
    #print(i)
    if(i==0): alphaTrue[i]=0
    else: alphaTrue[i]=np.sqrt(i)**(1/i) #alpha[i]=radice 2i di =i^(1/2i)

print(f"alphaTrue={alphaTrue}")
x=3 # Scegliamo un valore di x
y=f(x, alphaTrue) #calcoliamo f(x)
print(f"(x={x}, f(x)=y={y})")

#definiamo x[] con n x_i equispaziati fra 0 e 1
n=15 #n è il numero di punti campionati: è il numero di dati usati per determinare i coeff. del polinomio -----------------------------------------------PARAM DA CAMBIARE
#se n=d+1 ho un sistema quadrato, se n>d ho più eq che incognite, se n<d ho inf sols
x=np.linspace(0, 1, n)

e=np.random.normal(loc=0,scale=0.1, size=(n,)) #rumore, scale=sigma=0.1, size=1 o n??
print(f"rumore{np.linalg.norm(e)}")

#calcoliamo y[] per graficare gli n punti
y=np.zeros_like(x)
for i in range(n):
    y[i]=f(x[i],alphaTrue)+e[i]

#ridefiniamo f in maniera vettoriale
def f(x, alpha):
    d=alpha.shape[0] - 1 
    y=np.zeros_like(x) #unica riga diversa
    for i in range(d+1):
        y=y + alpha[i] * x ** i
    return y

#grafichiamo la "vera" curva
xx=np.linspace(0, 1, 100)
yy=f(xx, alphaTrue)

plt.plot(xx, yy, 'b')#funzione "vera" che passa per i punti
plt.plot(x, y, 'ro')#punti
plt.grid()
plt.show()

#-----------------------APPROX AI MINIMI QUADRATI con SVD ------------------------

#definiamo la X appartenente a R^n*d di vandermonde che è associata a x1,..,xn tc x[i][j]=x[i]**(j-1)
def vandermonde(x,d):
    n=x.shape[0]
    X=np.zeros((n,d+1))
    for i in range(d+1):
        X[:,i]=x**i #elementi da 0 a posizione i
    return X

X=vandermonde(x,d)

def SVD(X,y,d):
    #n numero punti, d=grado polinomio
    n=y.shape[0]
    U,s,Vt=np.linalg.svd(X) #X=U@Sigma@V
    Sigma=np.zeros((n,d+1)) #creazione di Sigma con val singolari sulla diag
    for i in range(d+1):
        Sigma[i,i]=s[i]
    alphaSVD=np.zeros((d+1,)) #alpha=sum ((u_i^T y)/s_i) v_i
    for i in range(d+1):
        alphaSVD+=(U[:,i].T @ y)/s[i]*Vt[i,:]
    return alphaSVD

alphaSVD=SVD(X,y,d)

#funzione per calcolo valore del residuo: r(alpha)=||Xalpha-y||^2
def residue(X,y,alpha):
    r=np.linalg.norm(X@alpha - y)
    return r
print(f"ResiduoSVD: {residue(X, y, alphaSVD)}.")
print(f"ResiduoTRUE: {residue(X, y, alphaTrue)}.") #residuo di alphaTrue interessante perche è circa il rumore e 

# Rappresentiamo tutte le soluzioni su grafico
xx=np.linspace(0, 1, 100)
yy_true=f(xx, alphaTrue)
yy_svd=f(xx, alphaSVD)

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