import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def vandermonde(x,d):
    n=x.shape[0]
    X=np.zeros((n,d+1))
    for i in range(d+1):
        X[:,i]=x**i #elementi da 0 a posizione i
    return X

######################################
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

######################################

def f(x, alpha):
    d = alpha.shape[0] - 1 
    y = np.zeros_like(x) #unica riga diversa
    for i in range(d+1):
        y = y + alpha[i] * x ** i
    return y

####################################

dataSet1=pd.read_csv("dataSets/SalaryDataset.csv")

print(f"Head dataset: \n{dataSet1.head()}")
#print(f"Info datset: \n {dataSet.info()}")
print(f"Descrizione dataset:\n{dataSet1.describe()}")

xRaw=dataSet1["YearsExperience"].values
yRaw=dataSet1["Salary"].values

print(f"Numero di righe Salary Dataset:{yRaw.shape[0]}")


d=3#-----------------------------------------------------------PARAM DA CAMBIARE

X=vandermonde(xRaw,d)

alphaSVD=SVD(X,yRaw,d)

xMin=xRaw.min()
xMax=xRaw.max()
xx=np.linspace(xMin-1, xMax+1,10000)
yy=f(xx,alphaSVD)


plt.plot(xRaw, yRaw,'r.')
plt.plot(xx,yy,'g')
plt.xlabel("Esperienza (Y)")
plt.ylabel("Salario (USD)")
plt.grid()
plt.show()
###########################################################################
dataSet2=pd.read_csv("dataSets/HousePrice.csv")

#rimuovo le colonne che non mi interessano, tengo solo: Square_Footage, House_Price
dataSet2.pop("Num_Bedrooms")
dataSet2.pop("Num_Bathrooms")
dataSet2.pop("Year_Built")
dataSet2.pop("Lot_Size")
dataSet2.pop("Garage_Size")
dataSet2.pop("Neighborhood_Quality")

xRaw=dataSet2["Square_Footage"].values
yRaw=dataSet2["House_Price"].values

print(f"Numero di righe Housing Dataset:{yRaw.shape[0]}")


d=5#-----------------------------------------------------------PARAM DA CAMBIARE

X=vandermonde(xRaw,d)

alphaSVD=SVD(X,yRaw,d)

xMin=xRaw.min()
xMax=xRaw.max()
xx=np.linspace(xMin-1, xMax+1,10000)
yy=f(xx,alphaSVD)


plt.plot(xRaw, yRaw,'r.')
plt.plot(xx,yy,'g')
plt.xlabel("Dimensioni (ft^2)")
plt.ylabel("Prezzo (USD)")
plt.grid()
plt.show()