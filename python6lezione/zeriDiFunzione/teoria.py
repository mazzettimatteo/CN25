import matplotlib.pyplot as plt
import numpy as np


#per prima cosa voglio fare un grafico (discreto) di una funzione
#creiamoarray delle ascisse e un corrispondente array delle coordinate
#1) creare array x[1..n]
#2) creare array y[1..n]=f(x[1..n])
#3) usare plot da matplotlib.pyplot per disegnare le coppie (x[i],y[i]) con i=1..n

a=0
b=2*np.pi
N=100

x=np.linspace(a,b,N)
y=np.exp(x)

plt.plot(x,y)

plt.title("Shao Bello") #titolo grafico
plt.xlabel("asse x")    #etichetta asse x
plt.grid()  #aggiunge griglia nel grafico
plt.ylim([4,6]) #limito l'asse y all'intervallo indicato

plt.show()


plt.subplot(1,2,1)
plt.plot(x,y,':',color='green') #x,y,carattere,colore,tipo di linea che congiunge i punti
plt.legend(['f(x)=e^x'])
plt.subplot(1,2,2)
plt.plot(x,x)
plt.legend(['f(x)=x'])

plt.show()

#guardare anche plt.figure