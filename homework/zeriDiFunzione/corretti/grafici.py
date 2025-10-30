import numpy as np
import matplotlib.pyplot as plt

f1=lambda x: np.log(x+1)-x
f2=lambda x: x**2-np.cos(x)
f3=lambda x: np.sin(x)-(x/2)
f4=lambda x: np.exp(x)-3*x

a3=-3; b3=3; 
x3=np.linspace(a3,b3,1000)
y3=f4(x3)
xAxis3=np.zeros_like(x3)

#plt.subplot(2,2,3)
plt.plot(x3,y3)
plt.plot(x3,xAxis3)
plt.grid()
plt.show()