import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from ProblemiInversi import utilities

x=plt.imread("spaBMP.bmp")
x = rgb2gray(x)
nx, ny = x.shape

# Normalizzazione (nel range [0, 1])
x = x / x.max()

plt.imshow(x, cmap="gray")
plt.title("Originale")
plt.axis("off")
plt.show()

U,S,Vt=np.linalg.svd(x)

p=1#rango per compressione----------------------------------------PARAM da cambiare
if p>min(nx,ny): #p non può essere mai maggiore del rango di x
    p=min(nx,ny)

#prendiamo solo le col di U, i primi sigma_p valori e le prime p righe di Vt
U_p=U[:,:p]
S_p=np.diag(S[:p])
Vt_p=Vt[:p,:]

x_p = U_p @ S_p @ Vt_p


Er=utilities.rel_err(x_p,x)
c_p=((1/p)*min(nx,ny))-1

print(f"Errore relativo = {Er}")
print(f"Fattore di compressione = {c_p}")



plt.imshow(x_p, cmap='gray')
plt.title(f"Ricostruzione (p={p})")
plt.axis("off")
plt.show()