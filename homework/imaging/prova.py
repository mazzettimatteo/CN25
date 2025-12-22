import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from ProblemiInversi import utilities

x=plt.imread("spaBMP.bmp")
x = rgb2gray(x)
nx, ny = x.shape
x = x / x.max()
U,S,Vt=np.linalg.svd(x)
p=50
if p>min(nx,ny):
    p=min(nx,ny)
U_p=U[:,:p]
S_p=np.diag(S[:p])
Vt_p=Vt[:p,:]
x_p = U_p @ S_p @ Vt_p
Er=utilities.rel_err(x_p,x)
c_p=p/(min(nx,ny)-1)
print(f"Errore relativo = {Er}")
print(f"Fattore di compressione = {c_p}")
