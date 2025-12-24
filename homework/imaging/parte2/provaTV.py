# CRAZIONE di un PROBLEMA TEST

from ProblemiInversi import operators, solvers, utilities
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray##############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Lettura dati da file immagine
x=plt.imread("redbullBMP.bmp")
x = rgb2gray(x)

nx, ny = x.shape

# Normalizzazione (nel range [0, 1])
x = x / x.max()


# Definizione kernel di blur e operatore associato
kernel = utilities.gaussian2d_kernel(k=11, sigma=3.5)
A = operators.ConvolutionOperator(kernel)

# Sfocatura dell'immagine e aggiunta di rumore
y = A(x)
y_delta = y + utilities.gaussian_noise(y, noise_level=0.05)


# Calcolo soluzione regolarizzata con Total Variation 
#  problema di minimo risolto con Discesa Gradiente

# Solver per Total Variation
gd_tv_solver = solvers.GDTotalVariation(A, beta=1e-3)


# Scelta di x0, kmax, tolf, tolx
x0 = np.zeros_like(x)
kmax = 30
tolf = 1e-8
tolx = 1e-8

#creo griglia per lambda

bestImg, obj_val, grad_norm = gd_tv_solver.solve(y_delta, 0.0001, x0, kmax, tolf, tolx)
   

# Visualizzazione ricostruzione


plt.imshow(x, cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(y_delta, cmap="gray")
plt.axis("off")
plt.show()


plt.imshow(bestImg, cmap="gray")
plt.axis("off")
plt.show()





