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

# Calcolo soluzione regolarizzata con il metodo di Tikhonov e lambda fissato
# Problema di minimo risolto con CGLS

# Scelta parametri Tikhonov (L e lambda)
L = operators.Identity()
lmbda = 0.1

# Costruzione operatore di Tikhonov
M = operators.TikhonovOperator(A, L, lmbda)

# Definizione dati estesi ybar e ybar_delta
"""ybar = np.pad(y, ((0, 512), (0, 0)))
ybar_delta = np.pad(y_delta, ((0, 512), (0, 0)))
"""
zeros = np.zeros_like(y)##############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ybar = np.vstack((y, zeros))
ybar_delta = np.vstack((y_delta, zeros))


# Solver CGLS + Tikhonov
cgls_tik_solver = solvers.CGLS(M)

# Scelta di x0, kmax, tolf, tolx
x0 = np.zeros_like(x)
kmax = 100
tolf = 1e-8
tolx = 1e-8

# Soluzione
x_tik = cgls_tik_solver.solve(ybar_delta, x0, kmax, tolf, tolx)

# Visualizzazione ricostruzione
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(x, cmap="gray")
plt.title("Originale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(y_delta, cmap="gray")
plt.axis("off")
plt.title("Corrotta")

plt.subplot(1, 3, 3)
plt.imshow(x_tik, cmap="gray")
plt.axis("off")
plt.title("Ricostruzione TIK")
plt.show()

#Calcolo metriche di errore
print('ER',utilities.rel_err(x_tik,x))
print('PSNR',utilities.psnr(x_tik,x))
print('SSIM',utilities.ssim(x_tik,x))