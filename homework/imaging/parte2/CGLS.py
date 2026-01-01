# CRAZIONE di un PROBLEMA TEST

from ProblemiInversi import operators, solvers, utilities
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray


x=plt.imread("redbullBMP.bmp")
x = rgb2gray(x)
nx, ny = x.shape

# Normalizzazione (nel range [0, 1])
x = x / x.max()

plt.imshow(x, cmap="gray")
plt.title("Originale")
plt.axis("off")
plt.show()

# Definizione kernel di blur e operatore associato
kernel = utilities.gaussian2d_kernel(k=7, sigma=2)#-----------------------
A = operators.ConvolutionOperator(kernel)

# Visualizzazione kernel di blur (PSF)
plt.imshow(kernel, cmap='hot')#summer
plt.axis('off')
plt.show()

# Sfocatura dell'immagine e aggiunta di rumore
y = A(x)
y_delta = y + utilities.gaussian_noise(y, noise_level=0.05)#-----------------------------------


#Calcolo della soluzione NAIVE come soluzione del problema dei minimi quadrati
# con le equazioni normali

# CGLS
cgls_solver = solvers.CGLS(A)

# Scelta di x0, kmax, atolf, tolx
x0 = np.zeros_like(x)
kmax = 30#-------------------------------------------------------------
tolf = 1e-8
tolx = 1e-8

# Soluzione
x_cgls = cgls_solver.solve(y_delta, x0, kmax, tolf, tolx)



# Visualizzazione ricostruzione
plt.imshow(x, cmap="gray")
plt.axis("off")
plt.title("Originale")
plt.show()

plt.imshow(y_delta, cmap="gray")
#plt.title("Corrotta")
plt.axis("off")
plt.show()


plt.imshow(x_cgls, cmap="gray")
plt.axis("off")
plt.show()

############################################

#Calcolo errore
print('ER',utilities.rel_err(x_cgls,x))
print('PSNR',utilities.psnr(x_cgls,x))
print('SSIM',utilities.ssim(x_cgls,x))
