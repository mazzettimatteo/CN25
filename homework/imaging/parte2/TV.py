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
kernel = utilities.gaussian2d_kernel(k=7, sigma=2)
A = operators.ConvolutionOperator(kernel)

# Sfocatura dell'immagine e aggiunta di rumore
y = A(x)
y_delta = y + utilities.gaussian_noise(y, noise_level=0.025)


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

lambdaVals=np.logspace(-5,-0.5,20)
errs=[]
reconstructedImgs=[]

print(f"Ricerca di lambda che minimizzi l'errore usano i possibili valori:{lambdaVals}")


for i,lmbda in enumerate(lambdaVals):
    x_TV, obj_val, grad_norm = gd_tv_solver.solve(y_delta, lmbda, x0, kmax, tolf, tolx)
    relErr=utilities.rel_err(x_TV,x)
    errs.append(relErr)
    reconstructedImgs.append(x_TV)
    
#ricerca lambda che minimizza:

bestIndex=np.argmin(errs)
bestLmbda=lambdaVals[bestIndex]
bestImg=reconstructedImgs[bestIndex]
bestError=errs[bestIndex]
    
print(f"Miglior lambda = {bestLmbda} con bestError = {bestError} bestIndex = {bestIndex}")


#grafico errore

plt.semilogx(lambdaVals, errs,'b-', label='Error')
plt.semilogx(bestLmbda,bestError,'r.',markersize=15,label="Best")
plt.title("Andamento dell'errore")
plt.show()




# Visualizzazione ricostruzione
plt.figure(figsize=(10, 4))

plt.imshow(x, cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(y_delta, cmap="gray")
plt.axis("off")
plt.show()


plt.imshow(bestImg, cmap="gray")
plt.axis("off")
plt.show()





#metriche di errore per bestImg
print('ER',utilities.rel_err(bestImg,x))
print('PSNR',utilities.psnr(bestImg,x))
print('SSIM',utilities.ssim(bestImg,x))