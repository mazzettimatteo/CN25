# CRAZIONE di un PROBLEMA TEST

from ProblemiInversi import operators, solvers, utilities
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray

# Lettura dati da file immagine
x=plt.imread("redbullBMP.bmp")
x = rgb2gray(x)

nx, ny = x.shape

# Normalizzazione (nel range [0, 1])
x = x / x.max()

# Definizione kernel di blur e operatore associato
kernel = utilities.gaussian2d_kernel(k=7, sigma=2) #----------------------------------------------PARAM
A = operators.ConvolutionOperator(kernel)

# Sfocatura dell'immagine e aggiunta di rumore
y = A(x)

# --- MODIFICA GENERAZIONE RUMORE ---
noise = utilities.gaussian_noise(y, noise_level=0.1) #----------------------------------------------PARAM
y_delta = y + noise 
norm_noise = np.linalg.norm(noise) 

# Definizione dati estesi ybar e ybar_delta
zeros = np.zeros_like(y)
ybar = np.vstack((y, zeros))
ybar_delta = np.vstack((y_delta, zeros))

# Calcolo soluzione regolarizzata con il metodo di Tikhonov e lambda fissato
# Problema di minimo risolto con CGLS

# Scelta parametri Tikhonov (L e lambda)
L = operators.Identity()

# Scelta di x0, kmax, tolf, tolx
x0 = np.zeros_like(x)
kmax = 50 
tolf = 1e-8
tolx = 1e-8

#creo griglia per lambda
lambdaVals=np.logspace(-3,1.5,20)
relErrs=[]
residueNorms=[] #qui salveremo la norma del residuo ||Ax - y|| per ogni lambda
reconstructedImgs=[]

print(f"Ricerca di lambda che minimizzi l'errore usano i possibili valori:{lambdaVals}")
print(f"Norma del rumore (Target Discrepanza): {norm_noise }") #Stampo il target per riferimento

for i,lmbda in enumerate(lambdaVals):
    M = operators.TikhonovOperator(A, L, lmbda)
    cgls_tik_solver = solvers.CGLS(M)

    x_TIK = cgls_tik_solver.solve(ybar_delta, x0, kmax, tolf, tolx) 
    reconstructedImgs.append(x_TIK)

    #errore rel (Ground Truth)
    relErr=utilities.rel_err(x_TIK,x)
    relErrs.append(relErr)
    
    #massima discrepanza: Il residuo va calcolato come || A(x_rec) - y_delta ||.
    current_residual = np.linalg.norm(A(x_TIK) - y_delta) 
    residueNorms.append(current_residual) 

    
# --- RICERCA MIGLIOR LAMBDA (MINIMO ERRORE RELATIVO - Ideale) ---
bestIndex=np.argmin(relErrs)
bestLmbda=lambdaVals[bestIndex]
bestImg=reconstructedImgs[bestIndex]
bestError=relErrs[bestIndex]
print(f"Miglior lambda (Min Error) = {bestLmbda } con Error = {bestError }")


# --- RICERCA MIGLIOR LAMBDA (PRINCIPIO DISCREPANZA - Reale) ---
tau = 1.01 #Fattore di sicurezza (tau > 1). Solitamente si sceglie tra 1.01 e 1.1
target_discrepancy = tau * norm_noise #cerchiamo il residuo più vicino a (tau * rumore)
diffFromTarget = np.abs(np.array(residueNorms) - target_discrepancy) #Calcolo la distanza tra i residui ottenuti e il target
bestIndexDisc = np.argmin(diffFromTarget) 
bestLmbdaDisc = lambdaVals[bestIndexDisc] 
bestImgDisc = reconstructedImgs[bestIndexDisc] 
bestResidDisc = residueNorms[bestIndexDisc] 
print(f"Miglior lambda (Discrepanza) = {bestLmbdaDisc } con Residuo = {bestResidDisc } (Target: {target_discrepancy })") 


# --- GRAFICI ---
plt.figure(figsize=(12, 5)) 

# Grafico 1: Errore Relativo
plt.subplot(1, 2, 1) 
plt.semilogx(lambdaVals, relErrs,'b-', label='Error Relativo')
plt.semilogx(bestLmbda,bestError,'r.',markersize=15,label="Best (Min Err)")
plt.semilogx(bestLmbdaDisc, relErrs[bestIndexDisc], 'g.', markersize=12, label="Scelta Discrepanza") # Mostro dove cade la scelta della discrepanza sulla curva dell'errore
plt.title("Andamento dell'errore (vs Ground Truth)")
plt.xlabel("Lambda") 
plt.legend() 

# Grafico 2: Principio di Discrepanza
plt.subplot(1, 2, 2) 
plt.semilogx(lambdaVals, residueNorms, 'k-', label='Norma Residuo ||Ax-y||') 
plt.axhline(y=target_discrepancy, color='r', linestyle='--', label=f'Target Rumore ({tau}*delta)') # Linea orizzontale che indica il livello di rumore
plt.semilogx(bestLmbdaDisc, bestResidDisc, 'g.', markersize=15, label="Scelta Discrepanza") #Punto scelto
plt.title("Principio di Discrepanza") 
plt.xlabel("Lambda") 
plt.legend() 

plt.tight_layout() 
plt.show()

# Visualizzazione ricostruzione (Confronto)
plt.figure(figsize=(15, 5)) 

plt.subplot(1, 4, 1) 
plt.imshow(x, cmap="gray")
plt.title("Originale")
plt.axis("off")

plt.subplot(1, 4, 2) 
plt.imshow(y_delta, cmap="gray")
plt.title("Corrotta")
plt.axis("off")

plt.subplot(1, 4, 3) 
plt.imshow(bestImg, cmap="gray")
plt.title(f"Best MinErr\nL={bestLmbda }") 
plt.axis("off")

plt.subplot(1, 4, 4) 
plt.imshow(bestImgDisc, cmap="gray")
plt.title(f"Best Discrepanza\nL={bestLmbdaDisc }")   
plt.axis("off")

plt.show()

#metriche di errore per bestImg (Discrepanza)
print("\nMetriche per soluzione da Discrepanza:")   
print('ER',utilities.rel_err(bestImgDisc,x))   
print('PSNR',utilities.psnr(bestImgDisc,x))   
print('SSIM',utilities.ssim(bestImgDisc,x)) 