import numpy as np
import scipy as sp

#problema test 1
B=np.array([[9,3,6],[3,4,6],[0,8,8]])
#devo risolvere il sistema LUx=b dove B=LU
P,L,U=sp.linalg.lu(B)   #fattorizzazione LU con pivoting PB=LU
err=np.linalg.norm(B-P@L@U)
print(f"norma errore LU: {err}")


#problema test 2
A=np.array([[3,-7,2,2],[-3,5,1,0],[6,-4,0,-5],[-9,5,-5,12]])
lu,piv=sp.linalg.lu_factor(A)
x_true=np.ones((4,))
#costruisco il problema test con soluzione esatta x_true
b=A@x_true
x=sp.linalg.lu_solve((lu,piv),b)#risolve i sitemi Ly=b e Ux=y
print(f"soluzione={x}")
print(np.linalg.norm(b-A@x))

#oppure si può usare la funzione di numpy np.linalg.solve
xx=np.linalg.solve(A,b)
print(f"soluzione={xx}")
print(np.linalg.norm(b-A@xx))

"""
funzioni da sapere:
scipy lu factor 
scipy lu_solve

numpy linalg.solve
"""