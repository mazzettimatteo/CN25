import numpy as np 

#ndarray è un array che contiene solo dati dello stesso tipo, che siano solo numerici
#per crearlo li modo più semplice è castarlo da una lista

a: list[int]=[1,2,3]
a_vec: np.array = np.array(a)
print(type(a_vec))
print(a_vec)
print(f"la dimensione dell'array è{ a_vec.shape }")
A=np.array([[-4,-5,-6],[1,2,3]])
print(f"la dimensione dell'array è{ A.shape }")
#.shape ritorna una tupla

print(A.dtype)
A2=np.array([2.3,4.5], dtype=np.float64)
print(A2.dtype)
#.dtype ritorna il tipo degli elementi

l=np.linspace(1,10,5) 
#crea vettore da 1 a 10 con ogni elemento distribuito uniformrmente secondo la distribuzione 5
print(l)

l1=np.arange(0,10,2)#crea array con elementi ogni 2 partendo da 0 e finendo a 10
print(l1)

l2=np.zeros((4,6))  #crea matrice, o array se non metti una tupla, di zeri
print(l2)

l3=np.zeros_like(l2)#crea matrice della stessa dim di l2
print(l3)

l4=np.diag(a_vec)  #crea matrice diagonale con a_vec lungo la diagonale di dim |a_vec|
print(l4)

#numpy ha delle sottolibrerie come ad eg: numpy.random o numpy.strings
l5=np.random.randn(3,4)
print(l5)

#posso fare operazioni fra ndarray elemento per elemento
L1=np.array([9,8,3])
L2=np.array([1,2,55])
pippo=L1+L2
print(pippo)
pluto=L1*L2
print(pluto)
#anche le funz base matematiche lavorano elemento per elemento
ciccio=np.cos(pippo)
print(ciccio)

#prodotto scalare standard fra m ed n si fa m@n, ???prod riga per colonna???
peppe=L1@L2 
print(peppe)

#vettore colonna
v_col=np.array([[2],[3],[4]])
print(v_col)
v_rig=v_col.T   #vettore riga si crea come trasposto del vettore colonna con l'operazione .T
print(v_rig)

#fra i vettori posso fare anche operazioni logiche
print(pippo<pluto)

#slicing con booleani
w1=np.array([1,2,3,4])
w2=np.array([False,True,True,False,])
w3=w1[w2]
print(w3)

#eliminazione valori negativi
w4=np.array([-1,2,-3,-55,32])
w5=w4[w4>0]
print(w5)
w4[w4<0]=999
print(w4)

#si può fare lo slicing anche con le matrici per ottenere delle sottomatrici
#esiste una sottolibreria numpy.linalg con le varie funzioni di algebra lineare
