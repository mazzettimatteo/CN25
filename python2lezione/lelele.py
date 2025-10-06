a=[2,5,"ciao",True]
print(a)
a[2]=4
print(a)
a.append(999)
a.insert(2,False)#indice,elem
print(a)
b=[33,44,55]
c = a+b #concatenazione
print(c)

x=(6,) #tupla con solo elemento: serve la virgola
print(type(x))
y=(4,5)
z=x+y#concatenamento tuple
print(z)

#le tuple sono salvate in memoria piu efficientemente


l1:list[str]=["la","meto","scipola?"]
print(l1)
l2=l1#l2 non è la copia di l1, punta  proprio alla stessa lista 
print(l2)
l1[-1]="picanto?" #indice -1 CORRISP A ULTIMO ELEM
print(l1)
print(l2)
l3=l1[:]
print(l1)
l1[-1]="latuga"
print(l1)
print(l3)

#piuttosto che usare l.append inizializzo lista di int vuota lunga 3 elem
p: list[int]= [0] *3
print(p)
#inserisco gli elementi adesso
p[0]=9
p[1]=2
p[2]=7
print(p)

#-------------------strutture condizionali e cicli-------------------
k=8
if(k<=7 and k>-1):
    print("k<=7")
elif(x==(6,)):
    print("x=(6,)")
else:
    print("else")


#for _var in iterable(lista,tupla o stringa in genere)

iterable=(2,1,4,9)

for i in iterable:
    print(i)

for i in range(3,5):#range crea tupla che parte da 3 e termina prima di 5
    print(i)

quit=False
temp=0
while(quit==False | temp<4):
    temp=temp+1
    print(temp)


#iterabili: oggetti 

def sommaPerX(k: int,n: int, X:int =2)->int:
    """
    per la funzione help si mette questo commento:
    sommaPerDue fa la somma di n e k e la moltiplica per X che di default è 2
    """

    #la tipizzazione non è necessaria neppure qui :(
    
    return (k+n)*2

print(sommaPerX(3,3))
print(sommaPerX(3,3,3))#default???

help(sommaPerX)