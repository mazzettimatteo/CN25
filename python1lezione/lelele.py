a = 3
s = "lelele"
b = True
print(type(b))

s1=f"il valore di a è:{a}"  #f-string prende variabili 
s2="il valore di a è:" + str(a)
print(s1)
print(s2)

x=0.1234567890

#print() stampa con tot cifre 

"""
/ divisione non intera
// divisione intera
** potenza
"""

a = None    #rappresenta l'assenza di dati, non è unvalore numerico, quindi si fa il controllo come segue fra 3 righe
b=5

print(a is None)
print(b is None)
print(b==5)

#tipizzazione

l=4
m: int = 4  #equinvalente a fare m=4 ma così specifico tipologia, funge solo da commento, posso cmq assegnarle un altro tipo:
n: int ="banana"
print(type(n))

#liste e tuple
"""
nelle liste i valori sono mutabili, nelle tuple no ma entrambe possono contenere elementi di diverso tipo
le liste le creo con [], le tuple con ()
anche le liste si possono tipizzare
"""
l1=[2,3,"zigga", False]
print(l1)
l2: int=[3,4,5,6]

t1=(3,"nano",[2.2,3.9,4.0])
print(type(t1))
print(t1)

print(f"l'elem di indice 2 è {l2[2]}")

l=[0,1,2,3,4,5,6,7,8,9]
pari=l[0:8:2]   #slicing: estrarre elementi da punto x a punto y con passo p, se il passo non è definito è 1 di default, se non metto nulla in y va fino alla fine
print(pari)
kebab=l[0::4]
print(kebab)
#indice di lista=-1 indica ultimo elemento
#la lunghezza della lista l si calcola con len(l)