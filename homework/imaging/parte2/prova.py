import numpy as np

lambdaVals=np.logspace(-5,-0.5,20)
target=0.002335721469090121
cont=0
for val in lambdaVals:
    if abs(target-val)==0:
        print(cont)
    cont+=1