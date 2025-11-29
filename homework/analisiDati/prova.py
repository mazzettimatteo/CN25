import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv("dataSets/Salary_Data.csv")
print(f"Head dataset:\n{dataSet.head()}")
xRaw=dataSet["YearsExperience"].values
yRaw=dataSet["Salary"].values
print(f"Head dataset: \n{dataSet.head()}")
print(f"Descrizione dataset:\n{dataSet.describe()}")
xRaw=dataSet["YearsExperience"].values
yRaw=dataSet["Salary"].values
print(f"Numero di righe:{yRaw.shape[0]}")
d=5
X=vandermonde(xRaw,d)
alphaSVD=SVD(X,yRaw,d)