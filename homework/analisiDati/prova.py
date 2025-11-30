#from sklearn.datasets import XXXXXXXXXXXXXXXXXXX
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


dataSet=pd.read_csv("dataSets/Advertising.csv")
print(f"Head dataset: \n{dataSet.head()}")
print(f"Info datset: \n {dataSet.info()}")
print(f"Descrizione dataset:\n{dataSet.describe()}")
xRaw=dataSet.drop("sales", axis=1)
yRaw=dataSet["sales"]
print(f"Numero di righe:{yRaw.shape[0]}")
print(f"Numero di features:{xRaw.shape[1]}\n")
xTrain, xTest, yTrain, yTest = train_test_split(xRaw, yRaw, test_size=0.2, random_state=89)
model = LinearRegression()
model.fit(xTrain, yTrain)
yPred = model.predict(xTest)
print("Mean Squared Error:", mean_squared_error(yTest, yPred))
print("R-squared:", r2_score(yTest, yPred))
print("\nEliminiamo la feature TV:")
xRaw=dataSet.drop(["sales","TV"], axis=1)
yRaw=dataSet["sales"]
xTrain, xTest, yTrain, yTest = train_test_split(xRaw, yRaw, test_size=0.2, random_state=89)
model = LinearRegression()
model.fit(xTrain, yTrain)
yPred = model.predict(xTest)
print("Mean Squared Error:", mean_squared_error(yTest, yPred))
print("R-squared:", r2_score(yTest, yPred))
print("\nEliminiamo la feature RADIO:")
xRaw=dataSet.drop(["sales","radio"], axis=1)
yRaw=dataSet["sales"]
xTrain, xTest, yTrain, yTest = train_test_split(xRaw, yRaw, test_size=0.2, random_state=89)
model = LinearRegression()
model.fit(xTrain, yTrain)
yPred = model.predict(xTest)
print("Mean Squared Error:", mean_squared_error(yTest, yPred))
print("R-squared:", r2_score(yTest, yPred))
print("\nEliminiamo la feature NEWSPAPER:")
xRaw=dataSet.drop(["sales","newspaper"], axis=1)
yRaw=dataSet["sales"]
xTrain, xTest, yTrain, yTest = train_test_split(xRaw, yRaw, test_size=0.2, random_state=89)
model = LinearRegression()
model.fit(xTrain, yTrain)
yPred = model.predict(xTest)
print("Mean Squared Error:", mean_squared_error(yTest, yPred))
print("R-squared:", r2_score(yTest, yPred))