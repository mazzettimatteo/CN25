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

X_train, X_test, y_train, y_test = train_test_split(xRaw, yRaw, test_size=0.2, random_state=89)

# Crea il modello di regressione lineare multipla
model = LinearRegression()

# Addestra il modello sul set di addestramento
model.fit(X_train, y_train)

# Valuta il modello sul set di test
# The model is used to predict the target variable for the test set.
y_pred = model.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

print("\nEliminiamo la feature TV:")
xRaw=dataSet.drop(["sales","TV"], axis=1)
yRaw=dataSet["sales"]
X_train, X_test, y_train, y_test = train_test_split(xRaw, yRaw, test_size=0.2, random_state=89)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

print("\nEliminiamo la feature RADIO:")
xRaw=dataSet.drop(["sales","radio"], axis=1)
yRaw=dataSet["sales"]
X_train, X_test, y_train, y_test = train_test_split(xRaw, yRaw, test_size=0.2, random_state=89)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

print("\nEliminiamo la feature NEWSPAPER:")
xRaw=dataSet.drop(["sales","newspaper"], axis=1)
yRaw=dataSet["sales"]
X_train, X_test, y_train, y_test = train_test_split(xRaw, yRaw, test_size=0.2, random_state=89)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))