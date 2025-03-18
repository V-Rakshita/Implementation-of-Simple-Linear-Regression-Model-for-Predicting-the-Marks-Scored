# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the numpy, pandas, matplotlib.pyplot, mean_squared_error and mean_absolute_error from sklearn.metrics, train_test_split from sklearn.model, LinearRegression from sklearn.linear_model.
2. Read the dataset.  Analyse the dataset, incase needed preprocessing has to be done.
3. Split the data into training and testing data using train_test_split() which will give four results therefore assign 4 variables for it.
4. Find the mean squared error, mean absolute error and root mean squared error using the necessary functions.
5. Plot the scatter plot and line plot using plt.scatter() and plt.plot()
6. Give an input for predicting and finally end the program.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: V RAKSHITA
RegisterNumber: 212224100049
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("student_scores.csv")
print(dataset.head())
print(dataset.tail())

dataset.info()

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)

mse = mean_squared_error(Y_test,Y_pred)
print("MSE= ",mse)
mae = mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

plt.scatter(X_test,Y_test,color = "blue")
plt.plot(X_test,Y_pred,color = "silver")
plt.title("Test set (H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
a=np.array([[13]])
y_predicted = reg.predict(a)
print(y_predicted)
```

## Output:

![Screenshot (7)](https://github.com/user-attachments/assets/9e353313-e3ca-48f5-be09-7b5a747a8056)
![Screenshot (8)](https://github.com/user-attachments/assets/c8f9dbd6-a5a8-4148-af5e-ce44ba133fe6)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
