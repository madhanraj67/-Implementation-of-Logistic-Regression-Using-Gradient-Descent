# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required. 
2.Read the dataset. 
3.Define X and Y array. 
4.Define a function for sigmoid, loss, gradient and predict and perform operations.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MADHANRAJ P
RegisterNumber: 212223220052
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")

dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta=np.random.randn(X.shape[1])

y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

```
## Output:
## Read and file and Display:
![328127425-cb244b14-56e7-491b-a316-3f7f6821e1a7](https://github.com/RamkumarGunasekaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870820/9dc2698a-c129-42c4-bbdb-da85c3dcea16)
## Categorizing columns:
![328127551-1419f9fd-7720-4c06-8e22-d42be9e58378](https://github.com/RamkumarGunasekaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870820/65895b94-4a98-489b-97be-f74e1811d892)
## Labelling columns and displaying dataset:
![328127646-0a1a2fb4-c8aa-4cf7-90fb-cc7f12b94e58](https://github.com/RamkumarGunasekaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870820/3bf40545-6285-4b15-bbef-e139c446e5fa)
## Display dependent variable:
![328127785-dcc8c7a9-045a-48db-b0f6-26b8816e7c59](https://github.com/RamkumarGunasekaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870820/39d7f490-0a65-4074-aa12-b1d5e287b0e1)
## Printing accuracy:
![328127872-43a1af40-a59b-4ee9-9adc-2fa98b0744b7](https://github.com/RamkumarGunasekaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870820/8b5835d0-5296-48e5-8ad1-2218f18dd715)
## Printing Y:
![328127995-d5812479-a68a-47e0-bd9d-971d61dfc2bd](https://github.com/RamkumarGunasekaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870820/3c4e6e4d-1c2c-477c-b422-0e6c87d9040b)
## Printing y_prednew:
![328128107-4731cafd-ee77-4464-aa1a-cf03ea783802](https://github.com/RamkumarGunasekaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870820/7f30598e-e6c9-499e-8164-b00c85081400)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

