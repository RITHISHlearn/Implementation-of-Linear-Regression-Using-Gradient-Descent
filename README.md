# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
   

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: RITHISH P
RegisterNumber:  212223230173
*/
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("ex1.txt",header=None)

plt.scatter(data[0],data[1])

plt.xticks(np.arange(5,30,step=5))

plt.yticks(np.arange(-5,30,step=5))

plt.xlabel("Population of City(10,000s)")

plt.ylabel("Profit ($10,000)")

plt.title("Profit Prediction")


def computeCost(X,y,theta):

    m=len(y) 
    
    h=X.dot(theta) 
    
    square_err=(h-y)**2
    
    return 1/(2*m)*np.sum(square_err) 


data_n=data.values

m=data_n[:,0].size

X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)

y=data_n[:,1].reshape(m,1)

theta=np.zeros((2,1))

computeCost(X,y,theta) 


def gradientDescent(X,y,theta,alpha,num_iters):

    m=len(y)
    
    J_history=[] #empty list
    
    for i in range(num_iters):
    
        predictions=X.dot(theta)
        
        error=np.dot(X.transpose(),(predictions-y))
        
        descent=alpha*(1/m)*error
        
        theta-=descent
        
        J_history.append(computeCost(X,y,theta))
        
    return theta,J_history


theta,J_history = gradientDescent(X,y,theta,0.01,1500)

print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")


plt.plot(J_history)

plt.xlabel("Iteration")

plt.ylabel("$J(\Theta)$")

plt.title("Cost function using Gradient Descent")


plt.scatter(data[0],data[1])

x_value=[x for x in range(25)]

y_value=[y*theta[1]+theta[0] for y in x_value]

plt.plot(x_value,y_value,color="r")

plt.xticks(np.arange(5,30,step=5))

plt.yticks(np.arange(-5,30,step=5))

plt.xlabel("Population of City(10,000s)")

plt.ylabel("Profit ($10,000)")

plt.title("Profit Prediction")


def predict(x,theta):

    predictions=np.dot(theta.transpose(),x)
    
    return predictions[0]


predict1=predict(np.array([1,3.5]),theta)*10000

print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))


predict2=predict(np.array([1,7]),theta)*10000

print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
## Output:
      Profit prediction:
![3 1](https://github.com/RITHISHlearn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446645/7ab9e05a-d9c0-4194-b6f6-b9e5daad6511)

      Function:
![3 2](https://github.com/RITHISHlearn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446645/aeb0af67-1be1-4c84-bff6-4b8b2dd4af43)

      GRADIENT DESCENT:
![3 3](https://github.com/RITHISHlearn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446645/576bbcdc-500a-49c4-bed5-9015777b7a78)

      COST FUNCTION USING GRADIENT DESCENT:
![3 4](https://github.com/RITHISHlearn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446645/ae66ffd8-23ea-4d7f-bfcc-b39e8f5d62fe)

      LINEAR REGRESSION USING PROFIT PREDICTION:
![3 5](https://github.com/RITHISHlearn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446645/492b1e0d-aaef-4852-b30a-9c187a64c44b)

      
      PROFIT PREDICTION FOR A POPULATION OF 35000:
![3 6](https://github.com/RITHISHlearn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446645/5fb72ad1-4249-4806-bdab-941d725ad171)

      ##PROFIT PREDICTION FOR A POPULATION OF 70000:
![3 7](https://github.com/RITHISHlearn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446645/8c636f8b-b418-4af7-bb3a-2c1487fe5ce8)


## Result:
Thus the program to implement the linear regression using gradient dProfit prediction:escent is written and verified using python programming.
