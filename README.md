# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Data Preparation
3. Hypothesis DefinitionCost Function
4. Parameter Update Rule
5. Iterative Training
6. Model Evaluation
7. End 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Visalan H
RegisterNumber: 212223240183
*/
```
```py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
X=df.drop(columns=['AveOccup','HousingPrice'])
y=df[['HousingPrice','AveOccup']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
ss= StandardScaler()
X_train_scaled=ss.fit_transform(X_train)
y_train_scaled=ss.fit_transform(y_train)
X_test_scaled=ss.fit_transform(X_test)
y_test_scaled=ss.fit_transform(y_test)

sgd=SGDRegressor(max_iter=5000,tol=1e+2)
multi_sgd= MultiOutputRegressor(sgd)
multi_sgd.fit(X_train_scaled,y_train_scaled)

predictions=multi_sgd.predict(X_test_scaled)
mse1=mean_squared_error(y_test_scaled,predictions)

y_pred_actual=ss.inverse_transform(predictions)
mse = mean_squared_error(y_test,y_pred_actual)
print("Mean Squared Error:",mse)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test['HousingPrice'].values, label='Actual Housing Prices', color='blue')
plt.plot(y_pred_actual[:, 0], label='Predicted Housing Prices', color='red', linestyle='dotted')
plt.xlabel('Samples')
plt.ylabel('Housing Prices')
plt.title('Actual vs Predicted Housing Prices')
plt.legend()
plt.show()
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
