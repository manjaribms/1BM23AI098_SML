#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[2]:


data = {
    "shear": [2160.70, 1680.15, 2318.00, 2063.30, 2209.50, 1710.30, 1786.70, 2577.00,
              2359.90, 2258.70, 2167.20, 2401.55, 1781.80, 2338.75, 1767.0, 2055.50,
              2416.40, 2202.50, 2656.20, 1755.70],
    
    "age": [15.50, 23.75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50,
            7.50, 11.00, 13.00, 3.75, 22.00, 18.00, 6.00, 12.50,
            2.00, 21.50, 9.00, 20.00] 
}

print(data.items())
df = pd.DataFrame(data)
df.head()
y = data['shear']
X = data['age']
X = sm.add_constant(X)
linear_regression = sm.OLS(y, X)

fitted_model = linear_regression.fit()
fitted_model.summary()
intercept = fitted_model.params[0]
Slope = fitted_model.params[1]
print("\nIntercept:",intercept)
print("Slope:",Slope)


# In[9]:


def gradient_descent(X, y, decay_rate=0.01, n_iterations=1000, initial_learning_rate = 0.01):
    m = len(y)
    theta = np.random.randn(2) 
    
    for i in range(n_iterations):
        gradients = (2/m) * X.T.dot(X.dot(theta) - y)
        learning_rate = initial_learning_rate / (1 + decay_rate * i)
        theta -= learning_rate * gradients
        error = np.mean((X.dot(theta)-y) ** 2)
        
        if error > 0.1:
            i += 1
        break
    return gradients
    return theta
theta_gd = gradient_descent(X, y)
print("\ngradient Descent:")
print(f"Intercept: {theta_gd[0]}, Slope: {theta_gd[1]}")    


# In[11]:


def stochastic_gradient_descent(X, y, lr=0.001, dr= 0.01, n_iterations=100000 ):
    m= len(y)
    tta= np.array([3000,-0.1])
    for i in range(n_iterations):
        for ii in range(m):
            random_index= np.random.randint(m)
            xi= X[random_index: random_index+1]
            yi= y[random_index: random_index+1]
            gradients= (2/m)* xi.T.dot(xi.dot(tta)-yi)
            t-=lr*gradients
    return gradients
    return tta 
tta_sgd = stochastic_gradient_descent( X, y)
print("\Stochastic Gradient Descent")
print(f"Intercept :{ tta_sgd[0]}, Slope:{tta_sgd[1]}")
            


# In[16]:


from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, train_test_split()=0.2, random_state=42)

sgd_reg= SDGRegressor(max_iter = 1000, tol= 1e-3)
sgd_reg.fit(x_train, y_train)


x_val= np.linspace(0,2,100)
y_pred= sgd_reg.predict(X)
print("Predictions", y_pred)

        

