#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
y = df['shear'].values
X = df['age'].values
X = sm.add_constant(X) 

def stochastic_gradient_descent(X, y, learning_rate=0.001, n_iterations=100000):
    m = len(y)
    theta = np.array([3000, -0.1]) 

    for iteration in range(n_iterations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]  
        yi = y[random_index:random_index+1]  
        
       
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        
        theta -= learning_rate * gradients.flatten() 
    return theta

theta_sgd = stochastic_gradient_descent(X, y)

print("\nStochastic Gradient Descent:")
print(f"Intercept: {theta_sgd[0]:.4f}, Slope: {theta_sgd[1]:.4f}")


from sklearn.linear_model import SGDRegressor 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

sgd_reg= SGDRegressor(max_iter=1000, tol=1e-3)
sgd_reg.fit(X_train, y_train)

x_values=np.linspace(0,25,100)
y_pred=sgd_reg.predict(X)
print("Predictions:", y_pred)


# In[ ]:




