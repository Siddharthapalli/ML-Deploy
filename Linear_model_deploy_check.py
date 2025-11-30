# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 08:46:13 2025

@author: siddh
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
tf.random.set_seed(20)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.autograph.set_verbosity(0)

data=np.loadtxt(r"C:\Users\siddh\Downloads\Files (2)\Files\data\data_w3_ex1.csv",delimiter=',')
x=data[:,0]
y=data[:,1]
x=x.reshape([-1,1])
y=y.reshape([-1,1])

fig,ax=plt.subplots(1,1,figsize=(5,3))
ax.scatter(x,y,marker='x',label='x vs y')
plt.legend()

x_train,x_,y_train,y_=train_test_split(x, y, test_size=0.40, random_state=1)
x_cv,x_test,y_cv,y_test=train_test_split(x_,y_,test_size=0.50, random_state=1)

del x_,y_

fig,ay=plt.subplots(1,1,figsize=(5,3))
ay.scatter(x_train,y_train, marker='x',color='blue')
ay.scatter(x_cv,y_cv,marker='o',color='red')
ay.scatter(x_test,y_test,marker='.',color='black')

scaler_linear=StandardScaler()

X_train_scaled=scaler_linear.fit_transform(x_train)
print("mean is", scaler_linear.mean_.squeeze())
print("standard dev", scaler_linear.scale_.squeeze())

linear_model=LinearRegression()
linear_model.fit(X_train_scaled,y_train)

yhat=linear_model.predict(X_train_scaled)
print(f"mse with training", (mean_squared_error(y_train,yhat)/2))

x_cv_scaled=scaler_linear.transform(x_cv)
yhat=linear_model.predict(x_cv_scaled)
print(f"mse with cv", (mean_squared_error(y_cv,yhat)/2))

poly=PolynomialFeatures(degree=2, include_bias=False)
x_train_mapped=poly.fit_transform(x_train)
print(x_train_mapped[:5])

scaler_poly=StandardScaler()
x_train_mapped_scaled=scaler_poly.fit_transform(x_train_mapped)

model=LinearRegression()
model.fit(x_train_mapped_scaled,y_train)

yhat=model.predict(x_train_mapped_scaled)
print("mse poly",(mean_squared_error(yhat, y_train)/2))

x_cv_mapped=poly.transform(x_cv)
x_cv_mapped_scaled=scaler_poly.transform(x_cv_mapped)
yhat=model.predict(x_cv_mapped_scaled)
print("mse with CV", mean_squared_error(yhat, y_cv)/2)

train_mses=[]
cv_mses=[]
models=[]
polys=[]
scalers=[]

for degree in range(1,11):
    poly=PolynomialFeatures(degree,include_bias=False)
    x_train_mapped=poly.fit_transform(x_train)
    polys.append(poly)
    
    scaler_poly=StandardScaler()
    x_train_mapped_scaled=scaler_poly.fit_transform(x_train_mapped)
    scalers.append(scaler_poly)
    
    model=LinearRegression()
    model.fit(x_train_mapped_scaled,y_train)
    models.append(model)
    
    yhat=model.predict(x_train_mapped_scaled)
    train_mse=mean_squared_error(yhat, y_train)/2
    train_mses.append(train_mse)
    
    x_cv_mapped=poly.transform(x_cv)
    x_cv_mapped_scaled=scaler_poly.transform(x_cv_mapped)
    
    yhat=model.predict(x_cv_mapped_scaled)
    cv_mse=mean_squared_error(yhat,y_cv)/2
    cv_mses.append(cv_mse)

degrees=range(1,11)    
fig,az=plt.subplots(1,1,figsize=(5,3))
az.plot(degrees,train_mses,cv_mses)


degree=np.argmin(cv_mses)+1
print(degree)

x_test_mapped=polys[degree-1].transform(x_test)
x_test_mapped_scaled=scalers[degree-1].transform(x_test_mapped)

yhat_test=models[degree-1].predict(x_test_mapped_scaled)
test_mse=mean_squared_error(yhat_test, y_test)/2

print(f'training mse:{train_mses[degree-1]:.2f}')
print(f'cv mses:{cv_mses[degree-1]:.2f}')
print(f'test mse:{test_mse:.2f}')


