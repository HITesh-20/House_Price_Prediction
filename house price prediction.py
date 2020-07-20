#HOUSE PRICE PREDICTION 
#IMPORT LIBRARIES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

#LOAD DATASET
df=pd.read_csv('c:/users/hitesh/desktop/machine learning/house price prediction/house data.csv')
print(df.shape)
print(df.columns)
drop=['id','date','view','condition','zipcode','lat','long','sqft_living15','sqft_lot15']
df=df.drop(drop,axis='columns')
print(df.shape)

#HANDLING MISSING VALUE
df=df.dropna()
print(df.isnull().sum())

#CONVERT SOME COLUMNS FROM FLOAT INTO INT 
df1=df.astype(int)
print(df1.dtypes)
df1.to_csv('HOUSE DATASET.csv')

#CORRELATION
print(df1.corr())

#ANALYSIS AND VISUALIZATION WILL BE DONE ON TABLEAU

#FEATURING 
X=df1.drop('price',axis='columns')
y=df1.price

#SPLITTING INTO TRAIN AND TEST SET
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#PREDICTION
#MULTIPLE LINEAR REGRESSION
mlr=LinearRegression().fit(X_train,y_train)
print("MULTIPLE LINEAR REGRESSION ACCURACY : ",mlr.score(X_train,y_train)*100)

#RANDOM FOREST REGRESSION
rfr=RandomForestRegressor(n_estimators=100,random_state=0).fit(X_train,y_train)
print("RANDOM FOREST REGREESSOR ACCURACY : ",rfr.score(X_train,y_train)*100)

#LASSO REGRESSION 
lr=LassoCV().fit(X_train,y_train)
print("LASSO REGRESSION ACCURACY : ",lr.score(X_train,y_train)*100)

#RIDGE REGRESSION
rr=RidgeCV().fit(X_train,y_train)
print("RIDGE REGRESSION ACCURACY : ",rr.score(X_train,y_train)*100)

#RANDOM FOREST REGRESSION IS THE BEST MACHINE LEARNING ALGO FOR THIS PROJECT

#PREDICTION
predicted=rfr.predict(X_test)
print(predicted)

#VISUALIZE
#PREDICTED GRAPH ACCOURDING TO BEDROOM
plt.scatter(X_test['bedrooms'],y_test,color='blue')
plt.scatter(X_test['bedrooms'],predicted,color='red')
plt.xlabel('BEDROOMS')
plt.ylabel('PRICE')
plt.show()




