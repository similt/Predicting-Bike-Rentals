The data can be downloaded from http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
bike_rentals=pd.read_csv("bike_rental_hour.csv")
bike_rentals["cnt"].hist()
print(bike_rentals.corr()["cnt"])

def assign_label(time): 
    if time >=6 and time <12: 
        return 1
    if time >=12 and time <18: 
        return 2
    if time >=18 and time <=24: 
        return 3
    if time >=0 and time <6: 
        return 4
bike_rentals["time_label"] = bike_rentals["hr"].apply(assign_label)

train=bike_rentals.sample(frac=0.8)
test=bike_rentals[~bike_rentals.index.isin(train.index)]

predictors = list(train.columns)
predictors.remove("cnt")
predictors.remove("casual")
predictors.remove("registered")
predictors.remove("dteday")
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
lr= LinearRegression()
lr=lr.fit(train[predictors], train[["cnt"]])
predictions=lr.predict(test[predictors])
mse=mean_squared_error(test['cnt'],predictions)
rmse=np.sqrt(mse)
print(rmse,mse)

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor  
from sklearn.metrics import roc_auc_score
clf = DecisionTreeRegressor(min_samples_leaf=5)
columns= predictors
clf.fit(train[columns], train["cnt"])
predictions=clf.predict(test[columns])
test_auc=mean_squared_error(test['cnt'],predictions)
print(test_auc)

clf = DecisionTreeRegressor(min_samples_leaf=2)
columns= predictors
clf.fit(train[columns], train["cnt"])
predictions=clf.predict(test[columns])
test_auc=mean_squared_error(test['cnt'],predictions)
print(test_auc)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(min_samples_leaf=5)
reg.fit(train[predictors], train["cnt"])
predictions = reg.predict(test[predictors])
fmse=mean_squared_error(test['cnt'],predictions)
print(fmse)
