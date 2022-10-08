import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import RidgeCV

data=pd.read_csv('回归\CCPP\data.csv')
print(data.head())

#step 准备数据
x = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]

#step 划分数据集

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

rigd=Ridge(alpha=1)
rigd.fit(X_train, y_train)

print (rigd.intercept_)
print (rigd.coef_)

#模型拟合测试集
y_pred = rigd.predict(X_test)

# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(rigd, X, y, cv=9)
# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y, predicted))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))


ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)

y_pred = ridgecv.predict(X_test)
# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))