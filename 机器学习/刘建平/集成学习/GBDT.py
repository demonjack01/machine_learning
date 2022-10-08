import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt

train = pd.read_csv('集成学习/train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
print(train['Disbursed'].value_counts())

x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

gbm0 = GradientBoostingClassifier(random_state=10)

gbm0.fit(X,y)
y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:,1]
print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                            min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
print('----'*15)

param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20, 
    max_features='sqrt', subsample=0.8, random_state=10), 
    param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch2.fit(X,y)
print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)