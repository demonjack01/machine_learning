import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import make_classification


# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                        n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)


sklearn_model = xgb.XGBClassifier(learning_rate= 0.5, verbosity=1, objective='binary:logistic',random_state=1)  #!! verbosity打印
sklearn_model=sklearn_model.set_params(early_stopping_rounds=10, eval_metric='error')
sklearn_model.fit(X_train, y_train,eval_set=[(X_test, y_test)])
print('---'*10)

sklearn_model_1 = xgb.XGBClassifier(learning_rate= 0.5, verbosity=1, objective='binary:logistic',random_state=1)
gsCv = GridSearchCV(sklearn_model_1,
                {'max_depth':[3,4,5],
                'n_estimators':[5,10,20]})
gsCv.fit(X_train,y_train)

print(gsCv.best_score_)
print(gsCv.best_params_)

sklearn_model_2 = xgb.XGBClassifier(max_depth=4,n_estimators=5,verbosity=1, objective='binary:logistic',random_state=1)
gsCv2 = GridSearchCV(sklearn_model_2, 
                {'learning_rate': [0.3,0.5,0.7]})
gsCv2.fit(X_train,y_train)

print(gsCv2.best_score_)
print(gsCv2.best_params_)

sklearn_model_final = xgb.XGBClassifier(max_depth=4,learning_rate= 0.7, verbosity=1, objective='binary:logistic',n_estimators=10)
sklearn_model_final.set_params(early_stopping_rounds=10, eval_metric="error")
sklearn_model_final.fit(X_train, y_train,eval_set=[(X_test, y_test)])

