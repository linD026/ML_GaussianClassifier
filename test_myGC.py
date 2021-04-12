from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn import neighbors, svm, naive_bayes 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import sklearn.datasets as ds
import numpy as np
from matplotlib import pyplot as plt

from selfGC import myGaussianClassifier 

# 載入資料集
data,target    = ds.load_breast_cancer(True)
 
# 宣告分類器
gauss_clf      = myGaussianClassifier()

gauss_clf_param={'alpha':[0.001,0.01,0.1,1,10,100]}

gauss_gs     = GridSearchCV(estimator=gauss_clf,param_grid = gauss_clf_param, scoring = 'accuracy', cv=5, n_jobs=-1, verbose=1)

kfold        = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

gauss_scores   = cross_val_score(gauss_gs, data, target, scoring='accuracy',cv = kfold, verbose=10) 


gauss_mean = gauss_scores.mean()

print(gauss_mean)
