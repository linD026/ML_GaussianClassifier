from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn import neighbors, svm, naive_bayes 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import sklearn.datasets as ds
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from selfGC import myGaussianClassifier

# 載入資料集
data,target    = ds.load_breast_cancer(True)
 
# 宣告分類器
gauss_clf      = myGaussianClassifier()
knn_clf        = neighbors.KNeighborsClassifier(n_neighbors=3,weights='uniform',algorithm='kd_tree',leaf_size=30)
svm_clf        = svm.SVC(kernel='linear', C=1, probability=True)
gaussnb_clf    = naive_bayes.GaussianNB()

# 定義超參數及其候選值
knn_clf_param = {'n_neighbors':[1,3,5,7]}
svm_clf_param = {'C':[0.01, 0.1, 1, 10]}
gauss_clf_param={'alpha':[0.001,0.01,0.1,1,10,100]}
gaussnb_clf_param={'var_smoothing':np.logspace(-5,2,6)}

# inner cross-validation for hyper-parameter tuning
# 當n_jobs=-1時，在Windows可能有Bug，那麼就改為n_jobs = 1
gauss_gs     = GridSearchCV(estimator=gauss_clf,param_grid = gauss_clf_param, scoring = 'accuracy', cv=5, n_jobs=-1, verbose=1)
knn_gs       = GridSearchCV(estimator=knn_clf,param_grid = knn_clf_param, scoring = 'accuracy', cv=5, n_jobs=-1, verbose=1)
svm_gs       = GridSearchCV(estimator=svm_clf,param_grid = svm_clf_param, scoring = 'accuracy', cv=5,  n_jobs=-1, verbose=1)
svm_pipeline = Pipeline([('scaler',MinMaxScaler()),('svm_gs',svm_gs)])
gaussnb_gs   = GridSearchCV(estimator=gaussnb_clf,param_grid = gaussnb_clf_param, scoring = 'accuracy', cv=5, n_jobs=-1, verbose=1)

# outer cross-validation for estimating the accuracy of the classifier
# the classifiers to be compared must be evaluated by the same k-fold CV
kfold        = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

#當n_jobs=-1時，在Windows可能有Bug，那麼就改為n_jobs = 1
gauss_scores   = cross_val_score(gauss_gs, data, target, scoring='accuracy',cv = kfold, verbose=10) 
knn_scores     = cross_val_score(knn_gs, data, target, scoring='accuracy',cv = kfold, verbose=10)
svm_scores     = cross_val_score(svm_pipeline, data, target, scoring='accuracy',cv = kfold, verbose=10)
gaussnb_scores = cross_val_score(gaussnb_gs, data, target, scoring='accuracy',cv = kfold, verbose=10)

#請同學接續寫完評比
# apply the paired t-test (Refer to ppt for Chapter 20 Design and Analysis of Machine Learning Experiments)

gauss_mean = gauss_scores.mean()
knn_mean = knn_scores.mean()
svm_mean = svm_scores.mean()
gaussnb_mean = gaussnb_scores.mean()

print("knn :{:6f}\nsvm :{:6f}\ngauss :{:6f}\ngaussNB :{:6f}".format(knn_mean, svm_mean, gauss_mean, gaussnb_mean))

result = {'gauss': gauss_scores, 'knn': knn_scores, 'svm': svm_scores, 'gaussNB' :gaussnb_scores};

largest = 'gauss'
for index in result :
  if (result[largest].mean() < result[index].mean()) :
    largest = index

for index in result :
  if (largest != index) :
    __t, __pvalue = stats.ttest_1samp(result[largest] - result[index], 0.05)
    print(largest + " & " + index + ":{:6f}".format(__pvalue))
