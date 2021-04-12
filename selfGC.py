import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import multivariate_normal

class myGaussianClassifier(BaseEstimator, ClassifierMixin): #必須繼承 BaseEstimator, ClassifierMixin
    def __init__(self,alpha=1.e-5):        # initializer函式參數必須包含所有需要設定的參數及其內定值
        if isinstance(self,myGaussianClassifier):
            super(myGaussianClassifier,self).__init__()  
        self.alpha = alpha

    def fit(self,train,target): # 不能缺
        #    
        N,d = train.shape
        label = np.sort(np.unique(target.ravel()))
        self.c_     = label.size
        self.d_     = d
        self.prior_ = np.zeros((self.c_,))
        self.mean_  = np.zeros((self.c_,self.d_))
        self.cov_   = np.zeros((self.c_,self.d_,self.d_))
        # 計算 mean, covariance
        for cid,y in enumerate(label):
            idx = np.nonzero(target.ravel()==y)
                        # np.cov(X[np.nonzero(target.ravel()==i)],rowvar=False)+alpha*np.eye(X.shape[1])
            self.cov_[cid] = np.cov(train[idx],rowvar=False)+self.alpha*np.eye(d)
            # 完成mean及prior
            # sample mean:  𝑚𝑖 =np.mean(X[np.nonzero(target.ravel()==i)],axis=0)
            # prior:  𝑃(𝐶=𝑖) = np.sum(target.ravel()==i)/target.size
            self.mean_[cid]  = np.mean(train[idx], axis= 0)
            self.prior_[cid] = np.sum(target[idx])/target.size

        return self #最後要傳回self這個物件

    ## 
    # predict(self,X,y=None)裡
    # 必須計算 𝑋 裡，每一列資料的事後機率 𝑃(𝐶=𝑖|𝑥),∀𝑖 
    # 選擇事後機率高的那個類別為x的類別。
    #
    #  𝑃(𝐶=𝑖|𝑥) = p(x|C=i) * p(C=i) / p(x)
    #  p(x|C=i) : p(x,C) / p(C)
    #  p(C=i)   : self.prior_
    #  p(X)     : self.mean_
    def predict(self, X, y=None): # 不能缺        
        # using cov_ and prior and mean
        x = self.predict_proba(X)
        
        return np.argmax(x, axis = 1)
        
    ##
    # predict_proba(self,X,y=None)
    # 計算每一列資料的事後機率 𝑃(𝐶|𝑥)
    def predict_proba(self, X, y=None): # 視需要        
        x = []
        for i in range(len(X)):
            index = []
            for j in range(self.c_):
                index.append( 1/( 2*np.pi**(self.d_/2) * np.linalg.det(self.cov_[j])**(1/2) ) * np.exp((-1/2)*np.linalg.solve(self.cov_[j], X[i] - self.mean_[j]).T.dot(X[i] - self.mean_[j])) )
            index = np.array(index)
            x.append(index / index.sum())
            
        return np.array(x)

    ##
    # score(self,X,y)
    # 計算這個模型資料為X，
    # 答案為y的得到的評分(越高分代表越好喔，例如準確度)
    def score(self, X, y): # 可有可無
        pass




