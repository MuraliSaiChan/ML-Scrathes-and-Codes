import numpy as np
from sklearn.model_selection import train_test_split
import warnings

class SimpleLR:
    def __init__(self, alpha=0.001, n_iter=50, random_state = None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.m = None
        self.b = None
        self.X = None
        self.y = None
        self.y_pred = None
        if random_state is not None:
            np.random.seed(random_state)
            
        
    def loss(self):
        return np.square(self.y-self.y_pred).sum()/self.y.shape[0]
    
    def coef_(self):
        return self.m, self.b 
    
    def score(self,X,y):
        self.y_pred = self.m*X+self.b
        ssr = np.square(y-self.y_pred).sum()
        self.y_pred = self.y.mean()
        sst =  np.square(y-self.y_pred).sum()
        return 1 - (ssr/sst)
        
    
    def update_weights(self):
        dl_dm = -2*(self.X.T.dot((self.y-self.m*self.X-self.b))).sum(axis=1)/self.y.shape[0]
        dl_dm = np.where(dl_dm>50,50,dl_dm)
        dl_dm = np.where(dl_dm<-50,-50,dl_dm)
        dl_db = -2*((self.y-self.m*self.X-self.b)).sum(axis=0)/self.y.shape[0]
        dl_db = np.where(dl_db>50,50,dl_db)
        dl_db = np.where(dl_db<-50,-50,dl_db)
        
        self.m = self.m - (dl_dm*self.alpha)
        self.b = self.b - (dl_db*self.alpha)
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.m = np.random.randn(X.shape[1]).reshape(1,-1)
        self.b = np.random.randn(X.shape[1]).reshape(1,-1)
        self.y_pred = (self.m*self.X+self.b).sum(axis=1).reshape(-1,1)
        for i in range(self.n_iter):
            self.update_weights()
            self.y_pred = (self.m*self.X+self.b).sum(axis=1).reshape(-1,1)
            # print("Cycle {0}: m:".format({i+1}),self.m,"b:",self.b,"loss:",self.loss())
        print(self.y)
        print(self.y_pred)
        print(self.loss())
        # print(y_pred)
        # print(y)
        
    def predict(self, X):
        return self.m*X+self.b
    
# warnings.filterwarnings('ignore')    

# lr = SimpleLR(random_state=None, n_iter=1000)
# X = np.arange(1,1000)
# y = X*5+5
# print(y.shape[0])
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)   

# lr.fit(X_train, y_train)

# print(lr.score(X_train, y_train))
# print(lr.score(X_test, y_test))

# print(lr.predict(X_test))
# print(y_test)    
    