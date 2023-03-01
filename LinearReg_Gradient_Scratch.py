import numpy as np
import warnings

warnings.filterwarnings('ignore')
class SimpleLR:
    def __init__(self, alpha=0.01, n_iter=50, random_state = None,clip=5):
        self.clip = clip
        self.alpha = alpha
        self.n_iter = n_iter
        self.m = None
        self.b = None
        self.b_track = []
        self.loss_track = []
        self.X = None
        self.y = None
        self.y_pred = None
        if random_state is not None:
            np.random.seed(random_state)
                
    def loss(self):
        return np.sqrt(np.square(self.y-self.y_pred).sum()/self.y.shape[0])
    
    def coef_(self):
        return self.m, self.b 
    
    def score(self,X,y):
        y_pred = self.predict(X)
        ssr = np.square(y-y_pred).sum()
        y_pred = y.mean()
        sst =  np.square(y-y_pred).sum()
        return 1 - (ssr/sst)
        
    
    def update_weights(self):
        dl_dm = -2*(self.X.T.dot(self.y-self.predict(self.X)))/self.y.shape[0]
        # dl_dm = (dl_dm-dl_dm.min())/(dl_dm.max()-dl_dm.min())
        dl_dm = np.where(dl_dm>self.clip,self.clip,dl_dm)
        dl_dm = np.where(dl_dm<-self.clip,-self.clip,dl_dm)
        dl_db = -2*((self.y-self.predict(self.X)).sum())/self.y.shape[0]
        dl_db = np.where(dl_db>self.clip,self.clip,dl_db)
        dl_db = np.where(dl_db<-self.clip,-self.clip,dl_db)
        
        self.m = self.m - (dl_dm*self.alpha)
        self.b = self.b - (dl_db*self.alpha)
    
    def fit(self,X,y,verbose = True):
        self.X = X
        self.y = y
        self.m = np.random.randn(X.shape[1])
        self.b = np.random.randn(1)
        self.b_track.append(self.b)
        self.y_pred = self.predict(self.X) #(self.m*self.X+self.b).sum(axis=1).reshape(-1,1)
        self.loss_track.append(self.loss())
        for i in range(self.n_iter):
            self.update_weights()
            self.b_track.append(self.b)
            self.y_pred = self.predict(self.X)#(self.m*self.X+self.b).sum(axis=1).reshape(-1,1)
            self.loss_track.append(self.loss())
            if verbose:
                print("Cycle {0}: m:".format({i+1}),self.m,"b:",self.b,"loss:",self.loss())
        # print(self.y)
        # print(self.y_pred)
        # print(self.loss())
        # print(y_pred)
        # print(y)
        
    def predict(self, X):
        return X.dot(self.m)+self.b#(self.m*X+self.b).sum(axis=1).reshape(-1,1)
    
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
    