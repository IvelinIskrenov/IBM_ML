import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split #!
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Logistic_model:
    def __init__(self):
        self.url = None
        self.churn_df = None
        self.X = None
        self.y = None
        self.X_norm = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.logisticRegressor = None
        self.yhat = None
        self.coefficients = None
        
        
    def loadData(self):
        if self.url == None:
            self.url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
        self.churn_df = pd.read_csv(self.url)
        self.churn_df.describe()
        #print(self.churn_df.sample(5))
    
    def preprocessing(self):
        print(self.churn_df.head(9))
        self.churn_df = self.churn_df[['tenure', 'income', 'ed', 'equip', 'churn']]#, 'equip'
        self.churn_df['churn'] = self.churn_df['churn'].astype('int') #set the predicted value as type int
        
        self.X = np.asarray(self.churn_df[['tenure',  'income', 'ed', 'equip']]) # ,'age', 'address',||, 'employ' ,,'wireless'
        self.y = np.asarray(self.churn_df['churn'])
        
        self.X_norm = StandardScaler().fit(self.X).transform(self.X)
    
    def splitDataSet(self, test_size: float = 0.2, random_state: int = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_norm, self.y, test_size = test_size, random_state = random_state
            )
        
    def train(self):
        self.logisticRegressor = LogisticRegression(solver='lbfgs', penalty='l2', C=1.0, max_iter=2).fit(self.X_train, self.y_train)
        
        self.yhat = self.logisticRegressor.predict(self.X_test)
        print(self.yhat[:10])
        
        yhat_prob = self.logisticRegressor.predict_proba(self.X_test) #see the probability
        print(yhat_prob[:10])
        
        coefficients = pd.Series(self.logisticRegressor.coef_[0], index=self.churn_df.columns[:-1])
        coefficients.sort_values().plot(kind='barh')
        plt.title("Feature Coefficients in Logistic Regression Churn Model")
        plt.xlabel("Coefficient Value")
        plt.show() 
        
        print(log_loss(self.y_test, yhat_prob))
    
    def accurancy(self):
        y_pred = self.logisticRegressor.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        
    def run(self):
        self.loadData()
        self.preprocessing()
        self.splitDataSet()
        self.train()
        self.accurancy()
        
        
if __name__ == '__main__':
    model = Logistic_model()
    model.run()
        
        
        
