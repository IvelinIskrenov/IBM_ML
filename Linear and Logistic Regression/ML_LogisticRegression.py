import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split , cross_val_score #!
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score


class Logistic_model:
    """
        Logistic regression model. 
        1.Load data.
        2.Preprocess and split data.
        3.Train the model.
        4.Evaluation and log-loss.
    """
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
        '''Load the data from the url and describe it'''
        if self.url == None:
            self.url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
        self.churn_df = pd.read_csv(self.url)
        self.churn_df.describe()
        #print(self.churn_df.sample(5))
    
    def preprocessing(self):
        '''Change the pred. value -> int, set the cols and standardizing it '''
        print(self.churn_df.head(9))
        self.churn_df = self.churn_df[['tenure', 'income', 'ed', 'equip', 'churn']]#, 'equip'
        self.churn_df['churn'] = self.churn_df['churn'].astype('int') #set the predicted value as type int
        
        self.X = np.asarray(self.churn_df[['tenure', 'income', 'ed', 'equip']]) # ,'age', 'address',||, 'employ' ,,'wireless'
        self.y = np.asarray(self.churn_df['churn'])
        
        self.X_norm = StandardScaler().fit(self.X).transform(self.X)
    
    def splitDataSet(self, test_size: float = 0.2, random_state: int = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_norm, self.y, test_size = test_size, random_state = random_state
            )
        
    def train(self):
        '''
            Train the LogisticRegression model
            Generates predicted labels - self.yhat && probabilities - yhat_prob for a test set
        '''
        self.logisticRegressor = LogisticRegression(
            solver='lbfgs', penalty='l2', C=1.0, max_iter=100
            ).fit(self.X_train, self.y_train) #l1 = lasso, choose the best features /solver='lbfgs', penalty='l2', C=1.0, max_iter=2/
        #With Lasso = l1 & solver='liblinear' there is a 0.5% better accurancy at cross validaton
        
        self.yhat = self.logisticRegressor.predict(self.X_test)
        #print(self.yhat[:10])
        
        yhat_prob = self.logisticRegressor.predict_proba(self.X_test) #see the probability
        print(yhat_prob[:4])
        
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
        
    def cross_validation(self):
        scores = cross_val_score(self.logisticRegressor, self.X_norm, self.y, cv=10)
        print("Cross-validation scores: ", scores)
        print("Mean cross-validation accuracy: {:.2f}%".format(scores.mean() * 100))
        
    def evaluate(self):
        self.accurancy()
        self.cross_validation()
        
    #Show us vision of that, which feature is better
    def drop_column_logloss(self):
        '''Show us which of the feature are better then others to help you to select the best features'''
        base_prob = self.logisticRegressor.predict_proba(self.X_test)
        base_loss = log_loss(self.y_test, base_prob)
        print(f"Base log-loss (with all features): {base_loss:.6f}")

        feat_names = list(self.churn_df.columns[:-1])
        results = []
        for i, feat in enumerate(feat_names):
            X_all = np.asarray(self.churn_df[feat_names])
            #drop column i
            mask = [j for j in range(X_all.shape[1]) if j != i]
            X_new = X_all[:, mask]
            X_new_scaled = StandardScaler().fit_transform(X_new)
            Xtr, Xte, ytr, yte = train_test_split(X_new_scaled, self.y, test_size=0.2, random_state=42)
            clf = LogisticRegression(solver='lbfgs', penalty='l2', C=1.0, max_iter=200)
            clf.fit(Xtr, ytr)
            loss = log_loss(yte, clf.predict_proba(Xte))
            results.append((feat, loss))
        df = pd.DataFrame(results, columns=['feature','log_loss']).set_index('feature').sort_values('log_loss')
        print(df)
        df['log_loss'].plot(kind='barh', title='Log-loss when dropping single feature (lower is better)')
        plt.xlabel('log_loss')
        plt.show()
        return df
        #Better acc when we droped some cols    
        
    def run(self):
        self.loadData()
        self.preprocessing()
        self.splitDataSet()
        self.train()
        self.evaluate()
        self.drop_column_logloss()
        
if __name__ == '__main__':
    model = Logistic_model()
    model.run()
        
        
        
