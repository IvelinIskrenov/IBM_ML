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
        self.__churn_df = None
        self.__X = None
        self.__y = None
        self.__X_norm = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__logisticRegressor = None
        self.__yhat = None
        self.__coefficients = None
        
        
    def loadData(self) -> None:
        '''Load the data from the url and describe it'''
        if self.url == None:
            self.url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
        self.__churn_df = pd.read_csv(self.url)
        self.__churn_df.describe()
        #print(self.churn_df.sample(5))
    
    def preprocessing(self) -> None:
        '''Change the pred. value -> int, set the cols and standardizing it '''
        try:
            print(self.__churn_df.head(9))
            self.__churn_df = self.__churn_df[['tenure', 'income', 'ed', 'equip', 'churn']]#, 'equip'
            self.__churn_df['churn'] = self.__churn_df['churn'].astype('int') #set the predicted value as type int
        
            self.__X = np.asarray(self.__churn_df[['tenure', 'income', 'ed', 'equip']]) # ,'age', 'address',||, 'employ' ,,'wireless'
            self.__y = np.asarray(self.__churn_df['churn'])
        
            self.__X_norm = StandardScaler().fit(self.__X).transform(self.__X)
        except Exception:
            print(f"Error in preprocessing !!!")
            
    def splitDataSet(self, test_size: float = 0.2, random_state: int = 42)  -> None:
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            self.__X_norm, self.__y, test_size = test_size, random_state = random_state
            )
        
    def train(self) -> None:
        '''
            Train the LogisticRegression model
            Generates predicted labels - self.yhat && probabilities - yhat_prob for a test set
        '''
        self.__logisticRegressor = LogisticRegression(
            solver='lbfgs', penalty='l2', C=1.0, max_iter=100
            ).fit(self.__X_train, self.__y_train) #l1 = lasso, choose the best features /solver='lbfgs', penalty='l2', C=1.0, max_iter=2/
        #With Lasso = l1 & solver='liblinear' there is a 0.5% better accurancy at cross validaton
        
        self.__yhat = self.__logisticRegressor.predict(self.__X_test)
        #print(self.yhat[:10])
        
        yhat_prob = self.__logisticRegressor.predict_proba(self.__X_test) #see the probability
        print(yhat_prob[:4])
        
        coefficients = pd.Series(self.__logisticRegressor.coef_[0], index=self.__churn_df.columns[:-1])
        coefficients.sort_values().plot(kind='barh')
        plt.title("Feature Coefficients in Logistic Regression Churn Model")
        plt.xlabel("Coefficient Value")
        plt.show() 
        
        print(log_loss(self.__y_test, yhat_prob))
    
    def accurancy(self) -> None:
        y_pred = self.__logisticRegressor.predict(self.__X_test)
        accuracy = accuracy_score(self.__y_test, y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        
    def cross_validation(self) -> None:
        scores = cross_val_score(self.__logisticRegressor, self.__X_norm, self.__y, cv=10)
        print("Cross-validation scores: ", scores)
        print("Mean cross-validation accuracy: {:.2f}%".format(scores.mean() * 100))
        
    def evaluate(self) -> None:
        self.accurancy()
        self.cross_validation()
        
    #Show us vision of that, which feature is better
    def drop_column_logloss(self):
        '''Show us which of the feature are better then others to help you to select the best features'''
        base_prob = self.__logisticRegressor.predict_proba(self.__X_test)
        base_loss = log_loss(self.__y_test, base_prob)
        print(f"Base log-loss (with all features): {base_loss:.6f}")

        feat_names = list(self.__churn_df.columns[:-1])
        results = []
        for i, feat in enumerate(feat_names):
            X_all = np.asarray(self.__churn_df[feat_names])
            #drop column i
            mask = [j for j in range(X_all.shape[1]) if j != i]
            X_new = X_all[:, mask]
            X_new_scaled = StandardScaler().fit_transform(X_new)
            Xtr, Xte, ytr, yte = train_test_split(X_new_scaled, self.__y, test_size=0.2, random_state=42)
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
        
    def run(self) -> None:
        self.loadData()
        self.preprocessing()
        self.splitDataSet()
        self.train()
        self.evaluate()
        self.drop_column_logloss()
        
if __name__ == '__main__':
    model = Logistic_model()
    model.run()
        
        
        
