import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class RegressionTree():
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None 
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        self.regressionTree = None
    def download_data(self):
        '''Download the data from the url'''
        if self.data == None:
            url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
            self.data = pd.read_csv(url)

    def data_analysis(self):
        '''Analisys the data, correleations ...'''
        self.data.describe()
        print(self.data.info())
        
        #important!!! - to see the correclation between data
        correlation_values = self.data.corr()['tip_amount'].drop('tip_amount')
        correlation_values.plot(kind="barh",figsize=(10, 6))
        plt.show()
        
        abs(correlation_values).sort_values(ascending=False)[:3] #!!!
    
    def preprocessing(self):
        '''Prepare data for training, normalize the feature matrix'''
        self.y = self.data[['tip_amount']].values.astype('float32')
        proc_data = self.data.drop(['tip_amount'], axis=1) # labeled matrix
        #from data analysis we saw the best features
        proc_data = self.data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)
        
        #get the feature matrix used for training
        self.X = proc_data.values # get only the values

        #normalize the feature matrix
        self.X = normalize(self.X, axis=1, norm='l1', copy=False) 
        
        
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
    
    def trainModel(self):
       self.regressionTree = DecisionTreeRegressor(criterion='squared_error', max_depth=8, random_state=34)
       self.regressionTree.fit(self.X_train, self.y_train)
       
    def evaluation(self):
        # run inference using the sklearn model
        y_pred = self.regressionTree.predict(self.X_test)

        # evaluate mean squared error on the test dataset
        mse_score = mean_squared_error(self.y_test, y_pred)
        print('MSE score : {0:.3f}'.format(mse_score))

        r2_score = self.regressionTree.score(self.X_test,self.y_test)
        print('R^2 score : {0:.3f}'.format(r2_score))
        
    def run(self):
        self.download_data()
        self.data_analysis()
        self.preprocessing()
        self.split_data()
        self.trainModel()
        self.evaluation()
        
if __name__ == '__main__':
    model = RegressionTree()
    model.run()
    