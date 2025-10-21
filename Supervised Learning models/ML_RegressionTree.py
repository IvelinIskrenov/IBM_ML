import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class RegressionTree():
    '''RegressionTree model which predicts the tip amount (tip_amount) for yellow cab rides in New York'''
    def __init__(self):
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_train = None 
        self.__X_test = None
        self.__y_train = None 
        self.__y_test = None
        self.__regressionTree = None
        
    def download_data(self) -> None:
        '''Download the data from the url'''
        if self.__data == None:
            url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
            self.__data = pd.read_csv(url)

    def data_analysis(self) -> None:
        '''Analisys the data, correleations ...'''
        self.__data.describe()
        print(self.__data.info())
        
        #important!!! - to see the correclation between data
        correlation_values = self.__data.corr()['tip_amount'].drop('tip_amount')
        correlation_values.plot(kind="barh",figsize=(10, 6))
        plt.show()
        
        abs(correlation_values).sort_values(ascending=False)[:3] #!!!
    
    def preprocessing(self) -> None:
        '''Prepare data for training, normalize the feature matrix'''
        self.__y = self.__data[['tip_amount']].values.astype('float32')
        proc_data = self.__data.drop(['tip_amount'], axis=1) # labeled matrix
        #from data analysis we saw the best features
        proc_data = self.__data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)
        
        #get the feature matrix used for training
        self.__X = proc_data.values # get only the values

        #normalize the feature matrix
        self.__X = normalize(self.__X, axis=1, norm='l1', copy=False) 
        
        
    def split_data(self) -> None:
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=0.3, random_state=42)
    
    def trainModel(self) -> None:
        '''Train the Decision Tree Regressor model with max-dedpth = 8'''
        self.__regressionTree = DecisionTreeRegressor(criterion='squared_error', max_depth=8, random_state=34)
        self.__regressionTree.fit(self.__X_train, self.__y_train)
       
    def evaluation(self) -> None:
        '''Evaluation with MSE and R^2'''
        # run inference using the sklearn model
        y_pred = self.__regressionTree.predict(self.__X_test)

        # evaluate mean squared error on the test dataset
        mse_score = mean_squared_error(self.__y_test, y_pred)
        print('MSE score : {0:.3f}'.format(mse_score))

        r2_score = self.__regressionTree.score(self.__X_test,self.__y_test)
        print('R^2 score : {0:.3f}'.format(r2_score))
        
    def run(self) -> None:
        self.download_data()
        self.data_analysis()
        self.preprocessing()
        self.split_data()
        self.trainModel()
        self.evaluation()
        
if __name__ == '__main__':
    model = RegressionTree()
    model.run()
    