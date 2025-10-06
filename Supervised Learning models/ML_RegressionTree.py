import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

class RegressionTree():
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        
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
    
    def preprocessing(self):
        '''Prepare data for training, normalize the feature matrix'''
        self.y = self.data[['tip_amount']].values.astype('float32')
        proc_data = self.data.drop(['tip_amount'], axis=1) # labeled matrix
        
        #get the feature matrix used for training
        self.X = proc_data.values # get only the values

        #normalize the feature matrix
        self.X = normalize(self.X, axis=1, norm='l1', copy=False) 
        
    def run(self):
        self.download_data()
        self.data_analysis()
        self.preprocessing()
        
if __name__ == '__main__':
    model = RegressionTree()
    model.run()
    