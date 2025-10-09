import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

class HousePrediction():
    '''
    Implemention Random Forest and XGBoost regression models for predicting house prices using the California Housing Dataset
        && compare the models
    '''
    def __init__(self):
        self.data = None
        self.model_RandomForest = None
        self.model_XGBoost = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        '''Load the data from sklearn.dataset'''
        if self.data == None:
            self.data = fetch_california_housing()
        self.X = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        self.y = self.data.target  
    
    def data_analysis(self):
        '''Describe the data - Print the first rows, get the cols info and DType'''
        print(self.X.head())
        
        print("\n5. Feature DataFrame Info:")
        self.X.info()
        
        print(self.X.describe())
        
        N_observations, N_features = self.X.shape
        print('Number of Observations: ' + str(N_observations))
        print('Number of Features: ' + str(N_features))
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
          
    def run(self):
        self.load_data()  
        self.data_analysis()
        
if __name__ == '__main__':
    model = HousePrediction()
    model.run()