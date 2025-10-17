import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew

class HousePricePrediting():
    '''
        Build Random Forest - evaluate its performance and use California Housing data set included in scikit-learn to 
        predict the median house price based on various attributes
    '''
    def __init__(self):
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_train = None 
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__model_RF = None
        
    def load_data(self) -> None:
        '''Loads data from sklearn.datasets'''
        try:
            self.__data = fetch_california_housing()
            self.__X, self.__y = self.__data.data, self.__data.target
        except Exception:
            print(f"Error while loading the data !!!")
            
    def split_data(self) -> None:
        '''Splits the data'''
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, random_state=42)
    
    def explore_train_data(self) -> None:
        eda = pd.DataFrame(data=self.__X_train)
        eda.columns = self.__data.feature_names
        eda['MedHouseVal'] = self.__y_train
        print(eda.describe())
    
    def distribute(self) -> None:
        '''median house prices distributed'''
        plt.hist(1e5*self.__y_train, bins=30, color='lightblue', edgecolor='black')
        plt.title(f'Median House Value Distribution\nSkewness: {skew(self.__y_train):.2f}')
        plt.xlabel('Median House Value')
        plt.ylabel('Frequency')
    
    def build_RF(self) -> None:
        '''Build Random Forest alg with n_estimators=100'''
        self.__model_RF = RandomForestRegressor(n_estimators=100, random_state=42)
        self.__model_RF.fit(self.__X_train, self.__y_train)
        
    def evaluation(self) -> None:
        '''Printing MAE, MSE, RMSE, R^2 for the model'''
        y_pred_test = self.__model_RF.predict(self.__X_test)
        
        mae = mean_absolute_error(self.__y_test, y_pred_test)
        mse = mean_squared_error(self.__y_test, y_pred_test)
        rmse = root_mean_squared_error(self.__y_test, y_pred_test)
        r2 = r2_score(self.__y_test, y_pred_test)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
    def pipeline(self) -> None:
        self.load_data()
        self.split_data()
        self.explore_train_data()
        self.distribute()
        self.build_RF()
        self.evaluation()
            
if __name__ == "__main__":
    model = HousePricePrediting()
    model.pipeline()