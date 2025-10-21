import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

class HousePrediction():
    '''
    Implemention Random Forest and XGBoost regression models for predicting house prices using the California Housing Dataset
        && compare the models
    '''
    def __init__(self):
        self.__data = None
        self.__model_RandomForest = None
        self.__model_XGBoost = None
        self.__X = None
        self.__y = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__n_estimators=100
        self.__std_y = None
        
    def load_data(self) -> None:
        '''Load the data from sklearn.dataset'''
        if self.__data == None:
            self.__data = fetch_california_housing()
        self.__X = pd.DataFrame(self.__data.data, columns=self.__data.feature_names)
        self.__y = self.__data.target  
    
    def data_analysis(self) -> None:
        '''Describe the data - Print the first rows, get the cols info and DType'''
        print(self.__X.head())
        
        print("\n5. Feature DataFrame Info:")
        self.__X.info()
        
        print(self.__X.describe())
        
        N_observations, N_features = self.__X.shape
        print('Number of Observations: ' + str(N_observations))
        print('Number of Features: ' + str(N_features))
    
    def split_data(self) -> None:
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=0.2, random_state=42)
          
    def Build_Train_RandomForest(self) -> None:
        '''Build and traing RandomForestRegressor model'''
        try:
            
            self.__model_RandomForest = RandomForestRegressor(n_estimators=self.__n_estimators, random_state=42)
            start_time_rf = time.time()
            self.__model_RandomForest.fit(self.__X_train, self.__y_train)
            end_time_rf = time.time()
            rf_train_time = end_time_rf - start_time_rf
            print(f'Random Forest train time = {rf_train_time:.4f}')
        except Exception:
            print(f"Error in train_build RandomForest model !!!")
            
    def Build_Train_model_XGBoost(self) -> None:
        '''Build and traing XGBoost model && XGBoost library should be installed'''
        try:
            
            self.__model_XGBoost = XGBRegressor(n_estimators=self.__n_estimators, random_state=42)
            start_time_xgb = time.time()
            self.__model_XGBoost.fit(self.__X_train, self.__y_train)
            end_time_xgb = time.time()
            xgb_train_time = end_time_xgb - start_time_xgb
            print(f'XGBoost train time = {xgb_train_time:.4f}')
        except Exception:
            print(f"Error in train_build XGBoost model !!!")
            
    def evaluation(self) -> None:
        '''Evaluation with MSE and R^2 and time printing'''
        try:
            #Measure prediction time for Random Forest
            start_time_RF = time.time()
            y_pred_RF = self.__model_RandomForest.predict(self.__X_test)
            end_time_RF = time.time()
            RF_pred_time = end_time_RF - start_time_RF
        
            #Measure prediciton time for XGBoost
            start_time_XGB = time.time()
            y_pred_XGB = self.__model_XGBoost.predict(self.__X_test)
            end_time_XGB = time.time()
            XGB_pred_time = end_time_XGB - start_time_XGB
        
            self.R2_MSE(y_pred_RF,y_pred_XGB)
            #print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')
            #print(f'      XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')
            self.__std_y = np.std(self.__y_test)
        
            print(f"Do you want visualize the models Y/N ?")
            vision = input()
            if vision == "Y":
                self.visualize(y_pred_RF, y_pred_XGB)
        except Exception:
            print(f"Error in evaluation !!!")
            
    def R2_MSE(self, y_pred_RF, y_pred_XGB) -> None:
        '''Calculate and printing the MSE and R^2 values for both models'''    
        mse_RF = mean_squared_error(self.__y_test, y_pred_RF)
        mse_XGB = mean_squared_error(self.__y_test, y_pred_XGB)
        r2_RF = r2_score(self.__y_test, y_pred_RF)
        r2_XGB = r2_score(self.__y_test, y_pred_XGB)
        
        print(f'Random Forest:  MSE = {mse_RF:.4f}, R^2 = {r2_RF:.4f}')
        print(f'      XGBoost:  MSE = {mse_XGB:.4f}, R^2 = {r2_XGB:.4f}')
    
    def visualize(self, y_pred_RF, y_pred_XGB) -> None:
        '''Visualiza the RandomForest && XGBoost models'''
        plt.figure(figsize=(14, 6))

        # Random Forest plot
        plt.subplot(1, 2, 1)
        plt.scatter(self.__y_test, y_pred_RF, alpha=0.5, color="blue",ec='k')
        plt.plot([self.__y_test.min(), self.__y_test.max()], [self.__y_test.min(), self.__y_test.max()], 'k--', lw=2,label="perfect model")
        plt.plot([self.__y_test.min(), self.__y_test.max()], [self.__y_test.min() + self.__std_y, self.__y_test.max() + self.__std_y], 'r--', lw=1, label="+/-1 Std Dev")
        plt.plot([self.__y_test.min(), self.__y_test.max()], [self.__y_test.min() - self.__std_y, self.__y_test.max() - self.__std_y], 'r--', lw=1, )
        plt.ylim(0,6)
        plt.title("Random Forest Predictions vs Actual")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()


        # XGBoost plot
        plt.subplot(1, 2, 2)
        plt.scatter(self.__y_test, y_pred_XGB, alpha=0.5, color="orange",ec='k')
        plt.plot([self.__y_test.min(), self.__y_test.max()], [self.__y_test.min(), self.__y_test.max()], 'k--', lw=2,label="perfect model")
        plt.plot([self.__y_test.min(), self.__y_test.max()], [self.__y_test.min() + self.__std_y, self.__y_test.max() + self.__std_y], 'r--', lw=1, label="+/-1 Std Dev")
        plt.plot([self.__y_test.min(), self.__y_test.max()], [self.__y_test.min() - self.__std_y, self.__y_test.max() - self.__std_y], 'r--', lw=1, )
        plt.ylim(0,6)
        plt.title("XGBoost Predictions vs Actual")
        plt.xlabel("Actual Values")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def run(self) -> None:
        self.load_data()  
        self.data_analysis()
        self.split_data()
        self.Build_Train_model_XGBoost()
        self.Build_Train_RandomForest()
        self.evaluation()
        
if __name__ == '__main__':
    model = HousePrediction()
    model.run()