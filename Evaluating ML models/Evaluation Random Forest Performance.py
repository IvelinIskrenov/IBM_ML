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
        self.__y_pred_test = None
        
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
        plt.show()
    
    def build_RF(self) -> None:
        '''Build Random Forest alg with n_estimators=100'''
        self.__model_RF = RandomForestRegressor(n_estimators=100, random_state=42)
        self.__model_RF.fit(self.__X_train, self.__y_train)
    
    def predict_test(self) -> None:
        self.__y_pred_test = self.__model_RF.predict(self.__X_test)
    
    def plot_AvsP_values(self) -> None:
        '''Plot Actual vs Predicted values'''
        plt.scatter(self.__y_test, self.__y_pred_test, alpha=0.5, color="blue")
        plt.plot([self.__y_test.min(), self.__y_test.max()], [self.__y_test.min(), self.__y_test.max()], 'k--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Random Forest Regression - Actual vs Predicted")
        plt.show()
    
    def plot_residual_errors(self) -> None:
        '''Plot histogram of the residual errors'''
        #Calculate the residual errors
        residuals = 1e5*(self.__y_test - self.__y_pred_test)

        #Plot the histogram of the residuals
        plt.hist(residuals, bins=30, color='lightblue', edgecolor='black')
        plt.title(f'Median House Value Prediction Residuals')
        plt.xlabel('Median House Value Prediction Error ($)')
        plt.ylabel('Frequency')
        print('Average error = ' + str(int(np.mean(residuals))))
        print('Standard deviation of error = ' + str(int(np.std(residuals))))
    
    def plot_model(self) -> None:
        '''Plot the model residual errors by median house value'''
        
        # Create a DataFrame to make sorting easy
        residuals = 1e5*(self.__y_test - self.__y_pred_test)
        residuals_df = pd.DataFrame({
            'Actual': 1e5*self.__y_test,
            'Residuals': residuals
        })

        #Sort the DataFrame by the actual target values
        residuals_df = residuals_df.sort_values(by='Actual')

        plt.scatter(residuals_df['Actual'], residuals_df['Residuals'], marker='o', alpha=0.4,ec='k')
        plt.title('Median House Value Prediciton Residuals Ordered by Actual Median Prices')
        plt.xlabel('Actual Values (Sorted)')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.show()
    
    def feature_importances(self) -> None:
        '''Display the feature importances as a bar chart'''
        importances = self.__model_RF.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = self.__data.feature_names

        # Plot feature importances
        plt.bar(range(self.__X.shape[1]), importances[indices],  align="center")
        plt.xticks(range(self.__X.shape[1]), [features[i] for i in indices], rotation=45)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Feature Importances in Random Forest Regression")
        plt.show()
        
    def evaluation(self) -> None:
        '''Printing MAE, MSE, RMSE, R^2 for the model'''
        
        mae = mean_absolute_error(self.__y_test, self.__y_pred_test)
        mse = mean_squared_error(self.__y_test, self.__y_pred_test)
        rmse = root_mean_squared_error(self.__y_test, self.__y_pred_test)
        r2 = r2_score(self.__y_test, self.__y_pred_test)
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
        
        self.predict_test()
        self.evaluation()
        
        self.plot_residual_errors()
        self.plot_AvsP_values()
        self.plot_model()
        self.feature_importances()
            
if __name__ == "__main__":
    model = HousePricePrediting()
    model.pipeline()