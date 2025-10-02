import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#class to analyze fuel cons. data and predict CO2 emissions
class FuelEmissions:
    def __init__(self, url=None):
        if url is None:
            url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
        self.url = url
        self.df = None
        self.cdf = None
        self.regressor = None
        self.regr = None

    def load_data(self):
        '''
        Loads the data from a CSV file and get ready for preprocessing
        '''
        url = self.url
        #Read the data file into a DataFrame.
        df = pd.read_csv(url) 
        self.df = df

        print(df.sample(5))

        #Selecting only the important cols
        cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
        self.cdf = cdf

    def run_regression_enginesize(self):
        '''Predicts emissions based on engine size.'''
        X = self.cdf.ENGINESIZE.to_numpy()
        y = self.cdf.CO2EMISSIONS.to_numpy()
        
        #Split the data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        regressor = linear_model.LinearRegression()
        self.regressor = regressor
        
        #train the model &&
        #reshape the data to fit the model's requirements.
        regressor.fit(X_train.reshape(-1, 1), y_train)
        
        print ('Coefficients: ', regressor.coef_[0])
        print ('Intercept: ',regressor.intercept_)
        
        #Plot the traing data
        plt.scatter(X_train, y_train, color='blue')
        plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
        plt.xlabel("Engine size")
        plt.ylabel("Emission")
        plt.show()
        
        #use the trained model to make predictions on the test data.
        y_test_ = regressor.predict(X_test.reshape(-1,1))

        #evaluate how well the model performed
        print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
        print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
        print("R2-score: %.2f" % r2_score(y_test, y_test_))
        
        #Plot the test data
        plt.scatter(X_test, y_test, color='blue')
        plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
        plt.xlabel("Engine size")
        plt.ylabel("Emission")
        plt.show()

    def run_regression_fuelconsumption(self):
        '''Predicts emissions based on fuel consumption.'''
        #set up the input and output data.
        X = self.cdf.FUELCONSUMPTION_COMB.to_numpy()
        y = self.cdf.CO2EMISSIONS.to_numpy()
        #Split the data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #Create and train the model.
        regr = linear_model.LinearRegression()
        self.regr = regr
        regr.fit(X_train.reshape(-1, 1), y_train)
        
        y_test_ = regr.predict(X_test.reshape(-1,1))
        
        #evaluate the model.
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
        print("R2-score: %.2f" % r2_score(y_test, y_test_))

    def run(self):
        '''Runs the entire analysis process'''
        self.load_data()
        # self.show_examples()  
        self.run_regression_enginesize()
        self.run_regression_fuelconsumption()


def main():
    fm = FuelEmissions()
    fm.run()

if __name__ == "__main__":
    #runs if  script is executable
    main()
