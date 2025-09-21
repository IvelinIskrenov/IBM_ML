import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        url = self.url
        df = pd.read_csv(url) #load in dataframe
        self.df = df

        df.sample(5)
        print(df.sample(5))

        cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
        cdf.sample(9)
        self.cdf = cdf

    def show_examples(self):
        #learn example (testing)
        """
        viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
        viz.hist() #generating histogram for each col
        plt.show()
        
        plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue') #creating dot diagram X = FC_COMB, Y = CO2E.
        plt.xlabel("FUELCONSUMPTION_COMB")
        plt.ylabel("Emission")
        plt.show()
        
        plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
        plt.xlabel("Engine size")
        plt.ylabel("Emission")
        plt.xlim(0,27) #zoom
        plt.show()
        
        plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
        plt.xlabel("CYLINDERS")
        plt.ylabel("CO2 Emission")
        plt.show()
        """
        pass

    def run_regression_enginesize(self):
        #extracting the input feature and target output variables - X and y from the dataset.
        X = self.cdf.ENGINESIZE.to_numpy()
        y = self.cdf.CO2EMISSIONS.to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        type(X_train), np.shape(X_train), np.shape(X_train)
        
        #create a model object
        regressor = linear_model.LinearRegression()
        self.regressor = regressor
        
        #shape (n_observations, n_features).
        #need to reshape it - We can let it infer the number of observations using '-1'. (From 1D to 2D)
        regressor.fit(X_train.reshape(-1, 1), y_train) # fit -> calculate theta (minimizing MSE)
        
        #print the coefficients
        print ('Coefficients: ', regressor.coef_[0]) #with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
        print ('Intercept: ',regressor.intercept_)
        
        plt.scatter(X_train, y_train,  color='blue')
        plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
        plt.xlabel("Engine size")
        plt.ylabel("Emission")
        plt.show()
        
        # Use the predict method to make test predictions
        y_test_ = regressor.predict(X_test.reshape(-1,1))

        # Evaluation
        print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
        print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
        print("R2-score: %.2f" % r2_score(y_test, y_test_))
        
        #example 2
        plt.scatter(X_test, y_test,  color='blue')
        plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
        plt.xlabel("Engine size")
        plt.ylabel("Emission")
        plt.show()

    def run_regression_fuelconsumption(self):
        X = self.cdf.FUELCONSUMPTION_COMB.to_numpy()
        y = self.cdf.CO2EMISSIONS.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        regr = linear_model.LinearRegression()
        self.regr = regr
        regr.fit(X_train.reshape(-1, 1), y_train)
        
        y_test_ = regr.predict(X_test.reshape(-1,1))
        
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
        print("R2-score: %.2f" % r2_score(y_test, y_test_))
        
        #LinearRegression 2 is more efficient then 1

    def run(self):
        self.load_data()
        # self.show_examples()  # kept as method to preserve the triple-quoted block; not called
        self.run_regression_enginesize()
        self.run_regression_fuelconsumption()


def main():
    fm = FuelEmissions()
    fm.run()

if __name__ == "__main__":
    main()
