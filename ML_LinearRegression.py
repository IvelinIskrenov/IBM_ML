import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    df=pd.read_csv(url) #load in dataframe
    
    df.sample(5)
    print(df.sample(5))

    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    cdf.sample(9)
    
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
    #extracting the input feature and target output variables - X and y from the dataset.
    X = cdf.ENGINESIZE.to_numpy()
    y = cdf.CO2EMISSIONS.to_numpy()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    type(X_train), np.shape(X_train), np.shape(X_train)
    
    from sklearn import linear_model
    #create a model object
    regressor = linear_model.LinearRegression()
    
    #shape (n_observations, n_features).
    #need to reshape it - We can let it infer the number of observations using '-1'.
    regressor.fit(X_train.reshape(-1, 1), y_train)
    
    # Print the coefficients
    print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
    print ('Intercept: ',regressor.intercept_)
    
    plt.scatter(X_train, y_train,  color='blue')
    plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()
    
    
    
if __name__ == "__main__":
    main()
