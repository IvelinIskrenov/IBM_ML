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
if __name__ == "__main__":
    main()
