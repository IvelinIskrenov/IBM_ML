import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    df=pd.read_csv(url) #load in dataframe  
    df.sample(5)
    
    df.describe()
    #df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1) #drop hte first col
    #print(df.corr())
    
    #choose strong correleation for CO2
    #corr_matrix = df.corr()
    #target_corr = corr_matrix['CO2EMISSIONS']
    #strong_features = target_corr[abs(target_corr) >= 0.8].drop('CO2EMISSIONS')
    
    df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
    print(df.head(9))
    
    axes = pd.plotting.scatter_matrix(df, alpha=0.2)
    # need to rotate axis labels so we can read them
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')

    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.show()
    #from this example, we see that the fuel con. and CO2 are no linear
    
    #extract cols
    X = df.iloc[:,[0,1]].to_numpy()
    y = df.iloc[:,[2]].to_numpy()
    
    # - Preprocess selected features - #
    from sklearn import preprocessing

    std_scaler = preprocessing.StandardScaler()
    X_std = std_scaler.fit_transform(X)
    
    pd.DataFrame(X_std).describe().round(2)
    
    # - Create train and test dataset # -
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.2,random_state=42)
    
    


    
main()