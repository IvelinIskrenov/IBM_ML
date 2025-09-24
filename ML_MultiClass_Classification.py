import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class MultiClass_Classification:
    
    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.prepped_data = None
    
    def  load_data(self):
        if self.data == None:
            file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
            self.data = pd.read_csv(file_path)
            
    def data_analysis(self):
        #Distribution of target variable
        sns.countplot(y='NObeyesdad', data=self.data)
        plt.title('Distribution of Obesity Levels')
        plt.show()
        
        print(self.data.isnull().sum())

        #Dataset summary
        print(self.data.info())
        print(self.data.describe())
    
    def preprocessing(self):
        #Standardizing continuous numerical features
        continuous_columns = self.data.select_dtypes(include=['float64']).columns.tolist()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data[continuous_columns])

        #Converting to a DataFrame
        scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

        #Combining with the original dataset
        self.scaled_data = pd.concat([self.data.drop(columns=continuous_columns), scaled_df], axis=1)
    
    #turnig categorical into numerical data
    def one_hot_encoding(self):
        #Identifying categorical columns
        categorical_columns = self.scaled_data.select_dtypes(include=['object']).columns.tolist()
        categorical_columns.remove('NObeyesdad')  #Exclude target column
        
        #Applying one-hot encoding
        encoder = OneHotEncoder(
            sparse_output=False, 
            drop='first' #avoid multicollinearity
            )
        encoded_features = encoder.fit_transform(self.scaled_data[categorical_columns])

        #Converting to a DF
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

        #Combining with the original dataset
        self.prepped_data = pd.concat([self.scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
        
    #inverse the process !!!
    
    def run(self):
        self.load_data()
        self.data_analysis()
        
if __name__ == '__main__':
    model = MultiClass_Classification()
    model.run()