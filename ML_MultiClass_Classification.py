import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MultiClass_Classification:
    
    def __init__(self):
        self.data = None
    
    def  load_data(self):
        if self.data == None:
            file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
            self.data = pd.read_csv(file_path)
            
    def data_analysis(self):
        #Distribution of target variable
        sns.countplot(y='NObeyesdad', data=self.data)
        plt.title('Distribution of Obesity Levels')
        plt.show()
        
    def run(self):
        self.load_data()
        self.data_analysis()
        
if __name__ == '__main__':
    model = MultiClass_Classification()
    model.run()