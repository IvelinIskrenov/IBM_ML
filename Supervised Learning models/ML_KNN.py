import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class KnnModel():
    '''Build and training classifier model KNN, which predict the service category for unknown cases'''
    def __init__(self):
        self.data = None
    
    def load_data(self):
        '''Load data from url'''
        print("Loading data ...")
        if self.data == None:
            url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
            self.data = pd.read_csv(url)

    def data_analysis(self):
        self.data['custcat'].value_counts() #distribution of the data set
        
        correlation_matrix = self.data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.show()
        
        correlation_values = abs(self.data.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
        print(correlation_values)      
    
    def run(self):
       self.load_data() 
       self.data_analysis()

if __name__ == '__main__':
    model = KnnModel()
    model.run()