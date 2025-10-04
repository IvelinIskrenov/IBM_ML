import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


class DecisionTreeModel():
    def __init__(self):
        self.data = None
        
    def download_data(self):
        if self.data == None :
            path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
        self.data = pd.read_csv(path)
    
    def data_analysis(self):
        '''Checks target distribution and visualizating the data'''
        #Dataset summary
        print(self.data.info()) #show us the type of data
        print(self.data.describe())
        # 4 features should be convert from categorical to numecric data
    
    def preprocessing(self):
        label_encoder = LabelEncoder()
        self.data['Sex'] = label_encoder.fit_transform(self.data['Sex']) 
        self.data['BP'] = label_encoder.fit_transform(self.data['BP'])
        self.data['Cholesterol'] = label_encoder.fit_transform(self.data['Cholesterol'])
        
        #self.data.isnull().sum()  #cheching to be sure
        
        #evaluate the correlation of the target variable with the input features
        custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
        self.data['Drug_num'] = self.data['Drug'].map(custom_map)
        
        #corr() function to find the correlation of the input variables with the target variable
        
    def run(self):
        self.download_data()
        self.data_analysis()
        #self.preprocessing()
        
if __name__ == '__main__':
    model = DecisionTreeModel()
    model.run()        