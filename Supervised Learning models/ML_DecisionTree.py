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
        self.X_trainset = None
        self.X_testset = None
        self.y_trainset = None
        self.y_testset = None
        self.drugTree = None
        
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
        print(self.data.drop('Drug',axis=1).corr()['Drug_num']) #!!!!!
        
        
    def distribution(self):
        '''distribution of the dataset by plotting the count of the records with each drug recommendation'''
        category_counts = self.data['Drug'].value_counts()

        #plot the count plot
        plt.bar(category_counts.index, category_counts.values, color='blue')
        plt.xlabel('Drug')
        plt.ylabel('Count')
        plt.title('Category Distribution')
        plt.xticks(rotation=45)  #Rotate labels for better readability if needed
        plt.show()

    def split_data(self):
        y = self.data['Drug']
        X = self.data.drop(['Drug','Drug_num'], axis=1)
        
        self.X_trainset, self.X_testset, self.y_trainset, self.y_testset = train_test_split(X, y, test_size=0.3, random_state=32)
    
    def train(self):
        self.drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
        self.drugTree.fit(self.X_trainset, self.y_trainset)
        
    def run(self):
        self.download_data()
        self.data_analysis()
        self.preprocessing()
        self.distribution()
        self.split_data()
        self.train()
        
if __name__ == '__main__':
    model = DecisionTreeModel()
    model.run()        