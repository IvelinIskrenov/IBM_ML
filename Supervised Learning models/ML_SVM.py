import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split

#from __future__ import print_function


class CreditCardFraudDetection():
    '''Build Decision Tree model and SVM model, then we compare it to see which model works better'''
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        
    def load_data(self):
        '''Load the data from the url'''
        if self.data == None:
            url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
            self.data=pd.read_csv(url)
    
    def data_analysis(self):
        # get the set of distinct classes
        labels = self.data['Class'].unique()

        # get the count of each class
        sizes = self.data['Class'].value_counts().values
        
        # plot the class value counts
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.3f%%')
        ax.set_title('Target Variable Value Counts')
        plt.show()
        
        correlation_values = self.data.corr()['Class'].drop('Class')
        correlation_values.plot(kind='barh', figsize=(10, 6))
        plt.show()

    def preprocessing(self):
        #standardize features by removing the mean and scaling to unit variance
        self.data.iloc[:, 1:30] = StandardScaler().fit_transform(self.data.iloc[:, 1:30])
        data_matrix = self.data.values
        
        self.X = data_matrix[:, 1:30]

        self.y = data_matrix[:, 30]

        self.X = normalize(self.X, norm="l1")
        
    def run(self):
        self.load_data()
        self.data_analysis()
        self.preprocessing()
    
if __name__ == '__main__':
    model = CreditCardFraudDetection()
    model.run()
