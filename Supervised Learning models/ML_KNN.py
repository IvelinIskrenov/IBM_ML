import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KnnModel():
    '''Build and training classifier model KNN, which predict the service category for unknown cases'''
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_norm = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_KNN = None
    
    def load_data(self):
        '''Load data from url'''
        print("Loading data ...")
        if self.data == None:
            url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
            self.data = pd.read_csv(url)

    def data_analysis(self):
        '''1. print distribution of data set
           2. print correlation matrix
           3. correlation values
        '''
        print(self.data['custcat'].value_counts()) #distribution of the data set
        
        correlation_matrix = self.data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.show()
        
        correlation_values = abs(self.data.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
        print(correlation_values) 
             
    
    def preprocessing(self):
        '''Standartising the values'''
        #keep the features with corrValue > 0.15
        correlation_values = abs(self.data.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
        features_to_keep = correlation_values[correlation_values > 0.15].index.tolist()
        self.X = self.data[features_to_keep] 
        self.y = self.data['custcat']
        
        self.X_norm = StandardScaler().fit_transform(self.X)
    
    def split_data(self):
        '''Splits data into train and test set'''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm, self.y, test_size=0.2, random_state=4)
        
    def build_train_KNN(self):
        '''Build and train the KNN model with k = 3 hyperparam'''
        #hyperparam k
        k = 9 # best accuracy with k = 94  is 0.445
        #Train Model and Predict  
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        self.model_KNN = knn_classifier.fit(self.X_train, self.y_train)
        
    def evaluatuion(self):
        '''evaluation with accuracy_score'''
        yhat = self.model_KNN.predict(self.X_test)
        print("Test set Accuracy: ", accuracy_score(self.y_test, yhat))
       
    def k_tuning(self):
        '''Tuning the hyperparam k, to optimize the best k value'''
        Ks = 100
        acc = np.zeros((Ks))
        std_acc = np.zeros((Ks))
        for n in range(1,Ks+1): 
            KNN_tuning_model = KNeighborsClassifier(n_neighbors = n).fit(self.X_train, self.y_train)
            yhat = KNN_tuning_model.predict(self.X_test) #
            acc[n-1] = accuracy_score(self.y_test, yhat) #
            std_acc[n-1] = np.std(yhat==self.y_test)/np.sqrt(yhat.shape[0]) #
        
        self.plot_k_tuning(Ks, acc, std_acc)

    def plot_k_tuning(self,Ks,acc,std_acc):
        '''Print the calculates from k_tuning() method which gives the best k value'''
        plt.plot(range(1,Ks+1),acc,'g')
        plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
        plt.legend(('Accuracy value', 'Standard Deviation'))
        plt.ylabel('Model Accuracy')
        plt.xlabel('Number of Neighbors (K)')
        plt.tight_layout()
        plt.show()
        print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 
        
    def run(self):
       self.load_data() 
       self.data_analysis()
       self.preprocessing()
       self.split_data()
       self.build_train_KNN()
       self.evaluatuion()
       self.k_tuning()
        
#improve the model!!!        

if __name__ == '__main__':
    model = KnnModel()
    model.run()