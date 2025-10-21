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
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_norm = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__model_KNN = None
    
    def load_data(self) -> None:
        '''Load data from url'''
        print("Loading data ...")
        if self.__data == None:
            url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
            self.__data = pd.read_csv(url)

    def data_analysis(self) -> None:
        '''1. print distribution of data set
           2. print correlation matrix
           3. correlation values
        '''
        print(self.__data['custcat'].value_counts()) #distribution of the data set
        
        correlation_matrix = self.__data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.show()
        
        correlation_values = abs(self.__data.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
        print(correlation_values) 
             
    
    def preprocessing(self) -> None:
        '''Standartising the values'''
        #keep the features with corrValue > 0.15
        correlation_values = abs(self.__data.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
        features_to_keep = correlation_values[correlation_values > 0.15].index.tolist()
        self.__X = self.__data[features_to_keep] 
        self.__y = self.__data['custcat']
        
        self.__X_norm = StandardScaler().fit_transform(self.__X)
    
    def split_data(self) -> None:
        '''Splits data into train and test set'''
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X_norm, self.__y, test_size=0.2, random_state=4)
        
    def build_train_KNN(self) -> None:
        '''Build and train the KNN model with k = 3 hyperparam'''
        #hyperparam k
        k = 9 # best accuracy with k = 94  is 0.445
        #Train Model and Predict  
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        self.__model_KNN = knn_classifier.fit(self.__X_train, self.__y_train)
        
    def evaluatuion(self) -> None:
        '''evaluation with accuracy_score'''
        yhat = self.__model_KNN.predict(self.__X_test)
        print("Test set Accuracy: ", accuracy_score(self.__y_test, yhat))
       
    def k_tuning(self) -> None:
        '''Tuning the hyperparam k, to optimize the best k value'''
        try:
            
            Ks = 100
            acc = np.zeros((Ks))
            std_acc = np.zeros((Ks))
            for n in range(1,Ks+1): 
                KNN_tuning_model = KNeighborsClassifier(n_neighbors = n).fit(self.__X_train, self.__y_train)
                yhat = KNN_tuning_model.predict(self.__X_test) #
                acc[n-1] = accuracy_score(self.__y_test, yhat) #
                std_acc[n-1] = np.std(yhat==self.__y_test)/np.sqrt(yhat.shape[0]) #
        
            self.plot_k_tuning(Ks, acc, std_acc)
        except Exception:
            print(f"Error in k_tuning !!!")
            
    def plot_k_tuning(self,Ks,acc,std_acc) -> None:
        '''Print the calculates from k_tuning() method which gives the best k value'''
        plt.plot(range(1,Ks+1),acc,'g')
        plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
        plt.legend(('Accuracy value', 'Standard Deviation'))
        plt.ylabel('Model Accuracy')
        plt.xlabel('Number of Neighbors (K)')
        plt.tight_layout()
        plt.show()
        print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 
        
    def run(self) -> None:
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