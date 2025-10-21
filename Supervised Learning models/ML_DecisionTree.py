import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


class DecisionTreeModel():
    '''DecisionTree model that determine which drug is most suitable'''
    def __init__(self):
        self.__data = None
        self.__X_trainset = None
        self.__X_testset = None
        self.__y_trainset = None
        self.__y_testset = None
        self.__drugTree = None
        
    def download_data(self) -> None:
        "downloading data"
        if self.__data == None :
            path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
        self.__data = pd.read_csv(path)
    
    def data_analysis(self) -> None:
        '''Checks target distribution and visualizating the data'''
        #Dataset summary
        print(self.__data.info()) #show us the type of data
        print(self.__data.describe())
        # 4 features should be convert from categorical to numecric data
        
    
    def preprocessing(self) -> None:
        '''convert the object cols into numeric format'''
        try:
            
            label_encoder = LabelEncoder()
            self.__data['Sex'] = label_encoder.fit_transform(self.__data['Sex']) 
            self.__data['BP'] = label_encoder.fit_transform(self.__data['BP'])
            self.__data['Cholesterol'] = label_encoder.fit_transform(self.__data['Cholesterol'])
        
            #self.data.isnull().sum()  #cheching to be sure
        
            #evaluate the correlation of the target variable with the input features
            custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
            self.__data['Drug_num'] = self.__data['Drug'].map(custom_map)
        
            #corr() function to find the correlation of the input variables with the target variable
            print(self.__data.drop('Drug',axis=1).corr()['Drug_num']) #!!!!!
        except Exception:
            print(f"Error in preprocessing")
        
    def distribution(self) -> None:
        '''distribution of the dataset by plotting the count of the records with each drug recommendation'''
        category_counts = self.__data['Drug'].value_counts()

        #plot the count plot
        plt.bar(category_counts.index, category_counts.values, color='blue')
        plt.xlabel('Drug')
        plt.ylabel('Count')
        plt.title('Category Distribution')
        plt.xticks(rotation=45)  #Rotate labels for better readability if needed
        plt.show()

    def split_data(self) -> None:
        y = self.__data['Drug']
        X = self.__data.drop(['Drug','Drug_num'], axis=1)
        
        self.__X_trainset, self.__X_testset, self.__y_trainset, self.__y_testset = train_test_split(X, y, test_size=0.3, random_state=32)
    
    def train(self) -> None:
        '''train the model Decision Tree Classifier with entropy and max-depth = 4'''
        self.__drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4) # you can use the params for better accuracy
        self.__drugTree.fit(self.__X_trainset, self.__y_trainset)
        
    def evaluation(self) -> None:
        '''evaluation with accurancy_score'''
        tree_predictions = self.__drugTree.predict(self.__X_testset)
        print("Decision Trees's Accuracy: ", metrics.accuracy_score(self.__y_testset, tree_predictions))
        
    def visualize(self) -> None:
        '''visualize tree'''
        plot_tree(self.__drugTree)
        plt.show()
        
    def run(self) -> None:
        self.download_data()
        self.data_analysis()
        self.preprocessing()
        self.distribution()
        self.split_data()
        self.train()
        self.evaluation()
        self.visualize()
        
if __name__ == '__main__':
    model = DecisionTreeModel()
    model.run()        