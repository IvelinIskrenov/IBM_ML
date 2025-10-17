import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import time

import warnings
warnings.filterwarnings('ignore')

#from stackoverflow
def timed(func):
    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = func(self, *args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__} took {t1-t0:.3f}s")
        return result
    return wrapper

class CreditCardFraudDetection():
    '''Build Decision Tree model and SVM model, then we compare it to see which model works better'''
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_SVM = None
        self.model_DecisionTree = None
        
    @timed    
    def load_data(self) -> None:
        '''Load the data from the url'''
        if self.data == None:
            url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
            self.data=pd.read_csv(url)
        print("Data loaded")
        
    @timed 
    def data_analysis(self) -> None:
        '''
        finds the best feature that correlate with the target and visul. the class sample counts
        finds top 6 features
        '''
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
        
        
        print("Top 6 correlation features: ")
        correlation_values = abs(self.data.corr()['Class']).drop('Class')
        correlation_values = correlation_values.sort_values(ascending=False)[:6]
        print(correlation_values)
        
    @timed 
    def preprocessing(self) -> None:
        #standardize features by removing the mean and scaling to unit variance
        self.data.iloc[:, 1:30] = StandardScaler().fit_transform(self.data.iloc[:, 1:30])
        data_matrix = self.data.values
        
        #self.X = data_matrix[:, 1:30]
        #set the features from the data_analysis()
        self.X = data_matrix[:,[3,10,12,14,16,17]]

        self.y = data_matrix[:, 30]

        self.X = normalize(self.X, norm="l1")
        
        print("Preprocess finished !!!")
     
    @timed    
    def split_data(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print("Split data done !!!")
    
    @timed     
    def buildTrainSVM(self) -> None:
        '''Builds and trains a balanced Linear SVM model'''
        self.model_SVM = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
        self.model_SVM.fit(self.X_train, self.y_train)
    
    @timed     
    def buildTrainDecisionTree(self) -> None:
        '''Builds and trains a Decision Tree (entropy, max_depth=4) with balanced sample weights'''
        w_train = compute_sample_weight('balanced', self.y_train)
        self.model_DecisionTree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=35)
        self.model_DecisionTree.fit(self.X_train, self.y_train, sample_weight=w_train)
    
    #roc_auc_score - how good models distinguish positive from negative ones
    def evaluationSVM(self) -> None:
        '''evaluation with roc_auc_score on a SVM model'''
        y_pred_svm = self.model_SVM.decision_function(self.X_test)
        roc_auc_svm = roc_auc_score(self.y_test, y_pred_svm)
        print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm)) 
        
    def evaluationDecisionTree(self) -> None:
        '''evaluation with roc_auc_score on a DecisionTree model'''
        y_pred_DecisionTree = self.model_DecisionTree.predict_proba(self.X_test)[:,1]
        roc_auc_DecisionTree = roc_auc_score(self.y_test, y_pred_DecisionTree)
        print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_DecisionTree))  
        
    def run(self) -> None:
        self.load_data()
        self.data_analysis()
        self.preprocessing()
        self.split_data()
        self.buildTrainDecisionTree()
        self.evaluationDecisionTree()
        self.buildTrainSVM()
        self.evaluationSVM() 
   
if __name__ == '__main__':
    model = CreditCardFraudDetection()
    model.run()
    
    
    #before feature selecting
    #Decision Tree ROC-AUC score : 0.912
    #SVM ROC-AUC score: 0.984
    
    #after feature selecting
    #Decision Tree ROC-AUC score : 0.927 max depth 10 
    #SVM ROC-AUC score: 0.929
    
    #testing other hyperparams
    #Decision Tree ROC-AUC score : 0.927 max depth 10 -> 0.946 max depth 4 -> (gini->entropy) 0.952
    
    #With a larger set of features, SVM performed relatively better in comparison to the Decision Trees
    #Decision Trees benefited from feature selection and performed better
    #SVMs may require higher feature dimensionality to create an efficient decision hyperplane
