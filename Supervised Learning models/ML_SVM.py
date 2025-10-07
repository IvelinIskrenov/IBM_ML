import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

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
        
    def load_data(self):
        '''Load the data from the url'''
        if self.data == None:
            url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
            self.data=pd.read_csv(url)
    
    def data_analysis(self):
        '''finds the best feature that correlate with the target and visul. the class sample counts'''
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
        
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
    def buildTrainSVM(self):
        self.model_SVM = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
        self.model_SVM.fit(self.X_train, self.y_train)
        
    def buildTrainDecisionTree(self):
        w_train = compute_sample_weight('balanced', self.y_train)
        self.model_DecisionTree = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=35)
        self.model_DecisionTree.fit(self.X_train, self.y_train, sample_weight=w_train)
    
    #roc_auc_score - how good models distinguish positive from negative ones
    def evaluationSVM(self):
        '''evaluation with roc_auc_score on a SVM model'''
        y_pred_svm = self.model_SVM.decision_function(self.X_test)
        roc_auc_svm = roc_auc_score(self.y_test, y_pred_svm)
        print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm)) 
        
    def evaluationDecisionTree(self):
        '''evaluation with roc_auc_score on a DecisionTree model'''
        y_pred_DecisionTree = self.model_DecisionTree.predict_proba(self.X_test)[:,1]
        roc_auc_DecisionTree = roc_auc_score(self.y_test, y_pred_DecisionTree)
        print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_DecisionTree))  
        
    def run(self):
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
