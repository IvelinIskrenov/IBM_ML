import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_selection import SelectKBest, f_classif

class MultiClass_Classification:
    '''
    Machine Learning class for Multi-class classification.
    Compares One-vs-Rest (OvR) and One-vs-One (OvO) strategies 
    using Logistic Regression to predict obesity levels.
    '''
    def __init__(self):
        self.__data = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__ovo = None
        self.__ova = None
        self.__feature_names = None

    def load_data(self) -> None:
        '''Load the dataset from a remote URL'''
        if self.__data is None:
            file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
            self.__data = pd.read_csv(file_path)

    def data_analysis(self) -> None:
        '''Perform exploratory data analysis: check distribution and null values'''
        # Visualize the distribution of the target variable to check for class imbalance
        sns.countplot(y='NObeyesdad', data=self.__data)
        plt.title('Distribution of Obesity Levels')
        plt.show()

        print(self.__data.info())
        print(self.__data.describe())

    def prepare_and_split(self) -> None:
        '''
        - Encode target variable.
        - Split data into features (X) and target (y).
        - Perform train/test split BEFORE preprocessing to prevent data leakage.
        '''
        # Encode target labels numerically 
        self.__data['NObeyesdad'] = self.__data['NObeyesdad'].astype('category').cat.codes
        
        X = self.__data.drop('NObeyesdad', axis=1)
        y = self.__data['NObeyesdad']
        
        # Split data (80% train, 20% test) with stratification to maintain class balance
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def preprocessing(self) -> None:
        '''
        Process Training and Testing sets separately to avoid Data Leakage.
        Scales numerical features and encodes categorical features.
        '''
        try:
            
            # numerical Feature Scaling
            num_cols = self.__X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
            scaler = StandardScaler()
            
            # fit only on training data, then transform both sets
            self.__X_train_num = scaler.fit_transform(self.__X_train[num_cols])
            self.__X_test_num = scaler.transform(self.__X_test[num_cols]) 

            # categorical Feature Encoding (One-Hot Encoding)
            cat_cols = self.__X_train.select_dtypes(include=['object']).columns.tolist()
            print(cat_cols)
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            
            # fit only on training data, then transform both sets
            self.__X_train_cat = encoder.fit_transform(self.__X_train[cat_cols])
            self.__X_test_cat = encoder.transform(self.__X_test[cat_cols]) 

            # combine processed numerical and categorical features back into arrays
            self.__X_train = np.hstack([self.__X_train_num, self.__X_train_cat])
            self.__X_test = np.hstack([self.__X_test_num, self.__X_test_cat])
            
            # store feature names for visualization purposes
            self.__feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
            
            selector = SelectKBest(score_func=f_classif, k=3)
            selector.fit(self.__X_train, self.__y_train) 
    
            features_to_keep = self.__X_train.columns[selector.get_support()].tolist()
            print("Selected features:", features_to_keep)
    
            self.__X_train = self.__X_train[features_to_keep]
            self.__X_test = self.__X_test[features_to_keep]
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")

    def ova(self) -> None:
        '''Train and evaluate using the One-vs-Rest (One-vs-All) strategy'''
        self.__ova = LogisticRegression(multi_class='ovr', max_iter=1000)
        self.__ova.fit(self.__X_train, self.__y_train)
        
        y_pred = self.__ova.predict(self.__X_test)
        print(f"One-vs-All Accuracy: {accuracy_score(self.__y_test, y_pred):.4f}")
        
        # Calculate feature importance based on coefficients
        importance = np.mean(np.abs(self.__ova.coef_), axis=0)
        self._plot_importance(importance, "One-vs-All")

    def ovo(self) -> None:
        '''Train and evaluate using the One-vs-One strategy'''
        self.__ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
        self.__ovo.fit(self.__X_train, self.__y_train)
        
        y_pred = self.__ovo.predict(self.__X_test)
        print(f"One-vs-One Accuracy: {accuracy_score(self.__y_test, y_pred):.4f}")
        
        # Aggregate coefficients from the multiple binary estimators used in OvO
        coefs = np.array([est.coef_[0] for est in self.__ovo.estimators_])
        importance = np.mean(np.abs(coefs), axis=0)
        self._plot_importance(importance, "One-vs-One")

    def _plot_importance(self, importance, title):
        '''Helper method to visualize feature importance'''
        plt.figure(figsize=(10, 6))
        # Showing only the first 20 features for better readability
        plt.barh(self.__feature_names[:20], importance[:20]) 
        plt.title(f"Feature Importance ({title})")
        plt.xlabel("Mean Absolute Coefficient Value")
        plt.show()

    def cross_validation(self, model):
        skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 32)
        scores = cross_val_score(estimator=model, X=self.__X_train, y=self.__y_train, cv = skf)
        
        print(f"Mean SKF CV accuracy for model {model} is: {scores.mean() * 100:.2f}%")
    
    def run(self):
        '''Execute the full ML pipeline'''
        self.load_data()
        self.data_analysis()
        self.prepare_and_split() 
        self.preprocessing()     
        self.ova()
        self.cross_validation(self.__ova)
        self.ovo()
        self.cross_validation(self.__ovo)

if __name__ == '__main__':
    model = MultiClass_Classification()
    model.run()