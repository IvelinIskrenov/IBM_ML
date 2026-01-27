import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier

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
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            
            # fit only on training data, then transform both sets
            self.__X_train_cat = encoder.fit_transform(self.__X_train[cat_cols])
            self.__X_test_cat = encoder.transform(self.__X_test[cat_cols]) 

            # combine processed numerical and categorical features back into arrays
            self.__X_train = np.hstack([self.__X_train_num, self.__X_train_cat])
            self.__X_test = np.hstack([self.__X_test_num, self.__X_test_cat])
            
            # store feature names for visualization purposes
            self.__feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")

    def ova(self) -> None:
        '''Train and evaluate using the One-vs-Rest (One-vs-All) strategy'''
        model = LogisticRegression(multi_class='ovr', max_iter=1000)
        model.fit(self.__X_train, self.__y_train)
        
        y_pred = model.predict(self.__X_test)
        print(f"One-vs-All Accuracy: {accuracy_score(self.__y_test, y_pred):.4f}")
        
        # Calculate feature importance based on coefficients
        importance = np.mean(np.abs(model.coef_), axis=0)
        self._plot_importance(importance, "One-vs-All")

    def ovo(self) -> None:
        '''Train and evaluate using the One-vs-One strategy'''
        model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
        model_ovo.fit(self.__X_train, self.__y_train)
        
        y_pred = model_ovo.predict(self.__X_test)
        print(f"One-vs-One Accuracy: {accuracy_score(self.__y_test, y_pred):.4f}")
        
        # Aggregate coefficients from the multiple binary estimators used in OvO
        coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])
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

    def run(self):
        '''Execute the full ML pipeline'''
        self.load_data()
        self.data_analysis()
        self.prepare_and_split() 
        self.preprocessing()     
        self.ova()
        self.ovo()

if __name__ == '__main__':
    model = MultiClass_Classification()
    model.run()