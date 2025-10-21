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
    '''ML Alg. - multi_class_classification, which compare One_vs_All and One_vs_One strategy using LogisticRegression - prediction of obesity levels'''
    def __init__(self):
        self.__data = None
        self.__scaled_data = None
        self.__prepped_data = None
        self.__X = None
        self.__y = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train =  None
        self.__y_test = None
        self.__model_ova = None
    
    def  load_data(self) -> None:
        '''Load the data from the url'''
        if self.__data == None:
            file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
            self.__data = pd.read_csv(file_path)
            
    def data_analysis(self) -> None:
        '''Checks target distribution and visualizating the data'''
        #Distribution of target variable (check if the data is disbalanced)
        sns.countplot(y='NObeyesdad', data=self.__data)
        plt.title('Distribution of Obesity Levels')
        plt.show()
        
        print(self.__data.isnull().sum())

        #Dataset summary
        print(self.__data.info())
        print(self.__data.describe())
    
    def preprocessing(self) -> None:
        '''Standartizing the contin. num. features && scales by standard deviation'''
        try:
            #Standardizing continuous numerical features
            continuous_columns = self.__data.select_dtypes(include=['float64']).columns.tolist()
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(self.__data[continuous_columns])

            #Converting to a DataFrame
            scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

            #Combining with the original dataset
            self.__scaled_data = pd.concat([self.__data.drop(columns=continuous_columns), scaled_df], axis=1)
        except Exception:
            print(f"Error in preprocessing !!!")
    
    #turnig categorical into numerical data
    def one_hot_encoding(self) -> None:
        '''Encodes categorical features && encodes the target var.
            Prepare the data and the target'''
        try:
            #Identifying categorical columns
            categorical_columns = self.__scaled_data.select_dtypes(include=['object']).columns.tolist()
            categorical_columns.remove('NObeyesdad')  #Exclude target column
        
            #Applying one-hot encoding
            encoder = OneHotEncoder(
                sparse_output=False, 
                drop='first' #avoid multicollinearity
                )
            encoded_features = encoder.fit_transform(self.__scaled_data[categorical_columns])

            #Converting to a DF
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

            #Combining with the original dataset
            self.__prepped_data = pd.concat([self.__scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

            #Encoding the target var
            self.__prepped_data['NObeyesdad'] = self.__prepped_data['NObeyesdad'].astype('category').cat.codes
            self.__prepped_data.head()

            self.__X = self.__prepped_data.drop('NObeyesdad', axis=1)
            self.__y = self.__prepped_data['NObeyesdad']
        except Exception:
            print(f"Error in encoding !!!")
        
    def split_data(self) -> None:
        '''Splits data into 80/20 % (train/test)'''
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=0.2, random_state=42, stratify=self.__y)
        
    #split data include
    def ova(self) -> None:
        '''
            Trains LogsticR using OvR (One vs All) strategy
            prints acc and visualizes feature importance based on coeff
        '''
        try:
            self.__model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
       
            self.__model_ova.fit(self.__X_train, self.__y_train)
    
            y_pred = self.__model_ova.predict(self.__X_test)
            print("One-vs-All Accuracy (Test Size 0.2):", accuracy_score(self.__y_test, y_pred))
            
            # Feature importance
            feature_importance = np.mean(np.abs(self.__model_ova.coef_), axis=0)
            plt.barh(self.__X.columns, feature_importance)
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.show()
        except Exception:
            print("Error in (One vs All) strategy")    

    def ovo(self) -> None:
        '''
            Trains LogsticR using OvO (One vs One) strategy
            prints acc and visualizes feature importance based on coeff
        '''
        try:
            
            model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
    
            model_ovo.fit(self.__X_train, self.__y_train)
    
            y_pred = model_ovo.predict(self.__X_test)
            print("One-vs-One Accuracy (Test Size 0.2):", accuracy_score(self.__y_test, y_pred))
            
            # For One vs One model
            # Collect all coefficients from each underlying binary classifier
            coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])

            # Now take the mean across all those classifiers
            feature_importance = np.mean(np.abs(coefs), axis=0)
            plt.barh(self.__X.columns, feature_importance)
            plt.title("Feature Importance (One-vs-One)")
            plt.xlabel("Importance")
            plt.show()
        except Exception:
            print("Error in (One vs One) strategy")        
    
    def run(self):
        '''Executes the complete ML pipeline'''
        self.load_data()
        self.data_analysis()
        self.preprocessing()
        self.one_hot_encoding()
        self.split_data()
        self.ova()
        self.ovo() # from the comparison result - One vs One is more efficiant
        
if __name__ == '__main__':
    model = MultiClass_Classification()
    model.run()