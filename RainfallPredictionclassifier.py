#from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


class RaindFallPredictor():
    '''
        original source of the data is Australian Government's Bureau of Meteorology 
        and the latest data can be gathered from http://www.bom.gov.au/climate/dwo/.
    '''
    def __init__(self):
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None   
        self.__numerical_features = None
        self.__categorical_features = None   
        self.__preprocessor = None
        self.__pipeline = None
        self.__param_grid = None
        
    def load_data(self):
        '''Loading data'''
        if self.__data == None:
            url ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
            self.__data = pd.read_csv(url)
    
    def data_analysis(self):
        print(f"Data head(5) :")
        print(self.__data.head(5))
        self.__data = self.__data.dropna()
        
        print(f"Data info :")
        print(self.__data.info())
        
        self.__data = self.__data.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })
    
    def location_selection(self):
        '''Reducing our attention to a smaller region'''
        print(f"Location selection: ")
        dataEx = self.__data[self.__data.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])] # they are near next to each other
        print(dataEx.info())
        
        self.__data = dataEx
    
    def date_to_season(self, data):
        '''Engineer a Season feature from Date and drop Date afterward, since it is most likely less informative than season'''
        month = data.month
        if month in (12, 1, 2):
            return 'Summer'
        elif month in (3, 4, 5):
            return 'Autumn'
        elif month in (6, 7, 8):
            return 'Winter'
        elif month in (9, 10, 11):
            return 'Spring'
        else:
            return np.nan
    
    def mapping(self):
        '''Map the dates to seasons and drop the Date col'''
        self.__data['Date'] = pd.to_datetime(self.__data['Date'])

        # Apply the function to the 'Date' column
        self.__data['Season'] = self.__data['Date'].apply(self.date_to_season)

        self.__data = self.__data.drop(columns='Date')

    def split_data(self):
        '''Splits the data into train/test data'''
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=0.2, random_state=42)
    
    def preprocessing(self):
        try:
            self.__X = self.__data.drop(columns='RainToday', axis=1)
            self.__y = self.__data['RainToday']
            
            print("How balanced are the classes!")
            print(self.__y.value_counts())
            
            self.__numerical_features = self.__X_train.select_dtypes(include=['number']).columns.tolist()
            self.__categorical_features = self.__X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Scale            
            self.__numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())]) 
            self.__categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            self.__preprocessor = ColumnTransformer(
                transformers=[
                    ('num', self.__numeric_transformer, self.__numerical_features),
                    ('cat', self.__categorical_transformer, self.__categorical_features)
                ]
            )
            
            self.__pipeline = Pipeline(steps=[
                ('preprocessor', self.__preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
            
            #use in a cross validation grid search model optimizer
            self.__param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }   
            
            #Performing grid search cross-validation and fit the best model
            cv = StratifiedKFold(n_splits=5, shuffle=True)
            
            
        except Exception:
            print(f"Error in preprocessing !!! ")
            
            
    def pipeline(self):
        self.load_data()
        self.data_analysis()  
        self.location_selection()
        self.mapping()  
        self.preprocessing()    
        
if __name__ == "__main__":
    print("Starting model ...")
    model = RaindFallPredictor()
    model.pipeline()