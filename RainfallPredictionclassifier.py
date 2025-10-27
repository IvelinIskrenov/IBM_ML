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
        self.__data['Date'] = pd.to_datetime(self.__data['Date'])

        # Apply the function to the 'Date' column
        self.__data['Season'] = self.__data['Date'].apply(self.date_to_season)

        self.__data = self.__data.drop(columns='Date')

    def preprocessing(self):
        try:
            self.__X = self.__data.drop(columns='RainToday', axis=1)
            self.__y = self.__data['RainToday']
            
            print("How balanced are the classes!")
            print(self.__y.value_counts())
            
            
        except Exception:
            print(f"Error in preprocessing !!! ")
            
            
    def pipeline(self):
        self.load_data()
        self.data_analysis()  
        self.location_selection()
        self.mapping()      
        
if __name__ == "__main__":
    print("Starting model ...")
    model = RaindFallPredictor()
    model.pipeline()