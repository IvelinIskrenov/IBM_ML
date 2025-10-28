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
        self.__grid_search_RF = None
        self.__grid_search_LR = None
        self.__best_model_LR = None
        
    def load_data(self):
        '''Loading data'''
        if self.__data == None:
            url ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
            self.__data = pd.read_csv(url)
    
    def data_analysis(self) -> None:
        ''' Printing head(5), info, and rename the cols in data'''
        print(f"Data head(5) :")
        print(self.__data.head(5))
        self.__data = self.__data.dropna()
        
        print(f"Data info :")
        print(self.__data.info())
        
        self.__data = self.__data.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })
    
    def location_selection(self) -> None:
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
    
    def mapping(self) -> None:
        '''Map the dates to seasons and drop the Date col'''
        self.__data['Date'] = pd.to_datetime(self.__data['Date'])

        # Apply the function to the 'Date' column
        self.__data['Season'] = self.__data['Date'].apply(self.date_to_season)

        self.__data = self.__data.drop(columns='Date')

    def split_data(self) -> None:
        '''Splits the data into train/test data'''
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=0.2, stratify=self.__y ,random_state=42)
    
    def preprocessing(self) -> None:
        '''
            Set X and y data, see value counts, splits data into train/test, identify numerical/categorical features,
            scale them and make pipeline, make pipeline (processor & classifier)
        '''
        try:
            self.__X = self.__data.drop(columns='RainToday', axis=1)
            self.__y = self.__data['RainToday']
            
            print("How balanced are the classes!")
            print(self.__y.value_counts())
            
            self.split_data()
            
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
            
        except Exception as e:
            print(f"Error in preprocessing: {type(e).__name__} - {e}")
   
    def train_RF(self) -> None:
        '''Set param grid, cv = StratifiedKFold, use GridSearchCV, print - best cross-val score, best param, test score'''
        #use in a cross validation grid search model optimizer
        try:
            self.__pipeline = Pipeline(steps=[
                ('preprocessor', self.__preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
            
            self.__param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }   
            
            #Performing grid search cross-validation and fit the best model
            cv = StratifiedKFold(n_splits=5, shuffle=True)
            
            self.__grid_search_RF = GridSearchCV(
                estimator=self.__pipeline,
                param_grid=self.__param_grid,
                cv=cv,
                scoring='accuracy',
                verbose=2
            )  
            self.__grid_search_RF.fit(self.__X_train, self.__y_train)
            
            print("\nBest parameters found: ", self.__grid_search_RF.best_params_)
            print("Best cross-validation score: {:.2f}".format(self.__grid_search_RF.best_score_))
            
            test_score = self.__grid_search_RF.score(self.__X_test, self.__y_test)  
            print("Test set score: {:.2f}".format(test_score))
            
        except Exception as e:
            print(f"Error in train(): {type(e).__name__} - {e}")  
                       
    def prediction(self, model) -> None:
        '''Print classification repport !'''
        if model == "LR":
            y_pred = self.__grid_search_LR.predict(self.__X_test)
        elif model == "RF":
            y_pred = self.__grid_search_RF.predict(self.__X_test)
            
        #y_pred = self.__grid_search_LR.predict(self.__X_test)
        print("\nClassification Report:")
        print(classification_report(self.__y_test, y_pred))
        
        conf_matrix = confusion_matrix(self.__y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
    
    def feature_importance(self) -> None:
        
        print(f"Feature importance :")
        feature_importances = self.__grid_search_RF.best_estimator_['classifier'].feature_importances_ 
        # Combine numeric and categorical feature names
        feature_names = self.__numerical_features + list(self.__grid_search_RF.best_estimator_['preprocessor']
                                                .named_transformers_['cat']
                                                .named_steps['onehot']
                                                .get_feature_names_out(self.__categorical_features))

        feature_importances = self.__grid_search_RF.best_estimator_['classifier'].feature_importances_

        importance_df = pd.DataFrame({'Feature': feature_names,
                                      'Importance': feature_importances
                                     }).sort_values(by='Importance', ascending=False)

        N = 20  # Change this number to display more or fewer features
        top_features = importance_df.head(N)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
        plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
        plt.xlabel('Importance Score')
        plt.show()

            
    def pipeline(self) -> None:
        self.load_data()
        self.data_analysis()  
        self.location_selection()
        self.mapping()  
        self.preprocessing() 
    
    def pipeline_RF(self) -> None:
        if self.__data is None:
            self.pipeline()
        self.train_RF() 
        self.prediction("RF")  
        self.feature_importance()
     
    def pipeline_LR(self) -> None:
        if self.__data is None:
            self.pipeline()
        try:
            #Replace classifier step in the pipeline
            #self.__pipeline.set_params(classifier=LogisticRegression(random_state=42, max_iter=1000))

            self.__pipeline = Pipeline(steps=[
                ('preprocessor', self.__preprocessor),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])

            #Ensure the GridSearchCV uses the updated pipeline as estimator
            if self.__grid_search_LR is None:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                self.__grid_search_LR = GridSearchCV(
                    estimator=self.__pipeline,
                    param_grid={},  # will set below
                    cv=cv,
                    scoring='accuracy',
                    verbose=2 
                )
            else:
                #update estimator in the existing GridSearchCV
                self.__grid_search_LR.estimator = self.__pipeline

            #New parameter grid for LR
            self.__param_grid = {
                'classifier__solver': ['liblinear'],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__class_weight': [None, 'balanced']
            }
            self.__grid_search_LR.param_grid = self.__param_grid

            #Ensure cv and other attrs are set
            self.__grid_search_LR.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            self.__grid_search_LR.scoring = 'accuracy'
            self.__grid_search_LR.verbose = 2
            #self.__grid_search.n_jobs = -1

            #Fit GridSearchCV
            self.__grid_search_LR.fit(self.__X_train, self.__y_train)

            print("\nBest LR parameters found: ", self.__grid_search_LR.best_params_)
            print("Best cross-validation accuracy: {:.4f}".format(self.__grid_search_LR.best_score_))

            test_score = self.__grid_search_LR.score(self.__X_test, self.__y_test)  
            print("Test set score: {:.2f}".format(test_score))
            
            ######
            # split 
            y_pred = self.__grid_search_LR.predict(self.__X_test)

            print("\nClassification Report (Logistic Regression):")
            print(classification_report(self.__y_test, y_pred))
            #Generate the confusion matrix 
            conf_matrix = confusion_matrix(self.__y_test, y_pred)
            plt.figure()
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

            plt.title('Titanic Classification Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            plt.tight_layout()
            plt.show()
            #Save best model for later use
            self.__best_model = self.__grid_search_LR.best_estimator_
            print(classification_report(self.__y_test, y_pred))

        except Exception as e:
            print(f"Error in pipeline_LR !!! {type(e).__name__}: {e}")
           
if __name__ == "__main__":
    print("Starting model ...")
    model = RaindFallPredictor()
    model.pipeline_LR()
    model.pipeline_RF()