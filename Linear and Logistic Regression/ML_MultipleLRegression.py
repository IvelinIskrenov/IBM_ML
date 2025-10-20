import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm


class FuelCO2Model:
    '''
    Multiple Linear Regression model to predict CO2 emissions from vehicle data.
    '''
    def __init__(self, url: str = None):
        self.url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_std = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__std_scaler = None
        self.__mRegression = None

    def load_data(self) -> None:
        '''loads data and performs init cleaning'''
        try:
            self.__data = pd.read_csv(self.url)
            print("Sample 5 rows:")
            print(self.__data.sample(5))
            print(self.__data.describe())

            # drop hte first col
            self.__data = self.__data.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',], axis=1)

            self.__data = self.__data.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',], axis=1)

            print(self.__data.head(9))
        except Exception:
            print(f"Error while loading data !!!")

    def visualize_pairplot(self) -> None:
        axes = pd.plotting.scatter_matrix(self.__data, alpha=0.2, figsize=(8, 8))
        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(90)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')

        plt.tight_layout()
        plt.gcf().subplots_adjust(wspace=0, hspace=0)
        plt.show()
        # from this example, we see that the fuel con. and CO2 are no linear

    def extract_cols(self) -> None:
        '''Extracts the features cols'''
        #self.X = self.df[['ENGINESIZE', 'VEHICLEWEIGHT']].to_numpy()
        #self.y = self.df[['CO2EMISSIONS']].to_numpy()
        self.__X = self.__data.iloc[:,[0,1]]
        self.__y = self.__data.iloc[:,[2]]

    # - Preprocess selected features - #
    def preprocess(self) -> None:
        '''Standardizes'''
        self.__std_scaler = StandardScaler()
        self.__X_std = self.__std_scaler.fit_transform(self.__X)

        print('\nStandardized features description:')
        print(pd.DataFrame(self.__X_std, columns=['ENGINESIZE', 'VEHICLEWEIGHT']).describe().round(2))

    # - Create train and test dataset - #
    def split(self, test_size: float = 0.2, random_state: int = 42) -> None:
        '''splits the data'''
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            self.__X_std, self.__y, test_size=test_size, random_state=random_state
        )

    # - Build ML Regression - #
    def train(self) -> None: 
        '''Trains and build the Multiple linear Regression'''
        try:
            self.__mRegression = lm.LinearRegression()
            self.__mRegression.fit(self.__X_train, self.__y_train)

            coef_ = self.__mRegression.coef_
            intercept_ = self.__mRegression.intercept_

            print('\nCoefficients (standardized space): ', coef_)
            print('Intercept (standardized space): ', intercept_)

            #standard deviation parameters
            means_ = self.__std_scaler.mean_
            std_devs_ = np.sqrt(self.__std_scaler.var_)

            #least squares param can be calculated relative to original in unstandardized feature space
            coef_original = coef_ / std_devs_
            intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

            print('\nCoefficients (original feature space): ', coef_original)
            print('Intercept (original feature space): ', intercept_original)
        except Exception:
            print(f"Error in training model !!!")
    #from stackoverflow(IBM VS)
    def visualize_3d(self) -> None:
        '''Visiz. the regression plane in 3D'''
        coef_ = self.__mRegression.coef_
        intercept_ = self.__mRegression.intercept_

        # Ensure X_test is numpy array
        X_test_arr = np.asarray(self.__X_test)
        # X1, X2
        X1 = X_test_arr[:, 0] if X_test_arr.ndim > 1 else X_test_arr
        X2 = X_test_arr[:, 1] if X_test_arr.ndim > 1 else np.zeros_like(X1)

        # Create a mesh grid for plotting the regression plane
        x1_surf, x2_surf = np.meshgrid(
            np.linspace(X1.min(), X1.max(), 100),
            np.linspace(X2.min(), X2.max(), 100),
        )

        # y_surf in standardized feature space
        y_surf = intercept_ + coef_[0, 0] * x1_surf + coef_[0, 1] * x2_surf

        # Make sure y_test and y_pred are 1D numpy arrays
        y_test_arr = np.asarray(self.__y_test).ravel()
        y_pred = self.__mRegression.predict(self.__X_test)
        y_pred_arr = np.asarray(y_pred).ravel()

        # Now boolean masks (1D numpy)
        above_plane = y_test_arr >= y_pred_arr
        below_plane = ~above_plane

        # Plotting
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the data points above and below the plane in different colors
        ax.scatter(X1[above_plane], X2[above_plane], y_test_arr[above_plane],
                   label="Above Plane", s=70, alpha=.7, ec='k')
        ax.scatter(X1[below_plane], X2[below_plane], y_test_arr[below_plane],
                   label="Below Plane", s=50, alpha=.3, ec='k')

        # Plot the regression plane
        ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21)

        # Set view and labels
        ax.view_init(elev=10)
        ax.legend(fontsize='x-large', loc='upper center')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect(None, zoom=0.75)
        ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
        ax.set_ylabel('VEHICLEWEIGHT', fontsize='xx-large')
        ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
        ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
        plt.tight_layout()
        plt.show()


    def simple_2d_plots_vis(self) -> None:
        '''2D plots for individual feature relationships'''
        coef_ = self.__mRegression.coef_
        intercept_ = self.__mRegression.intercept_

        plt.scatter(self.__X_train[:, 0], self.__y_train, color='blue')
        plt.plot(self.__X_train[:, 0], coef_[0, 0] * self.__X_train[:, 0] + intercept_[0], '-r')
        plt.xlabel("Engine size")
        plt.ylabel("Emission")
        plt.show()

        plt.scatter(self.__X_train[:, 1], self.__y_train, color='blue')
        plt.plot(self.__X_train[:, 1], coef_[0, 1] * self.__X_train[:, 1] + intercept_[0], '-r')
        plt.xlabel("VEHICLEWEIGHT")
        plt.ylabel("Emission")
        plt.show()

    def run(self) -> None:
        self.load_data()
        #self.visualize_pairplot()
        self.extract_cols()
        self.preprocess()
        self.split()
        self.train()
        self.visualize_3d()
        self.simple_2d_plots_vis()


if __name__ == '__main__':
    model = FuelCO2Model()
    model.run()
