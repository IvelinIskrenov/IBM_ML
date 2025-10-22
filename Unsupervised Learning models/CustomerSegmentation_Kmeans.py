import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation():
    '''
        Build and train K-means one group might contain customers who are high-profit and low-risk, 
        or more likely to purchase products, or subscribe to a service. A business task is to retain those customers      
        K-means++ : selects initial cluster centres for k-means clustering in a smart way to speed up convergence
        n_init : n times - k-means Alg. will be run with different centroid seeds
    '''
    def __init__(self):
        self.__data = None
        self.__std_data = None
        self.__n_clusters = None # this is the K in k-means
        self.__n_init = None
        self.__X = None
        self.__y = None
        self.__k_means = None
        self.__labels = None
        
        
    def load_data(self):
        if self.__data == None:
            self.__data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
        
    def preprocessing(self):
        '''Drop the address col to remains only numerical data && Standartizing the data'''
        self.__data = self.__data.drop('Address', axis=1)
        self.__data = self.__data.dropna()
        print(self.__data.info())
        
        self.__X = self.__data.values[:,1:] #leaves out - Customer ID
        self.__std_data = StandardScaler().fit_transform(self.__X)
              
    def visualizing_data(self):
        '''Visualize the data clusters i plot'''
        np.random.seed(0)
        plt.scatter(self.__X[:, 0], self.__X[:, 1], marker='.',alpha=0.3,ec='k',s=80)
        plt.show()

    def train_model_Kmeans(self,n__clusters = 3, n__init = 12):
        '''Train and build k-means++ with 3 clusters and 12 inits (default)'''
        self.__n_clusters = n__clusters
        self.__n_init = n__init
        
        self.__k_means = KMeans(init="k-means++", n_clusters = self.__n_clusters, n_init = self.__n_init)
        self.__k_means.fit(self.__X)
        self.__labels = self.__k_means.labels_
        
    def data_exploration(self):
        '''Add new col "Clus_km" && show the clusters in 2d'''
        self.__data["Clus_km"] = self.__labels
        self.__data.groupby('Clus_km').mean()
        
        area = np.pi * ( self.__X[:, 1])**2  
        plt.scatter(self.__X[:, 0], self.__X[:, 3], s=area, c=self.__labels.astype(float), cmap='tab10', ec='k',alpha=0.5)
        plt.xlabel('Age', fontsize=18)
        plt.ylabel('Income', fontsize=16)
        plt.show()  
    
    def visual_plot_3D(self):
        fig = px.scatter_3d(self.__X, x=1, y=0, z=3, opacity=0.7, color=self.__labels.astype(float))

        fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
        fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
                xaxis=dict(title='Education'),
                yaxis=dict(title='Age'),
                zaxis=dict(title='Income')
            ))  # Remove color bar, resize plot

        fig.show()       
    
    def create_profile_foreach_Group(self):
        '''considering the common characteristics of each cluster based on your observations'''
        cust_df_sub = self.__data[['Age', 'Edu','Income','Clus_km']].copy() 
        sns.pairplot(cust_df_sub, hue='Clus_km', palette='viridis', diag_kind='kde') 
        plt.suptitle('Pairwise Scatter Plot with K-means Clusters', y=1.02)
        plt.show()
        # 3- clusters can be
        # LATE CAREER, AFFLUENT, AND EDUCATED
        # MID CAREER AND MIDDLE INCOME
        # EARLY CAREER AND LOW INCOME          
        
    def run(self):
        self.load_data()
        self.preprocessing()
        self.train_model_Kmeans()
        self.data_exploration()
        self.visual_plot_3D()
        self.create_profile_foreach_Group()

if __name__ == "__main__":
    model = CustomerSegmentation()
    model.run()