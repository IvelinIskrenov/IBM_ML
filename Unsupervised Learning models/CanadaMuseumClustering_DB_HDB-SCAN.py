import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler

# geographical tools
import geopandas as gpd  # pandas dataframe-like geodataframes for geographical data
import contextily as ctx  # used for obtianing a basemap of Canada
from shapely.geometry import Point
import requests
import zipfile
import io
import os
import warnings
warnings.filterwarnings('ignore')
 
# from IBM lab 
def download_extract_data(zip_file_url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip"):
    '''Download and extrackting data'''
    try:
        output_dir = './'
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Download the ZIP file
        response = requests.get(zip_file_url)
        response.raise_for_status()  # Ensure the request was successful
        # Step 2: Open the ZIP file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Step 3: Iterate over the files in the ZIP
            for file_name in zip_ref.namelist():
                if file_name.endswith('.tif'):  # Check if it's a TIFF file
                    # Step 4: Extract the TIFF file
                    zip_ref.extract(file_name, output_dir)
                    print(f"Downloaded and extracted: {file_name}")   
    except Exception:
        print(f"Error while download and unzipping the data !!!")  

#from IBM lab
def plot_clustered_locations(df,  title='Museums Clustered by Proximity'):
    """
    Plots clustered locations and overlays on a basemap.
    
    Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """
    
    # Load the coordinates intto a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    
    # Reproject to Web Mercator to align with basemap 
    gdf = gdf.to_crs(epsg=3857)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Separate non-noise, or clustered points from noise, or unclustered points
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]
    
    # Plot noise points 
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    
    # Plot clustered points, colured by 'Cluster' number
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)
    
    # Add basemap of  Canada
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)
    
     # Format plot
    plt.title(title, )
    plt.xlabel('Longitude', )
    plt.ylabel('Latitude', )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
class MuseumClusterer():
    '''Build and train DBSCAN and HDBSCAN models, then compare them'''
    def __init__(self):
        self.__data = None
        self.__X = None
        self.__y = None
        self.__DBSCAN = None
        self.__HDBSCAN = None
        self.__coords_scaled = None
    
        # DBSCAN && HDBSCAN params
        self.__min_samples = None #minimum number of samples needed to form a neighbourhood 
        self.__eps = None #neighbourhood search radius
        self.__metric = None #distance measure
        
        #HDBSCAN param
        self.__min_cluster_size = None
        

    def load_data(self, url= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'):
        '''Loading data form cvs url'''
        try:
            if self.__data == None:
                self.__data = pd.read_csv(url, encoding = "ISO-8859-1")
        except Exception:
            print(f"Error while loading the data !!!")
    
    def exploar_data(self):
        '''Explore data -> describe(), info(), print some samples, print value counts'''
        print(f"Describe data: ")
        print(self.__data.describe())
        
        print(f"Data info: ")
        print(self.__data.info())
        
        print(f"Some samples: ")
        print(self.__data.head(5))
        
        print(f"Value counts: ")
        print(self.__data.ODCAF_Facility_Type.value_counts())
    
    def filter_data(self):
        self.__data = self.__data[self.__data.ODCAF_Facility_Type == 'museum']
        self.__data.ODCAF_Facility_Type.value_counts()
        
        self.__data = self.__data[['Latitude', 'Longitude']]
        print(self.__data.info())
        
        self.__data = self.__data[self.__data.Latitude!='..']
    
    def preprocessing(self):
        self.filter_data()
        self.__data[['Latitude','Longitude']] = self.__data[['Latitude','Longitude']].astype('float')
        
        #Using standardization would be an error becaues we aren't using the full range of the lat/lng feature coordinates
        #Since latitude has a range of +/- 90 degrees and longitude ranges from 0 to 360 degrees, 
        #the correct scaling is to double the longitude coordinates (or half the Latitudes)
        self.__coords_scaled = self.__data.copy()
        self.__coords_scaled["Latitude"] = 2*self.__coords_scaled["Latitude"]
    
    def build_DBSCAN(self, _min_samples = 3, _eps=1.0, _metric = 'euclidean'):
        '''Build and train the DBSCAN model with params -  min_samles, epsilon, metric = euclidean (default)'''
        
        print(f"Building DBSCAN model ... ")
        
        self.__min_samples = _min_samples 
        self.__eps = _eps
        self.__metric = _metric 

        self.__DBSCAN = DBSCAN(eps=self.__eps, min_samples=self.__min_samples, metric=self.__metric).fit(self.__coords_scaled)
        
        self.__data['Cluster'] = self.__DBSCAN.fit_predict(self.__coords_scaled)  # Assign the cluster labels

        print(self.__data['Cluster'].value_counts())
    
    def build_HDBSCAN(self, _min_samples = None, _min_cluster_size = 3, _metric = 'euclidean'):
        '''Build and train the HDBSCAN model with params -  min_samles = None, min_cluster_size = 3, metric = euclidean (default)'''
        print(f"Building HDBSCAN model ... ")
        
        self.__min_samples = _min_samples
        self.__min_cluster_size = _min_cluster_size
        self.__metric = _metric
        self.__HDBSCAN = hdbscan.HDBSCAN(min_samples = self.__min_samples, min_cluster_size = self.__min_cluster_size, metric='euclidean')
        
        self.__data['Cluster'] = self.__HDBSCAN.fit_predict(self.__coords_scaled)

        print(self.__data['Cluster'].value_counts())
    
    def compare_HDB_DB_SCAN(self):
        '''Pipeline of the HDB/DB SCAN compare'''
        self.load_data()
        self.preprocessing()
        self.build_DBSCAN()
        plot_clustered_locations(self.__data)
        
        self.build_HDBSCAN()
        plot_clustered_locations(self.__data)
        
    def run(self):
        '''Pipline on one of the models'''
        self.load_data()
        self.exploar_data()
        self.preprocessing()
        
        print("Which model do you want to build DBSCAN/HDBSCAN ?")
        model = input()
        
        if model == "DBSCAN":
            self.build_DBSCAN()
        elif model == "HDBSCAN":    
            self.build_HDBSCAN()
        else:
            print(f"Invalid model")
            
        plot_clustered_locations(self.__data)
            
if __name__ == "__main__":
    download_extract_data()
    model = MuseumClusterer()
    model.compare_HDB_DB_SCAN()
    #model.run()
    #plot_clustered_locations(model.data)
    #1. from plot_clustered_locations we see that the clusters are not uniformly dense
    #2. DBSCAN agglomerates neighboring clusters together when they are close enough
    
