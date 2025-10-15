import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests
import zipfile
import io
import os
import warnings
warnings.filterwarnings('ignore')
 
# from IBM lab 
def download_extract_data(zip_file_url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip"):
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
                        

    
class MuseumClusterer():
    '''Build and train DBSCAN and HDBSCAN models, then compare them'''
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.DBSCAN = None
        self.HDBSCAN = None
    
    def load_data(self, url= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'):
        '''Loading data form cvs url'''
        try:
            if self.data == None:
                self.data = pd.read_csv(url, encoding = "ISO-8859-1")
        except Exception:
            print(f"Error while loading the data !!!")
    
    def exploar_data(self):
        '''Explore data -> describe(), info(), print some samples, print value counts'''
        print(f"Describe data: ")
        print(self.data.describe())
        
        print(f"Data info: ")
        print(self.data.info())
        
        print(f"Some samples: ")
        print(self.data.head(5))
        
        print(f"Value counts: ")
        print(self.data.ODCAF_Facility_Type.value_counts())
    
    def run(self):
        self.load_data()
        self.exploar_data()
            
if __name__ == "__main__":
    download_extract_data()
    #plot_clustered_locations()
    model = MuseumClusterer()
    model.run()