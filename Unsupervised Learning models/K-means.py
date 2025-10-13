import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans 

class K_means():
    '''
        Build and train K-means on syntetic data
        K-means++ : selects initial cluster centres for k-means clustering in a smart way to speed up convergence
        n_init : Number of times the k-means Alg. will be run with different centroid seeds
        The final results will be the best output of n_init consecutive runs in terms of inertia
    '''
    def __init__(self):
        self.k = None
        self.n_clusters = None
        self.n_init = None
        self.X = None
        self.y = None
        
    def create_syntetic_data(self):
        '''Create syntetic data with make_blobs function'''
        self.X, self.y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
    
    def visualizing_data(self):
        '''Visualize the data clusters i plot'''
        np.random.seed(0)
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='.',alpha=0.3,ec='k',s=80)
        plt.show()

    def run(self):
        self.create_syntetic_data()
        self.visualizing_data()

if __name__ == "__main__":
    model = K_means()
    model.run()