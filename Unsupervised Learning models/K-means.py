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
        self.model = None
        self.k_means_labels = None
        self.k_means_cluster_centers = None
        
        
    def create_syntetic_data(self):
        '''Create syntetic data with make_blobs function'''
        self.X, self.y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
    
    def visualizing_data(self):
        '''Visualize the data clusters i plot'''
        np.random.seed(0)
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='.',alpha=0.3,ec='k',s=80)
        plt.show()

    def train_model(self,n__clusters = 4, n__init = 12):
        self.n_clusters = n__clusters
        self.n_init = n__init
        
        self.model = KMeans(init = "k-means++", n_clusters = self.n_clusters, n_init = self.n_init)
        self.model.fit(self.X)
        
        k_means_labels = self.model.labels_
        print(k_means_labels)
        
        k_means_cluster_centers = self.model.cluster_centers_
        print(k_means_cluster_centers)
        
    def visual_plot_Kmeans(self):
        #Init the plot with the specified dimensions
        fig = plt.figure(figsize=(6, 4))

        #Colors uses a color map, which will produce an array of colors based on the number of labels there are
        #We use set(k_means_labels) to get the unique labels
        colors = plt.cm.tab10(np.linspace(0, 1, len(set(self.k_means_labels))))

        #Create a plot
        ax = fig.add_subplot(1, 1, 1)

        #plots the data points and centroids
        #k will range from 0-3, which will match the possible clusters that each data point is in
        for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

            #Create a list of all data p., where the data p. that are  in the cluster (ex. cluster 0) are labeled as true, else - false.
            my_members = (self.k_means_labels == k)

            #def the centroid or cluster center
            cluster_center = self.k_means_cluster_centers[k]

            #plots the data p. with color col.
            ax.plot(self.X[my_members, 0], self.X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

            #Plots the centroids with specified color (with a darker outline)
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

        ax.set_title('KMeans')

        #Remove x-axis && y-axis ticks
        ax.set_xticks(())
        ax.set_yticks(())
        plt.show()         
        
    def run(self):
        self.create_syntetic_data()
        self.visualizing_data()
        self.train_model()
        self.visual_plot_Kmeans()

if __name__ == "__main__":
    model = K_means()
    model.run()