import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

class APP_PCA():
    '''
        Using PCA to project 2-D data onto its principal axes
        &&
        ...
    '''
    def __init__(self):
        self.data = None
        self.X = None
        self.model_pca = None
        self.components = None
        
    def create_data(self):
        '''
            2-dimensional dataset containing two linearly correlated features
            Using a bivariate normal distribution
            Both features, X1 and X2, will have zero mean and a covariance given by the (symmetric) covariance matrix
        '''
        np.random.seed(42)
        mean = [0, 0]
        cov = [[3, 2], [2, 2]]
        self.X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)
        
    def visualize_relationship(self):
        '''visualize relationship between the points'''
        #Scatter plot of the two features
        plt.figure()
        plt.scatter(self.X[:, 0], self.X[:, 1],  edgecolor='k', alpha=0.7)
        plt.title("Scatter Plot of Bivariate Normal Distribution")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        
    def perform_PCA(self):
        self.model_pca = PCA(n_components=2)
        X_pca = self.model_pca.fit_transform(self.X)
        
    def principal_components(self):
        '''
        principal components are the principal axes, represented in feature space coordinates,
        which align with the directions of maximum variance in the data
        '''
        print(f"Princial components: ")
        self.components = self.model_pca.components_
        print(self.components)
        
        print(f"Explained variance ration: ")
        print(self.model_pca.explained_variance_ratio_) # first component explains over 91% of the variance in the data
    
    def projection(self):
        '''Project the data onto its principal component axes
            && plot the projections along PC1 && PC2'''
        projection_pc1 = np.dot(self.X, self.components[0])
        projection_pc2 = np.dot(self.X, self.components[1])
        
        # now we can represent the projections of each data poin, along the principal directions in the original feature space
        x_pc1 = projection_pc1 * self.components[0][0]
        y_pc1 = projection_pc1 * self.components[0][1]
        x_pc2 = projection_pc2 * self.components[1][0]
        y_pc2 = projection_pc2 * self.components[1][1]
    
        # Plot original data
        plt.figure()
        plt.scatter(self.X[:, 0], self.X[:, 1], label='Original Data', ec='k', s=50, alpha=0.6)

        # Plot the projections along PC1 and PC2
        plt.scatter(x_pc1, y_pc1, c='r', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 1')
        plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 2')
        plt.title('Linearly Correlated Data Projected onto Principal Components', )
        plt.xlabel('Feature 1',)
        plt.ylabel('Feature 2',)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def pipline(self):
        self.create_data()
        self.visualize_relationship()
        self.perform_PCA()
        self.principal_components()
        self.projection()
        
if __name__ == "__main__":
    pca_alg = APP_PCA()
    pca_alg.pipline()
    