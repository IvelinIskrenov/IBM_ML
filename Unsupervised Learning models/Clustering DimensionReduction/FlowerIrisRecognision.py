import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

class IrisRecognition():
    '''
        Build and use PCA Alg. to project the four-dimensional Iris feature data set down onto a two-dimensional feature space
    '''
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.model_pca = None
        self.X_pca = None
        self.components = None
        
    def load_data(self):
        '''
            Load iris (flower) data set
        '''
        self.data = datasets.load_iris()
        self.X = self.data.data
        self.y = self.data.target

        

    def preprocessing(self):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
    
    def build_PCA(self):
        '''Initialize a PCA model and reduce the data set dimensionality to two components'''
        self.model_pca = PCA(n_components=2)
        self.X_pca = self.model_pca.fit_transform(self.X_scaled)
        
    def projection(self):
        '''plot data in 2d'''
        plt.figure(figsize=(8,6))

        colors = ['navy', 'turquoise', 'darkorange']
        lw = 1

        for color, i, target_name in zip(colors, [0, 1, 2], self.data.target_names):
            plt.scatter(self.X_pca[self.y == i, 0], self.X_pca[self.y == i, 1], color=color, s=50, ec='k',alpha=0.7, lw=lw,
                        label=target_name)

        plt.title('PCA 2-dimensional reduction of IRIS dataset',)
        plt.xlabel("PC1",)
        plt.ylabel("PC2",)
        plt.legend(loc='best', shadow=False, scatterpoints=1,)
        # plt.grid(True)
        plt.show()    
    
    def explained_variance(self):
        print(f"Percentage of the original feature space variance do these two combined principal components: ")
        print(100*self.model_pca.explained_variance_ratio_.sum())
    
    #This method is for fixing
    def plott(self):
        '''visualize relationship between the points'''
        explained_variance_ratio = self.model_pca.explained_variance_ratio_

        # Plot explained variance ratio for each component
        plt.figure(figsize=(10,6))
        plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio, alpha=1, align='center', label='PC explained variance ratio' )
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.title('Explained Variance by Principal Components')

        # Plot cumulative explained variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        plt.step(range(1, 5), cumulative_variance, where='mid', linestyle='--', lw=3,color='red', label='Cumulative Explained Variance')
        # Only display integer ticks on the x-axis
        plt.xticks(range(1, 5))
        plt.legend()
        plt.grid(True)
        plt.show()
        
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
    

    def pipline(self):
        self.load_data()
        self.preprocessing()
        self.build_PCA()
        self.projection()
        self.explained_variance()
        self.principal_components()
        
if __name__ == "__main__":
    pca_alg = IrisRecognition()
    pca_alg.pipline()