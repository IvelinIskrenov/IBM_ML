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
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_scaled = None
        self.__model_pca = None
        self.__X_pca = None
        self.__components = None
        
    def load_data(self):
        '''
            Load iris (flower) data set
        '''
        self.__data = datasets.load_iris()
        self.__X = self.__data.data
        self.__y = self.__data.target      

    def preprocessing(self):
        scaler = StandardScaler()
        self.__X_scaled = scaler.fit_transform(self.__X)
    
    def build_PCA(self):
        '''Initialize a PCA model and reduce the data set dimensionality to two components'''
        self.__model_pca = PCA(n_components=2)
        self.__X_pca = self.__model_pca.fit_transform(self.__X_scaled)
        
    def projection(self):
        '''plot data in 2d'''
        plt.figure(figsize=(8,6))

        colors = ['navy', 'turquoise', 'darkorange']
        lw = 1

        for color, i, target_name in zip(colors, [0, 1, 2], self.__data.target_names):
            plt.scatter(self.__X_pca[self.__y == i, 0], self.__X_pca[self.__y == i, 1], color=color, s=50, ec='k',alpha=0.7, lw=lw,
                        label=target_name)

        plt.title('PCA 2-dimensional reduction of IRIS dataset',)
        plt.xlabel("PC1",)
        plt.ylabel("PC2",)
        plt.legend(loc='best', shadow=False, scatterpoints=1,)
        # plt.grid(True)
        plt.show()    
    
    def explained_variance(self):
        print(f"Percentage of the original feature space variance do these two combined principal components: ")
        print(100*self.__model_pca.explained_variance_ratio_.sum())
    
    #This method is for fixing
    def plott(self):
        '''visualize relationship between the points'''
        explained_variance_ratio = self.__model_pca.explained_variance_ratio_

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
        self.__components = self.__model_pca.components_
        print(self.__components)
        
        print(f"Explained variance ration: ")
        print(self.__model_pca.explained_variance_ratio_) # first component explains over 91% of the variance in the data
    

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