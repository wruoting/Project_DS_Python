
from mnist.loader import MNIST
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os

def convert_pca(data, dimx=28, dimy=28, n=1):
    data = np.array(data).reshape(dimx, dimy)
    U, D, V = linalg.svd(data)
    # Convert D into a square matrix
    D_diag = np.diag(D)
    # First apply then reduce down to pca # N
    matmul_1 = np.matmul(U[:, 0:n], D_diag[0:n, 0:n])
    x_hat = np.matmul(matmul_1, V[0:n, :])
    return x_hat, D

# Use KNN with and without PCA and compare results
def main():
    # MNIST path data
    data_path = './Data/'
    mndata = MNIST(data_path)
    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()

    # PCA
    standardized_training = './Standardized_training.csv'
    standardized_training_pca = './PCA_training.csv'
    dimensions = len(images_training), len(images_training[0])

    if os.path.exists(standardized_training):
        with open(standardized_training, 'r') as file_in:
            images_training_standardized = pd.read_csv(standardized_training)
            images_training_standardized = images_training_standardized.to_numpy()
    else:
        with open(standardized_training, 'w') as file_out:
            images_training_standardized = StandardScaler().fit_transform(images_training).reshape(dimensions)
            pd.DataFrame(images_training_standardized).to_csv(standardized_training, index=False)

    if os.path.exists(standardized_training_pca):
        with open(standardized_training_pca, 'r') as file_in:
            images_training_standardized_pca = pd.read_csv(standardized_training_pca)
            images_training_standardized_pca = images_training_standardized_pca.to_numpy()
    else:
        with open(standardized_training_pca, 'w') as file_out:
            images_training_standardized_pca = np.array([convert_pca(image, n=10)[0].flatten() for image in images_training_standardized])
            pd.DataFrame(images_training_standardized_pca).to_csv(standardized_training_pca, index=False)
    
    # print(images_training_standardized_pca)

    # for n in range(1, 25):
    #     knn_classifier = KNeighborsClassifier(n_neighbors=n)
    
    



if __name__ == "__main__":
    main()