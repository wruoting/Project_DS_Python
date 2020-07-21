from mnist.loader import MNIST
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import matplotlib.pyplot as plt

def convert_pca(data, dimx=28, dimy=28, n=1):
    data = np.array(data).reshape(dimx, dimy)
    U, D, V = linalg.svd(data)
    # Only take components greater than N
    pca_components_greater_than_n = np.sum(np.array([D > n], dtype=np.bool))
    # Convert D into a square matrix
    D_diag = np.diag(D)
    # First apply then reduce down to pca # N
    matmul_1 = np.matmul(U[:, 0:pca_components_greater_than_n], D_diag[0:pca_components_greater_than_n, 0:pca_components_greater_than_n])
    x_hat = np.matmul(matmul_1, V[0:pca_components_greater_than_n, :])
    return x_hat, D

def plot_demo(data, x_hat):
    # Plotting Demo
    f, axarr = plt.subplots(2,1) 
    axarr[0].imshow(data)
    axarr[1].imshow(x_hat)
    plt.show()

def plot_scree(D, n=10):
    sum_all_components = np.sum(D)
    first_n_components = np.sum(D[0:n-1])
    percent_contribution = np.round(np.multiply(100, np.divide(first_n_components, sum_all_components)), 2)
    plt.plot(D)
    plt.title('Scree Plot Sample Image Top n={} contribution: {}%'.format(n, percent_contribution))
    plt.ylabel('EigenValue')
    plt.xlabel('Principal Component Number')
    plt.savefig(fname='Scree Plot')
    plt.show()


def main():
    # MNIST path data
    data_path = './Data/'
    mndata = MNIST(data_path)
    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()

    images_training_standardized = StandardScaler().fit_transform(images_training)

    # This converts and plots the pca of a c
    x_hat, D = convert_pca(images_training_standardized[1])

    data = np.array(images_training[1]).reshape(28, 28)
    plot_demo(data, x_hat)

    # We can plot our scree plot to determine whether 
    plot_scree(D)


if __name__ == "__main__":
    main()