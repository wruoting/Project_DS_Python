from mnist.loader import MNIST
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import matplotlib.pyplot as plt
from collections import deque
from noise import create_noise
from sklearn import preprocessing

def convert_pca(data, dimx=28, dimy=28, n=10):
    data = np.array(data).reshape(dimx, dimy)
    U, D, V = linalg.svd(data)
    # Convert D into a square matrix
    D_diag = np.diag(D)
    # First apply then reduce down to pca # N
    matmul_1 = np.matmul(U[:, 0:n], D_diag[0:n, 0:n])
    x_hat = np.matmul(matmul_1, V[0:n, :])
    return x_hat, D


def plot_demo(data, x_hat, fname='No_Noise_Comparison'):
    # Plotting Demo
    f, axarr = plt.subplots(2,1) 
    axarr[0].imshow(data)
    axarr[1].imshow(x_hat)
    plt.savefig(fname=fname)
    plt.show()

def plot_scree(D, n=10, name='Scree_Plot'):
    sum_all_components = np.sum(D)
    first_n_components = np.sum(D[0:n-1])
    percent_contribution = np.round(np.multiply(100, np.divide(first_n_components, sum_all_components)), 2)
    plt.plot(D)
    plt.title('Scree Plot Sample Image Top n={} contribution: {}%'.format(n, percent_contribution))
    plt.ylabel('EigenValue')
    plt.xlabel('Principal Component Number')
    plt.savefig(fname=name)
    plt.show()


def main():
    # MNIST path data
    data_path = './Data/'
    mndata = MNIST(data_path)
    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()

    images_training = np.asarray(images_training)
    images_testing = np.asarray(images_testing)
    # Normalize data
    images_training_normalize = preprocessing.normalize(images_training)
    labels_training = np.asarray(labels_training)

    # # This converts and plots the pca of a c
    x_hat, D = convert_pca(images_training_normalize[1])

    data = np.array(images_training_normalize[1]).reshape(28, 28)
    plot_demo(data, x_hat)

    # We can plot our scree plot to determine what percentage of our variance is attributed to those features
    plot_scree(D)

    # Adding noise
    images_training_noise = create_noise(images_training)
    images_training_noise = np.array(images_training_noise)
    images_training_noise = preprocessing.normalize(images_training_noise)

    x_hat, D = convert_pca(images_training_noise[1])
    data = np.array(images_training_noise[1]).reshape(28, 28)
    plot_demo(data, x_hat, fname='Noise_Comparison')
    plot_scree(D, n=15, name='Scree_Plot_Noise')

if __name__ == "__main__":
    main()