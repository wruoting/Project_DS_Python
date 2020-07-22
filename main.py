
from mnist.loader import MNIST
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

def convert_pca(data, dimx=28, dimy=28, n=1):
    data = np.array(data).reshape(dimx, dimy)
    U, D, V = linalg.svd(data)
    # Convert D into a square matrix
    D_diag = np.diag(D)
    # First apply then reduce down to pca # N
    matmul_1 = np.matmul(U[:, 0:n], D_diag[0:n, 0:n])
    x_hat = np.matmul(matmul_1, V[0:n, :])
    return x_hat, D


def create_training_and_testing_data(images_training, labels_training, images_testing, labels_testing, name="", n=10):
     # PCA
    standardized_training_pca = './PCA_training_{}.csv'.format(name)
    standardized_testing_pca = './PCA_testing_{}.csv'.format(name)

    # Training PCA
    if os.path.exists(standardized_training_pca):
        with open(standardized_training_pca, 'r') as file_in:
            images_training_standardized_pca = pd.read_csv(standardized_training_pca)
    else:
        with open(standardized_training_pca, 'w') as file_out:
            images_training_standardized_pca = np.array([convert_pca(image, n=n)[0].flatten() for image in images_training])
            pd.DataFrame(images_training_standardized_pca).to_csv(standardized_training_pca, index=False)
    # Testing PCA
    if os.path.exists(standardized_testing_pca):
        with open(standardized_testing_pca, 'r') as file_in:
            images_testing_standardized_pca = pd.read_csv(standardized_testing_pca)
    else:
        with open(standardized_testing_pca, 'w') as file_out:
            images_testing_standardized_pca = np.array([convert_pca(image, n=n)[0].flatten() for image in images_testing])
            pd.DataFrame(images_testing_standardized_pca).to_csv(standardized_testing_pca, index=False)
    return images_training_standardized_pca, images_testing_standardized_pca
    
# Use KNN with and without PCA and compare results
def main():
    # MNIST path data
    data_path = './Data/'
    mndata = MNIST(data_path)
    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()

    images_training = np.asarray(images_training)
    images_testing = np.asarray(images_testing)
    labels_training = np.asarray(labels_training)
    labels_testing = np.asarray(labels_testing)

    images_training_pca, images_testing_pca = create_training_and_testing_data(images_training, labels_training, images_testing, labels_testing, name="Random_Forest")
    total_range = range(5, 6)
    pca_accuracy_list = deque()
    accuracy_list = deque()
    print('Starting classification')
    # for n in total_range:    
        # knn_classifier = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
        # knn_classifier.fit(images_training_standardized[0:4000], labels_training[0:4000])
        # prediction_custom = knn_classifier.predict(images_testing)
        # accuracy = np.round(np.multiply(np.mean(labels_testing == prediction_custom), 100))
        # accuracy_list.append(accuracy)

        # knn_classifier_pca = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
        # knn_classifier_pca.fit(images_training_standardized_pca[0:4000], labels_training[0:4000])
        # prediction_custom_pca = knn_classifier_pca.predict(images_testing)
        # accuracy_pca = np.round(np.multiply(np.mean(labels_testing == prediction_custom_pca), 100))
        # pca_accuracy_list.append(accuracy_pca)

    # Random Forest Accuracies
    clf_score = deque()
    clf_pca_score = deque()
    for n in range(1, 10):
        clf = RandomForestClassifier(n_estimators=100)
        clf = clf.fit(images_training, labels_training)
        prediction_custom = clf.predict(images_testing)
        clf_score.append(clf.score(images_testing, labels_testing))

        clf_pca = RandomForestClassifier(n_estimators=100)
        clf_pca = clf_pca.fit(images_training_pca, labels_training)
        clf_pca_score.append(clf_pca.score(images_testing_pca, labels_testing))
    clf_score = np.array(clf_score)
    clf_pca_score = np.array(clf_pca_score)
    accuracies_df = pd.DataFrame(
            {
                'Scores': clf_score,
                'PCA_Scores': clf_pca_score
            }).to_csv('RandomForestAccuracies.csv', index=False)

    # Decision Tree Accuracies
    dtc_score = deque()
    dtc_pca_score = deque()
    for n in range(1, 10):
        dtc = tree.DecisionTreeClassifier()
        dtc = dtc.fit(images_training, labels_training)
        prediction_custom = dtc.predict(images_testing)
        dtc_score.append(dtc.score(images_testing, labels_testing))

        dtc_pca = tree.DecisionTreeClassifier()
        dtc_pca = dtc_pca.fit(images_training_pca, labels_training)
        dtc_pca_score.append(dtc_pca.score(images_testing_pca, labels_testing))

    dtc_score = np.array(dtc_score)
    dtc_pca_score = np.array(dtc_pca_score)
    accuracies_df = pd.DataFrame(
            {
                'Scores': dtc_score,
                'PCA_Scores': dtc_pca_score
            }).to_csv('DecisionTreeAccuracies.csv', index=False)


if __name__ == "__main__":
    main()