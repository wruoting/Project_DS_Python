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
from main import convert_pca, create_training_and_testing_data

def create_noise(image_data):
    image_data_noise = deque()
    for image in image_data:
        mean = np.mean(image)
        std = np.std(image)
        noisy_img = image + np.random.normal(mean, std, image.shape)
        noisy_img_clipped = np.round(np.clip(noisy_img, 0, 255)).astype(int)
        image_data_noise.append(noisy_img_clipped)
    image_data_noise = np.array(image_data_noise)
    return image_data_noise

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

    images_training_noise = create_noise(images_training)
    images_testing_noise = create_noise(images_testing)

    images_training_pca_noise, images_testing_pca_noise = create_training_and_testing_data(images_training_noise, labels_training, images_testing_noise, labels_testing, n=15)

    pca_accuracy_list = deque()
    accuracy_list = deque()
    print('Starting classification')

    # KNN
    print('Starting KNN')
    knn_score = deque()
    knn_pca_score = deque()
    knn = KNeighborsClassifier()
    knn = knn.fit(images_training_noise, labels_training)
    knn_score.append(knn.score(images_testing_noise, labels_testing))

    knn_pca = KNeighborsClassifier()
    knn_pca = knn_pca.fit(images_training_pca_noise, labels_training)
    knn_pca_score.append(knn_pca.score(images_testing_pca_noise, labels_testing))
    knn_score = np.array(knn_score)
    knn_pca_score = np.array(knn_pca_score)
    accuracies_df = pd.DataFrame(
            {
                'Scores': knn_score,
                'PCA_Scores': knn_pca_score
            }).to_csv('KNNAccuracies_noise.csv', index=False)


    # Random Forest Accuracies
    clf_score = deque()
    clf_pca_score = deque()
    for n in range(1, 10):
        print('Iteration {} of Random Forest Classifier'.format(n))
        clf = RandomForestClassifier(n_estimators=100)
        clf = clf.fit(images_training_noise, labels_training)
        clf_score.append(clf.score(images_testing_noise, labels_testing))

        clf_pca = RandomForestClassifier(n_estimators=100)
        clf_pca = clf_pca.fit(images_training_pca_noise, labels_training)
        clf_pca_score.append(clf_pca.score(images_testing_pca_noise, labels_testing))
    clf_score = np.array(clf_score)
    clf_pca_score = np.array(clf_pca_score)
    accuracies_df = pd.DataFrame(
            {
                'Scores': clf_score,
                'PCA_Scores': clf_pca_score
            }).to_csv('RandomForestAccuracies_noise.csv', index=False)

    # Decision Tree Accuracies
    dtc_score = deque()
    dtc_pca_score = deque()
    for n in range(1, 10):
        print('Iteration {} of Decision Tree Accuracies'.format(n))
        dtc = tree.DecisionTreeClassifier()
        dtc = dtc.fit(images_training_noise, labels_training)
        dtc_score.append(dtc.score(images_testing_noise, labels_testing))

        dtc_pca = tree.DecisionTreeClassifier()
        dtc_pca = dtc_pca.fit(images_training_pca_noise, labels_training)
        dtc_pca_score.append(dtc_pca.score(images_testing_pca_noise, labels_testing))

    dtc_score = np.array(dtc_score)
    dtc_pca_score = np.array(dtc_pca_score)
    accuracies_df = pd.DataFrame(
            {
                'Scores': dtc_score,
                'PCA_Scores': dtc_pca_score
            }).to_csv('DecisionTreeAccuracies_noise.csv', index=False)


if __name__ == "__main__":
    main()