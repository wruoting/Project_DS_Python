import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def file_to_np(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as file_in:
            file_name = pd.read_csv(file_name)
            file_name = file_name.to_numpy()
        return file_name
    return None

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def main():
    decision_trees_path = './DecisionTreeAccuracies.csv'
    decision_trees_noise_path = './DecisionTreeAccuracies_noise.csv'
    random_forest_path = './RandomForestAccuracies.csv'
    random_forest_noise_path = './RandomForestAccuracies_noise.csv'
    knn_path = './KNNAccuracies.csv'
    knn_noise_path = './KNNAccuracies_noise.csv'

    decision_trees = file_to_np(decision_trees_path).T
    decision_trees_noise = file_to_np(decision_trees_noise_path).T
    random_forest = file_to_np(random_forest_path).T
    random_forest_noise = file_to_np(random_forest_noise_path).T
    knn = file_to_np(knn_path).T
    knn_noise = file_to_np(knn_noise_path).T

    # No Noise Data
    means = np.round(np.array([np.mean(decision_trees[0]), np.mean(random_forest[0]), np.mean(knn[0])]), 2)
    means_pca = np.round(np.array([np.mean(decision_trees[1]), np.mean(random_forest[1]), np.mean(knn[1])]), 2)
    sigma = np.round(np.array([np.std(decision_trees[0]), np.std(random_forest[0]), np.std(knn[0])]), 2)
    sigma_pca = np.round(np.array([np.std(decision_trees[1]), np.std(random_forest[1]), np.std(knn[1])]), 2)

    means_noise = np.round(np.array([np.mean(decision_trees_noise[0]), np.mean(random_forest_noise[0]), np.mean(knn_noise[0])]), 2)
    means_noise_pca = np.round(np.array([np.mean(decision_trees_noise[1]), np.mean(random_forest_noise[1]), np.mean(knn_noise[1])]), 2)
    sigma_noise = np.round(np.array([np.std(decision_trees_noise[0]), np.std(random_forest_noise[0]), np.std(knn_noise[0])]), 2)
    sigma_noise_pca = np.round(np.array([np.std(decision_trees_noise[1]), np.std(random_forest_noise[1]), np.std(knn_noise[1])]), 2)

    x_labels = ['Decision Trees', 'Random Forest', 'KNN']
    x = np.arange(3) 
    width = 0.35

    # Plots with no noise
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means, width, yerr=sigma, alpha=0.5, ecolor='black', capsize=10, label='Unprocessed')
    rects2 = ax.bar(x + width/2, means_pca, width, yerr=sigma_pca, alpha=0.5, ecolor='black', capsize=10, label='PCA')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.set_size_inches(11, 8)
    fig.savefig(fname='Accuracy_by_classifier_no_noise')
    fig.tight_layout()
    plt.show()
    plt.close()

    # Plots with noise
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means_noise, width, yerr=sigma_noise, alpha=0.5, ecolor='black', capsize=10, label='Unprocessed')
    rects2 = ax.bar(x + width/2, means_noise_pca, width, yerr=sigma_noise_pca, alpha=0.5, ecolor='black', capsize=10, label='PCA')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by classifier with noise')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.set_size_inches(11, 8)
    fig.savefig(fname='Accuracy_by_classifier_with_noise')
    fig.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()