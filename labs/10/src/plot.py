# Визуализация кластеров
import numpy as np
from matplotlib import pyplot as plt


def plot_clusters(x, y_true, kmeans_labels, centers, filename="kmeans_result.png"):
    plt.figure(figsize=(12, 5))

    # Истинные метки
    plt.subplot(1, 2, 1)
    for lab in np.unique(y_true):
        mask = y_true == lab
        plt.scatter(x[mask, 0], x[mask, 1], s=15, label=f"true {lab}")
    plt.title("Истинные метки")
    plt.legend()

    # Найденные кластеры
    plt.subplot(1, 2, 2)
    for lab in np.unique(kmeans_labels):
        mask = kmeans_labels == lab
        plt.scatter(x[mask, 0], x[mask, 1], s=15, label=f"km {lab}")
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, edgecolor='k', linewidth=1.5, label='centers')
    plt.title("KMeans: найденные кластеры")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)


# Визуализация предсказаний классификаторов
def plot_classification_predictions(x, y_knn, y_gnb, title: str, filename: str):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for lab in np.unique(y_knn):
        mask = y_knn == lab
        plt.scatter(x[mask, 0], x[mask, 1], s=25, label=f"knn {lab}")
    plt.title(f"k-NN: {title}")
    plt.legend()

    plt.subplot(1, 2, 2)
    for lab in np.unique(y_gnb):
        mask = y_gnb == lab
        plt.scatter(x[mask, 0], x[mask, 1], s=25, label=f"gnb {lab}")
    plt.title(f"GaussianNB: {title}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
