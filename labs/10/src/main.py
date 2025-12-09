import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from console import print_classification_result, print_cluster_correspondence
from plot import plot_classification_predictions, plot_clusters


# Генерация данных
def generate_data(n=200, random_state=42):
    np.random.seed(random_state)

    centers = np.array([[3, 3], [9, 2], [9, 6]])
    covs = [np.diag([1.5, 1.5]), np.diag([1.0, 1.0]), np.diag([1.0, 1.0])]

    x_list, y_list = [], []
    for i, (c, cov) in enumerate(zip(centers, covs)):
        Xi = np.random.multivariate_normal(mean=c, cov=cov, size=n)
        yi = np.full(n, i)
        x_list.append(Xi)
        y_list.append(yi)

    x = np.vstack(x_list)
    y = np.concatenate(y_list)
    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["label"] = y

    return x, y, df


# Кластеризация KMeans
def run_kmeans(X, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return labels, centers


# Классификация k-NN и Naive Bayes
def classify_knn_gnb(x, y, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred_gnb = gnb.predict(x_test)

    y_train_knn = knn.predict(x_train)
    y_train_gnb = gnb.predict(x_train)

    return (x_test, y_test, y_pred_knn, y_pred_gnb, y_train_knn, y_train_gnb, x_train)


# Оценка классификации
def evaluate_classification(y_test, y_pred_knn, y_pred_gnb, save_csv="classification_results.csv"):
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    report_gnb = classification_report(y_test, y_pred_gnb, output_dict=True)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_gnb = accuracy_score(y_test, y_pred_gnb)

    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_gnb = confusion_matrix(y_test, y_pred_gnb)

    results = pd.DataFrame({
        "method": ["k-NN (k=5)", "GaussianNB"],
        "accuracy": [acc_knn, acc_gnb],
        "precision_macro": [report_knn['macro avg']['precision'], report_gnb['macro avg']['precision']],
        "recall_macro": [report_knn['macro avg']['recall'], report_gnb['macro avg']['recall']],
        "f1_macro": [report_knn['macro avg']['f1-score'], report_gnb['macro avg']['f1-score']]
    })

    results.to_csv(save_csv, index=False)

    return results, cm_knn, cm_gnb


# Основной сценарий
if __name__ == "__main__":
    x, y_true, df = generate_data()

    kmeans_labels, centers_km = run_kmeans(x)
    plot_clusters(x, y_true, kmeans_labels, centers_km)

    # Добавлен вызов функции печати таблицы соответствия
    cross_tab, row_percent, mapping = print_cluster_correspondence(y_true, kmeans_labels)

    x_test, y_test, y_pred_knn, y_pred_gnb, y_train_knn, y_train_gnb, x_train = classify_knn_gnb(x, y_true)
    results, cm_knn, cm_gnb = evaluate_classification(y_test, y_pred_knn, y_pred_gnb)

    print_classification_result(results, cm_knn, cm_gnb)

    plot_classification_predictions(x_test, y_pred_knn, y_pred_gnb, "предсказания на тесте",
                                    "classification_predictions.png")
    plot_classification_predictions(x_train, y_train_knn, y_train_gnb, "обучающая выборка", "train_result.png")
