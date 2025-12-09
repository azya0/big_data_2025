# Таблица соответствия между истинными метками и кластерами KMeans
import numpy as np
import pandas as pd


def print_cluster_correspondence(y_true: np.ndarray, kmeans_labels: np.ndarray):
    """
    Создает и печатает таблицу соответствия между истинными метками
    и найденными кластерами KMeans.
    """
    # Фикс порядка
    remap = {0: 2, 1: 0, 2: 1}
    y_true = np.array([remap[x] for x in y_true])
    # Создаем DataFrame для анализа соответствия
    correspondence_df = pd.DataFrame({
        'true_label': y_true,
        'kmeans_cluster': kmeans_labels
    })

    # Создаем кросс-таблицу (contingency table)
    cross_tab = pd.crosstab(
        correspondence_df['true_label'],
        correspondence_df['kmeans_cluster'],
        margins=True,
        margins_name="Total"
    )

    # Добавляем проценты по строкам
    row_percent = pd.crosstab(
        correspondence_df['true_label'],
        correspondence_df['kmeans_cluster'],
        normalize='index'
    ) * 100

    print("\n" + "=" * 60)
    print("ТАБЛИЦА СООТВЕТСТВИЯ: Истинные метки vs KMeans кластеры")
    print("=" * 60)

    print("\nАбсолютные частоты:")
    print(cross_tab.to_string())

    print("\n\nПроцентное распределение по строкам (истинные метки -> кластеры):")
    for true_label in row_percent.index:
        print(f"\nИстинная метка {true_label}:")
        for cluster in row_percent.columns:
            percent = row_percent.loc[true_label, cluster]
            if percent > 0:  # Показываем только ненулевые значения
                print(f"  -> Кластер {cluster}: {percent:.1f}%")

    # Определяем доминирующее соответствие
    print("\n\nДОМИНИРУЮЩЕЕ СООТВЕТСТВИЕ:")
    mapping = {}
    for true_label in np.unique(y_true):
        mask = y_true == true_label
        cluster_counts = pd.Series(kmeans_labels[mask]).value_counts()
        if len(cluster_counts) > 0:
            dominant_cluster = cluster_counts.idxmax()
            count = cluster_counts.max()
            total = mask.sum()
            percentage = 100 * count / total
            mapping[true_label] = dominant_cluster
            print(f"Истинная метка {true_label} -> Кластер {dominant_cluster}: "
                  f"{count}/{total} ({percentage:.1f}%)")

    # Проверка уникальности соответствия
    if len(set(mapping.values())) == len(mapping):
        print("\n✓ Соответствие взаимно-однозначное (биекция)")
    else:
        print("\n⚠ Внимание: несколько истинных меток соответствуют одному кластеру")

    return cross_tab, row_percent, mapping


def print_classification_result(results: pd.DataFrame, cm_knn: np.ndarray, cm_gnb: np.ndarray):
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("=" * 60)
    print("\n=== Results table ===")
    print(results)

    print("\n=== Confusion matrix k-NN ===")
    print(cm_knn)

    print("\n=== Confusion matrix GaussianNB ===")
    print(cm_gnb)
