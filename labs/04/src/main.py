import dataclasses

import numpy as np
import matplotlib.pyplot as plt

@dataclasses.dataclass
class SigmaRuleResult:
    lower_bound: float
    upper_bound: float
    first_three: np.ndarray
    first_three_anomalies: np.ndarray
    last_three: np.ndarray
    last_three_anomalies: np.ndarray

# 1. Генерация данных
def generate_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    xx = np.random.normal(0, 1, 195)

    additional_values = np.array([5.0, -4, 3.3, 2.99, -3], dtype=np.float64)
    data = np.append(xx, additional_values)

    return data, np.sort(data)

# 1. Правило трёх сигм
def apply_three_sigma(data: np.ndarray, sorted_data: np.ndarray) -> SigmaRuleResult:
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    first_three = sorted_data[:3]
    first_three_anomalies = np.array([x for x in first_three if x < lower_bound or x > upper_bound])

    last_three = sorted_data[-3:]
    last_three_anomalies = np.array([x for x in last_three if x < lower_bound or x > upper_bound])

    print("Результаты правила трёх сигм:")
    print(f"Среднее: {mean:.2f}, Стандартное отклонение: {std:.2f}")
    print(f"Границы: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print("Первые три порядковые статистики:", first_three)
    print("Аномалии среди первых трёх:", first_three_anomalies)
    print("Последние три порядковые статистики:", last_three)
    print("Аномалии среди последних трёх:", last_three_anomalies)

    return SigmaRuleResult(
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        first_three=first_three,
        first_three_anomalies=first_three_anomalies,
        last_three=last_three,
        last_three_anomalies=last_three_anomalies,
    )


# 2. Результаты
def print_conclusions(result: SigmaRuleResult, outliers: np.ndarray) -> None:
    three_sigma_anomalies = np.concatenate((result.first_three_anomalies, result.last_three_anomalies))
    print("\nСравнение:")
    print("Аномалии по правилу трёх сигм (среди проверенных крайних значений):", three_sigma_anomalies)
    print("Выбросы по боксплоту Тьюки:", outliers)

# 2. Boxplot
def print_tukey_boxplot(data: np.ndarray) -> np.ndarray:
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data, vert=False, patch_artist=True)

    outliers = box['fliers'][0].get_xydata()[:, 0]
    print("\nВыбросы по боксплоту Тьюки:", outliers)

    plt.title("Боксплот Тьюки для сгенерированных данных")
    plt.xlabel("Значения")
    plt.show(block=False)
    plt.savefig("tukey-boxplot.png")

    return outliers


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    data, sorted_data = generate_data()
    print_conclusions(apply_three_sigma(data, sorted_data), print_tukey_boxplot(data))
