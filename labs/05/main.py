import numpy as np

np.random.seed(18022004)

num_trials = 10000
n = 100

# Оценка Хубера
def huber_estimator(data, k=1.44, tol=1e-6, max_iter=100):
    theta = np.median(data)
    mad = np.median(np.abs(data - theta))
    if mad == 0:
        mad = np.mean(np.abs(data - theta))
        return theta
    iter_count = 0
    prev_theta = theta + 2 * tol
    while abs(theta - prev_theta) > tol and iter_count < max_iter:
        prev_theta = theta
        u = (data - theta) / mad
        abs_u = np.abs(u)
        w = np.where(abs_u <= k, 1, k / abs_u)
        theta = np.sum(w * data) / np.sum(w)
        iter_count += 1
    return theta

# Двухэтапная оценка (боксплот Тьюки с IQR, как в лекции)
def two_step_estimator(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = data[(data >= lower) & (data <= upper)]
    if len(filtered) == 0:
        return np.mean(data)  # Редкий случай
    return np.mean(filtered)

# Основная функция для Монте-Карло
def monte_carlo(distribution_func, n, num_trials):
    means = []
    medians = []
    huber_estimates = []
    two_step_estimates = []

    for _ in range(num_trials):
        sample = distribution_func(n)
        means.append(np.mean(sample))
        medians.append(np.median(sample))
        huber_estimates.append(huber_estimator(sample))
        two_step_estimates.append(two_step_estimator(sample))

    return (
        np.mean(means), np.var(means),
        np.mean(medians), np.var(medians),
        np.mean(huber_estimates), np.var(huber_estimates),
        np.mean(two_step_estimates), np.var(two_step_estimates)
    )

# Распределения
def normal_distribution(n):
    return np.random.normal(0, 1, n)

def cauchy_distribution(n):
    return np.random.standard_cauchy(n)

def mixture_distribution(n):
    return np.where(np.random.rand(n) < 0.9, np.random.normal(0, 1, n), np.random.standard_cauchy(n))

# Запуск и вывод в таблице
distributions = {
    "Стандартное нормальное распределение N(0,1)": normal_distribution,
    "Распределение Коши C(0,1)": cauchy_distribution,
    "\"Смесь\" 0.9N(0,1) + 0.1C(0,1)": mixture_distribution
}

for dist_name, dist_func in distributions.items():
    results = monte_carlo(dist_func, n, num_trials)

    print(f"Данные по {dist_name}:")
    print("            | Среднее | Дисперсия")
    print("============|=========|==========")
    print(f"Выб. среднее| {results[0]:7.4f} | {results[1]:7.4f}")
    print(f"Выб. медиана| {results[2]:7.4f} | {results[3]:7.4f}")
    print(f"Хубер       | {results[4]:7.4f} | {results[5]:7.4f}")
    print(f"Двухэтап.   | {results[6]:7.4f} | {results[7]:7.4f}")
    print()
