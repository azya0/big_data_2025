from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.axes import Axes


np.random.seed(18022004)


COMMAND_TEXT: str = """
one - Всё на одном графике
different - Все данные на разных графиках
exit - Выход
"""


@dataclass
class ModelSeries:
    base:       np.ndarray[int]
    true_trend: np.ndarray[float]
    data:       np.ndarray[float]


def generate_series(index: int = 500, h: float = 0.05) -> ModelSeries:
    base = np.arange(index + 1)
    true_trend = np.sqrt(base * h)
    noise = np.random.normal(0, 1, size=base.size)
    data = true_trend + noise
    
    return ModelSeries(base, true_trend, data)


@dataclass
class BaseProps:
    series:         ModelSeries
    window_sizes:   list[int]
    m_values:       list[int]


# Moving Average Trend extraction
def MAT(series: ModelSeries, m_value: int) -> np.ndarray[float]:
    result = np.zeros(series.data.size)

    for index in range(series.data.size):
        start = max(0, index - m_value)
        end = min(series.data.size, index + m_value + 1)

        window_size = end - start
        
        if window_size <= 0:
            continue
        
        result[index] = sum(series.data[start:end]) / window_size
    
    return result


def MM(series: ModelSeries, m_value: int) -> np.ndarray[float]:
    result = np.zeros(series.data.size)

    for index in range(series.data.size):
        start = max(0, index - m_value)
        end = min(series.data.size, index + m_value + 1)
        window_size = end - start

        if window_size <= 0:
            continue

        window = series.data[start:end]

        result[index] = np.median(window)
    
    return result


@dataclass
class TrendFunctionProps:
    props:  BaseProps
    func:   Callable[[ModelSeries, int], np.ndarray]


def get_trend(data: TrendFunctionProps) -> dict[int, np.ndarray[float]]:
    result = {}
    for window, m_value in zip(data.props.window_sizes, data.props.m_values):
        result[window] = data.func(data.props.series, m_value)
    
    return result


@dataclass
class ResultProps:
    props:      BaseProps
    mat_data:   dict[int, np.ndarray[float]]
    mm_data:    dict[int, np.ndarray[float]]


def print_graphs(data: ResultProps):
    # Для тайп-хинтв в vs-code
    def get_axes_by_index(data: np.ndarray[np.ndarray[Axes]], i: int, j: int) -> Axes:
        return data[i, j]
    
    fig, axes = pyplot.subplots(2, len(data.props.window_sizes), figsize=(15, 10))
    fig.suptitle("Сравнение трендов с исходным рядом", fontsize=16)

    get_axes = lambda i, j: get_axes_by_index(axes, i, j)

    for index, window in enumerate(data.props.window_sizes):
        get_axes(0, index).plot(
            data.props.series.base,
            data.props.series.data,
            "bo", alpha=0.7, label="Исходный ряд x_k",
            linewidth=0.2, ms=2
        )
        get_axes(0, index).plot(
            data.props.series.base,
            data.mat_data[window],
            "r-", label=f"Скользящее среднее (окно {window})",
            linewidth=1.5
        )
        get_axes(0, index).plot(
            data.props.series.base,
            data.props.series.true_trend,
            "g--", label="Истинный тренд",
            linewidth=2
        )
        get_axes(0, index).set_title(f"Скользящее среднее, окно {window}")
        get_axes(0, index).set_xlabel("data.props.series.base")
        get_axes(0, index).set_ylabel("Значение")
        get_axes(0, index).legend()
        get_axes(0, index).grid(True)

        get_axes(1, index).plot(
            data.props.series.base,
            data.props.series.data,
            "bo", alpha=0.7,
            label="Исходный ряд x_k",
            linewidth=0.8, ms=2
        )
        get_axes(1, index).plot(
            data.props.series.base,
            data.mat_data[window],
            "orange", label=f"Медиана (окно {window})",
            linewidth=1.5
        )
        get_axes(1, index).plot(
            data.props.series.base,
            data.props.series.true_trend,
            "g--", label="Истинный тренд",
            linewidth=2
        )
        get_axes(1, index).set_title(f"Медиана, окно {window}")
        get_axes(1, index).set_xlabel("data.props.series.base")
        get_axes(1, index).set_ylabel("Значение")
        get_axes(1, index).legend()
        get_axes(1, index).grid(True)

    pyplot.tight_layout()
    pyplot.show()


def print_graph(data: ResultProps):
    pyplot.figure(figsize=(12, 8))
    pyplot.plot(
        data.props.series.base,
        data.props.series.data,
        "bo", alpha=0.5, label="Исходный ряд x_k",
        linewidth=0.8, ms=2
    )
    pyplot.plot(
        data.props.series.base,
        data.props.series.true_trend,
        "g--", label="Истинный тренд",
        linewidth=2
    )
    
    for window in data.props.window_sizes:
        pyplot.plot(
            data.props.series.base,
            data.mat_data[window],
            "--", label=f"Скользящее среднее, окно {window}",
            linewidth=1.5
        )
        pyplot.plot(
            data.props.series.base,
            data.mm_data[window],
            "-", label=f"Медиана окно {window}",
            linewidth=1.5
        )
    
    pyplot.title("Все тренды на синхронной оси k")
    pyplot.xlabel("k")
    pyplot.ylabel("Значение")
    pyplot.legend()
    pyplot.grid(True)
    pyplot.show()


def checkout(data: ResultProps):
    def count_turning_points(series):
        n = len(series)
        if n < 3:
            return 0
        p = 0
        for i in range(1, n - 1):
            if (series[i - 1] < series[i] > series[i + 1]) or (series[i - 1] > series[i] < series[i + 1]):
                p += 1
        return p


    def expected_var_p(n):
        if n < 3:
            return 0, 0
        ep = (2 / 3) * (n - 2)
        return ep


    def kendall_tau_time(series, time):
        from scipy.stats import kendalltau

        tau, p_value = kendalltau(time, series)
        return tau, p_value
    
    @dataclass
    class ResultData:
        p:      int
        ep:     float
        tau:    float

    results: dict[str, dict[int, ResultData]] = {}

    for method, trends in [("SMA", data.mat_data), ("Median", data.mm_data)]:
        results[method] = {}
        for window in data.props.window_sizes:
            trend = trends[window]
            residuals = data.props.series.data - trend

            p = count_turning_points(residuals)
            ep = expected_var_p(data.props.series.data.size)

            tau, _ = kendall_tau_time(residuals, data.props.series.base)

            results[method][window] = ResultData(p, ep, tau)

    print("Проверка остатков на случайность:")
    for method, trend_results in results.items():
        print(f"\nМетод: {method}")
        for window, result in trend_results.items():
            print(f"\tОкно {window}:")
            print(f"\t\tПоворотные точки: p={result.p:.1f}, E(p)={result.ep:.1f}")
            print(f"\t\tКендалл tau: τ={result.tau:.4f}")
    

    def MSE(first: np.ndarray[float], second: np.ndarray[float]) -> float:
        return np.mean((first - second) ** 2)

    get_MSE = lambda series: MSE(series, data.props.series.true_trend)

    print("\nMSE для скользящего среднего:")
    for window in data.props.window_sizes:
        mat_mse = get_MSE(data.mat_data[window])
        mm_mse  = get_MSE(data.mm_data[window])
        
        print(f"Окно {window}:\tMAT: {mat_mse:.4f} | MM: {mm_mse:.4f}")
    

    pyplot.figure(figsize=(12, 6))
    residuals_example = data.props.series.data - data.mat_data[51]
    pyplot.plot(
        data.props.series.base,
        residuals_example,
        "b-", alpha=0.7,
        label="Остатки (SMA, окно 51)"
    )
    pyplot.axhline(y=0, color="r", linestyle="--", label="Нулевая линия")
    pyplot.title("Пример остатков после вычитания тренда")
    pyplot.xlabel("k")
    pyplot.ylabel("Остатки")
    pyplot.legend()
    pyplot.grid(True)
    pyplot.show()


def main():
    series = generate_series()
    props = BaseProps(series, [21, 51, 111], [10, 25, 55])

    mat_data = get_trend(TrendFunctionProps(
        props,
        MAT
    ))

    mm_data = get_trend(TrendFunctionProps(
        props,
        MM
    ))

    result_props = ResultProps(
        props,
        mat_data,
        mm_data
    )

    while (command := input(COMMAND_TEXT).strip()) != "exit":
        match (command):
            case "one":
                print_graph(result_props)
            case "different":
                print_graphs(result_props)
            case _:
                print("Wrong command")

    print("Обработка результатов...")
    
    checkout(result_props)


if __name__ == "__main__":
    main()
