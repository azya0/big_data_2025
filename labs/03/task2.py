import matplotlib.pyplot as pyplot
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


INPUT_COMMAND_TEST: str = """
print - Показать столбик температур (DEBUG)
0 - Визуализация исходных данных
1 - Анализ тренда, сезонности и остатков
2 - Дополнительный анализ: скользящее среднее для тренда
3 - Амплитудный спектр Фурье
exit - Выход
"""


def process_error(old_function):
    def new_function(*args, **kwargs):
        try:
            return old_function(*args, **kwargs)
        except (KeyboardInterrupt, EOFError):
            return
    
    return new_function


def parse_csv(filename: str, year: str) -> pd.Series:
    assert filename.split(".")[-1] == "csv"

    result: dict[str, float] = {}

    with open(filename, "r", encoding="utf-8") as file:
        for month, line in enumerate(file):
            for day, value in enumerate(map(float, line.strip().split(","))):
                result[f"{year:0>4}-{month + 1:0>2}-{day + 1:0>2}"] = value

    return pd.Series(result)


def get_data() -> pd.DataFrame:
    def get_filename(data: str) -> str:
        result: str = input(data)

        assert result[-4:] == ".csv"
        assert result.count("_") == 2

        return result

    def get_year_from_filename(filename: str) -> str:
        reversed = filename[::-1]

        start:  int = len(filename) - reversed.find("_")
        end:    int = len(filename) - reversed.find(".")

        return filename[start:end - 1]
    
    first:  str = get_filename("Путь к первому файлу: ")
    second: str = get_filename("Путь ко второму файлу: ")

    # first = "weather_27333_2023.csv"
    # second = "weather_27333_2024.csv"

    df_f: pd.Series = parse_csv(first, get_year_from_filename(first))
    df_s: pd.Series = parse_csv(second, get_year_from_filename(second))

    concated = pd.concat([df_f, df_s])

    df = pd.DataFrame({
        "date": list(concated.keys()),
        "temp": concated.values
    })

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.asfreq("D") 

    return df


@process_error
def process_data(data: pd.DataFrame):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(data["temp"], model="additive", period=365)

    # Проверка тренда
    from scipy.stats import linregress
    trend = decomposition.trend.dropna()
    slope, _, _, p_value, _ = linregress(range(len(trend)), trend)
    if p_value < 0.05:
        print("Статистически значимый тренд (p-value < 0.05)")
    else:
        print("Нет статистически значимого тренда")

    print("Частота: ", slope)

    # Анализ сезонности по месяцам
    seasonal_amplitude = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()

    print("\n" + "=" * 30)
    print("СЕЗОННАЯ КОМПОНЕНТА ТЕМПЕРАТУР")
    print("=" * 30)

    months = [
        "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
        "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"
    ]

    print(f"{"Месяц":<15} {"Температура":<12}")
    print("-" * 30)

    for month_num in range(1, 13):
        temp = seasonal_amplitude[month_num]
        deviation = f"{temp:+.2f}°C"
        
        print(f"{months[month_num-1]:<15} {deviation:<12}")

    print("-" * 30)


def print_base_graph(df: pd.DataFrame):
    pyplot.figure(figsize=(15, 6))
    pyplot.plot(df.index, df["temp"])
    pyplot.title("Среднесуточная температура")
    pyplot.xlabel("Дата")
    pyplot.ylabel("Температура (°C)")
    pyplot.grid(True)
    pyplot.show()


def print_trend_graph(df: pd.DataFrame):
    # Анализ тренда, сезонности и остатков

    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df["temp"], model="additive", period=365)

    pyplot.figure(figsize=(15, 15))
    pyplot.subplot(4, 1, 1)
    pyplot.plot(decomposition.observed)
    pyplot.title("Исходный ряд")

    pyplot.subplot(4, 1, 2)
    pyplot.plot(decomposition.trend)
    pyplot.title("Тренд")

    pyplot.subplot(4, 1, 3)
    pyplot.plot(decomposition.seasonal)
    pyplot.title("Сезонная компонента")

    pyplot.subplot(4, 1, 4)
    pyplot.scatter(decomposition.resid.index, decomposition.resid, s=30, alpha=0.5)
    pyplot.title("Остатки")
    pyplot.tight_layout()
    pyplot.show()


def print_moving_avg_graph(df: pd.DataFrame):
    WINDOW = 30
    df["moving_avg"] = df["temp"].rolling(window=WINDOW).mean()

    pyplot.figure(figsize=(15, 6))
    pyplot.plot(df.index, df["temp"], alpha=0.7, label="Исходные данные")
    pyplot.plot(df.index, df["moving_avg"], label=f"Скользящее среднее ({WINDOW} дней)")
    pyplot.title("Выделение тренда методом скользящего среднего")
    pyplot.legend()
    pyplot.show()


def print_fourier_spectrum(df: pd.DataFrame):
    # Для тайп-хинтв в vs-code
    def get_axes_by_index(data: np.ndarray[np.ndarray[Axes]], i: int, j: int) -> Axes:
        return data[i, j]
    
    data = np.asarray(df)
    data = data[~np.isnan(data)]
    n = len(data)
    
    if n == 0:
        print("Нет данных для анализа")
        return None
    
    window = np.hanning(n)
    data_windowed = data * window
    
    fft_result = np.fft.fft(data_windowed)
    
    amplitudes = np.abs(fft_result)
    
    amplitudes_half = amplitudes[:n//2]
    
    freqs = np.fft.fftfreq(n, d=1.0)[:n//2]
    
    valid_indices = np.where(freqs > 0.001)[0]
    if len(valid_indices) > 0:
        main_freq_idx = valid_indices[np.argmax(amplitudes_half[valid_indices])]
        main_freq = freqs[main_freq_idx]
        main_amplitude = amplitudes_half[main_freq_idx]
        period = 1 / main_freq if main_freq > 0 else np.inf
    else:
        main_freq = 0
        main_amplitude = 0
        period = np.inf
    
    axes: np.ndarray[np.ndarray[Axes]]
    _, axes = pyplot.subplots(2, 2, figsize=(15, 10))

    get_axes = lambda i, j: get_axes_by_index(axes, i, j)
    
    get_axes(0, 0).plot(data, color="blue", alpha=0.7, linewidth=1)
    get_axes(0, 0).set_title("Исходный временной ряд температуры", fontsize=12, fontweight="bold")
    get_axes(0, 0).set_xlabel("Дни")
    get_axes(0, 0).set_ylabel("Температура")
    get_axes(0, 0).grid(True, alpha=0.3)
    
    get_axes(0, 1).plot(freqs, amplitudes_half, color="red", linewidth=1.5)
    get_axes(0, 1).set_title("Амплитудный спектр Фурье", fontsize=12, fontweight="bold")
    get_axes(0, 1).set_xlabel("Частота (1/день)")
    get_axes(0, 1).set_ylabel("Амплитуда")
    get_axes(0, 1).grid(True, alpha=0.3)
    
    if main_freq > 0:
        get_axes(0, 1).axvline(
            x=main_freq,
            color="green",
            linestyle="--",
            alpha=0.7, 
            label=f"Главная частота: {main_freq:.8f}"
        )

        get_axes(0, 1).plot(
            main_freq,
            main_amplitude,
            "go",
            markersize=10, 
            label=f"Амплитуда: {main_amplitude:.2f}"
        )
        
        get_axes(0, 1).legend()
    
    get_axes(1, 0).semilogy(freqs, amplitudes_half, color="purple", linewidth=1.5)
    get_axes(1, 0).set_title(
        "Амплитудный спектр (логарифмическая шкала)", 
        fontsize=12, fontweight="bold"
    )
    get_axes(1, 0).set_xlabel("Частота (1/день)")
    get_axes(1, 0).set_ylabel("Амплитуда (log)")
    get_axes(1, 0).grid(True, alpha=0.3)
    
    if main_freq > 0:
        get_axes(1, 0).axvline(x=main_freq, color="green", linestyle="--", alpha=0.7)
        get_axes(1, 0).plot(main_freq, main_amplitude, "go", markersize=10)
    
    valid_amps = amplitudes_half.copy()
    valid_amps[0] = 0
    
    top_indices = np.argsort(valid_amps)[-5:][::-1]
    top_frequencies = freqs[top_indices]
    top_amplitudes = amplitudes_half[top_indices]
    top_periods = 1 / top_frequencies
    
    results_text = "Топ-5 частот:\n"
    for i, (freq, amp, period) in enumerate(zip(top_frequencies, top_amplitudes, top_periods)):
        results_text += f"{i+1}. Частота: {freq:.8f} 1/день, "
        results_text += f"Период: {period:.1f} дней, "
        results_text += f"Амплитуда: {amp:.2f}\n"
    
    results_text += f"\nВсего точек: {n}\n"
    results_text += f"Главная частота: {main_freq:.8f} 1/день\n"
    results_text += f"Период главной частоты: {period:.1f} дней"
    
    get_axes(1, 1).text(
        0.05, 0.95,
        results_text,
        transform=get_axes(1, 1).transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    get_axes(1, 1).set_title("Результаты анализа", fontsize=12, fontweight="bold")
    get_axes(1, 1).axis("off")
    
    pyplot.tight_layout()
    pyplot.show()


def print_data(df: pd.DataFrame):
    print()
    print(list := df["temp"].tolist())
    print(f"{len(list)} dots")
    print()


@process_error
def process_command(data: pd.DataFrame) -> bool | None:    
    command = input(INPUT_COMMAND_TEST)
    
    match (command):
        case "print":
            print_data(data)
        case "0":
            print_base_graph(data)
        case "1":
            print_trend_graph(data)
        case "2":
            print_moving_avg_graph(data)
        case "3":
            print_fourier_spectrum(data)
        case "exit":
            return True
        case _:
            print(f"Команды \"{command}\" нет в списке!")


def main():
    data = get_data()
    process_data(data)
    
    while True:
        if process_command(data):
            break


if __name__ == "__main__":
    main()
