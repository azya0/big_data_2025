import matplotlib.pyplot as pyplot
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


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


def process_data(data: pd.DataFrame):
    decomposition = seasonal_decompose(data["temp"], model="additive", period=365)

    # Проверка тренда
    from scipy.stats import linregress
    trend = decomposition.trend.dropna()
    slope, intercept, r_value, p_value, std_err = linregress(range(len(trend)), trend)
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

    decomposition = seasonal_decompose(df['temp'], model='additive', period=365)

    pyplot.figure(figsize=(15, 15))
    pyplot.subplot(4, 1, 1)
    pyplot.plot(decomposition.observed)
    pyplot.title('Исходный ряд')

    pyplot.subplot(4, 1, 2)
    pyplot.plot(decomposition.trend)
    pyplot.title('Тренд')

    pyplot.subplot(4, 1, 3)
    pyplot.plot(decomposition.seasonal)
    pyplot.title('Сезонная компонента')

    pyplot.subplot(4, 1, 4)
    pyplot.scatter(decomposition.resid.index, decomposition.resid, s=30, alpha=0.5)
    pyplot.title('Остатки')
    pyplot.tight_layout()
    pyplot.show()


def print_moving_avg_graph(df: pd.DataFrame):
    # Скользящее среднее для тренда

    WINDOW = 30
    df['moving_avg'] = df['temp'].rolling(window=WINDOW).mean()

    pyplot.figure(figsize=(15, 6))
    pyplot.plot(df.index, df['temp'], alpha=0.7, label='Исходные данные')
    pyplot.plot(df.index, df['moving_avg'], label=f'Скользящее среднее ({WINDOW} дней)')
    pyplot.title('Выделение тренда методом скользящего среднего')
    pyplot.legend()
    pyplot.show()


def main():
    data = get_data()
    process_data(data)

    while True:
        command = input("0 - Визуализация исходных данных\n1 - Анализ тренда, сезонности и остатков\n2 - Дополнительный анализ: скользящее среднее для тренда\nexit - Выход\n")
        
        match (command):
            case "0":
                print_base_graph(data)
            case "1":
                print_trend_graph(data)
            case "2":
                print_moving_avg_graph(data)
            case "exit":
                break
            case _:
                print(f"Команды \"{command}\" нет в списке!")


if __name__ == "__main__":
    main()
