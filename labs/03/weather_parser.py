from typing import Callable, Any

from bs4 import BeautifulSoup
import csv
import requests
from tqdm import tqdm


def parse_page(soup: BeautifulSoup) -> list[float]:
    result: list[float] = []

    table = soup.find("table")

    if table is None:
        raise Exception("Parser miss table")
    
    for row in table.find_all("tr")[2:]:
        cells = row.find_all("td")

        try:
            result.append(float(cells[2].get_text(strip=True)))
        except ValueError:
            break

    return result


def get_htmls(url: Callable[[Any], str]) -> list[list[float]]:
    result: list[list[float]] = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for month in (bar := tqdm(range(1, 13), desc="Получение данных по месяцам...")):
        full_url = url(month)

        bar.set_description_str(full_url)

        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        
        result.append(parse_page(BeautifulSoup(response.text, 'html.parser')))
    
    return result


def save_data_as_csv(data: list[list[float]], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def main():
    baseurl: str = "http://www.pogodaiklimat.ru"

    print(f"Парсер сайта: {baseurl}")
    id:     int = int(input("Введите id города: "))
    year:   int = int(input("Введите год: "))

    url = lambda month: f"{baseurl}/monitor.php?id={id}&month={month}&year={year}"

    data = get_htmls(url)

    save_data_as_csv(data, f"weather_{id}_{year}.csv")


if __name__ == "__main__":
    main()
