from dataclasses import dataclass

# Required PyQt6 (UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown pyplot.show())
import matplotlib.pyplot as pyplot
import numpy as np


FIELD_TO_STRING: dict[str, str] = {
    "A_i":      "Амплитуда",
    "omega_i":  "Частота",
    "phi_i":    "Начальная фаза",
    "lambda_i": "Коэффициент затухания",
    "delta_t":  "Период дескретизации сигнала",
}


@dataclass
class PronyApproximation:
    # Амплитуда
    A_i:        np.ndarray[complex]
    # Частота
    omega_i:    np.ndarray[complex]
    # Начальная фаза
    phi_i:      np.ndarray[complex]
    # Коэффициент затухания
    lambda_i:   np.ndarray[complex]
    # Период дескретизации сигнала
    delta_t:    complex


def prony(x: np.ndarray[complex], m: int, delta_t: float = 1.0) -> PronyApproximation:
    n = len(x)
    rows = n - m

    X = np.zeros((rows, m), dtype=complex)
    y = np.zeros(rows,      dtype=complex)
    x = np.array(x,         dtype=complex)
    
    for index_row in range(rows):
        for index_col in range(m):
            X[index_row, index_col] = x[m + index_row - index_col - 1]
        y[index_row] = x[m + index_row]
    
    a = np.linalg.lstsq(X, -y, rcond=None)[0]

    coeffs = np.concatenate(([1.], a))
    z = np.roots(coeffs)
    lambda_i = np.log(np.abs(z)) / delta_t
    omega_i = np.arctan2(np.imag(z), np.real(z)) / (2 * np.pi * delta_t)
    
    V = np.zeros((n, m), dtype=complex)
    
    for k in range(n):
        for r in range(m):
            V[k, r] = z[r] ** k
    
    h = np.linalg.lstsq(V, x, rcond=None)[0]
    A = np.abs(h)
    phi_i = np.arctan2(np.imag(h), np.real(h))

    return PronyApproximation(A, omega_i, phi_i, lambda_i, delta_t)


def get_model_row_from(data: PronyApproximation, n: int, m: int) -> np.ndarray[complex]:
    result: np.ndarray[complex] = np.zeros((n, ), dtype=complex)

    for result_index in range(n):
        for sum_index in range(m):
            h_i: np.ndarray[complex] = data.A_i[sum_index] * np.exp(1j * data.phi_i[sum_index])
            z_i: np.ndarray[complex] = np.exp((data.lambda_i[sum_index] + 2j * np.pi * data.omega_i[sum_index]) * data.delta_t)
            
            result[result_index] += h_i * z_i ** result_index
    
    return result


def get_model_row(n: int = 200, h: float = 0.02) -> np.ndarray[complex]:
    i = np.arange(1, n + 1)
    x = np.zeros(n)

    for k in range(1, 4):
        x += k * np.exp(-h * i / k) * np.cos(4 * np.pi * h * i * k + np.pi / k)
    
    return x


def print_graph(x_original: np.ndarray[complex], x_restored: np.ndarray[complex], n: int):
    indexes = np.arange(1, n + 1)

    pyplot.figure(figsize=(15, 10))
    pyplot.plot(indexes, x_original, label='Оригинальный',      color="red")
    pyplot.plot(indexes, x_restored, label='Восстановленный',   color="blue", linestyle='--')
    
    pyplot.legend()

    pyplot.title('Оригинальный и восстановленный модельный ряд $x_i$')
    pyplot.xlabel('i')
    pyplot.ylabel('$x_i$')

    pyplot.show()


def compare_rows(x_original: np.ndarray[complex], x_restored: np.ndarray[complex]):
    return np.mean((x_original - x_restored) ** 2)


def main():
    N: int = 200

    model_row = get_model_row(N)
    print(f"Первые 12 элементов изначального ряда:\n{model_row[:12]}")

    M: int = 6
    data: PronyApproximation = prony(model_row, M)
    
    print("\nДанные апроксимации методом Прони:")
    for key, value in data.__dict__.items():
        value_output: str = value if isinstance(value, float) else ",\t".join(map(lambda item: f"{item:.3f}", value))
        print(f"{key:<20}| {FIELD_TO_STRING[key]:<30} | {value_output}")

    restored_row = np.real(get_model_row_from(data, N, M))
    print(f"\nПервые 12 элементов восстановленного ряда:\n{restored_row[:12]}")

    print(f"\nMSE: {compare_rows(model_row, restored_row)}")

    print_graph(model_row, restored_row, N)


if __name__ == "__main__":
    main()
