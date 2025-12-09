import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
import pandas as pd
import os

from plot import plot_task1, plot_task2


def generate_data():
    """Генерация исходных данных."""
    x = np.array([-2, -1, 0, 1, 2]).reshape(-1, 1)
    y_true = np.array([-7, 0, 1, 2, 9])
    x_plot = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)
    y_true_func = x_plot ** 3 + 1
    return x, y_true, x_plot, y_true_func


# ======================================================================
#  Анализ коэффициентов
# ======================================================================


def analyze_coefficients(x, y_true, alphas):
    """
    Анализ коэффициентов Ridge и Lasso:
    - случай без шума
    - случай с шумом N(0, 0.1)
    - вывод всех коэффициентов 0..11 степени (без отсечения)
    """

    print("\n\n" + "=" * 80)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ КОЭФФИЦИЕНТОВ (БЕЗ ШУМА + С ШУМОМ)")
    print("=" * 80)

    # ---------------------------------------------------------
    # Подготовка данных
    # ---------------------------------------------------------
    degree = 11
    noise_std = 0.1
    y_noisy = y_true + np.random.normal(0, noise_std, size=y_true.shape)

    # Точка, где будем хранить таблицы
    coef_ridge_clean = pd.DataFrame()
    coef_lasso_clean = pd.DataFrame()
    coef_ridge_noisy = pd.DataFrame()
    coef_lasso_noisy = pd.DataFrame()

    # ---------------------------------------------------------
    # Обработка для каждого alpha
    # ---------------------------------------------------------
    for alpha in alphas:

        # ------- Ridge: без шума -------
        ridge_clean = make_pipeline(
            PolynomialFeatures(degree), Ridge(alpha=max(alpha, 1e-15))
        )
        ridge_clean.fit(x, y_true)
        coef_ridge_clean.loc["x^0", f"Ridge_α={alpha}"] = ridge_clean.named_steps[
            "ridge"
        ].intercept_
        for i in range(1, degree + 1):
            coef_ridge_clean.loc[f"x^{i}", f"Ridge_α={alpha}"] = (
                ridge_clean.named_steps["ridge"].coef_[i - 1]
            )

        # ------- Ridge: с шумом -------
        ridge_noisy = make_pipeline(
            PolynomialFeatures(degree), Ridge(alpha=max(alpha, 1e-15))
        )
        ridge_noisy.fit(x, y_noisy)
        coef_ridge_noisy.loc["x^0", f"Ridge_α={alpha}"] = ridge_noisy.named_steps[
            "ridge"
        ].intercept_
        for i in range(1, degree + 1):
            coef_ridge_noisy.loc[f"x^{i}", f"Ridge_α={alpha}"] = (
                ridge_noisy.named_steps["ridge"].coef_[i - 1]
            )

        # ------- Lasso -------
        # без шума
        lasso_clean = make_pipeline(
            PolynomialFeatures(degree), Lasso(alpha=alpha, max_iter=50000, tol=1e-5)
        )
        lasso_clean.fit(x, y_true)
        coef_lasso_clean.loc["x^0", f"Lasso_α={alpha}"] = lasso_clean.named_steps[
            "lasso"
        ].intercept_
        for i in range(1, degree + 1):
            coef_lasso_clean.loc[f"x^{i}", f"Lasso_α={alpha}"] = (
                lasso_clean.named_steps["lasso"].coef_[i - 1]
            )

        # с шумом
        lasso_noisy = make_pipeline(
            PolynomialFeatures(degree), Lasso(alpha=alpha, max_iter=50000, tol=1e-5)
        )
        lasso_noisy.fit(x, y_noisy)
        coef_lasso_noisy.loc["x^0", f"Lasso_α={alpha}"] = lasso_noisy.named_steps[
            "lasso"
        ].intercept_
        for i in range(1, degree + 1):
            coef_lasso_noisy.loc[f"x^{i}", f"Lasso_α={alpha}"] = (
                lasso_noisy.named_steps["lasso"].coef_[i - 1]
            )

    # ---------------------------------------------------------
    # Вывод результатов
    # ---------------------------------------------------------

    pd.set_option("display.float_format", "{: .6e}".format)

    print("\n\n" + "-" * 80)
    print("Ridge — коэффициенты без шума")
    print("-" * 80)
    print(coef_ridge_clean)

    print("\n\n" + "-" * 80)
    print("Ridge — коэффициенты с шумом N(0, 0.1):")
    print("-" * 80)
    print(coef_ridge_noisy)

    print("\n\n" + "-" * 80)
    print("Lasso — коэффициенты без шума (все степени 0..11):")
    print("-" * 80)
    print(coef_lasso_clean)

    print("\n\n" + "-" * 80)
    print("Lasso — коэффициенты с шумом N(0, 0.1):")
    print("-" * 80)
    print(coef_lasso_noisy)


# ======================================================================
#  Главная функция
# ======================================================================


def main():
    alphas = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
    noises = [0.1, 0.2, 0.3]
    np.random.seed(42)

    x, y_true, x_plot, y_true_func = generate_data()

    os.makedirs("plots", exist_ok=True)

    plot_task1(x, y_true, x_plot, y_true_func, alphas)
    plot_task2(x, y_true, x_plot, y_true_func, alphas, noises)
    analyze_coefficients(x, y_true, alphas)


if __name__ == "__main__":
    main()
