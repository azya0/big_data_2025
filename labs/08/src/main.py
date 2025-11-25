import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
import os

# Исходные данные
x = np.array([-2, -1, 0, 1, 2]).reshape(-1, 1)
y_true = np.array([-7, 0, 1, 2, 9])

# Точки для гладкого графика
x_plot = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)

# Истинная функция: y = x^3 + 1
y_true_func = x_plot ** 3 + 1

# Список alpha
alphas = [0, 0.001, 0.01, 0.1, 1.0, 10.0]

# ------------------- Задача 1: Без шума -------------------

fig, axes = plt.subplots(2, 3, figsize=(19, 11))
axes = axes.ravel()

for idx, alpha in enumerate(alphas):
    ax = axes[idx]

    ridge_model = make_pipeline(PolynomialFeatures(11), Ridge(alpha=max(alpha, 1e-15)))  # защита от alpha=0
    ridge_model.fit(x, y_true)
    y_ridge = ridge_model.predict(x_plot)
    ax.plot(x_plot, y_ridge, color='blue', lw=2.7, label='Ridge', alpha=0.9)

    if alpha == 0:
        ax.plot(x_plot, y_ridge, color='green', lw=2.7, linestyle='--', label='МНК (Lasso не поддерживает α=0)',
                alpha=0.9)
        title = 'α = 0 (МНК)'
    else:
        lasso_model = make_pipeline(
            PolynomialFeatures(11),
            Lasso(alpha=alpha, max_iter=100000, tol=1e-5, warm_start=True)
        )
        lasso_model.fit(x, y_true)
        y_lasso = lasso_model.predict(x_plot)
        ax.plot(x_plot, y_lasso, color='green', lw=2.5, linestyle='--', label='Lasso', alpha=0.85)
        title = f'α = {alpha}'

    ax.scatter(x, y_true, color='red', s=100, zorder=5, edgecolors='k', label='Данные')
    ax.plot(x_plot, x_plot.ravel() ** 3 + 1, 'k:', lw=2.5, label='Истинная: x³+1')

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylim(-12, 16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

plt.suptitle("Задача 1: Сравнение Ridge и Lasso при p=11 (без шума)", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("plots/task1_comparison.png", dpi=300, bbox_inches='tight')
plt.show(block=False)

# ------------------- Задача 2: С шумом -------------------
noises = [0.1, 0.2, 0.3]
np.random.seed(42)  # для воспроизводимости

for noise_std in noises:
    y_noisy = y_true + np.random.normal(0, noise_std, size=y_true.shape)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]

        # Ridge
        ridge_pipe = make_pipeline(PolynomialFeatures(11), Ridge(alpha=alpha))
        ridge_pipe.fit(x, y_noisy)
        y_ridge = ridge_pipe.predict(x_plot)

        ax.scatter(x, y_noisy, color='red', s=80, label='Данные с шумом', zorder=5)
        ax.plot(x_plot, y_true_func, 'k--', lw=2, label='Истинная: x³+1')
        ax.plot(x_plot, y_ridge, color='blue', lw=2.5, label='Ridge')

        # Lasso (кроме alpha=0)
        if alpha > 0:
            lasso_pipe = make_pipeline(PolynomialFeatures(11),
                                       Lasso(alpha=alpha, max_iter=20000, tol=1e-4))
            lasso_pipe.fit(x, y_noisy)
            y_lasso = lasso_pipe.predict(x_plot)
            ax.plot(x_plot, y_lasso, color='green', lw=2, alpha=0.8, label='Lasso')

        ax.set_title(f"α = {alpha}" + (" (Ridge только)" if alpha == 0 else " (Ridge + Lasso)"))
        ax.set_ylim(-15, 20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.suptitle(f"Задача 2: Шум N(0, {noise_std}) — Ridge и Lasso (p=11)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"plots/task2_noise_{noise_std}.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
