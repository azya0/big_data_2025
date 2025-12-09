import numpy as np
from matplotlib import pyplot as plt

from base import compute_coefficients, predict_models_for_alpha
from console import print_coeff_table


def plot_task1(x, y_true, x_plot, y_true_func, alphas):
    print("ЗАДАЧА 1: БЕЗ ШУМА")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    axes = axes.ravel()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]

        models = predict_models_for_alpha(alpha, x, y_true, x_plot)

        ridge, y_ridge = models["ridge"]
        ax.plot(x_plot, y_ridge, color="blue", lw=2.5, label="Ridge")

        # печать коэфов
        df_r = compute_coefficients(ridge, "Ridge", alpha)
        print_coeff_table(df_r, "Ridge", alpha)

        if "lasso" in models:
            lasso, y_lasso = models["lasso"]
            ax.plot(x_plot, y_lasso, color="green", lw=2, linestyle="--", label="Lasso")
            df_l = compute_coefficients(lasso, "Lasso", alpha)
            print_coeff_table(df_l, "Lasso", alpha)
        else:
            ax.plot(x_plot, y_ridge, "g--", lw=2.3, label="МНК")

        ax.scatter(x, y_true, color="red", s=80, edgecolors="k")
        ax.plot(x_plot, y_true_func, "k:", lw=2, label="Истинная")

        ax.set_title(f"α = {alpha}")
        ax.set_ylim(-12, 16)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/task1_comparison.png", dpi=300)


def plot_task2(x, y_true, x_plot, y_true_func, alphas, noises):
    for noise_std in noises:

        y_noisy = y_true + np.random.normal(0, noise_std, size=y_true.shape)

        print(f"\n\n{'=' * 60}")
        print(f"ЗАДАЧА 2: шум N(0, {noise_std})")
        print(f"{'=' * 60}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        for idx, alpha in enumerate(alphas):

            ax = axes[idx]
            models = predict_models_for_alpha(alpha, x, y_noisy, x_plot)

            ridge, y_ridge = models["ridge"]
            ax.plot(x_plot, y_ridge, lw=2.5, label="Ridge")

            df_r = compute_coefficients(ridge, "Ridge", alpha)
            print_coeff_table(df_r, "Ridge", alpha)

            if alpha > 0 and "lasso" in models:
                lasso, y_lasso = models["lasso"]
                ax.plot(x_plot, y_lasso, "g--", lw=2, label="Lasso")
                df_l = compute_coefficients(lasso, "Lasso", alpha)
                print_coeff_table(df_l, "Lasso", alpha)

            ax.scatter(x, y_noisy, color="red", s=80)
            ax.plot(x_plot, y_true_func, "k--", lw=2)

            ax.set_title(f"α = {alpha}")
            ax.set_ylim(-15, 20)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"plots/task2_noise_{noise_std}.png", dpi=300)
