import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

# Turning points detection
def turning_points(series):
    s = np.array(series)
    return np.where(np.diff(np.sign(np.diff(s))) != 0)[0] + 1

# 1. Generation
def generate_series(K=500, h=0.1, seed=0):
    np.random.seed(seed)
    k = np.arange(K + 1)
    true_trend = 0.5 * np.sin(k * h)
    noise = np.random.normal(0, 1, size=k.size)
    x = true_trend + noise
    return k, true_trend, x

# 2. EMA
def compute_ema(series, alphas):
    return {alpha: pd.Series(series).ewm(alpha=alpha, adjust=False).mean() for alpha in alphas}

# 3. Comparison
def compare_trends(true_trend, ema_dict):
    results = {}
    for alpha, ema in ema_dict.items():
        mse = mean_squared_error(true_trend, ema.values)
        corr = np.corrcoef(true_trend, ema.values)[0, 1]
        kendall, _ = stats.kendalltau(true_trend, ema.values)
        diff = ema.values - true_trend
        results[alpha] = dict(mse=mse, corr=corr, kendall=kendall, diff=diff)
    return results

# 4. Fourier
def fourier_spectrum(series, h):
    n = len(series)
    fft_vals = np.fft.rfft(series)
    fft_freqs = np.fft.rfftfreq(n, d=h)
    amp = np.abs(fft_vals) / n
    main_idx = np.argmax(amp[1:]) + 1 if len(amp) > 1 else 0
    return fft_freqs, amp, fft_freqs[main_idx]

# 5. Runs test

def runs_test(residuals):
    med = np.median(residuals)
    signs = residuals > med
    runs = 1 + np.sum(signs[:-1] != signs[1:])
    n1 = np.sum(signs)
    n2 = len(signs) - n1
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    expected = 1 + (2 * n1 * n2) / (n1 + n2)
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z = (runs - expected) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

# Tests

def test_residuals(series, ema_dict):
    results = {}
    for alpha, ema in ema_dict.items():
        resid = np.array(series) - ema.values
        t_stat, p_t = stats.ttest_1samp(resid, 0)
        sh_stat, p_sh = stats.shapiro(resid)
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_p = lb['lb_pvalue'].iloc[0]
        z_runs, p_runs = runs_test(resid)
        results[alpha] = dict(mean=resid.mean(), std=resid.std(ddof=1), p_t=p_t,
                              p_sh=p_sh, ljung_p=lb_p, runs_p=p_runs)
    return results

# Plotting

def print_trend_comparison(k, x, true_trend, ema_dict, alphas, figsize=(14, 12)):
    n_alphas = len(alphas)
    fig, axes = plt.subplots(n_alphas + 1, 1, figsize=figsize)

    ax = axes[0]
    ax.plot(k, x, 'b-', alpha=0.5, linewidth=0.8, label='Original series')
    ax.plot(k, true_trend, 'r-', linewidth=2, label='True trend')
    ax.set_title('Original and true trend')
    ax.legend(); ax.grid(True, alpha=0.3)

    for i, alpha in enumerate(alphas, 1):
        ax = axes[i]
        ema = ema_dict[alpha]
        ax.plot(k, x, 'b-', alpha=0.3, linewidth=0.5)
        ax.plot(k, true_trend, 'r-', linewidth=1.5)
        ax.plot(k, ema.values, 'g-', linewidth=2, label=f'EMA α={alpha}')

        mse = mean_squared_error(true_trend, ema.values)
        corr = np.corrcoef(true_trend, ema.values)[0, 1]
        kendall, _ = stats.kendalltau(true_trend, ema.values)

        ax.text(0.02, 0.95,
                f'MSE: {mse:.4f}\nCorr: {corr:.4f}\nKendall: {kendall:.4f}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_title(f'Trend comparison α={alpha}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trend_comparison.png')

    # Errors plot
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8)); axes2 = axes2.flatten()
    for i, alpha in enumerate(alphas):
        ax = axes2[i]
        ema = ema_dict[alpha]
        error = ema.values - true_trend
        ax.plot(k, error, 'm-', linewidth=1)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'Error α={alpha}')

        mean_err = error.mean(); std_err = error.std()
        ax.text(0.02, 0.95, f'Mean:{mean_err:.4f}\nStd:{std_err:.4f}', transform=ax.transAxes, va='top')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trend_comparison_errors.png")

# Main

def main():
    k, true_trend, x = generate_series()
    alphas = [0.01, 0.05, 0.1, 0.3]
    ema_dict = compute_ema(x, alphas)
    comparison = compare_trends(true_trend, ema_dict)
    print("Comparison:")
    for a, r in comparison.items():
        print(f"α={a}: MSE={r['mse']:.4f}, Corr={r['corr']:.4f}, Kendall={r['kendall']:.4f}")
    print_trend_comparison(k, x, true_trend, ema_dict, alphas)
    freqs, amp, main_freq = fourier_spectrum(x, 0.1)
    print(f"Main freq: {main_freq:.5f}")
    residual_tests = test_residuals(x, ema_dict)
    for a, r in residual_tests.items():
        print(f"α={a}: mean={r['mean']:.4f}, std={r['std']:.4f}, p_t={r['p_t']:.4f}, p_sh={r['p_sh']:.4f}, ljung_p={r['ljung_p']}, runs_p={r['runs_p']:.4f}")

if __name__ == '__main__':
    main()
