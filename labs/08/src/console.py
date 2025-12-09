import pandas as pd


def print_coeff_table(df_coeffs, model_name, alpha):
    df_coeffs = df_coeffs[1:]
    
    """Печать таблицы коэффициентов."""
    print(f"\n{'=' * 60}")
    print(f"Модель: {model_name}, α = {alpha}")
    print(f"{'=' * 60}")

    print(df_coeffs.to_string(index=False))

    print("\nСводка:")
    nnz = (df_coeffs["|Коэф|"] > 1e-12).sum()
    print(f"Ненулевых коэффициентов: {nnz - 1}")
    print(f"Сумма |коэф|: {df_coeffs['|Коэф|'].sum():.4e}")
    print(f"Максимальный |коэф|: {df_coeffs['|Коэф|'].max():.4e}")


def print_coefficients(model, model_name, alpha, degree=11):
    """Вывод коэффициентов модели в читаемом формате"""

    if model_name == "Ridge":
        coefs = model.named_steps["ridge"].coef_
        intercept = model.named_steps["ridge"].intercept_
    else:
        coefs = model.named_steps["lasso"].coef_
        intercept = model.named_steps["lasso"].intercept_

    coef_dict = {"Степень": [], "Коэффициент": [], "|Коэф|": []}

    coef_dict["Степень"].append("Intercept")
    coef_dict["Коэффициент"].append(intercept)
    coef_dict["|Коэф|"].append(abs(intercept))

    for _, i in enumerate(range(degree + 1)):
        if i == 0:
            continue
        coef_value = coefs[i - 1] if i - 1 < len(coefs) else 0
        coef_dict["Степень"].append(f"x^{i}")
        coef_dict["Коэффициент"].append(coef_value)
        coef_dict["|Коэф|"].append(abs(coef_value))

    df = pd.DataFrame(coef_dict)

    print(f"\n{'=' * 60}")
    print(f"Модель: {model_name}, α = {alpha}")
    print(f"{'=' * 60}")
    print(df[df["|Коэф|"] > 1e-6].to_string(index=False))

    print(f"\nСводная информация:")
    print(f"Количество ненулевых коэффициентов: {(df['|Коэф|'] > 1e-6).sum() - 1}")
    print(f"Сумма |коэффициентов|: {df['|Коэф|'].sum():.4f}")
    print(f"Максимальный |коэффициент|: {df['|Коэф|'].max():.4f}")

    return df
