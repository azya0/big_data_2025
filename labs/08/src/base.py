import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def build_model(model_type, alpha):
    """Создает модель Ridge или Lasso."""
    if model_type == "Ridge":
        return make_pipeline(PolynomialFeatures(11), Ridge(alpha=max(alpha, 1e-15)))
    else:
        return make_pipeline(
            PolynomialFeatures(11),
            Lasso(alpha=alpha, max_iter=100000, tol=1e-5, warm_start=True),
        )


def fit_and_predict(model, x, y, x_plot):
    """Обучает модель и возвращает предсказания."""
    model.fit(x, y)
    y_pred = model.predict(x_plot)
    return y_pred, model


def predict_models_for_alpha(alpha, x, y, x_plot):
    """Строит Ridge и Lasso (если возможно) модели и делает предсказания."""

    results = {}

    # Ridge
    ridge = build_model("Ridge", alpha)
    y_ridge, ridge = fit_and_predict(ridge, x, y, x_plot)
    results["ridge"] = (ridge, y_ridge)

    # Lasso (если alpha > 0)
    lasso = build_model("Lasso", alpha)
    y_lasso, lasso = fit_and_predict(lasso, x, y, x_plot)
    results["lasso"] = (lasso, y_lasso)

    return results


def compute_coefficients(model, model_type, alpha, degree=11):
    """Формирует DataFrame со всеми коэффициентами модели."""

    if model_type == "Ridge":
        coefs = model.named_steps["ridge"].coef_
        intercept = model.named_steps["ridge"].intercept_
    else:
        coefs = model.named_steps["lasso"].coef_
        intercept = model.named_steps["lasso"].intercept_

    data = {"Степень": [], "Коэф": [], "|Коэф|": []}

    # Интерсепт
    data["Степень"].append("x^0")
    data["Коэф"].append(intercept)
    data["|Коэф|"].append(abs(intercept))

    # Остальные коэфы
    for i in range(1, degree + 1):
        v = coefs[i - 1] if i - 1 < len(coefs) else 0
        data["Степень"].append(f"x^{i - 1}")
        data["Коэф"].append(v)
        data["|Коэф|"].append(abs(v))

    df = pd.DataFrame(data)
    df["alpha"] = alpha
    df["model"] = model_type

    return df
