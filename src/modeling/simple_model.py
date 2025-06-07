import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from utils.plotting_utils import plot_predictions_vs_actual_scatter, plot_predictions_over_samples
from utils.data_preprocessing import prepare_and_split_data_stratified
from sklearn.preprocessing import PolynomialFeatures # NOWY IMPORT
from sklearn.pipeline import Pipeline # NOWY IMPORT

def train_evaluate_linear_model(df: pd.DataFrame, feature_cols: list, target_col: str, plots_dir: str, plot_results: bool = True):
    """
    Trenuje i ocenia model regresji liniowej używając stratyfikowanego podziału danych.

    Args:
        df (pd.DataFrame): Ramka danych zawierająca dane.
        feature_cols (list): Lista nazw kolumn cech.
        target_col (str): Nazwa kolumny docelowej.
        plots_dir (str): Katalog do zapisywania wykresów.
        plot_results (bool): Czy generować wykresy.

    Returns:
        tuple: (wytrenowany model, słownik z metrykami) lub (None, {}) jeśli błąd.
    """
    X_train, X_test, y_train, y_test = prepare_and_split_data_stratified(
        df, 
        feature_cols, 
        target_col,
        test_size=0.2, 
        random_state=42,
        regime_threshold=0.2 # Próg do wykrywania skoku temperatury
    )

    if X_train is None or X_test is None or y_train is None or y_test is None:
        print(f"Nie udało się przygotować lub podzielić danych dla {target_col}. Pomijanie modelu liniowego.")
        return None, {}
    
    if X_train.empty or X_test.empty:
        print(f"Zbiór treningowy lub testowy jest pusty dla {target_col} po podziale. Pomijanie modelu liniowego.")
        return None, {}

    # Trenowanie modelu
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ewaluacja modelu
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nModel Liniowy dla celu: {target_col}")
    print(f"Wybrane cechy: {feature_cols}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Porównanie kilku pierwszych predykcji z wartościami rzeczywistymi
    comparison_df = pd.DataFrame({'Rzeczywiste': y_test.values, 'Predykowane': predictions})
    print("\nPrzykładowe porównanie wartości rzeczywistych i predykowanych (zbiór testowy):")
    print(comparison_df.head())

    if plot_results:
        model_name = "Linear_Regression"
        plot_predictions_vs_actual_scatter(y_test, predictions, target_col, model_name, plots_dir)
        plot_predictions_over_samples(y_test, predictions, target_col, model_name, plots_dir)
        
    metrics = {'mse': mse, 'r2': r2}
    return model, metrics

def train_evaluate_polynomial_model(df: pd.DataFrame, feature_cols: list, target_col: str, plots_dir: str, plot_results: bool = True, degree: int = 2):
    """
    Trenuje i ocenia model regresji wielomianowej używając stratyfikowanego podziału danych.

    Args:
        df (pd.DataFrame): Ramka danych zawierająca dane.
        feature_cols (list): Lista nazw kolumn cech.
        target_col (str): Nazwa kolumny docelowej.
        plots_dir (str): Katalog do zapisywania wykresów.
        plot_results (bool): Czy generować wykresy.
        degree (int): Stopień wielomianu.

    Returns:
        tuple: (wytrenowany model, słownik z metrykami) lub (None, {}) jeśli błąd.
    """
    X_train, X_test, y_train, y_test = prepare_and_split_data_stratified(
        df,
        feature_cols,
        target_col,
        test_size=0.2,
        random_state=42,
        regime_threshold=0.2
    )

    if X_train is None or X_test is None or y_train is None or y_test is None:
        print(f"Nie udało się przygotować lub podzielić danych dla {target_col}. Pomijanie modelu wielomianowego.")
        return None, {}

    if X_train.empty or X_test.empty:
        print(f"Zbiór treningowy lub testowy jest pusty dla {target_col} po podziale. Pomijanie modelu wielomianowego.")
        return None, {}

    # Tworzenie pipeline: cechy wielomianowe -> regresja liniowa
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    
    model = Pipeline([
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression)
    ])

    # Trenowanie modelu
    model.fit(X_train, y_train)

    # Ewaluacja modelu
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nModel Wielomianowy (stopień {degree}) dla celu: {target_col}")
    print(f"Wybrane cechy: {feature_cols}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    comparison_df = pd.DataFrame({'Rzeczywiste': y_test.values, 'Predykowane': predictions})
    print("\nPrzykładowe porównanie wartości rzeczywistych i predykowanych (zbiór testowy):")
    print(comparison_df.head())

    if plot_results:
        model_name = f"Polynomial_Regression_Deg{degree}"
        plot_predictions_vs_actual_scatter(y_test, predictions, target_col, model_name, plots_dir)
        plot_predictions_over_samples(y_test, predictions, target_col, model_name, plots_dir)

    metrics = {'mse': mse, 'r2': r2}
    return model, metrics