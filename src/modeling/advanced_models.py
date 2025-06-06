import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from utils.plotting_utils import plot_predictions_vs_actual_scatter, plot_predictions_over_samples # Importuj nowe funkcje

def preprocess_data_for_modeling(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Przygotowuje dane do modelowania: wybór kolumn, konwersja, obsługa NaN."""
    all_required_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        print(f"Brakujące kolumny w DataFrame: {missing_cols}. Pomijanie modelowania dla {target_col}.")
        return None, None

    data = df[all_required_cols].copy()
    
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    data.dropna(inplace=True)

    if data.empty or len(data) < 2:
        print(f"Brak wystarczających danych dla modelowania celu {target_col} po usunięciu NaN. Pomijanie.")
        return None, None

    X = data[feature_cols]
    y = data[target_col]

    if X.empty or len(X) < 2:
        print(f"Niewystarczająca ilość danych (X) dla {target_col} po preprocessingu. Pomijanie.")
        return None, None
    return X, y

def train_evaluate_random_forest_model(df: pd.DataFrame, feature_cols: list, target_col: str, plots_dir: str, plot_results: bool = True):
    """Trenuje i ocenia model Random Forest Regressor."""
    X, y = preprocess_data_for_modeling(df, feature_cols, target_col)
    if X is None or y is None:
        return None, {}

    actual_test_size = 0.2
    if len(X) < 5:
        print(f"Ostrzeżenie: Mała ilość danych ({len(X)} próbek) dla {target_col} (Random Forest).")
        if len(X) <= 1: return None, {}
        if int(len(X) * actual_test_size) == 0: actual_test_size = 1 / len(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=actual_test_size, random_state=42, shuffle=True)

    if X_train.empty or X_test.empty:
        print(f"Zbiór treningowy lub testowy jest pusty dla {target_col} (Random Forest). Pomijanie.")
        return None, {}

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nModel Random Forest dla celu: {target_col}")
    print(f"Wybrane cechy: {feature_cols}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    comparison_df = pd.DataFrame({'Rzeczywiste': y_test.values, 'Predykowane': predictions})
    print("\nPrzykładowe porównanie (Random Forest):")
    print(comparison_df.head())

    if plot_results:
        model_name = "Random_Forest"
        plot_predictions_vs_actual_scatter(y_test, predictions, target_col, model_name, plots_dir)
        plot_predictions_over_samples(y_test, predictions, target_col, model_name, plots_dir)
        
    metrics = {'mse': mse, 'r2': r2}
    return model, metrics

def train_evaluate_gradient_boosting_model(df: pd.DataFrame, feature_cols: list, target_col: str, plots_dir: str, plot_results: bool = True):
    """Trenuje i ocenia model Gradient Boosting Regressor."""
    X, y = preprocess_data_for_modeling(df, feature_cols, target_col)
    if X is None or y is None:
        return None, {}

    actual_test_size = 0.2
    if len(X) < 5:
        print(f"Ostrzeżenie: Mała ilość danych ({len(X)} próbek) dla {target_col} (Gradient Boosting).")
        if len(X) <= 1: return None, {}
        if int(len(X) * actual_test_size) == 0: actual_test_size = 1 / len(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=actual_test_size, random_state=42, shuffle=True)

    if X_train.empty or X_test.empty:
        print(f"Zbiór treningowy lub testowy jest pusty dla {target_col} (Gradient Boosting). Pomijanie.")
        return None, {}

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nModel Gradient Boosting dla celu: {target_col}")
    print(f"Wybrane cechy: {feature_cols}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    comparison_df = pd.DataFrame({'Rzeczywiste': y_test.values, 'Predykowane': predictions})
    print("\nPrzykładowe porównanie (Gradient Boosting):")
    print(comparison_df.head())

    if plot_results:
        model_name = "Gradient_Boosting"
        plot_predictions_vs_actual_scatter(y_test, predictions, target_col, model_name, plots_dir)
        plot_predictions_over_samples(y_test, predictions, target_col, model_name, plots_dir)

    metrics = {'mse': mse, 'r2': r2}
    return model, metrics