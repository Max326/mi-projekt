import pandas as pd
# from sklearn.model_selection import train_test_split # Już niepotrzebne bezpośrednio
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from utils.plotting_utils import plot_predictions_vs_actual_scatter, plot_predictions_over_samples
from utils.data_preprocessing import prepare_and_split_data_stratified # NOWY IMPORT

# Usunięto funkcję preprocess_data_for_modeling, ponieważ jej logika jest teraz w prepare_and_split_data_stratified

def train_evaluate_random_forest_model(df: pd.DataFrame, feature_cols: list, target_col: str, plots_dir: str, plot_results: bool = True):
    """Trenuje i ocenia model Random Forest Regressor używając stratyfikowanego podziału danych."""
    X_train, X_test, y_train, y_test = prepare_and_split_data_stratified(
        df, 
        feature_cols, 
        target_col,
        test_size=0.2,
        random_state=42,
        regime_threshold=0.2
    )

    if X_train is None or X_test is None or y_train is None or y_test is None:
        print(f"Nie udało się przygotować lub podzielić danych dla {target_col}. Pomijanie Random Forest.")
        return None, {}

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
    """Trenuje i ocenia model Gradient Boosting Regressor używając stratyfikowanego podziału danych."""
    X_train, X_test, y_train, y_test = prepare_and_split_data_stratified(
        df, 
        feature_cols, 
        target_col,
        test_size=0.2,
        random_state=42,
        regime_threshold=0.2
    )

    if X_train is None or X_test is None or y_train is None or y_test is None:
        print(f"Nie udało się przygotować lub podzielić danych dla {target_col}. Pomijanie Gradient Boosting.")
        return None, {}

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