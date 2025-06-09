import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
from utils.plotting_utils import plot_predictions_vs_actual_scatter, plot_predictions_over_samples
from utils.data_preprocessing import prepare_and_split_data_stratified, create_arx_data
from utils.metrics_utils import calculate_dynamic_static_metrics, calculate_adj_r2

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
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    n_test = len(y_test)
    p_features = X_train.shape[1] 
    adj_r2 = calculate_adj_r2(r2, n_test, p_features)

    metrics = {'mse': mse, 'r2': r2, 'mae': mae, 'rmse': rmse, 'adj_r2': adj_r2}
    calculate_dynamic_static_metrics(y_test, predictions, metrics)
        
    print(f"\nModel Random Forest dla celu: {target_col}")
    print(f"Wybrane cechy: {feature_cols}")
    print(f"Global MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"Global R2: {r2:.4f}, Adjusted R2: {adj_r2:.4f}")
    print(f"Dynamic Change MSE: {metrics.get('mse_dynamic_change', np.nan):.4f}, MAE: {metrics.get('mae_dynamic_change', np.nan):.4f}, R2: {metrics.get('r2_dynamic_change', np.nan):.4f}")
    print(f"Static Continuation MSE: {metrics.get('mse_static_cont', np.nan):.4f}, MAE: {metrics.get('mae_static_cont', np.nan):.4f}, R2: {metrics.get('r2_static_cont', np.nan):.4f}")

    comparison_df = pd.DataFrame({'Rzeczywiste': y_test.values, 'Predykowane': predictions})
    #print("\nPrzykładowe porównanie (Random Forest):")
    #print(comparison_df.head())

    if plot_results:
        model_name = "Random_Forest"
        plot_predictions_vs_actual_scatter(y_test, predictions, target_col, model_name, plots_dir)
        plot_predictions_over_samples(y_test, predictions, target_col, model_name, plots_dir)
        
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
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    n_test = len(y_test)
    p_features = X_train.shape[1]
    adj_r2 = calculate_adj_r2(r2, n_test, p_features)

    metrics = {'mse': mse, 'r2': r2, 'mae': mae, 'rmse': rmse, 'adj_r2': adj_r2}
    calculate_dynamic_static_metrics(y_test, predictions, metrics)

    print(f"\nModel Gradient Boosting dla celu: {target_col}")
    print(f"Wybrane cechy: {feature_cols}")
    print(f"Global MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"Global R2: {r2:.4f}, Adjusted R2: {adj_r2:.4f}")
    print(f"Dynamic Change MSE: {metrics.get('mse_dynamic_change', np.nan):.4f}, MAE: {metrics.get('mae_dynamic_change', np.nan):.4f}, R2: {metrics.get('r2_dynamic_change', np.nan):.4f}")
    print(f"Static Continuation MSE: {metrics.get('mse_static_cont', np.nan):.4f}, MAE: {metrics.get('mae_static_cont', np.nan):.4f}, R2: {metrics.get('r2_static_cont', np.nan):.4f}")
    
    comparison_df = pd.DataFrame({'Rzeczywiste': y_test.values, 'Predykowane': predictions})
    #print("\nPrzykładowe porównanie (Gradient Boosting):")
    #print(comparison_df.head())

    if plot_results:
        model_name = "Gradient_Boosting"
        plot_predictions_vs_actual_scatter(y_test, predictions, target_col, model_name, plots_dir)
        plot_predictions_over_samples(y_test, predictions, target_col, model_name, plots_dir)

    return model, metrics


def train_evaluate_dynamic_arx_model(
    df: pd.DataFrame, 
    input_features: list,
    target_col: str,
    arx_params: dict,
    model_type: str,
    train_sessions: list,
    test_sessions: list,
    plots_dir: str,
    plot_results: bool = True
):
    """
    Uproszczona i poprawna wersja funkcji do trenowania modelu ARX.
    """
    na = arx_params.get('na', 2)
    nb = arx_params.get('nb', 2)
    nk = arx_params.get('nk', 1)
    poly_degree = arx_params.get('poly_degree', 2)

    print(f"\n--- Przygotowanie danych ARX dla celu: {target_col} ---")
    
    active_inputs = [col for col in input_features if col in df.columns]
    
    full_arx_df, X_cols = create_arx_data(
        df=df.dropna(subset=active_inputs + [target_col]),
        input_cols=active_inputs,
        output_cols=[target_col],
        na=na, nb=nb, nk=nk
    )

    if full_arx_df.empty:
        print(f"Nie udało się stworzyć danych ARX dla {target_col}.")
        return

    # Podział na zbiory jest teraz prosty i pewny
    train_df = full_arx_df[full_arx_df['session_id'].isin(train_sessions)]
    test_df = full_arx_df[full_arx_df['session_id'].isin(test_sessions)]

    if train_df.empty or test_df.empty:
        print(f"Błąd: Zbiór treningowy lub testowy jest pusty dla {target_col}.")
        return

    X_train, y_train = train_df[X_cols], train_df[target_col]
    X_test, y_test = test_df[X_cols], test_df[target_col]

    print(f"Dane treningowe z sesji: {train_sessions}. Liczba próbek: {len(X_train)}")
    print(f"Dane testowe z sesji: {test_sessions}. Liczba próbek: {len(X_test)}")
    
    # Budowanie pipeline'u BEZ StandardScaler, zgodnie z prośbą
    if model_type == "dynamic_arx_linear":
        pipeline = Pipeline([('linear_regression', LinearRegression())])
        model_name = f"ARX_Linear_NoScale_{target_col.replace(' ', '_')[:10]}"
    elif model_type == "dynamic_arx_polynomial":
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('linear_regression', LinearRegression())
        ])
        model_name = f"ARX_Poly_NoScale_deg{poly_degree}_{target_col.replace(' ', '_')[:10]}"
    else:
        print(f"Nieznany typ modelu dynamicznego: {model_type}")
        return

    print(f"Trening modelu: {model_name}")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Logika metryk (przeniesiona z usuniętej _run_single_arx_evaluation)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    n_test, p_features = len(y_test), X_train.shape[1]
    if n_test == 0:
        print("Brak danych w zbiorze testowym do obliczenia metryk.")
        return
    if 'poly' in pipeline.named_steps:
        p_features = pipeline.named_steps['poly'].n_output_features_
        
    adj_r2 = calculate_adj_r2(r2, n_test, p_features)
    metrics = {'mse': mse, 'r2': r2, 'mae': mae, 'rmse': rmse, 'adj_r2': adj_r2}
    calculate_dynamic_static_metrics(y_test, predictions, metrics)

    print(f"Global MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"Global R2: {r2:.4f}, Adjusted R2: {adj_r2:.4f}")
    
    if plot_results:
        plot_predictions_vs_actual_scatter(y_test, predictions, target_col, model_name, plots_dir)
        plot_predictions_over_samples(y_test, predictions, target_col, model_name, plots_dir)
