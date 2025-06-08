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
    output_features: list,
    arx_params: dict,
    model_type: str, # "dynamic_arx_linear" lub "dynamic_arx_polynomial"
    plots_dir: str,
    plot_results: bool = True
):
    """
    Kompleksowa funkcja do trenowania i oceny dynamicznego modelu ARX.
    Cała logika jest tutaj, z dala od main.py.
    """
    na = arx_params.get('na', 2)
    nb = arx_params.get('nb', 2)
    nk = arx_params.get('nk', 1)
    poly_degree = arx_params.get('poly_degree', 2)

    print(f"\n--- Przygotowanie danych dla modelu dynamicznego ARX ---")
    
    active_inputs = [col for col in input_features if col in df.columns]
    active_outputs = [col for col in output_features if col in df.columns]

    X_arx, y_arx = create_arx_data(
        df=df.dropna(subset=active_inputs + active_outputs),
        input_cols=active_inputs,
        output_cols=active_outputs,
        na=na, nb=nb, nk=nk
    )

    if X_arx.empty:
        print("Nie udało się stworzyć danych ARX. Modelowanie przerwane.")
        return

    print(f"Stworzono macierz regresorów ARX o kształcie: {X_arx.shape}")

    for target_col in active_outputs:
        print(f"\n--- Modelowanie ARX dla celu: {target_col} ---")
        
        y_target = y_arx[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X_arx, y_target, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Budowanie pipeline'u modelu
        if model_type == "dynamic_arx_linear":
            model_pipeline = Pipeline([('linear_regression', LinearRegression())])
            model_name_str = f"ARX_Linear_na{na}_nb{nb}_nk{nk}"
        elif model_type == "dynamic_arx_polynomial":
            model_pipeline = Pipeline([
                ('polynomialfeatures', PolynomialFeatures(degree=poly_degree, include_bias=False)),
                ('linear_regression', LinearRegression())
            ])
            model_name_str = f"ARX_Poly_deg{poly_degree}_na{na}_nb{nb}_nk{nk}"
        else:
            print(f"Nieznany typ modelu dynamicznego: {model_type}. Pomijanie.")
            continue
            
        # Wywołanie generycznej funkcji do treningu i oceny
        _run_single_arx_evaluation(
            X_train, y_train, X_test, y_test,
            model_pipeline=model_pipeline,
            target_col=target_col,
            model_name=model_name_str,
            plots_dir=plots_dir,
            plot_results=plot_results
        )


def _run_single_arx_evaluation(X_train, y_train, X_test, y_test, model_pipeline, target_col, model_name, plots_dir, plot_results):
    """Helper function to fit, predict, and print metrics for a single ARX model run."""
    print(f"Trening modelu: {model_name}")
    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)
    
    # Tutaj wklej logikę obliczania i drukowania metryk z Twoich istniejących funkcji
    # (mean_squared_error, r2_score, calculate_adj_r2, etc.)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    n_test = len(y_test)
    try:
        p_features = model_pipeline.named_steps['polynomialfeatures'].n_output_features_
    except KeyError:
        p_features = X_train.shape[1]
    adj_r2 = calculate_adj_r2(r2, n_test, p_features)

    metrics = {'mse': mse, 'r2': r2, 'mae': mae, 'rmse': rmse, 'adj_r2': adj_r2}
    calculate_dynamic_static_metrics(y_test, predictions, metrics)

    print(f"Global MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"Global R2: {r2:.4f}, Adjusted R2: {adj_r2:.4f}")
    print(f"Dynamic Change MSE: {metrics.get('mse_dynamic_change', np.nan):.4f}, MAE: {metrics.get('mae_dynamic_change', np.nan):.4f}, R2: {metrics.get('r2_dynamic_change', np.nan):.4f}")
    print(f"Static Continuation MSE: {metrics.get('mse_static_cont', np.nan):.4f}, MAE: {metrics.get('mae_static_cont', np.nan):.4f}, R2: {metrics.get('r2_static_cont', np.nan):.4f}")

    if plot_results:
        plot_predictions_vs_actual_scatter(y_test, predictions, target_col, model_name, plots_dir)
        plot_predictions_over_samples(y_test, predictions, target_col, model_name, plots_dir)