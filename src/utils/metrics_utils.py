import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_dynamic_static_metrics(y_true_series: pd.Series, y_pred_array: np.ndarray, metrics: dict, epsilon: float = 1e-6):
    """
    Oblicza metryki dla dynamicznych i statycznych części danych.
    Dodaje wyniki do przekazanego słownika `metrics`.
    """
    if not isinstance(y_true_series, pd.Series):
        y_true_series = pd.Series(y_true_series) # Upewnij się, że to Series dla .diff()

    if len(y_true_series) <= 1:
        for m in ['mse', 'mae', 'rmse', 'r2']:
            metrics[f'{m}_dynamic_change'] = np.nan
            metrics[f'{m}_static_cont'] = np.nan
        return

    # Używamy .values, aby uniknąć problemów z indeksami po diff() i filtrowaniu
    y_true_for_diff = y_true_series.values 
    y_pred_for_eval = y_pred_array

    # Różnice są obliczane na oryginalnej serii, ale maska jest stosowana do danych bez pierwszego elementu
    diff_abs = np.abs(np.diff(y_true_for_diff))
    mask_static_continuation = diff_abs < epsilon
    
    # Dopasowujemy y_true_eval i y_pred_eval do długości maski
    y_true_eval = y_true_for_diff[1:]
    y_pred_eval = y_pred_for_eval[1:] # Zakładamy, że y_pred_array ma tę samą długość co y_true_series

    if len(y_true_eval) != len(mask_static_continuation):
        # To nie powinno się zdarzyć, jeśli y_pred_array ma tę samą długość co y_true_series
        print("Ostrzeżenie: Niezgodność długości w calculate_dynamic_static_metrics.")
        # Ustawiamy wszystkie metryki na NaN w przypadku problemu
        for m in ['mse', 'mae', 'rmse', 'r2']:
            metrics[f'{m}_dynamic_change'] = np.nan
            metrics[f'{m}_static_cont'] = np.nan
        return

    y_true_static_cont = y_true_eval[mask_static_continuation]
    y_pred_static_cont = y_pred_eval[mask_static_continuation]

    y_true_dynamic_change = y_true_eval[~mask_static_continuation]
    y_pred_dynamic_change = y_pred_eval[~mask_static_continuation]

    if len(y_true_static_cont) > 0:
        metrics['mse_static_cont'] = mean_squared_error(y_true_static_cont, y_pred_static_cont)
        metrics['mae_static_cont'] = mean_absolute_error(y_true_static_cont, y_pred_static_cont)
        metrics['rmse_static_cont'] = np.sqrt(metrics['mse_static_cont'])
        if len(np.unique(y_true_static_cont)) > 1 and len(y_true_static_cont) >=2: # R2 wymaga co najmniej 2 próbek i wariancji w y_true
            metrics['r2_static_cont'] = r2_score(y_true_static_cont, y_pred_static_cont)
        else:
            metrics['r2_static_cont'] = np.nan # Lub 0.0 jeśli preferowane dla braku wariancji
    else:
        metrics['mse_static_cont'] = metrics['mae_static_cont'] = metrics['rmse_static_cont'] = metrics['r2_static_cont'] = np.nan
            
    if len(y_true_dynamic_change) > 0:
        metrics['mse_dynamic_change'] = mean_squared_error(y_true_dynamic_change, y_pred_dynamic_change)
        metrics['mae_dynamic_change'] = mean_absolute_error(y_true_dynamic_change, y_pred_dynamic_change)
        metrics['rmse_dynamic_change'] = np.sqrt(metrics['mse_dynamic_change'])
        if len(np.unique(y_true_dynamic_change)) > 1 and len(y_true_dynamic_change) >=2:
            metrics['r2_dynamic_change'] = r2_score(y_true_dynamic_change, y_pred_dynamic_change)
        else:
            metrics['r2_dynamic_change'] = np.nan
    else:
        metrics['mse_dynamic_change'] = metrics['mae_dynamic_change'] = metrics['rmse_dynamic_change'] = metrics['r2_dynamic_change'] = np.nan

def calculate_adj_r2(r2: float, n: int, p: int) -> float:
    """Oblicza skorygowane R-kwadrat."""
    if n - p - 1 > 0:
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return np.nan # Lub 0.0, w zależności od preferencji dla przypadków brzegowych