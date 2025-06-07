import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def ensure_dir(directory_path: str):
    """Upewnia się, że katalog istnieje, tworząc go w razie potrzeby."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Utworzono katalog: {directory_path}")

def plot_predictions_vs_actual_scatter(y_test: pd.Series, predictions: np.ndarray, target_col: str, model_name: str, plots_dir: str):
    """Generuje i zapisuje wykres rozrzutu wartości rzeczywistych vs predykowanych."""
    ensure_dir(plots_dir)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.7, edgecolors='k', label='Predykcje')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Idealna predykcja')
    plt.xlabel('Wartości rzeczywiste')
    plt.ylabel('Wartości predykowane')
    plt.title(f'Rozrzut: {model_name} dla {target_col}')
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(plots_dir, f"scatter_pred_vs_actual_{model_name.replace(' ', '_')}_{target_col.replace(' ', '_').replace('-', '_')}.png")
    plt.savefig(plot_filename)
    print(f"Wykres rozrzutu zapisano jako: {plot_filename}")
    plt.close()

def plot_predictions_over_samples(y_test: pd.Series, predictions: np.ndarray, target_col: str, model_name: str, plots_dir: str):
    """Generuje i zapisuje wykres wartości rzeczywistych i predykowanych w funkcji indeksu próbki (tylko punkty)."""
    ensure_dir(plots_dir)
    plt.figure(figsize=(12, 7))
    sample_indices = np.arange(len(y_test))
    
    # Resetowanie indeksu y_test, aby zapewnić spójność z sample_indices, jeśli y_test ma niestandardowy indeks
    y_test_values = y_test.values 
    
    # Zaznaczanie punktów dla wartości rzeczywistych i predykowanych
    plt.scatter(sample_indices, y_test_values, label='Wartości rzeczywiste', alpha=0.8, marker='o', edgecolors='k', s=50) # Increased size
    plt.scatter(sample_indices, predictions, label='Wartości predykowane', alpha=0.8, marker='x', color='r', s=50) # Increased size
    
    plt.xlabel('Indeks próbki w zbiorze testowym')
    plt.ylabel(target_col)
    plt.title(f'Predykcje vs Rzeczywiste (punkty): {model_name} dla {target_col}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = os.path.join(plots_dir, f"scatter_pred_over_samples_{model_name.replace(' ', '_')}_{target_col.replace(' ', '_').replace('-', '_')}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Wykres (punkty) zapisano jako: {plot_filename}")
    plt.close()

def plot_outlier_visualization(
    series_data: pd.Series, 
    column_name: str, 
    plots_dir: str,
    outlier_mask: pd.Series = None, 
    method_name: str = None, 
    lower_b: float = None, 
    upper_b: float = None
):
    """
    Generuje i zapisuje wykres wizualizujący dane kolumny.
    Jeśli outlier_mask jest dostarczona, zaznacza outliery i granice.
    W przeciwnym razie, rysuje tylko surowe dane.
    """
    ensure_dir(plots_dir)
    plt.figure(figsize=(12, 7))
    
    safe_col_name = column_name.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
    plot_title = f'Wizualizacja danych dla: {column_name}'
    base_filename = f"input_data_{safe_col_name}"

    # Rysowanie wszystkich ważnych punktów danych
    valid_data_indices = series_data.notna()
    plt.scatter(series_data.index[valid_data_indices], series_data.loc[valid_data_indices], color='blue', label='Dane', alpha=0.6, s=30)

    if outlier_mask is not None and method_name and lower_b is not None and upper_b is not None:
        # Rysowanie tylko tych punktów, które są outlierami (i nie są NaN)
        outliers_to_plot = series_data[outlier_mask & series_data.notna()]
        if not outliers_to_plot.empty:
            # Nadpisz niebieskie punkty, które są outlierami, czerwonymi markerami
            plt.scatter(outliers_to_plot.index, outliers_to_plot.values, color='red', label=f'Outliery ({method_name})', alpha=0.8, s=50, marker='x', zorder=3)
        
        plt.axhline(y=lower_b, color='gray', linestyle='--', label=f'Dolna granica ({method_name}): {lower_b:.2f}')
        plt.axhline(y=upper_b, color='gray', linestyle='--', label=f'Górna granica ({method_name}): {upper_b:.2f}')
        plot_title = f'Wizualizacja Outlierów ({method_name}) dla: {column_name}'
        base_filename = f"outliers_{method_name}_{safe_col_name}"
    
    plt.title(plot_title)
    plt.xlabel('Indeks próbki')
    plt.ylabel('Wartość')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = os.path.join(plots_dir, f"{base_filename}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Wykres wizualizacji danych/outlierów zapisano jako: {plot_filename}")
    plt.close()