import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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
    plt.scatter(sample_indices, y_test_values, label='Wartości rzeczywiste', alpha=0.8, marker='o', edgecolors='k')
    plt.scatter(sample_indices, predictions, label='Wartości predykowane', alpha=0.8, marker='x', color='r')
    
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