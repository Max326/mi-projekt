import pandas as pd
import numpy as np
import os
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_data_for_csv(df: pd.DataFrame, target_columns: List[str]) -> Dict[str, pd.Series]:
    numeric_df = df.select_dtypes(include=[np.number])
    all_correlations_data: Dict[str, pd.Series] = {}
    for target_col in target_columns:
        if target_col not in numeric_df.columns:
            print(f"Ostrzeżenie: Kolumna docelowa '{target_col}' nie została znaleziona w danych numerycznych.")
            continue
        correlations_series = numeric_df.corr()[target_col]
        correlations_series = correlations_series.drop(target_col, errors='ignore')
        sorted_correlations = correlations_series.sort_values(ascending=False)
        all_correlations_data[target_col] = sorted_correlations
    return all_correlations_data

def aggregate_daily_correlations_to_summary_files(
    sheet_names_processed: List[str], 
    target_columns: List[str], 
    input_csv_dir: str, 
    output_summary_dir: str
) -> None:
    os.makedirs(output_summary_dir, exist_ok=True)
    all_correlations_side_A = []
    all_correlations_side_B = []
    for sheet_name in sheet_names_processed:
        for target_col in target_columns:
            safe_target_name = target_col.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
            csv_filename = os.path.join(input_csv_dir, f"correlations_{sheet_name}_{safe_target_name}.csv")
            if os.path.exists(csv_filename):
                try:
                    daily_corr_df = pd.read_csv(csv_filename)
                    daily_corr_df['Day'] = sheet_name 
                    if "strona A" in target_col:
                        all_correlations_side_A.append(daily_corr_df)
                    elif "strona B" in target_col:
                        all_correlations_side_B.append(daily_corr_df)
                except pd.errors.EmptyDataError:
                    print(f"Ostrzeżenie: Plik CSV {csv_filename} jest pusty i zostanie pominięty.")
                except Exception as e:
                    print(f"Błąd podczas wczytywania pliku {csv_filename}: {e}")
            else:
                print(f"Ostrzeżenie: Nie znaleziono pliku CSV {csv_filename}. Pomijam.")
    if all_correlations_side_A:
        summary_df_A = pd.concat(all_correlations_side_A, ignore_index=True)
        summary_df_A = summary_df_A.sort_values(by=['Feature', 'Day'])
        summary_filename_A = os.path.join(output_summary_dir, "summary_correlations_strona_A.csv")
        summary_df_A.to_csv(summary_filename_A, index=False, float_format='%.4f')
        print(f"Zapisano zbiorczy plik korelacji dla strony A: {summary_filename_A}")
    else:
        print("Nie znaleziono danych korelacyjnych dla strony A do agregacji.")
    if all_correlations_side_B:
        summary_df_B = pd.concat(all_correlations_side_B, ignore_index=True)
        summary_df_B = summary_df_B.sort_values(by=['Feature', 'Day'])
        summary_filename_B = os.path.join(output_summary_dir, "summary_correlations_strona_B.csv")
        summary_df_B.to_csv(summary_filename_B, index=False, float_format='%.4f')
        print(f"Zapisano zbiorczy plik korelacji dla strony B: {summary_filename_B}")
    else:
        print("Nie znaleziono danych korelacyjnych dla strony B do agregacji.")

def plot_significant_correlations_for_days(
    sheet_names_processed: List[str],
    target_columns: List[str],
    input_csv_dir: str,
    correlation_threshold: float = 0.2
) -> None:
    print(f"\n{'='*20} Tworzenie wykresów znaczących korelacji (próg: |r| > {correlation_threshold}) {'='*20}")
    output_plot_dir = "significant_correlation_plots"
    os.makedirs(output_plot_dir, exist_ok=True)
    for target_main_name in ["strona A", "strona B"]:
        actual_target_col = ""
        for tc in target_columns:
            if target_main_name in tc:
                actual_target_col = tc
                break
        if not actual_target_col:
            print(f"Nie znaleziono pełnej nazwy kolumny docelowej dla '{target_main_name}'. Pomijam.")
            continue
        list_of_daily_dfs_for_target = []
        for sheet_name in sheet_names_processed:
            safe_target_name = actual_target_col.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
            csv_filename = os.path.join(input_csv_dir, f"correlations_{sheet_name}_{safe_target_name}.csv")
            if os.path.exists(csv_filename):
                try:
                    daily_corr_df = pd.read_csv(csv_filename)
                    daily_corr_df['Day'] = sheet_name
                    significant_daily_corr = daily_corr_df[daily_corr_df['CorrelationCoefficient'].abs() > correlation_threshold]
                    list_of_daily_dfs_for_target.append(significant_daily_corr)
                except pd.errors.EmptyDataError:
                    print(f"Ostrzeżenie: Plik CSV {csv_filename} jest pusty i zostanie pominięty przy tworzeniu wykresu dla {target_main_name}.")
                except Exception as e:
                    print(f"Błąd podczas wczytywania pliku {csv_filename} dla {target_main_name}: {e}")
            else:
                print(f"Ostrzeżenie: Nie znaleziono pliku CSV {csv_filename} dla {target_main_name}. Pomijam przy tworzeniu wykresu.")
        if not list_of_daily_dfs_for_target:
            print(f"Brak danych do stworzenia wykresu dla {target_main_name}.")
            continue
        combined_df_for_target = pd.concat(list_of_daily_dfs_for_target, ignore_index=True)
        if combined_df_for_target.empty:
            print(f"Brak znaczących korelacji (powyżej {correlation_threshold}) dla {target_main_name} we wszystkich analizowanych dniach.")
            continue
        combined_df_for_target = combined_df_for_target.sort_values(by=['Feature', 'CorrelationCoefficient'], ascending=[True, False])
        plt.figure(figsize=(12, max(8, len(combined_df_for_target['Feature'].unique()) * 0.5)))
        sns.barplot(data=combined_df_for_target, 
                    x='CorrelationCoefficient', 
                    y='Feature', 
                    hue='Day', 
                    palette='viridis')
        plt.title(f'Znaczące korelacje (|r| > {correlation_threshold}) z "{actual_target_col}"', fontsize=15)
        plt.xlabel('Współczynnik korelacji Pearsona', fontsize=12)
        plt.ylabel('Cecha', fontsize=12)
        plt.legend(title='Dzień', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_filename = os.path.join(output_plot_dir, f'significant_correlations_plot_{target_main_name.replace(" ", "_").lower()}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres znaczących korelacji jako: {plot_filename}")
        plt.close()