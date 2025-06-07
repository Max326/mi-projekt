import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import os

def ensure_dir(directory_path: str):
    """Upewnia się, że katalog istnieje, tworząc go w razie potrzeby."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Utworzono katalog: {directory_path}")

def exploratory_data_analysis(df: pd.DataFrame, target_columns: List[str], sheet_name_for_save: str, output_dir: str = "eda_plots") -> None:
    """Przeprowadź eksploracyjną analizę danych, zapisując wykresy z unikalną nazwą arkusza."""
    ensure_dir(output_dir)
    print(f"=== EKSPLORACYJNA ANALIZA DANYCH (Arkusz: {sheet_name_for_save}) ===")
    for target in target_columns:
        if target in df.columns:
            plt.figure(figsize=(6, 4))
            df[target].plot(kind='line', alpha=0.7)
            plt.title(f'{target}\n(Arkusz: {sheet_name_for_save})')
            plt.xlabel('Próbka')
            plt.ylabel('Temperatura [°C]')
            plt.tight_layout()
            safe_target_name = target.replace(" ", "_").replace("-", "_").lower()
            filename = os.path.join(output_dir, f'eda_{sheet_name_for_save}_{safe_target_name}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Zapisano wykres EDA jako: {filename}")
            plt.close()

def plot_downsampled_data_comparison(original_df: pd.DataFrame, downsampled_df: pd.DataFrame, target_columns: List[str], sheet_name_for_save: str, output_dir: str = "downsampled_comparison_plots") -> None:
    """Porównuje dane oryginalne i po downsamplingu, zapisując wykresy."""
    ensure_dir(output_dir)
    print(f"=== PORÓWNANIE DANYCH ORYGINALNYCH I PO DOWNSAMPLINGU (Arkusz: {sheet_name_for_save}) ===")
    for target in target_columns:
        if target in original_df.columns and target in downsampled_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(original_df[target], label='Tp=2s', alpha=0.7)
            plt.plot(downsampled_df[target], label='Tp=10s', alpha=0.7)
            plt.title(f'Porównanie danych: {target}\n(Arkusz: {sheet_name_for_save})')
            plt.xlabel('Próbka')
            plt.ylabel('Temperatura [°C]')
            plt.legend()
            plt.tight_layout()
            safe_target_name = target.replace(" ", "_").replace("-", "_").lower()
            filename = os.path.join(output_dir, f'comparison_{sheet_name_for_save}_{safe_target_name}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Zapisano wykres porównawczy jako: {filename}")
            plt.close()