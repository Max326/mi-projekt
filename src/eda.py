import matplotlib.pyplot as plt
from typing import List
import pandas as pd

def exploratory_data_analysis(df: pd.DataFrame, target_columns: List[str], sheet_name_for_save: str) -> None:
    """Przeprowadź eksploracyjną analizę danych, zapisując wykresy z unikalną nazwą arkusza."""
    print(f"=== EKSPLORACYJNA ANALIZA DANYCH (Arkusz: {sheet_name_for_save}) ===")
    for target in target_columns:
        if target in df.columns:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            df[target].hist(bins=30, alpha=0.7)
            plt.title(f'Rozkład {target}\n(Arkusz: {sheet_name_for_save})')
            plt.xlabel('Temperatura [°C]')
            plt.ylabel('Częstość')
            plt.subplot(1, 2, 2)
            df[target].plot(kind='line', alpha=0.7)
            plt.title(f'Seria czasowa {target}\n(Arkusz: {sheet_name_for_save})')
            plt.xlabel('Próbka')
            plt.ylabel('Temperatura [°C]')
            plt.tight_layout()
            safe_target_name = target.replace(" ", "_").replace("-", "_").lower()
            filename = f'eda_{sheet_name_for_save}_{safe_target_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Zapisano wykres EDA jako: {filename}")
            plt.close()