import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

def load_excel_data(file_path: str) -> dict:
    """Wczytaj dane z pliku Excel do słownika DataFrames."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plik {file_path} nie został znaleziony.")
    
    excel_data = pd.read_excel(file_path, sheet_name=None)
    return excel_data

def get_full_dataset(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Załaduj pełny dataset z określonego arkusza."""
    excel_data = load_excel_data(file_path)
    
    if sheet_name not in excel_data:
        raise ValueError(f"Arkusz '{sheet_name}' nie istnieje w pliku. "
                         f"Dostępne arkusze: {list(excel_data.keys())}")
    
    return excel_data[sheet_name]

def exploratory_data_analysis(df: pd.DataFrame, target_columns: List[str]) -> None:
    """Przeprowadź eksploracyjną analizę danych."""
    print("=== EKSPLORACYJNA ANALIZA DANYCH ===")
    print(f"Wymiary datasetu: {df.shape}")
    print(f"Brakujące wartości:\n{df.isnull().sum().sum()} total")
    
    # Podstawowe statystyki dla zmiennych docelowych
    for target in target_columns:
        if target in df.columns:
            print(f"\nStatystyki dla {target}:")
            print(df[target].describe())
            
            # Wykres rozkładu
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            df[target].hist(bins=30, alpha=0.7)
            plt.title(f'Rozkład {target}')
            plt.xlabel('Temperatura [°C]')
            plt.ylabel('Częstość')
            
            plt.subplot(1, 2, 2)
            df[target].plot(kind='line', alpha=0.7)
            plt.title(f'Seria czasowa {target}')
            plt.xlabel('Próbka')
            plt.ylabel('Temperatura [°C]')
            
            plt.tight_layout()
            plt.savefig(f'eda_{target.replace(" ", "_").replace("-", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()

def correlation_analysis_sorted_plots(df: pd.DataFrame, target_columns: List[str]) -> Dict[str, pd.Series]:
    """
    Oblicza korelacje wszystkich zmiennych numerycznych z podanymi kolumnami docelowymi.
    Tworzy posortowane wykresy słupkowe dla każdej kolumny docelowej.
    """
    print("\n=== ANALIZA KORELACJI (POSORTOWANE WYKRESY SŁUPKOWE) ===")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    all_correlations_data: Dict[str, pd.Series] = {}
    
    for target_col in target_columns:
        if target_col not in numeric_df.columns:
            print(f"Ostrzeżenie: Kolumna docelowa '{target_col}' nie została znaleziona w danych numerycznych.")
            continue
            
        print(f"\nObliczanie korelacji dla: {target_col}")
        
        # Oblicz korelacje wszystkich cech numerycznych z bieżącą kolumną docelową
        # Używamy .corr() na całym numeric_df, a następnie wybieramy kolumnę target_col
        # Alternatywnie, dla dużych ramek danych, numeric_df.corrwith(numeric_df[target_col]) może być bardziej wydajne
        # ale numeric_df.corr()[target_col] jest prostsze i częściej stosowane [3]
        correlations_series = numeric_df.corr()[target_col]
        
        # Usuń korelację zmiennej docelowej samej ze sobą (która zawsze wynosi 1.0)
        correlations_series = correlations_series.drop(target_col, errors='ignore')
        
        # Posortuj korelacje od najwyższej do najniższej
        sorted_correlations = correlations_series.sort_values(ascending=False)
        
        all_correlations_data[target_col] = sorted_correlations
        
        print(f"Top 5 najwyższych korelacji z {target_col}:")
        print(sorted_correlations.head(5))
        print(f"\nTop 5 najniższych (najbardziej ujemnych) korelacji z {target_col}:")
        print(sorted_correlations.tail(5))
        
        # Tworzenie wykresu słupkowego
        plt.figure(figsize=(12, max(8, len(sorted_correlations) * 0.3))) # Dynamiczna wysokość
        sns.barplot(x=sorted_correlations.values, y=sorted_correlations.index, palette="coolwarm_r")
        plt.title(f'Korelacja zmiennych z "{target_col}" (posortowane)', fontsize=15)
        plt.xlabel('Współczynnik korelacji Pearsona', fontsize=12)
        plt.ylabel('Zmienne', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Zapisz wykres
        filename = f'correlations_plot_{target_col.replace(" ", "_").replace("-", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres korelacji jako: {filename}")
        plt.show()
        
    return all_correlations_data

def main():
    """Główna funkcja analizy."""
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_name = "d2"  # Zmień na odpowiedni arkusz
    
    # Zmienne docelowe (temperatura spalin)
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]
    
    try:
        print("Ładowanie danych...")
        df = get_full_dataset(file_path, sheet_name)
        
        # 1. Eksploracyjna analiza danych
        exploratory_data_analysis(df, target_columns)
        
        # 2. Analiza korelacji
        correlations = correlation_analysis_sorted_plots(df, target_columns)
        
        print("\n" + "="*60)
        print("ANALIZA ZAKOŃCZONA - sprawdź wygenerowane wykresy!")
        print("="*60)
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
