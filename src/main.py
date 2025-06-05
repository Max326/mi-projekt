import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

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

def correlation_analysis(df: pd.DataFrame, target_columns: List[str]) -> dict:
    """Analiza korelacji między zmiennymi, uwzględnia wszystkie kolumny numeryczne."""
    print("\n=== ANALIZA KORELACJI ===")
    
    # Wybierz tylko kolumny numeryczne
    numeric_df = df.select_dtypes(include=[np.number])
    
    correlations = {}
    
    for target in target_columns:
        if target in numeric_df.columns:
            # Oblicz korelacje z target variable
            corr_with_target = numeric_df.corr()[target].abs().sort_values(ascending=False)
            
            # Zachowaj wszystkie zmienne (bez progu)
            significant_vars = corr_with_target.drop(target)
            
            correlations[target] = significant_vars
            
            print(f"\nZmienne skorelowane z {target} (wszystkie):")
            for var, corr in significant_vars.head(10).items():
                print(f"  {var}: {corr:.3f}")
            
            # Mapa ciepła dla wszystkich zmiennych (lub max 30, żeby było czytelnie)
            n_vars_heatmap = min(len(numeric_df.columns), 30)  # Użyj wszystkich kolumn, ale max 30
            top_vars = list(corr_with_target.head(n_vars_heatmap).index)
            corr_matrix = numeric_df[top_vars].corr()
            
            plt.figure(figsize=(16, 14))  # Zwiększ rozmiar dla czytelności
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                        square=True, fmt='.2f', annot_kws={"size": 8})  # Zmniejsz rozmiar czcionki
            plt.title(f'Mapa korelacji dla {target}')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'correlation_heatmap_{target.replace(" ", "_").replace("-", "_")}.png', 
                        dpi=300, bbox_inches='tight')
            plt.show()
    
    return correlations

def main():
    """Główna funkcja analizy."""
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_name = "d3"  # Zmień na odpowiedni arkusz
    
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
        correlations = correlation_analysis(df, target_columns)
        
        print("\n" + "="*60)
        print("ANALIZA ZAKOŃCZONA - sprawdź wygenerowane wykresy!")
        print("="*60)
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
