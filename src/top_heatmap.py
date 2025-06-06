import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_excel_data(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plik {file_path} nie został znaleziony.")
    
    # Load all sheets into a dictionary of DataFrames
    excel_data = pd.read_excel(file_path, sheet_name=None)
    return excel_data

def get_full_dataset(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Załaduj pełny dataset z określonego arkusza."""
    excel_data = load_excel_data(file_path)
    if sheet_name not in excel_data:
        raise ValueError(f"Arkusz '{sheet_name}' nie istnieje w pliku. "
                         f"Dostępne arkusze: {list(excel_data.keys())}")
    # Ta funkcja teraz tylko wczytuje, konwersja będzie w main
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

def correlation_analysis(df: pd.DataFrame, target_columns: List[str], threshold: float = 0.3) -> dict:
    """Analiza korelacji między zmiennymi, uwzględnia wszystkie kolumny numeryczne."""
    print("\n=== ANALIZA KORELACJI ===")
    
    # Wybierz tylko kolumny numeryczne
    numeric_df = df.select_dtypes(include=[np.number])
    
    correlations = {}
    
    for target in target_columns:
        if target in numeric_df.columns:
            # Oblicz korelacje z target variable
            corr_with_target = numeric_df.corr()[target].abs().sort_values(ascending=False)
            
            # Wybierz zmienne o korelacji powyżej threshold
            # UWAGA: Teraz bierzemy wszystkie zmienne, niezależnie od progu
            significant_vars = corr_with_target.drop(target)
            
            correlations[target] = significant_vars
            
            print(f"\nZmienne skorelowane z {target} (wszystkie):")
            for var, corr in significant_vars.head(10).items():
                print(f"  {var}: {corr:.3f}")
            
            # Mapa ciepła dla wszystkich zmiennych (lub max 30, żeby było czytelnie)
            n_vars_heatmap = min(30, len(significant_vars))
            top_vars = list(significant_vars.head(n_vars_heatmap).index) + [target]
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

def analyze_inter_feature_correlations(
    df_day: pd.DataFrame, 
    feature_list: List[str], 
    day_name: str, 
    corr_threshold: float = 0.7
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Analizuje korelacje między podanymi cechami (wejściami).
    Rysuje heatmapę korelacji i zwraca listę silnie skorelowanych par.

    Args:
        df_day: DataFrame z danymi dla konkretnego dnia.
        feature_list: Lista nazw kolumn (wejść) do analizy.
        day_name: Nazwa dnia (np. "d2") dla tytułów wykresów/logów.
        corr_threshold: Próg korelacji (wartość bezwzględna) do uznania pary za silnie skorelowaną.

    Returns:
        Tuple: (macierz korelacji między wybranymi cechami, lista silnie skorelowanych par)
               Lista par ma format (cecha1, cecha2, współczynnik_korelacji).
    """
    print(f"\n=== ANALIZA KORELACJI MIĘDZY WYBRANYMI WEJŚCIAMI (Dzień: {day_name}) ===")
    
    # Wybierz tylko te kolumny z feature_list, które faktycznie istnieją w df_day
    available_features = [col for col in feature_list if col in df_day.columns]
    if not available_features:
        print(f"Żadne z podanych wejść nie zostały znalezione w danych dla dnia {day_name}.")
        return pd.DataFrame(), []
        
    df_subset = df_day[available_features].copy() # Używamy .copy() aby uniknąć SettingWithCopyWarning
    
    # Usuń kolumny, które mają tylko jedną unikalną wartość (lub same NaN), bo .corr() da błąd lub NaN
    for col in df_subset.columns:
        if df_subset[col].nunique(dropna=True) <= 1: # nunique bez NaN
            print(f"Ostrzeżenie: Kolumna '{col}' ma <= 1 unikalną wartość w dniu {day_name} i zostanie pominięta w analizie korelacji między wejściami.")
            df_subset.drop(col, axis=1, inplace=True)
            
    if df_subset.shape[1] < 2:
        print(f"Niewystarczająca liczba wejść (po filtracji) do analizy korelacji w dniu {day_name}.")
        return pd.DataFrame(), []

    inter_corr_matrix = df_subset.corr()
    
    # Rysowanie heatmapy
    plt.figure(figsize=(max(10, len(available_features)*0.6), max(8, len(available_features)*0.5)))
    sns.heatmap(inter_corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, annot_kws={"size": 8})
    plt.title(f'Heatmapa korelacji między wybranymi wejściami - Dzień {day_name}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Zapisz wykres
    output_plot_dir = "inter_feature_correlation_plots"
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_filename = os.path.join(output_plot_dir, f'inter_feature_corr_plot_{day_name}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Zapisano heatmapę korelacji między wejściami jako: {plot_filename}")
    plt.show() # Możesz zakomentować, jeśli nie chcesz wyświetlać interaktywnie

    # Znajdowanie silnie skorelowanych par
    strong_pairs = []
    # Używamy .abs() aby brać pod uwagę zarówno silne korelacje dodatnie, jak i ujemne
    abs_corr_matrix = inter_corr_matrix.abs() 
    upper = abs_corr_matrix.where(np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool))
    
    for column in upper.columns:
        for index in upper.index:
            # upper.loc[index, column] to już wartość bezwzględna
            if pd.notna(upper.loc[index, column]) and upper.loc[index, column] > corr_threshold:
                # Dodajemy oryginalną wartość korelacji (z znakiem)
                original_corr_value = inter_corr_matrix.loc[index, column]
                strong_pairs.append((index, column, original_corr_value)) 
                
    if strong_pairs:
        print(f"\nSilnie skorelowane pary wejść (|r| > {corr_threshold}) dla dnia {day_name}:")
        for pair in strong_pairs:
            print(f"- {pair[0]} ORAZ {pair[1]}: {pair[2]:.2f}")
    else:
        print(f"Nie znaleziono par wejść o korelacji |r| > {corr_threshold} dla dnia {day_name}.")
        
    return inter_corr_matrix, strong_pairs


def main():
    """Główna funkcja analizy."""
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    # Możesz iterować po dniach lub wybrać konkretny
    sheet_names_to_analyze = ['d2', 'd3', 'd5', 'd6'] # Przykładowe dni
    
    # Twoja lista "top wejść"
    top_inputs_list = [
        "temperatura mieszanki za młynem A", "temperatura mieszanki za młynem F", 
        "temperatura mieszanki za młynem E",
        "kąt wychylenia palnika róg #1", "kąt wychylenia palnika róg #2", 
        "kąt wychylenia palnika róg #3", "kąt wychylenia palnika róg #4",
        "klapy wentylatora podmuchu - strona A", # Zakładamy, że strona B jest identyczna lub ją dodajemy
        "przepływ powietrza pierwotnego",
        "zawór zdmuchiwaczy sadzy - strona L", "zawór zdmuchiwaczy sadzy - strona P",
        "ciśnienie wody wtryskowej do pary wtórnej",
        "temperatura za wtryskiem pary wtórnej - strona L", 
        "temperatura za wtryskiem pary wtórnej - strona P"
    ]
    # Dodaj "klapy wentylatora podmuchu - strona B" jeśli chcesz ją analizować osobno
    # top_inputs_list.append("klapy wentylatora podmuchu - strona B")


    # Zmienne docelowe (temperatura spalin) - nadal potrzebne dla exploratory_data_analysis
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]

    # Próg dla silnej korelacji między wejściami
    inter_feature_corr_threshold = 0.8 # Możesz dostosować, np. 0.7, 0.8, 0.9
    
    for sheet_name in sheet_names_to_analyze:
        print(f"\n{'='*30} ANALIZA DLA DNIA: {sheet_name} {'='*30}")
        try:
            print("Ładowanie danych...")
            df = get_full_dataset(file_path, sheet_name)
            
            # 1. Eksploracyjna analiza danych (nadal przydatna dla wyjść)
            # exploratory_data_analysis(df, target_columns) # Możesz odkomentować, jeśli chcesz EDA
            
            # 2. Analiza korelacji między wybranymi wejściami
            _, _ = analyze_inter_feature_correlations(
                df_day=df, 
                feature_list=top_inputs_list, 
                day_name=sheet_name,
                corr_threshold=inter_feature_corr_threshold
            )
            
        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku {file_path}. Upewnij się, że plik istnieje.")
            break 
        except ValueError as ve:
            print(f"BŁĄD podczas przetwarzania arkusza {sheet_name}: {ve}")
        except Exception as e:
            print(f"Nieoczekiwany błąd podczas przetwarzania arkusza {sheet_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("ANALIZA KORELACJI MIĘDZY WEJŚCIAMI ZAKOŃCZONA.")
    print(f"Sprawdź wygenerowane heatmapy w katalogu: 'inter_feature_correlation_plots'")
    print("="*60)

if __name__ == "__main__":
    main()