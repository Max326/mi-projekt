import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple # Upewnij się, że Dict i Tuple są zaimportowane
import warnings
warnings.filterwarnings('ignore')

def load_excel_data(file_path: str) -> dict:
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
    # Ta funkcja teraz tylko wczytuje, konwersja będzie w main
    return excel_data[sheet_name]

# Twoja funkcja exploratory_data_analysis (bez zmian)
def exploratory_data_analysis(df: pd.DataFrame, target_columns: List[str]) -> None:
    print("=== EKSPLORACYJNA ANALIZA DANYCH ===")
    # ... (reszta kodu funkcji bez zmian) ...
    # Upewnij się, że ta funkcja poprawnie obsługuje NaN przy plotowaniu, 
    # np. df[target].hist() i df[target].plot() powinny sobie z tym radzić.
    print(f"Wymiary datasetu: {df.shape}")
    print(f"Brakujące wartości (po konwersji 'Bad Data' na NaN):\n{df.isnull().sum().sum()} total")
    
    for target in target_columns:
        if target in df.columns:
            print(f"\nStatystyki dla {target}:")
            print(df[target].describe()) # describe() również dobrze obsługuje NaN
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            df[target].hist(bins=30, alpha=0.7) # hist() ignoruje NaN
            plt.title(f'Rozkład {target}')
            plt.xlabel('Temperatura [°C]')
            plt.ylabel('Częstość')
            
            plt.subplot(1, 2, 2)
            df[target].plot(kind='line', alpha=0.7) # plot() dla serii czasowej pokaże przerwy dla NaN
            plt.title(f'Seria czasowa {target}')
            plt.xlabel('Próbka')
            plt.ylabel('Temperatura [°C]')
            
            plt.tight_layout()
            save_path = f'eda_{target.replace(" ", "_").replace("-", "_")}.png'
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Zapisano wykres EDA jako {save_path}")
            except Exception as e:
                print(f"Nie udało się zapisać wykresu EDA {save_path}: {e}")
            plt.show()


# Twoja funkcja analyze_inter_feature_correlations (z drobną modyfikacją)
def analyze_inter_feature_correlations(
    df_day: pd.DataFrame, 
    feature_list: List[str], 
    day_name: str, 
    corr_threshold: float = 0.7
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    print(f"\n=== ANALIZA KORELACJI MIĘDZY WYBRANYMI WEJŚCIAMI (Dzień: {day_name}) ===")
    
    available_features = [col for col in feature_list if col in df_day.columns]
    if not available_features:
        print(f"Żadne z podanych wejść nie zostały znalezione w danych dla dnia {day_name}.")
        return pd.DataFrame(), []
        
    # Tutaj df_day już powinno mieć NaN zamiast 'Bad Data'
    # df.select_dtypes(include=[np.number]) nie jest już potrzebne jeśli wszystkie kolumny są już numeryczne (lub NaN)
    # ale dla bezpieczeństwa, można je zostawić lub użyć df_subset = df_day[available_features].copy()
    # i upewnić się, że te kolumny są faktycznie numeryczne
    
    df_subset_numeric_only = df_day[available_features].select_dtypes(include=[np.number])
    
    # Aktualizacja listy dostępnych cech po selekcji tylko numerycznych
    available_features = df_subset_numeric_only.columns.tolist()

    if not available_features:
        print(f"Żadne z podanych wejść (po konwersji 'Bad Data' i selekcji numerycznych) nie pozostały do analizy w dniu {day_name}.")
        return pd.DataFrame(), []

    # Usuń kolumny, które mają tylko jedną unikalną wartość (lub same NaN), bo .corr() da błąd lub NaN
    # Ta pętla jest nadal ważna
    valid_features_for_corr = []
    for col in available_features: # Iterujemy po kolumnach, które są numeryczne
        if df_subset_numeric_only[col].nunique(dropna=True) > 1: # Sprawdzamy unikalne wartości bez NaN
            valid_features_for_corr.append(col)
        else:
            print(f"Ostrzeżenie: Kolumna '{col}' ma <= 1 unikalną wartość numeryczną (po usunięciu NaN) w dniu {day_name} i zostanie pominięta w analizie korelacji między wejściami.")
            
    if len(valid_features_for_corr) < 2:
        print(f"Niewystarczająca liczba wejść (po filtracji) do analizy korelacji w dniu {day_name}.")
        return pd.DataFrame(), []

    df_for_corr = df_subset_numeric_only[valid_features_for_corr]
    inter_corr_matrix = df_for_corr.corr() # .corr() domyślnie obsłuży NaN (use='pairwise')
    
    # --- reszta kodu funkcji analyze_inter_feature_correlations bez zmian ---
    # (rysowanie heatmapy, znajdowanie silnie skorelowanych par)
    plt.figure(figsize=(max(10, len(valid_features_for_corr)*0.6), max(8, len(valid_features_for_corr)*0.5)))
    sns.heatmap(inter_corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, annot_kws={"size": 8})
    plt.title(f'Heatmapa korelacji między wybranymi wejściami - Dzień {day_name}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_plot_dir = "inter_feature_correlation_plots"
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_filename = os.path.join(output_plot_dir, f'inter_feature_corr_plot_{day_name}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Zapisano heatmapę korelacji między wejściami jako: {plot_filename}")
    plt.show()

    strong_pairs = []
    abs_corr_matrix = inter_corr_matrix.abs() 
    upper = abs_corr_matrix.where(np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool))
    
    for column in upper.columns:
        for index in upper.index:
            if pd.notna(upper.loc[index, column]) and upper.loc[index, column] > corr_threshold:
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
    sheet_names_to_analyze = ['d2', 'd3', 'd5', 'd6'] 
    
    top_inputs_list = [
        "temperatura mieszanki za młynem A", "temperatura mieszanki za młynem F", 
        "temperatura mieszanki za młynem E",
        "kąt wychylenia palnika róg #1", "kąt wychylenia palnika róg #2", 
        "kąt wychylenia palnika róg #3", "kąt wychylenia palnika róg #4",
        "klapy wentylatora podmuchu - strona A", "klapy wentylatora podmuchu - strona B", # Dodaję obie dla pewności
        "przepływ powietrza pierwotnego",
        "zawór zdmuchiwaczy sadzy - strona L", "zawór zdmuchiwaczy sadzy - strona P",
        "ciśnienie wody wtryskowej do pary wtórnej",
        "temperatura za wtryskiem pary wtórnej - strona L", 
        "temperatura za wtryskiem pary wtórnej - strona P"
    ]
    
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]
    inter_feature_corr_threshold = 0.8
    
    for sheet_name in sheet_names_to_analyze:
        print(f"\n{'='*30} ANALIZA DLA DNIA: {sheet_name} {'='*30}")
        try:
            print("Ładowanie danych...")
            df_original = get_full_dataset(file_path, sheet_name)
            df = df_original.copy() # Pracuj na kopii, aby nie modyfikować wczytanego oryginału wielokrotnie

            # --- DODANY KROK: Konwersja 'Bad Data' na NaN ---
            print("Konwertowanie 'Bad Data' na NaN i typów kolumn na numeryczne...")
            for col in df.columns:
                # Próba konwersji każdej kolumny na typ numeryczny.
                # Wartości, których nie da się skonwertować (np. ' Bad Data') staną się NaN.
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # -------------------------------------------------

            # Opcjonalnie: Sprawdź, ile NaN powstało
            # print("Liczba wartości NaN w każdej kolumnie po konwersji:")
            # print(df.isnull().sum())

            # 1. Eksploracyjna analiza danych (teraz z obsłużonymi NaN)
            # exploratory_data_analysis(df, target_columns) # Odkomentuj w razie potrzeby
            
            # 2. Analiza korelacji między wybranymi wejściami (teraz z obsłużonymi NaN)
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
