import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns # Nadal potrzebne dla exploratory_data_analysis jeśli używa
from typing import List, Dict # Dodano Dict

# Ignorowanie ostrzeżeń, jeśli chcesz
import warnings
warnings.filterwarnings('ignore')

def load_excel_data(file_path: str) -> dict:
    """Wczytuje wszystkie arkusze z pliku Excel do słownika DataFrame."""
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

def exploratory_data_analysis(df: pd.DataFrame, target_columns: List[str], sheet_name_for_save: str) -> None:
    """Przeprowadź eksploracyjną analizę danych, zapisując wykresy z unikalną nazwą arkusza."""
    print(f"=== EKSPLORACYJNA ANALIZA DANYCH (Arkusz: {sheet_name_for_save}) ===")
    # print(f"Wymiary datasetu: {df.shape}") # Można odkomentować w razie potrzeby
    # print(f"Brakujące wartości:\n{df.isnull().sum().sum()} total")
    
    for target in target_columns:
        if target in df.columns:
            # print(f"\nStatystyki dla {target} (Arkusz: {sheet_name_for_save}):") # Można odkomentować
            # print(df[target].describe())
            
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
            # Modyfikacja nazwy pliku, aby uwzględnić arkusz
            safe_target_name = target.replace(" ", "_").replace("-", "_").lower()
            filename = f'eda_{sheet_name_for_save}_{safe_target_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Zapisano wykres EDA jako: {filename}")
            plt.close() # Zamknij figurę po zapisaniu, aby nie wyświetlać interaktywnie
            # plt.show() # Odkomentuj, jeśli chcesz wyświetlać interaktywnie

def correlation_data_for_csv(df: pd.DataFrame, target_columns: List[str]) -> Dict[str, pd.Series]:
    """
    Oblicza korelacje wszystkich zmiennych numerycznych z podanymi kolumnami docelowymi.
    Nie tworzy wykresów, zwraca dane do zapisu CSV.
    """
    # print("\n=== OBLICZANIE KORELACJI ===") # Można odkomentować w razie potrzeby
    
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
        
        # print(f"Top 5 najwyższych korelacji z {target_col}:") # Można odkomentować
        # print(sorted_correlations.head(5))
        # print(f"\nTop 5 najniższych (najbardziej ujemnych) korelacji z {target_col}:")
        # print(sorted_correlations.tail(5))
        
    return all_correlations_data

def aggregate_daily_correlations_to_summary_files(
    sheet_names_processed: List[str], 
    target_columns: List[str], 
    input_csv_dir: str, 
    output_summary_dir: str
) -> None:
    """
    Agreguje indywidualne pliki CSV z korelacjami (z różnych dni)
    w dwa zbiorcze pliki CSV (jeden dla strony A, jeden dla strony B).
    Każda korelacja będzie miała przypisany swój dzień.
    """
    print(f"\n{'='*20} Agregowanie korelacji do plików zbiorczych {'='*20}")
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
                    # Dodajemy kolumnę 'Day' i 'OriginalTarget' (jeśli nie ma)
                    daily_corr_df['Day'] = sheet_name 
                    # daily_corr_df['OriginalTarget'] = target_col # Nazwa oryginalnego targetu, jeśli potrzebne
                    
                    # Zmieniamy nazwę kolumny cechy, aby uwzględnić dzień dla unikalności
                    # np. 'poziom wody w walczaku' -> 'poziom wody w walczaku d2'
                    # Ta modyfikacja jest opcjonalna i zależy od tego, jak chcesz mieć dane w końcowym CSV
                    # W tej wersji łączymy wiersze, więc nie modyfikujemy Feature
                    # daily_corr_df['Feature_with_Day'] = daily_corr_df['Feature'] + f" {sheet_name}"
                    
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
        # Opcjonalnie sortowanie dla lepszej czytelności
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
    """
    Wczytuje dzienne pliki CSV z korelacjami, filtruje te o wartości bezwzględnej
    powyżej progu i tworzy porównawcze wykresy słupkowe dla każdej strony (A i B).
    """
    print(f"\n{'='*20} Tworzenie wykresów znaczących korelacji (próg: |r| > {correlation_threshold}) {'='*20}")

    # Katalog do zapisu wykresów
    output_plot_dir = "significant_correlation_plots"
    os.makedirs(output_plot_dir, exist_ok=True)

    for target_main_name in ["strona A", "strona B"]: # Iterujemy po "strona A" i "strona B"
        
        # Znajdź pełną nazwę kolumny docelowej
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
                    # Dodajemy kolumnę 'Day'
                    daily_corr_df['Day'] = sheet_name
                    # Filtrujemy znaczące korelacje
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

        # Łączenie danych ze wszystkich dni dla danego targetu
        combined_df_for_target = pd.concat(list_of_daily_dfs_for_target, ignore_index=True)

        if combined_df_for_target.empty:
            print(f"Brak znaczących korelacji (powyżej {correlation_threshold}) dla {target_main_name} we wszystkich analizowanych dniach.")
            continue
            
        # Sortowanie dla lepszej czytelności na wykresie (opcjonalne, ale pomaga)
        # Sortujemy najpierw po Feature, potem po wartości korelacji, aby ułatwić porównanie tej samej cechy w różne dni
        # Można też sortować tylko po wartości korelacji, jeśli chcemy widzieć ogólnie najsilniejsze
        combined_df_for_target = combined_df_for_target.sort_values(by=['Feature', 'CorrelationCoefficient'], ascending=[True, False])

        # Tworzenie wykresu
        plt.figure(figsize=(12, max(8, len(combined_df_for_target['Feature'].unique()) * 0.5))) # Dynamiczna wysokość
        
        # Używamy 'dodge=True' jeśli chcemy słupki obok siebie, lub 'dodge=False' (domyślnie) jeśli mogą się nakładać lub grupować inaczej.
        # Dla porównania dni, 'hue' i 'dodge=True' (lub domyślne zachowanie barplot) jest zwykle dobre.
        sns.barplot(data=combined_df_for_target, 
                    x='CorrelationCoefficient', 
                    y='Feature', 
                    hue='Day', 
                    palette='viridis') # 'viridis' to przykładowa paleta, możesz wybrać inną
        
        plt.title(f'Znaczące korelacje (|r| > {correlation_threshold}) z "{actual_target_col}"', fontsize=15)
        plt.xlabel('Współczynnik korelacji Pearsona', fontsize=12)
        plt.ylabel('Cecha', fontsize=12)
        plt.legend(title='Dzień', bbox_to_anchor=(1.05, 1), loc='upper left') # Legenda na zewnątrz
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Dopasowanie, aby legenda się zmieściła

        # Zapisz wykres
        plot_filename = os.path.join(output_plot_dir, f'significant_correlations_plot_{target_main_name.replace(" ", "_").lower()}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres znaczących korelacji jako: {plot_filename}")
        plt.close() # Zamknij figurę
        # plt.show() # Odkomentuj, jeśli chcesz wyświetlać interaktywnie

# Modyfikacja funkcji main, aby wywołać nową funkcję plotującą
def main():
    """Główna funkcja analizy dla wielu dni."""
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_names_to_process = ['d2', 'd3', 'd5', 'd6'] # Dni do analizy
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]
    
    daily_csv_dir = "correlation_results_csv"
    # summary_csv_dir = "summary_correlation_results" # Jeśli używasz agregacji

    os.makedirs(daily_csv_dir, exist_ok=True)
    # os.makedirs(summary_csv_dir, exist_ok=True) # Jeśli używasz agregacji

    print("Rozpoczynanie analizy korelacji dla wielu dni...")

    for sheet_name in sheet_names_to_process:
        print(f"\n{'='*20} Przetwarzanie danych dla dnia: {sheet_name} {'='*20}")
        try:
            df_day = get_full_dataset(file_path, sheet_name)
            exploratory_data_analysis(df_day, target_columns, sheet_name)
            correlations_for_day = correlation_data_for_csv(df_day, target_columns)
            
            for target_col, corr_series in correlations_for_day.items():
                safe_target_name = target_col.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
                csv_filename = os.path.join(daily_csv_dir, f"correlations_{sheet_name}_{safe_target_name}.csv")
                corr_df_to_save = corr_series.reset_index()
                corr_df_to_save.columns = ['Feature', 'CorrelationCoefficient']
                corr_df_to_save.to_csv(csv_filename, index=False, float_format='%.4f')
                # print(f"Zapisano korelacje dla '{target_col}' z dnia '{sheet_name}' do: {csv_filename}") # Można wykomentować dla krótszego logu
        except Exception as e:
            print(f"Problem podczas przetwarzania arkusza {sheet_name}: {e}")
    
    print("\nPrzetwarzanie dziennych korelacji zakończone.")
    
    # Wywołanie funkcji agregującej (jeśli nadal jej potrzebujesz)
    # aggregate_daily_correlations_to_summary_files(
    #     sheet_names_processed=sheet_names_to_process,
    #     target_columns=target_columns,
    #     input_csv_dir=daily_csv_dir,
    #     output_summary_dir=summary_csv_dir
    # )
    
    # Wywołanie nowej funkcji do tworzenia wykresów znaczących korelacji
    plot_significant_correlations_for_days(
        sheet_names_processed=sheet_names_to_process,
        target_columns=target_columns,
        input_csv_dir=daily_csv_dir,
        correlation_threshold=0.2 # Możesz zmienić ten próg
    )
    
    print("\n" + "="*60)
    print("ANALIZA DLA WSZYSTKICH DNI ORAZ TWORZENIE WYKRESÓW ZNACZĄCYCH KORELACJI ZAKOŃCZONE.")
    print(f"Sprawdź indywidualne pliki CSV w katalogu: '{daily_csv_dir}'")
    # print(f"Sprawdź zbiorcze pliki CSV w katalogu: '{summary_csv_dir}'") # Jeśli używasz agregacji
    print(f"Sprawdź wykresy znaczących korelacji w katalogu: 'significant_correlation_plots'")
    print("Oraz wykresy EDA (przebiegi temperatur) w głównym katalogu skryptu.")
    print("="*60)

if __name__ == "__main__":
    main()