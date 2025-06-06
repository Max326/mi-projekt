import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns # Dodano dla potencjalnych przyszłych wykresów
from typing import List, Dict # Upewnij się, że Dict jest importowane, jeśli używasz starszej wersji Python
import warnings

# Ignorowanie ostrzeżeń, jeśli chcesz
warnings.filterwarnings('ignore')

# --- Funkcje pomocnicze (bez zmian) ---
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
        raise ValueError(f"Arkusz {sheet_name} nie istnieje w pliku {file_path}.")
    return excel_data[sheet_name]

def handle_outliers_iqr_capping(df: pd.DataFrame, columns: List[str] = None, threshold: float = 1.5) -> pd.DataFrame:
    """
    Obsługuje wartości odstające w określonych (lub wszystkich numerycznych) kolumnach DataFrame
    za pomocą metody IQR i zastępuje je wartościami granicznymi (capping).
    """
    df_processed = df.copy()
    if columns is None:
        columns_to_check = df_processed.select_dtypes(include=np.number).columns.tolist()
    else:
        columns_to_check = [col for col in columns if col in df_processed.columns] # Sprawdź czy kolumny istnieją

    for col in columns_to_check:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            continue
        if df_processed[col].nunique() < 2 or df_processed[col].isnull().all():
            continue

        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        original_values_count = len(df_processed)
        df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
        
        num_outliers_handled = (df[col] < lower_bound).sum() + (df[col] > upper_bound).sum()

        if num_outliers_handled > 0:
            print(f"  W kolumnie '{col}': Obsłużono {num_outliers_handled} wartości odstających.")
            
    return df_processed

def exploratory_data_analysis(df: pd.DataFrame, target_columns: List[str], sheet_name_for_save: str) -> None:
    """Przeprowadź eksploracyjną analizę danych, zapisując wykresy z unikalną nazwą arkusza."""
    # print(f"=== EKSPLORACYJNA ANALIZA DANYCH (Arkusz: {sheet_name_for_save}) ===") # Mniej gadatliwe
    plots_dir = "eda_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    for target in target_columns:
        if target in df.columns:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.histplot(df[target], kde=True)
            plt.title(f'Histogram: {target} (Arkusz: {sheet_name_for_save})')
            
            plt.subplot(1, 2, 2)
            df[target].plot()
            plt.title(f'Przebieg czasowy: {target} (Arkusz: {sheet_name_for_save})')
            
            plt.tight_layout()
            safe_target_name = target.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
            plot_filename = os.path.join(plots_dir, f"eda_{sheet_name_for_save}_{safe_target_name}.png")
            plt.savefig(plot_filename)
            plt.close()
            # print(f"Zapisano wykres EDA dla {target} (arkusz: {sheet_name_for_save}) do: {plot_filename}")
        else:
            print(f"Ostrzeżenie (EDA): Kolumna docelowa '{target}' nie znaleziona w arkuszu {sheet_name_for_save}.")


def correlation_data_for_csv(df: pd.DataFrame, target_columns: List[str]) -> Dict[str, pd.Series]:
    """Oblicza korelacje (lag=0) wszystkich zmiennych numerycznych z podanymi kolumnami docelowymi."""
    numeric_df = df.select_dtypes(include=[np.number])
    all_correlations_data: Dict[str, pd.Series] = {}
    
    for target_col in target_columns:
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col].drop(target_col) # Usunięcie korelacji samej ze sobą
            all_correlations_data[target_col] = correlations.sort_values(ascending=False)
        else:
            print(f"Ostrzeżenie (Korelacje lag=0): Kolumna docelowa '{target_col}' nie znaleziona.")
            all_correlations_data[target_col] = pd.Series(dtype='float64')
            
    return all_correlations_data

def analyze_and_report_lagged_correlations(
    df: pd.DataFrame,
    input_features: List[str],
    target_columns: List[str],
    sample_lags_for_sheet: List[int], # Lista opóźnień w próbkach dla bieżącego arkusza
    time_lags_seconds_map: Dict[int, int], # Mapa: opóźnienie w próbkach -> opóźnienie w sekundach
    sheet_name: str,
    output_dir: str = "lagged_correlation_results_csv"
) -> None:
    """
    Analizuje korelacje opóźnione, wypisuje top N i zapisuje wszystkie do CSV.
    """
    print(f"\n--- Analiza korelacji opóźnionych dla arkusza: {sheet_name} ---")
    os.makedirs(output_dir, exist_ok=True)

    if not all(isinstance(lag, int) and lag > 0 for lag in sample_lags_for_sheet):
        print(f"  Ostrzeżenie (Arkusz: {sheet_name}): Lista opóźnień (sample_lags_for_sheet) jest nieprawidłowa lub pusta. Pomijam analizę opóźnień.")
        return

    for target_col in target_columns:
        if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
            print(f"  Ostrzeżenie (Arkusz: {sheet_name}): Kolumna docelowa '{target_col}' nie istnieje lub nie jest numeryczna. Pomijam dla niej analizę opóźnień.")
            continue
        
        print(f"\n  Korelacje opóźnione dla targetu: {target_col} (Arkusz: {sheet_name})")
        
        all_lags_data_for_csv = [] # Lista do przechowywania wyników dla CSV dla bieżącego targetu

        for lag_samples in sample_lags_for_sheet:
            time_lag_sec_approx = time_lags_seconds_map.get(lag_samples, "N/A")
            # print(f"    Opóźnienie: {lag_samples} próbek (ok. {time_lag_sec_approx}s)") # Mniej gadatliwe
            
            current_lag_correlations = {}
            for feature in input_features:
                if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
                    continue
                if feature == target_col: 
                    continue

                lagged_feature_series = df[feature].shift(lag_samples)
                
                try:
                    if pd.api.types.is_numeric_dtype(lagged_feature_series): # Dodatkowe sprawdzenie
                        correlation = df[target_col].corr(lagged_feature_series)
                        if pd.notna(correlation):
                             current_lag_correlations[feature] = correlation
                             all_lags_data_for_csv.append({
                                 'Feature_Original': feature,
                                 'Lag_Samples': lag_samples,
                                 'Lag_Seconds_Approx': time_lag_sec_approx,
                                 'CorrelationCoefficient': correlation
                             })
                except Exception: # Ogólny wyjątek, można uszczegółowić
                    pass # Cicha obsługa błędów korelacji dla pojedynczej pary
            
            if current_lag_correlations:
                sorted_lag_corr = sorted(current_lag_correlations.items(), key=lambda item: abs(item[1]), reverse=True)
                # print(f"      Top korelacje (opóźnienie {lag_samples} próbek, ok. {time_lag_sec_approx}s):")
                # for i, (feat_name, corr_val) in enumerate(sorted_lag_corr[:3]): 
                #     print(f"        - {feat_name}: {corr_val:.4f}")
            # else:
                # print(f"      Brak obliczonych korelacji dla opóźnienia {lag_samples} próbek.")

        # Zapis do CSV dla bieżącego target_col
        if all_lags_data_for_csv:
            df_lag_output = pd.DataFrame(all_lags_data_for_csv)
            df_lag_output = df_lag_output.sort_values(by=['Feature_Original', 'Lag_Samples'])
            
            safe_target_name = target_col.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
            csv_filename = os.path.join(output_dir, f"lagged_correlations_{sheet_name}_{safe_target_name}.csv")
            try:
                df_lag_output.to_csv(csv_filename, index=False, float_format='%.4f')
                print(f"  Zapisano opóźnione korelacje dla '{target_col}' (Arkusz: {sheet_name}) do: {csv_filename}")
            except Exception as e:
                print(f"  Błąd podczas zapisu pliku CSV dla opóźnionych korelacji ({csv_filename}): {e}")
        elif sample_lags_for_sheet: # Jeśli były próby obliczenia opóźnień
             print(f"  Brak obliczonych korelacji opóźnionych do zapisu dla '{target_col}' (Arkusz: {sheet_name}).")


def plot_significant_correlations_for_days(
    sheet_names_processed: List[str],
    target_columns: List[str],
    input_csv_dir: str, # Katalog z wynikami korelacji (lag=0)
    correlation_threshold: float = 0.2,
    output_plot_dir: str = "significant_correlation_plots"
) -> None:
    """Tworzy wykresy słupkowe dla korelacji (lag=0) przekraczających próg."""
    # print("\n=== TWORZENIE WYKRESÓW ZNACZĄCYCH KORELACJI (LAG=0) ===") # Mniej gadatliwe
    os.makedirs(output_plot_dir, exist_ok=True)

    for target_col in target_columns:
        plt.figure(figsize=(15, 8))
        all_days_significant_corr = {} # Słownik: {feature: [corr_d2, corr_d3, ...]}

        for sheet_name in sheet_names_processed:
            safe_target_name = target_col.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
            csv_filename = os.path.join(input_csv_dir, f"correlations_{sheet_name}_{safe_target_name}.csv")
            
            if not os.path.exists(csv_filename):
                # print(f"Plik {csv_filename} nie istnieje, pomijam dla wykresu.")
                continue
            try:
                day_corr_df = pd.read_csv(csv_filename)
                # Upewnij się, że kolumny mają oczekiwane nazwy
                if 'Feature' not in day_corr_df.columns or 'CorrelationCoefficient' not in day_corr_df.columns:
                    print(f"Brak oczekiwanych kolumn w {csv_filename}. Pomijam.")
                    continue

                significant_for_day = day_corr_df[abs(day_corr_df['CorrelationCoefficient']) >= correlation_threshold]
                
                for _, row in significant_for_day.iterrows():
                    feature = row['Feature']
                    corr_val = row['CorrelationCoefficient']
                    if feature not in all_days_significant_corr:
                        all_days_significant_corr[feature] = {s: 0.0 for s in sheet_names_processed} # Inicjalizuj dla wszystkich dni
                    all_days_significant_corr[feature][sheet_name] = corr_val
            except Exception as e:
                print(f"Błąd podczas przetwarzania {csv_filename} dla wykresu: {e}")
                continue
        
        if not all_days_significant_corr:
            print(f"Brak znaczących korelacji (lag=0) dla {target_col} do wykreślenia.")
            plt.close()
            continue

        plot_df = pd.DataFrame(all_days_significant_corr).T # Transpozycja dla łatwiejszego plotowania
        plot_df = plot_df.fillna(0) # Wypełnij brakujące korelacje (poniżej progu w danym dniu) zerem dla wykresu
        
        if plot_df.empty:
            print(f"Brak danych do wykreślenia dla {target_col} po przetworzeniu.")
            plt.close()
            continue

        plot_df.plot(kind='bar', figsize=(max(15, len(plot_df)*0.5), 8)) # Dynamiczna szerokość
        plt.title(f'Znaczące korelacje (lag=0, |corr| >= {correlation_threshold}) dla: {target_col}')
        plt.ylabel('Współczynnik korelacji Pearsona')
        plt.xlabel('Cecha')
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        
        safe_target_name_plot = target_col.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
        plot_filename = os.path.join(output_plot_dir, f"significant_correlations_lag0_{safe_target_name_plot}.png")
        plt.savefig(plot_filename)
        plt.close()
        # print(f"Zapisano wykres znaczących korelacji (lag=0) dla {target_col} do: {plot_filename}")

# --- Główna funkcja sterująca ---
def main():
    """Główna funkcja analizy dla wielu dni."""

    # === Flagi sterujące ===
    PERFORM_OUTLIER_HANDLING = True
    APPLY_MA_AND_DOWNSAMPLE_D6 = True # True: d6 będzie miało próbkowanie 10s. False: d6 pozostanie 2s.
    PERFORM_LAG_ANALYSIS = True
    PERFORM_EDA_PLOTS = True
    PERFORM_LAG0_CORRELATION_PLOTS = True
    # =======================

    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_names_to_process = ['d2', 'd3', 'd5', 'd6'] 
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]
    
    # Lista cech do analizy opóźnień (można dostosować)
    features_for_lag_analysis = [
        "całkowity przepływ pary", "przepływ węgla do młyna A", "przepływ węgla do młyna B", 
        "przepływ węgla do młyna C", "przepływ węgla do młyna D", "przepływ węgla do młyna E", 
        "przepływ węgla do młyna F", "przepływ powietrza pierwotnego", 
        "przepływ powietrza wtórnego - strona A", "przepływ powietrza wtórnego - strona B",
        "kąt wychylenia palnika róg #1", "kąt wychylenia palnika róg #2", 
        "kąt wychylenia palnika róg #3", "kąt wychylenia palnika róg #4",
        "tlen w spalinach - strona A", "tlen w spalinach - strona B"
    ]
    # Definicja opóźnień w SEKUNDACH, które chcemy analizować
    time_lags_to_check_seconds = [10, 20, 30, 60, 120, 180, 300, 600] # np. do 10 minut

    # Definicja początkowych częstotliwości próbkowania (w sekundach)
    initial_sampling_rates_seconds = { 'd2': 10, 'd3': 10, 'd5': 10, 'd6': 2 }
    effective_sampling_rates = initial_sampling_rates_seconds.copy() # Będzie aktualizowane

    # Katalogi wyjściowe
    lag0_correlation_csv_dir = "correlation_results_csv_lag0"
    lagged_correlation_csv_dir = "lagged_correlation_results_csv"
    os.makedirs(lag0_correlation_csv_dir, exist_ok=True)
    if PERFORM_LAG_ANALYSIS:
        os.makedirs(lagged_correlation_csv_dir, exist_ok=True)


    print("Rozpoczynanie analizy dla wielu dni...")
    processed_sheets_for_plots = []

    for sheet_name in sheet_names_to_process:
        print(f"\n{'='*20} Przetwarzanie danych dla dnia: {sheet_name} {'='*20}")
        try:
            df_day_raw = get_full_dataset(file_path, sheet_name)
            
            # Krok: Średnia ruchoma i downsampling dla d6 (jeśli włączone)
            if sheet_name == 'd6' and APPLY_MA_AND_DOWNSAMPLE_D6:
                print(f"--- Stosowanie średniej ruchomej dla arkusza: {sheet_name} (okno 5 próbek) ---")
                numeric_cols_d6 = df_day_raw.select_dtypes(include=np.number).columns
                for col in numeric_cols_d6:
                    df_day_raw[col] = df_day_raw[col].rolling(window=5, min_periods=1).mean()
                
                # Downsampling
                print(f"--- Downsampling danych dla arkusza: {sheet_name} (do 10s) ---")
                df_day_raw = df_day_raw.iloc[::5, :].reset_index(drop=True)
                effective_sampling_rates[sheet_name] = 10 # Aktualizacja efektywnej częstotliwości
            
            current_sheet_sampling_rate = effective_sampling_rates[sheet_name]
            print(f"Efektywna częstotliwość próbkowania dla '{sheet_name}': {current_sheet_sampling_rate}s")

            # Krok: Obsługa wartości odstających (jeśli włączone)
            if PERFORM_OUTLIER_HANDLING:
                print(f"--- Obsługa wartości odstających dla arkusza: {sheet_name} ---")
                df_day = handle_outliers_iqr_capping(df_day_raw) # Domyślnie wszystkie numeryczne
            else:
                df_day = df_day_raw.copy() # Pracuj na kopii jeśli nie ma outlier handlingu

            # Krok: Eksploracyjna analiza danych (jeśli włączone)
            if PERFORM_EDA_PLOTS:
                exploratory_data_analysis(df_day, target_columns, sheet_name)

            # Krok: Analiza korelacji opóźnionych (jeśli włączone)
            if PERFORM_LAG_ANALYSIS:
                sample_lags_for_sheet = []
                time_lags_map_for_sheet = {}
                if current_sheet_sampling_rate > 0:
                    for time_lag_s in time_lags_to_check_seconds:
                        num_samples = max(1, int(round(time_lag_s / current_sheet_sampling_rate)))
                        if num_samples not in sample_lags_for_sheet:
                            sample_lags_for_sheet.append(num_samples)
                            time_lags_map_for_sheet[num_samples] = time_lag_s
                    sample_lags_for_sheet.sort()
                    
                    # Ostrzeżenie jeśli analizujemy d6 bez downsamplingu i jego rate jest inny
                    if sheet_name == 'd6' and not APPLY_MA_AND_DOWNSAMPLE_D6 and current_sheet_sampling_rate != initial_sampling_rates_seconds.get('d2',10): # Porównaj z typowym rate
                         print(f"  OSTRZEŻENIE (Arkusz: {sheet_name}): Analiza opóźnień użyje próbkowania {current_sheet_sampling_rate}s. "
                               "Opóźnienia w 'próbkach' będą odpowiadać innym czasom rzeczywistym niż dla arkuszy z próbkowaniem np. 10s.")
                
                    analyze_and_report_lagged_correlations(
                        df_day, 
                        input_features=features_for_lag_analysis, 
                        target_columns=target_columns, 
                        sample_lags_for_sheet=sample_lags_for_sheet,
                        time_lags_seconds_map=time_lags_map_for_sheet,
                        sheet_name=sheet_name,
                        output_dir=lagged_correlation_csv_dir
                    )
                else:
                    print(f"  Ostrzeżenie (Arkusz: {sheet_name}): Nieprawidłowa częstotliwość próbkowania ({current_sheet_sampling_rate}s). Pomijam analizę opóźnień.")

            # Krok: Obliczanie "standardowych" korelacji (lag=0)
            # print(f"--- Obliczanie korelacji (lag=0) dla arkusza: {sheet_name} ---") # Mniej gadatliwe
            correlations_lag0_for_day = correlation_data_for_csv(df_day, target_columns)
            for target_col, corr_series in correlations_lag0_for_day.items():
                safe_target_name = target_col.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
                csv_filename = os.path.join(lag0_correlation_csv_dir, f"correlations_lag0_{sheet_name}_{safe_target_name}.csv")
                
                if not corr_series.empty:
                    corr_series.name = 'CorrelationCoefficient'
                    corr_series.index.name = 'Feature'
                    corr_series.to_csv(csv_filename, header=True, float_format='%.4f')
                    # print(f"Zapisano korelacje (lag=0) dla {target_col} (arkusz: {sheet_name}) do: {csv_filename}")
            
            processed_sheets_for_plots.append(sheet_name)

        except FileNotFoundError as e:
            print(f"BŁĄD KRYTYCZNY: {e}")
        except ValueError as e:
            print(f"BŁĄD DANYCH: {e}")
        except Exception as e:
            print(f"Nieoczekiwany problem podczas przetwarzania arkusza {sheet_name}: {e}")
            import traceback
            traceback.print_exc() 
    
    print("\nPrzetwarzanie dzienne zakończone.")
    
    # Krok: Tworzenie wykresów znaczących korelacji (lag=0) (jeśli włączone)
    if PERFORM_LAG0_CORRELATION_PLOTS and processed_sheets_for_plots:
        plot_significant_correlations_for_days(
            sheet_names_processed=processed_sheets_for_plots,
            target_columns=target_columns,
            input_csv_dir=lag0_correlation_csv_dir, # Użyj katalogu z korelacjami lag=0
            correlation_threshold=0.2 
        )
if __name__ == "__main__":
    main()