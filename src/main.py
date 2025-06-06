import os
import pandas as pd
import numpy as np
from id.id import process_identification 
from data_load.data_loader import load_excel_data 
from modeling.simple_model import train_evaluate_linear_model 
from modeling.advanced_models import train_evaluate_random_forest_model, train_evaluate_gradient_boosting_model 
from utils.plotting_utils import ensure_dir
from utils.data_preprocessing import downsample_dataframe 

def main():
    # --- Konfiguracja ---
    MODEL_TYPE = "random_forest"  # Choose your fighter: "linear", "random_forest", "gradient_boosting"
    PLOT_RESULTS = True
    PLOTS_DIR = "plots_output" # Katalog na wykresy
    DOWNSAMPLE_SHEET_NAME = 'd6' # Arkusz do downsamplingu
    DOWNSAMPLE_WINDOW = 5       # Okno dla downsamplingu (z 2s na 10s to okno 5)
    # --- Dane wejściowe ---
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_names_to_process = ['d2', 'd3', 'd5', 'd6']
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]
    # daily_csv_dir = "correlation_results_csv" # Odkomentuj jeśli używasz
    if PLOT_RESULTS:
        ensure_dir(PLOTS_DIR)

    # Identyfikacja (EDA + korelacje)
    # process_identification(file_path, sheet_names_to_process, target_columns, daily_csv_dir)
    print("Faza identyfikacji pominięta w tej konfiguracji.")

    # Modelowanie 
    print(f"\n--- Rozpoczęcie modelowania z użyciem: {MODEL_TYPE} ---")
    
    try:
        all_sheets_data = load_excel_data(file_path)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Wystąpił błąd podczas ładowania danych: {e}")
        return

    data_frames_to_concat = []
    for sheet_name in sheet_names_to_process:
        if sheet_name in all_sheets_data:
            sheet_df = all_sheets_data[sheet_name].copy() 

            if sheet_name == DOWNSAMPLE_SHEET_NAME:
                print(f"Downsampling arkusza: {sheet_name} z oknem {DOWNSAMPLE_WINDOW} używając średniej.")
                sheet_df = downsample_dataframe(sheet_df, window_size=DOWNSAMPLE_WINDOW, aggregation_func='mean')
                
                if sheet_df.empty:
                    print(f"Ostrzeżenie: Arkusz {sheet_name} jest pusty po downsamplingu. Pomijanie.")
                    continue
            
            data_frames_to_concat.append(sheet_df)
        else:
            print(f"Ostrzeżenie: Arkusz '{sheet_name}' nie znaleziony w pliku {file_path}.")

    if not data_frames_to_concat:
        print("Nie załadowano żadnych danych do modelowania. Zakończenie.")
        return

    combined_df = pd.concat(data_frames_to_concat, ignore_index=True)
    
    # Usuwanie kolumny 'Date/Time' PRZED przekazaniem do modeli, jeśli nie jest cechą
    # Modele oczekują danych numerycznych (poza specjalnymi przypadkami)
    # Jeśli 'Date/Time' zostało zachowane i jest potrzebne do sortowania, zrób to wcześniej.
    # Tutaj zakładamy, że do modelowania idą tylko cechy numeryczne.
    if 'Date/Time' in combined_df.columns:
        # Sprawdź, czy 'Date/Time' nie jest jedną z cech, zanim usuniesz
        # (na razie zakładamy, że nie jest cechą numeryczną dla modelu)
        combined_df_for_modeling = combined_df.drop(columns=['Date/Time'], errors='ignore')
    else:
        combined_df_for_modeling = combined_df.copy()
        
    # Definicja cech na podstawie analizy korelacji
    features_target_A = [
        "całkowity przepływ pary", "ciśnienie wody wtryskowej do pary wtórnej",
        "klapy wentlatora podmuchu - strona A", "kąt wychylenia palnika róg #2",
        "przepływ węgla do młyna B", "temperatura wlotowa powietrza - strona A"
    ]

    features_target_B = [
        "całkowity przepływ pary", "ciśnienie wody wtryskowej do pary wtórnej",
        "ciśnienie wody wtryskowej do pary świeżej", "przepływ powietrza pierwotnego",
        "przepływ węgla do młyna E", "tlen w spalinach - strona B"
    ]

    target_A = "temperatura wylotowa spalin - strona A"
    target_B = "temperatura wylotowa spalin - strona B"

    targets_and_features = {
        target_A: features_target_A,
        target_B: features_target_B
    }

    for target_col, feature_cols in targets_and_features.items():
        print(f"\n--- Modelowanie dla: {target_col} ---")
        
        model = None
        metrics = {}

        # Użyj kopii, aby uniknąć modyfikacji między iteracjami celów, jeśli df jest modyfikowany w funkcjach modelujących
        current_df_for_modeling = combined_df_for_modeling.copy()

        if MODEL_TYPE == "linear":
            model, metrics = train_evaluate_linear_model(
                current_df_for_modeling, 
                feature_cols, 
                target_col,
                plots_dir=PLOTS_DIR,
                plot_results=PLOT_RESULTS
            )
        elif MODEL_TYPE == "random_forest":
            model, metrics = train_evaluate_random_forest_model(
                current_df_for_modeling, 
                feature_cols, 
                target_col,
                plots_dir=PLOTS_DIR,
                plot_results=PLOT_RESULTS
            )
        elif MODEL_TYPE == "gradient_boosting":
            model, metrics = train_evaluate_gradient_boosting_model(
                current_df_for_modeling, 
                feature_cols, 
                target_col,
                plots_dir=PLOTS_DIR,
                plot_results=PLOT_RESULTS
            )
        else:
            print(f"Nieznany typ modelu: {MODEL_TYPE}. Wybierz 'linear', 'random_forest' lub 'gradient_boosting'.")
            continue

        if model:
            print(f"Metryki dla {target_col} ({MODEL_TYPE}): {metrics}")
        else:
            print(f"Nie udało się wytrenować modelu {MODEL_TYPE} dla {target_col}.")

if __name__ == "__main__":
    main()