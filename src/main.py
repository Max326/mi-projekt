import os
import pandas as pd
from id.id import process_identification
from data_load.data_loader import load_excel_data
from modeling.simple_model import train_evaluate_linear_model

def main():
    # Dane wejściowe
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_names_to_process = ['d2', 'd3', 'd5', 'd6']
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]
    daily_csv_dir = "correlation_results_csv"

    # Identyfikacja (EDA + korelacje)
    # process_identification(file_path, sheet_names_to_process, target_columns, daily_csv_dir)
    print("Faza identyfikacji pominięta w tej konfiguracji.")

    # Modelowanie 
    print("\n--- Rozpoczęcie modelowania ---")
    
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
            sheet_df = all_sheets_data[sheet_name]
            data_frames_to_concat.append(sheet_df)
        else:
            print(f"Ostrzeżenie: Arkusz '{sheet_name}' nie znaleziony w pliku {file_path}.")

    if not data_frames_to_concat:
        print("Nie załadowano żadnych danych do modelowania. Zakończenie.")
        return

    combined_df = pd.concat(data_frames_to_concat, ignore_index=True)

    if 'Date/Time' in combined_df.columns:
        combined_df_for_modeling = combined_df.drop(columns=['Date/Time'])
    else:
        combined_df_for_modeling = combined_df.copy()
        
    # Definicja cech na podstawie analizy korelacji
    features_target_A = [
        "całkowity przepływ pary",
        "ciśnienie wody wtryskowej do pary wtórnej",
        "klapy wentlatora podmuchu - strona A",
        "kąt wychylenia palnika róg #2",
        "przepływ węgla do młyna B",
        "temperatura wlotowa powietrza - strona A"
    ]

    features_target_B = [
        "całkowity przepływ pary",
        "ciśnienie wody wtryskowej do pary wtórnej",
        "ciśnienie wody wtryskowej do pary świeżej",
        "przepływ powietrza pierwotnego",
        "przepływ węgla do młyna E",
        "tlen w spalinach - strona B"
    ]

    target_A = "temperatura wylotowa spalin - strona A"
    target_B = "temperatura wylotowa spalin - strona B"

    print(f"\n--- Modelowanie dla: {target_A} ---")
    model_A, metrics_A = train_evaluate_linear_model(combined_df_for_modeling.copy(), features_target_A, target_A)
    if model_A:
        print(f"Metryki dla {target_A}: {metrics_A}")

    print(f"\n--- Modelowanie dla: {target_B} ---")
    model_B, metrics_B = train_evaluate_linear_model(combined_df_for_modeling.copy(), features_target_B, target_B)
    if model_B:
        print(f"Metryki dla {target_B}: {metrics_B}")

if __name__ == "__main__":
    main()