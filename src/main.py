import os
import pandas as pd
import numpy as np # Upewnij się, że ten import jest na górze pliku
from data_load.data_loader import load_excel_data 
from modeling.simple_model import train_evaluate_linear_model, train_evaluate_polynomial_model # Dodano train_evaluate_polynomial_model
from modeling.advanced_models import train_evaluate_random_forest_model, train_evaluate_gradient_boosting_model, train_evaluate_dynamic_arx_model
from utils.plotting_utils import ensure_dir, plot_outlier_visualization
from utils.data_preprocessing import downsample_dataframe, remove_outliers_iqr 
from id.eda import exploratory_data_analysis, plot_downsampled_data_comparison 

def main():
    # --- Config ---
    MODEL_TYPE = "dynamic_arx_linear"  # "linear", "polynomial", "random_forest", "gradient_boosting", "dynamic_arx_linear", "dynamic_arx_polynomial"
    
    # Dynamic ARX model configuration
    ARX_PARAMS = {
        "na": 3,
        "nb": 2,
        "nk": 1,
        "poly_degree": 2
    }

    TARGET_SAMPLING_PERIOD = '10s' # Konieczne do ujednolicenia danych
    
    POLYNOMIAL_DEGREE = 2
    
    PLOT_RESULTS = True 
    PLOTS_DIR = "plots_output" 
    
    DOWNSAMPLE_SHEET_NAME = 'd6' 
    DOWNSAMPLE_WINDOW = 5       
    
    EDA_OUTPUT_DIR = "eda_plots" 
    DOWNSAMPLED_COMPARISON_DIR = "downsampled_comparison_plots" 

    PLOT_INPUT_DATA_VISUALIZATION = False # use it to veirfy new inputs!
    INPUT_VISUALIZATION_DIR = "input_data_visualizations" 
    HIGHLIGHT_OUTLIERS_ON_INPUT_PLOTS = True 

    APPLY_OUTLIER_REMOVAL = False # used with current inputs may only cause harm xd
    COLUMNS_FOR_VISUALIZATION_AND_OUTLIERS = [
        "całkowity przepływ pary", 
        "ciśnienie wody wtryskowej do pary wtórnej",
        "ciśnienie wody wtryskowej do pary świeżej",
        "przepływ powietrza pierwotnego",
        "przepływ węgla do młyna B",
        "przepływ węgla do młyna E",
        "temperatura wlotowa powietrza - strona A",
        "tlen w spalinach - strona B",
        "temperatura wylotowa spalin - strona A", 
        "temperatura wylotowa spalin - strona B"
    ]
    IQR_MULTIPLIER_FOR_OUTLIERS = 4.5

    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_names_to_process = ['d2', 'd3', 'd5', 'd6']
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]

    # --- Ładowanie i przygotowanie danych ---
    try:
        all_sheets_data = load_excel_data(file_path)
    except Exception as e:
        print(f"Błąd ładowania danych: {e}")
        return

    # <<< NOWA, KLUCZOWA SEKCJA: CZYSZCZENIE DANYCH >>>
    print("\n--- Rozpoczęcie czyszczenia danych ---")
    cleaned_sheets_data = {}
    for sheet_name, df in all_sheets_data.items():
        if sheet_name in sheet_names_to_process:
            print(f"Czyszczenie arkusza: {sheet_name}")
            df_copy = df.copy()
            
            # Krok 1: Zastąp tekst ' Bad Data' wartością NaN, aby można było z nim pracować
            df_copy.replace(to_replace=r'.*Bad Data.*', value=np.nan, regex=True, inplace=True)
            
            # Krok 2: Upewnij się, że wszystkie kolumny, które powinny być numeryczne, są numeryczne
            # Wybieramy wszystkie kolumny oprócz 'Date/Time' do konwersji
            cols_to_convert = [col for col in df_copy.columns if col != 'Date/Time']
            for col in cols_to_convert:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

            # Krok 3: Inteligentna imputacja - wypełnianie brakujących danych (NaN)
            # Używamy forward fill, który propaguje ostatnią poprawną wartość.
            # To jest standardowe i dobre podejście dla danych procesowych, gdzie wartości
            # nie zmieniają się gwałtownie z próbki na próbkę.
            df_copy.ffill(inplace=True)
            
            # Krok 4: Na wszelki wypadek, jeśli na samym początku były NaN, używamy back fill
            df_copy.bfill(inplace=True)
            
            cleaned_sheets_data[sheet_name] = df_copy
            
    # Zastępujemy słownik z danymi na ten wyczyszczony
    all_sheets_data = cleaned_sheets_data
    # <<< KONIEC NOWEJ SEKCJI >>>

    data_frames_to_concat = []
    for sheet_name in sheet_names_to_process:
        if sheet_name in all_sheets_data:
            sheet_df = all_sheets_data[sheet_name] # Nie trzeba już .copy()
            
            sheet_df['session_id'] = sheet_name

            if 'Date/Time' in sheet_df.columns:
                sheet_df['Date/Time'] = pd.to_datetime(sheet_df['Date/Time'], dayfirst=True)
                sheet_df.set_index('Date/Time', inplace=True)
                
                # Resampling teraz działa na czystych danych
                numeric_cols = sheet_df.select_dtypes(include='number').columns
                resampled_numeric = sheet_df[numeric_cols].resample(TARGET_SAMPLING_PERIOD).mean()
                resampled_session = sheet_df[['session_id']].resample(TARGET_SAMPLING_PERIOD).first().ffill()
                sheet_df = pd.concat([resampled_numeric, resampled_session], axis=1).reset_index()
            
            data_frames_to_concat.append(sheet_df)


    if not data_frames_to_concat:
        print("Nie załadowano danych. Zakończenie.")
        return

    combined_df = pd.concat(data_frames_to_concat, ignore_index=True)
    print(f"Rozmiar połączonego DataFrame przed przetwarzaniem outlierów: {combined_df.shape}")

    if PLOT_INPUT_DATA_VISUALIZATION and not combined_df.empty:
        print("\n--- Generowanie wizualizacji danych wejściowych/outlierów ---")
        for col_name in COLUMNS_FOR_VISUALIZATION_AND_OUTLIERS:
            if col_name in combined_df.columns and pd.api.types.is_numeric_dtype(combined_df[col_name]):
                series_data = combined_df[col_name]
                
                outlier_mask_param = None
                method_name_param = None
                lower_b_param = None
                upper_b_param = None

                if APPLY_OUTLIER_REMOVAL and HIGHLIGHT_OUTLIERS_ON_INPUT_PLOTS:
                    original_col_data_no_nan = series_data.dropna()
                    if not original_col_data_no_nan.empty:
                        Q1 = original_col_data_no_nan.quantile(0.25)
                        Q3 = original_col_data_no_nan.quantile(0.75)
                        IQR_val = Q3 - Q1
                        if IQR_val > 0: 
                            lower_b_param = Q1 - IQR_MULTIPLIER_FOR_OUTLIERS * IQR_val
                            upper_b_param = Q3 + IQR_MULTIPLIER_FOR_OUTLIERS * IQR_val
                            
                            outlier_mask_param = pd.Series(False, index=series_data.index)
                            non_na_indices = series_data.notna()
                            condition = (series_data.loc[non_na_indices] < lower_b_param) | \
                                        (series_data.loc[non_na_indices] > upper_b_param)
                            outlier_mask_param.loc[non_na_indices] = condition
                            method_name_param = "IQR"
                        else:
                            print(f"IQR dla '{col_name}' wynosi 0, nie można zaznaczyć outlierów na wizualizacji.")
                
                plot_outlier_visualization(
                    series_data=series_data,
                    column_name=col_name,
                    plots_dir=INPUT_VISUALIZATION_DIR,
                    outlier_mask=outlier_mask_param,
                    method_name=method_name_param,
                    lower_b=lower_b_param,
                    upper_b=upper_b_param
                )
            else:
                print(f"Kolumna '{col_name}' nie istnieje w DataFrame lub nie jest numeryczna - pomijanie wizualizacji.")

    if APPLY_OUTLIER_REMOVAL and not combined_df.empty:
        print("\n--- Usuwanie outlierów z połączonego DataFrame ---")
        actual_cols_for_removal = [col for col in COLUMNS_FOR_VISUALIZATION_AND_OUTLIERS if col in combined_df.columns]
        if not actual_cols_for_removal:
            print("Brak zdefiniowanych kolumn do usuwania outlierów lub kolumny nie istnieją. Pomijanie.")
        else:
            combined_df = remove_outliers_iqr(
                combined_df, 
                actual_cols_for_removal, 
                iqr_multiplier=IQR_MULTIPLIER_FOR_OUTLIERS
            )
            print(f"Rozmiar DataFrame po usunięciu outlierów: {combined_df.shape}")
            if combined_df.empty:
                print("DataFrame jest pusty po usunięciu outlierów. Zatrzymywanie przetwarzania.")
                return
            
            current_target_cols_for_eda_after_outliers = [tc for tc in target_columns if tc in combined_df.columns]
            if PLOT_RESULTS and current_target_cols_for_eda_after_outliers:
                exploratory_data_analysis(combined_df, current_target_cols_for_eda_after_outliers, "combined_after_outliers", output_dir=EDA_OUTPUT_DIR)

    if MODEL_TYPE.startswith("dynamic_arx"):
        print(f"\n--- Rozpoczęcie modelowania dynamicznego: {MODEL_TYPE} ---")

        # KLUCZOWA ZMIANA: Osobne wejścia dla każdego celu!
        # Poniżej przykład - musisz je dobrać na podstawie wiedzy o obiekcie.
        features_target_A = [
            "kąt wychylenia palnika róg #1", "kąt wychylenia palnika róg #2", "kąt wychylenia palnika róg #3", "kąt wychylenia palnika róg #4",
            "klapy wentlatora podmuchu - strona A", "klapy wentlatora podmuchu - strona B",
            "przepływ powietrza pierwotnego", "ciśnienie wody wtryskowej do pary świeżej",
            "temperatura za wtryskiem pary wtórnej - strona L ", 
            "temperatura za wtryskiem pary wtórnej - strona P"
        ]
        target_A = "temperatura wylotowa spalin - strona A"

        features_target_B = [
            "kąt wychylenia palnika róg #1", "kąt wychylenia palnika róg #2", "kąt wychylenia palnika róg #3", "kąt wychylenia palnika róg #4",
            "klapy wentlatora podmuchu - strona A", "klapy wentlatora podmuchu - strona B",
            "przepływ powietrza pierwotnego", "ciśnienie wody wtryskowej do pary świeżej",
            "temperatura za wtryskiem pary wtórnej - strona L ", 
            "temperatura za wtryskiem pary wtórnej - strona P"
        ]
        target_B = "temperatura wylotowa spalin - strona B"
        
        targets_and_features_arx = {
            target_A: features_target_A,
            target_B: features_target_B
        }

        TRAIN_DAYS = ['d3', 'd6']
        TEST_DAYS = ['d2']

        # Pętla trenująca osobny model dla każdego celu
        for target_col, input_features in targets_and_features_arx.items():
            # ZMIANA: Przekazujemy oryginalny combined_df bez resetowania indeksu
            train_evaluate_dynamic_arx_model(
                df=combined_df, 
                input_features=input_features,
                target_col=target_col,
                arx_params=ARX_PARAMS,
                model_type=MODEL_TYPE,
                train_sessions=TRAIN_DAYS,
                test_sessions=TEST_DAYS,
                plots_dir=PLOTS_DIR,
                plot_results=PLOT_RESULTS
            )
    
    else: # Logika dla modeli statycznych
        df_for_modeling = combined_df.drop(columns=['Date/Time', 'session_id'], errors='ignore')
            
        features_target_A = [
            # "temperatura mieszanki za młynem A", "temperatura mieszanki za młynem F", 
            # "temperatura mieszanki za młynem E",
            "kąt wychylenia palnika róg #1", "kąt wychylenia palnika róg #2", 
            "kąt wychylenia palnika róg #3", "kąt wychylenia palnika róg #4",
            "klapy wentlatora podmuchu - strona A", "klapy wentlatora podmuchu - strona B", # Dodaję obie dla pewności
            "przepływ powietrza pierwotnego",
            "ciśnienie wody wtryskowej do pary świeżej",
            "temperatura za wtryskiem pary wtórnej - strona L", 
            "temperatura za wtryskiem pary wtórnej - strona P"
        ]
        target_A = "temperatura wylotowa spalin - strona A"

        features_target_B = [
            # "temperatura mieszanki za młynem A", "temperatura mieszanki za młynem F", 
            # "temperatura mieszanki za młynem E",
            "kąt wychylenia palnika róg #1", "kąt wychylenia palnika róg #2", 
            "kąt wychylenia palnika róg #3", "kąt wychylenia palnika róg #4",
            "klapy wentlatora podmuchu - strona A", "klapy wentlatora podmuchu - strona B", # Dodaję obie dla pewności
            "przepływ powietrza pierwotnego",
            "ciśnienie wody wtryskowej do pary świeżej",
            "temperatura za wtryskiem pary wtórnej - strona L", 
            "temperatura za wtryskiem pary wtórnej - strona P"
        ]
        target_B = "temperatura wylotowa spalin - strona B"

        targets_and_features_all = {target_A: features_target_A, target_B: features_target_B}
        active_targets_and_features = {}

        for target_col, feature_cols in targets_and_features_all.items():
            if target_col in df_for_modeling.columns:
                existing_features = [f_col for f_col in feature_cols if f_col in df_for_modeling.columns]
                if len(existing_features) == len(feature_cols):
                    active_targets_and_features[target_col] = existing_features
                elif existing_features:
                    print(f"Ostrzeżenie: Dla celu '{target_col}' modelowanie z częścią cech: {existing_features}.")
                    active_targets_and_features[target_col] = existing_features
                else:
                    print(f"Ostrzeżenie: Dla celu '{target_col}' brak cech. Pomijanie.")
            else:
                print(f"Cel '{target_col}' nie istnieje w danych. Pomijanie.")

        for target_col, feature_cols in active_targets_and_features.items():
            print(f"\n--- Modelowanie dla: {target_col} ---")
            
            if df_for_modeling.empty:
                print(f"DataFrame dla modelowania celu {target_col} jest pusty. Pomijanie.")
                continue
            
            if target_col not in df_for_modeling.columns or not all(f_col in df_for_modeling.columns for f_col in feature_cols):
                print(f"Brak wymaganych kolumn dla {target_col} w df_for_modeling. Pomijanie.")
                continue

            model = None
            metrics = {}

            if MODEL_TYPE == "linear":
                model, metrics = train_evaluate_linear_model(
                    df_for_modeling, feature_cols, target_col, PLOTS_DIR, PLOT_RESULTS
                )
            elif MODEL_TYPE == "polynomial": # NOWA GAŁĄŹ
                model, metrics = train_evaluate_polynomial_model(
                    df_for_modeling, 
                    feature_cols, 
                    target_col, 
                    PLOTS_DIR, 
                    PLOT_RESULTS,
                    degree=POLYNOMIAL_DEGREE
                )
            elif MODEL_TYPE == "random_forest":
                model, metrics = train_evaluate_random_forest_model(
                    df_for_modeling, feature_cols, target_col, PLOTS_DIR, PLOT_RESULTS
                )
            elif MODEL_TYPE == "gradient_boosting":
                model, metrics = train_evaluate_gradient_boosting_model(
                    df_for_modeling, feature_cols, target_col, PLOTS_DIR, PLOT_RESULTS
                )
            else:
                print(f"Nieznany typ modelu: {MODEL_TYPE}.")
                continue

if __name__ == "__main__":
    main()