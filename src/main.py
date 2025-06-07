import os
import pandas as pd
from data_load.data_loader import load_excel_data 
from modeling.simple_model import train_evaluate_linear_model, train_evaluate_polynomial_model # Dodano train_evaluate_polynomial_model
from modeling.advanced_models import train_evaluate_random_forest_model, train_evaluate_gradient_boosting_model 
from utils.plotting_utils import ensure_dir, plot_outlier_visualization
from utils.data_preprocessing import downsample_dataframe, remove_outliers_iqr 
from id.eda import exploratory_data_analysis, plot_downsampled_data_comparison 

def main():
    # --- Config ---
    MODEL_TYPE = "polynomial"  # "linear", "polynomial", "random_forest", "gradient_boosting"
    POLYNOMIAL_DEGREE = 5
    
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
    sheet_names_to_process = ['d2', 'd3', 'd5'] 
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]

    if PLOT_RESULTS:
        ensure_dir(PLOTS_DIR)
    ensure_dir(EDA_OUTPUT_DIR) 
    if DOWNSAMPLE_SHEET_NAME in sheet_names_to_process and PLOT_RESULTS: 
        ensure_dir(DOWNSAMPLED_COMPARISON_DIR)
    if PLOT_INPUT_DATA_VISUALIZATION: 
        ensure_dir(INPUT_VISUALIZATION_DIR)

    print("Faza identyfikacji pominięta w tej konfiguracji.")
    if MODEL_TYPE == "polynomial":
        print(f"\n--- Rozpoczęcie modelowania z użyciem: {MODEL_TYPE} (stopień: {POLYNOMIAL_DEGREE}) ---")
    else:
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

            if sheet_name == 'd2':
                all_b_cols_to_drop = [col for col in sheet_df.columns if 'strona B' in col]
                existing_b_cols_to_drop = [col for col in all_b_cols_to_drop if col in sheet_df.columns]
                if existing_b_cols_to_drop:
                    sheet_df.drop(columns=existing_b_cols_to_drop, inplace=True, errors='ignore')
                    print(f"Wykluczono kolumny dotyczące strony B dla arkusza {sheet_name}: {existing_b_cols_to_drop}")
            
            current_target_cols_for_eda = [tc for tc in target_columns if tc in sheet_df.columns]
            if current_target_cols_for_eda:
                 exploratory_data_analysis(sheet_df, current_target_cols_for_eda, f"{sheet_name}_original", output_dir=EDA_OUTPUT_DIR)

            if sheet_name == DOWNSAMPLE_SHEET_NAME and DOWNSAMPLE_SHEET_NAME in sheet_names_to_process:
                print(f"Downsampling arkusza: {sheet_name} z oknem {DOWNSAMPLE_WINDOW} używając średniej.")
                original_for_comparison = sheet_df.copy()
                downsampled_df = downsample_dataframe(sheet_df, window_size=DOWNSAMPLE_WINDOW, aggregation_func='mean')
                if not downsampled_df.empty:
                    current_target_cols_for_downsample_plot = [tc for tc in target_columns if tc in original_for_comparison.columns and tc in downsampled_df.columns]
                    if current_target_cols_for_downsample_plot and PLOT_RESULTS:
                        plot_downsampled_data_comparison(original_for_comparison, downsampled_df, current_target_cols_for_downsample_plot, sheet_name, output_dir=DOWNSAMPLED_COMPARISON_DIR)
                    sheet_df = downsampled_df
                else:
                    print(f"Ostrzeżenie: Arkusz {sheet_name} jest pusty po downsamplingu. Pomijanie.")
                    continue
            
            data_frames_to_concat.append(sheet_df)
        else:
            print(f"Ostrzeżenie: Arkusz '{sheet_name}' nie znaleziony w pliku {file_path}.")

    if not data_frames_to_concat:
        print("Nie załadowano żadnych danych do modelowania. Zakończenie.")
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

    df_for_modeling = combined_df.drop(columns=['Date/Time'], errors='ignore')
        
    features_target_A = [
        "całkowity przepływ pary", "ciśnienie wody wtryskowej do pary wtórnej",
        "klapy wentlatora podmuchu - strona A", "kąt wychylenia palnika róg #2",
        "przepływ węgla do młyna B", "temperatura wlotowa powietrza - strona A"
    ]
    target_A = "temperatura wylotowa spalin - strona A"

    features_target_B = [
        "całkowity przepływ pary", "ciśnienie wody wtryskowej do pary wtórnej",
        "ciśnienie wody wtryskowej do pary świeżej", "przepływ powietrza pierwotnego",
        "przepływ węgla do młyna E", "tlen w spalinach - strona B"
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

        if model:
            print(f"Metryki dla {target_col} ({MODEL_TYPE}): {metrics}")
        else:
            print(f"Nie udało się wytrenować modelu {MODEL_TYPE} dla {target_col}.")

if __name__ == "__main__":
    main()