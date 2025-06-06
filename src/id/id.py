import os
from data_load.data_loader import get_full_dataset
from id.eda import exploratory_data_analysis
from id.correlation import correlation_data_for_csv

def process_identification(file_path: str, sheet_names: list, target_columns: list, output_dir: str) -> None:
    """Przeprowadza identyfikację (EDA + korelacje) dla każdego arkusza."""
    os.makedirs(output_dir, exist_ok=True)

    for sheet_name in sheet_names:
        print(f"\n{'='*20} Przetwarzanie danych dla dnia: {sheet_name} {'='*20}")
        try:
            df_day = get_full_dataset(file_path, sheet_name)
            exploratory_data_analysis(df_day, target_columns, sheet_name)
            correlations_for_day = correlation_data_for_csv(df_day, target_columns)
            for target_col, corr_series in correlations_for_day.items():
                safe_target_name = target_col.replace(" ", "_").replace("-", "_").replace("/", "_").lower()
                csv_filename = os.path.join(output_dir, f"correlations_{sheet_name}_{safe_target_name}.csv")
                corr_df_to_save = corr_series.reset_index()
                corr_df_to_save.columns = ['Feature', 'CorrelationCoefficient']
                corr_df_to_save.to_csv(csv_filename, index=False, float_format='%.4f')
        except Exception as e:
            print(f"Problem podczas przetwarzania arkusza {sheet_name}: {e}")

    print("\nPrzetwarzanie dziennych korelacji zakończone.")