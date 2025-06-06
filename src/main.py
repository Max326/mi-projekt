import os
from data_loader import get_full_dataset
from eda import exploratory_data_analysis
from correlation import (
    correlation_data_for_csv,
    aggregate_daily_correlations_to_summary_files,
    plot_significant_correlations_for_days
)

def main():
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_names_to_process = ['d2', 'd3', 'd5', 'd6']
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]
    daily_csv_dir = "correlation_results_csv"
    os.makedirs(daily_csv_dir, exist_ok=True)

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
        except Exception as e:
            print(f"Problem podczas przetwarzania arkusza {sheet_name}: {e}")

    print("\nPrzetwarzanie dziennych korelacji zakończone.")

    # Jeśli chcesz agregować do zbiorczych plików, odkomentuj poniższe:
    # summary_csv_dir = "summary_correlation_results"
    # os.makedirs(summary_csv_dir, exist_ok=True)
    # aggregate_daily_correlations_to_summary_files(
    #     sheet_names_processed=sheet_names_to_process,
    #     target_columns=target_columns,
    #     input_csv_dir=daily_csv_dir,
    #     output_summary_dir=summary_csv_dir
    # )

"""     plot_significant_correlations_for_days(
        sheet_names_processed=sheet_names_to_process,
        target_columns=target_columns,
        input_csv_dir=daily_csv_dir,
        correlation_threshold=0.2
    ) """
if __name__ == "__main__":
    main()