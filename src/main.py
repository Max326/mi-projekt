import os
from id.id import process_identification

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
    process_identification(file_path, sheet_names_to_process, target_columns, daily_csv_dir)

    # Modelowanie 

if __name__ == "__main__":
    main()