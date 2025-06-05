import pandas as pd
import os
from typing import List, Union, Optional

def load_excel_data(file_path: str) -> dict:
    """
    Load all worksheets from the Excel file into a dictionary of DataFrames.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Dictionary with worksheet names as keys and pandas DataFrames as values
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plik {file_path} nie został znaleziony.")
    
    # Load all sheets into a dictionary of DataFrames
    excel_data = pd.read_excel(file_path, sheet_name=None)
    return excel_data

def get_column_data(file_path: str, column_name: str, sheet_name: str) -> pd.Series:
    """
    Get data from a specific column in a specific worksheet.
    
    Args:
        file_path: Path to the Excel file
        column_name: Name of the column to extract
        sheet_name: Name of the worksheet
        
    Returns:
        Series containing the column data
    """
    # Load all the data
    excel_data = load_excel_data(file_path)
    
    # Check if the sheet exists
    if sheet_name not in excel_data:
        raise ValueError(f"Arkusz '{sheet_name}' nie istnieje w pliku. "
                         f"Dostępne arkusze: {list(excel_data.keys())}")
    
    # Get the specified worksheet
    df = excel_data[sheet_name]
    
    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Kolumna '{column_name}' nie istnieje w arkuszu '{sheet_name}'. "
                         f"Dostępne kolumny: {list(df.columns)}")
    
    # Return the column data
    return df[column_name]

def main():
    """Main function to demonstrate how to use the data extraction."""
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    
    try:
        # Example usage
        column_name = "całkowity przepływ pary"  # przykładowa nazwa kolumny
        sheet_name = "d2"  # przykładowa nazwa arkusza
        
        data = get_column_data(file_path, column_name, sheet_name)
        print(f"Dane z kolumny '{column_name}' w arkuszu '{sheet_name}':")
        print(data.head())  # Pokazuje pierwsze 5 wierszy
        
        # Możesz również wykorzystać te dane do dalszej analizy
        print(f"Średnia wartość: {data.mean()}")
        print(f"Minimalna wartość: {data.min()}")
        print(f"Maksymalna wartość: {data.max()}")
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    main()