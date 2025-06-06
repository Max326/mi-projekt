import pandas as pd
import numpy as np

def downsample_dataframe(df: pd.DataFrame, window_size: int = 5, aggregation_func='mean') -> pd.DataFrame:
    """
    Przeprowadza downsampling DataFrame, stosując funkcję agregującą (domyślnie średnia krocząca)
    do kolumn numerycznych i wybierając co `window_size`-ty wiersz dla kolumn nienumerycznych.

    Args:
        df (pd.DataFrame): Ramka danych do przetworzenia.
        window_size (int): Rozmiar okna dla średniej kroczącej i krok próbkowania.
        aggregation_func (str): Funkcja agregująca dla .rolling() ('mean', 'median', 'sum', etc.).

    Returns:
        pd.DataFrame: Przetworzona ramka danych.
    """
    if df.empty:
        print("Ostrzeżenie: Przekazano pusty DataFrame do downsamplingu.")
        return df

    df_copy = df.copy()
    
    datetime_col_data = None
    numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols_data = {}

    # Obsługa kolumny 'Date/Time' osobno, jeśli istnieje
    if 'Date/Time' in df_copy.columns:
        try:
            # Próba konwersji na datetime, jeśli jeszcze nie jest
            if not pd.api.types.is_datetime64_any_dtype(df_copy['Date/Time']):
                df_copy['Date/Time'] = pd.to_datetime(df_copy['Date/Time'])
            datetime_col_data = df_copy['Date/Time'].iloc[(window_size - 1)::window_size].reset_index(drop=True)
        except Exception as e:
            print(f"Ostrzeżenie: Nie udało się przetworzyć kolumny 'Date/Time' podczas downsamplingu: {e}")
            # Jeśli 'Date/Time' nie jest numeryczna i nie da się przekonwertować, potraktuj jak inną nienumeryczną
            if 'Date/Time' in numeric_cols: # Powinno być rzadkie, ale dla pewności
                numeric_cols.remove('Date/Time')
            if 'Date/Time' not in non_numeric_cols_data and 'Date/Time' in df_copy.columns:
                 non_numeric_cols_data['Date/Time'] = df_copy['Date/Time'].iloc[(window_size - 1)::window_size].reset_index(drop=True)


    # Próbkowanie innych kolumn nienumerycznych
    for col in df_copy.columns:
        if col not in numeric_cols and col != 'Date/Time': # Upewnij się, że nie przetwarzamy ponownie Date/Time
            non_numeric_cols_data[col] = df_copy[col].iloc[(window_size - 1)::window_size].reset_index(drop=True)

    # Downsampling kolumn numerycznych
    if not numeric_cols:
        print("Ostrzeżenie: Brak kolumn numerycznych do downsamplingu.")
        downsampled_numeric_df = pd.DataFrame()
    else:
        numeric_df = df_copy[numeric_cols]
        # Użycie .agg() dla elastyczności funkcji agregującej
        downsampled_numeric_df = numeric_df.rolling(window=window_size, min_periods=window_size).agg(aggregation_func).iloc[(window_size - 1)::window_size, :].reset_index(drop=True)

    # Łączenie przetworzonych części
    final_parts = []
    if datetime_col_data is not None:
        final_parts.append(datetime_col_data)
    
    # Dodawanie pozostałych kolumn nienumerycznych, dbając o kolejność
    original_non_numeric_order = [col for col in df.columns if col in non_numeric_cols_data]
    for col_name in original_non_numeric_order:
        final_parts.append(non_numeric_cols_data[col_name])
        
    if not downsampled_numeric_df.empty:
        final_parts.append(downsampled_numeric_df)
    
    if not final_parts:
        print("Ostrzeżenie: Brak danych po próbie downsamplingu.")
        return pd.DataFrame(columns=df.columns) # Zwróć pusty DF z oryginalnymi kolumnami

    downsampled_df = pd.concat(final_parts, axis=1)
    
    # Upewnienie się, że kolumny są w oryginalnej kolejności (o ile to możliwe)
    # To może być skomplikowane, jeśli niektóre kolumny zostały usunięte lub zmienione
    # Prostsze podejście: połącz i pozwól pandas zdecydować, lub zdefiniuj oczekiwaną kolejność
    # Na razie zachowujemy kolejność z `final_parts`
    
    # Usuń wiersze, które mogły stać się całkowicie NaN w części numerycznej
    if not downsampled_numeric_df.empty:
        downsampled_df.dropna(how='all', subset=downsampled_numeric_df.columns, inplace=True)
    
    return downsampled_df