import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

def prepare_and_split_data_stratified(
    df: pd.DataFrame, 
    feature_cols: list, 
    target_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42,
    regime_threshold: float = 0.2
):
    """
    Przygotowuje dane (czyści, konwertuje) i dzieli na zbiory uczący oraz testowy
    metodą stratyfikowaną względem wykrytego "reżimu" (zmiany w kolumnie docelowej).

    Args:
        df (pd.DataFrame): Ramka danych wejściowych.
        feature_cols (list): Lista nazw kolumn cech.
        target_col (str): Nazwa kolumny docelowej.
        test_size (float): Procent danych przeznaczony na zbiór testowy.
        random_state (int): Ziarno losowości dla powtarzalności.
        regime_threshold (float): Próg dla `diff()` do wykrycia zmiany reżimu.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) lub (None, None, None, None) w przypadku błędu.
    """
    all_required_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        print(f"Brakujące kolumny w DataFrame: {missing_cols} dla celu {target_col}. Pomijanie.")
        return None, None, None, None

    data = df[all_required_cols].copy()

    # Konwersja na typ numeryczny, błędy zamieniane na NaN
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.dropna(inplace=True)

    if data.empty or len(data) < 2:
        print(f"Brak wystarczających danych dla {target_col} po usunięciu NaN. Pomijanie.")
        return None, None, None, None

    if target_col not in data.columns:
        print(f"Kolumna docelowa {target_col} nie istnieje po czyszczeniu danych. Pomijanie.")
        return None, None, None, None
        
    # Wykrywanie reżimu
    # Używamy oryginalnej serii target_col z DataFrame `data` przed ekstrakcją X, y
    diff_series = data[target_col].diff().abs().fillna(0) 
    change_idx = diff_series > regime_threshold
    data['regime'] = change_idx.cumsum().clip(upper=1)
    
    # Sprawdzenie, czy mamy co najmniej dwa różne reżimy do stratyfikacji
    # train_test_split wymaga co najmniej 2 klas w `stratify` array, jeśli jest używany.
    if data['regime'].nunique() < 2:
        print(f"Ostrzeżenie: Wykryto mniej niż 2 reżimy dla {target_col} (liczba unikalnych reżimów: {data['regime'].nunique()}). "
              "Przeprowadzam standardowy podział losowy zamiast stratyfikowanego.")
        stratify_labels = None
    else:
        stratify_labels = data['regime']

    X = data[feature_cols]
    y = data[target_col]

    if X.empty or len(X) < 2:
        print(f"Niewystarczająca ilość danych (X) dla {target_col} po preprocessingu. Pomijanie.")
        return None, None, None, None

    # Dostosowanie test_size dla małych zbiorów danych
    actual_test_size = test_size
    min_samples_for_split = 2 # train_test_split wymaga co najmniej 1 próbki w każdym zbiorze, jeśli stratify jest używane
    if stratify_labels is not None: # Dla stratyfikacji, każda klasa musi mieć co najmniej 2 próbki
        min_samples_per_class = data['regime'].value_counts().min()
        if min_samples_per_class < 2 :
             print(f"Ostrzeżenie: Co najmniej jedna klasa reżimu dla {target_col} ma mniej niż 2 próbki ({data['regime'].value_counts().to_dict()}). "
                   "Przeprowadzam standardowy podział losowy.")
             stratify_labels = None # Wracamy do standardowego podziału
    
    if len(X) < 5: # Ogólne ostrzeżenie dla małych zbiorów
        print(f"Ostrzeżenie: Mała ilość danych ({len(X)} próbek) dla {target_col}. Podział może nie być optymalny.")
        if len(X) <= 1:
            print(f"Nie można podzielić danych dla {target_col} - za mało próbek. Pomijanie.")
            return None, None, None, None
        # Jeśli po wszystkich sprawdzeniach nadal chcemy wymusić test_size
        if int(len(X) * actual_test_size) == 0 and len(X) > 1 :
             actual_test_size = 1 / len(X) # Zapewnia co najmniej 1 próbkę w teście

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=actual_test_size, 
            random_state=random_state, 
            stratify=stratify_labels, # Użyj etykiet reżimu do stratyfikacji
            shuffle=True # Mieszanie jest nadal zalecane, nawet przy stratyfikacji
        )
    except ValueError as e:
        print(f"Błąd podczas podziału danych dla {target_col} (prawdopodobnie z powodu stratyfikacji i małej liczby próbek w klasach): {e}. "
              "Próba standardowego podziału losowego.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=actual_test_size, random_state=random_state, shuffle=True
            )
        except Exception as final_e:
            print(f"Nie udało się podzielić danych dla {target_col} nawet standardowo: {final_e}")
            return None, None, None, None


    if X_train.empty or X_test.empty:
        print(f"Zbiór treningowy lub testowy jest pusty dla {target_col} po podziale. Pomijanie.")
        return None, None, None, None
        
    return X_train, X_test, y_train, y_test