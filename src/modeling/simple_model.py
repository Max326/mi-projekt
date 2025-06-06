import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_evaluate_linear_model(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Trenuje i ocenia model regresji liniowej.

    Args:
        df (pd.DataFrame): Ramka danych zawierająca dane.
        feature_cols (list): Lista nazw kolumn cech.
        target_col (str): Nazwa kolumny docelowej.

    Returns:
        tuple: ( wytrenowany model, słownik z metrykami ) lub (None, {}) jeśli błąd.
    """
    all_required_cols = feature_cols + [target_col]
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        print(f"Brakujące kolumny w DataFrame: {missing_cols}. Pomijanie modelowania dla {target_col}.")
        return None, {}

    data = df[all_required_cols].copy()
    
    # Konwersja na typ numeryczny, błędy zamieniane na NaN
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    data.dropna(inplace=True) # Proste usunięcie wierszy z brakującymi wartościami

    if data.empty or len(data) < 2: # Potrzeba co najmniej 2 próbek
        print(f"Brak wystarczających danych dla modelowania celu {target_col} po usunięciu NaN. Pomijanie.")
        return None, {}

    X = data[feature_cols]
    y = data[target_col]

    if X.empty or len(X) < 2:
        print(f"Niewystarczająca ilość danych do trenowania modelu dla {target_col} po preprocessingu. Pomijanie.")
        return None, {}
    
    # Podział danych
    # test_size=0.2 wymaga co najmniej 2 próbki w X i y, aby uniknąć pustego zbioru testowego dla małych danych
    # Jeśli jest bardzo mało danych (np. <5), podział może być problematyczny.
    actual_test_size = 0.2
    if len(X) < 5: # Minimalna liczba próbek, aby podział miał sens
        print(f"Ostrzeżenie: Mała ilość danych ({len(X)} próbek) dla {target_col}. Podział może nie być optymalny.")
        if len(X) <= 1: # Nie można podzielić
             print(f"Nie można podzielić danych dla {target_col} - za mało próbek. Pomijanie.")
             return None, {}
        # Dla bardzo małych zbiorów, można rozważyć mniejszy test_size lub inną strategię walidacji
        # Na razie zostawiamy 0.2, ale sklearn może rzucić błąd, jeśli test set będzie pusty.
        # Alternatywnie, można wymusić co najmniej 1 próbkę w teście, jeśli to możliwe.
        if int(len(X) * actual_test_size) == 0 and len(X) > 1:
            actual_test_size = 1 / len(X) # Zapewnia co najmniej 1 próbkę w teście, jeśli X ma > 1


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=actual_test_size, random_state=42, shuffle=True)

    if X_train.empty or X_test.empty:
        print(f"Zbiór treningowy lub testowy jest pusty dla {target_col} po podziale. Pomijanie.")
        return None, {}

    # Trenowanie modelu
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ewaluacja modelu
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nModel dla celu: {target_col}")
    print(f"Wybrane cechy: {feature_cols}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    metrics = {'mse': mse, 'r2': r2}
    return model, metrics