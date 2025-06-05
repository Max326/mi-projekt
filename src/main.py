import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from typing import List, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_excel_data(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plik {file_path} nie został znaleziony.")
    
    # Load all sheets into a dictionary of DataFrames
    excel_data = pd.read_excel(file_path, sheet_name=None)
    return excel_data

def get_full_dataset(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Załaduj pełny dataset z określonego arkusza."""
    excel_data = load_excel_data(file_path)
    
    if sheet_name not in excel_data:
        raise ValueError(f"Arkusz '{sheet_name}' nie istnieje w pliku. "
                         f"Dostępne arkusze: {list(excel_data.keys())}")
    
    return excel_data[sheet_name]

def exploratory_data_analysis(df: pd.DataFrame, target_columns: List[str]) -> None:
    """Przeprowadź eksploracyjną analizę danych."""
    print("=== EKSPLORACYJNA ANALIZA DANYCH ===")
    print(f"Wymiary datasetu: {df.shape}")
    print(f"Brakujące wartości:\n{df.isnull().sum().sum()} total")
    
    # Podstawowe statystyki dla zmiennych docelowych
    for target in target_columns:
        if target in df.columns:
            print(f"\nStatystyki dla {target}:")
            print(df[target].describe())
            
            # Wykres rozkładu
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            df[target].hist(bins=30, alpha=0.7)
            plt.title(f'Rozkład {target}')
            plt.xlabel('Temperatura [°C]')
            plt.ylabel('Częstość')
            
            plt.subplot(1, 2, 2)
            df[target].plot(kind='line', alpha=0.7)
            plt.title(f'Seria czasowa {target}')
            plt.xlabel('Próbka')
            plt.ylabel('Temperatura [°C]')
            
            plt.tight_layout()
            plt.savefig(f'eda_{target.replace(" ", "_").replace("-", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()

def correlation_analysis(df: pd.DataFrame, target_columns: List[str], threshold: float = 0.3) -> dict:
    """Analiza korelacji między zmiennymi."""
    print("\n=== ANALIZA KORELACJI ===")
    
    # Usuń kolumny niebędące liczbami
    numeric_df = df.select_dtypes(include=[np.number])
    
    correlations = {}
    
    for target in target_columns:
        if target in numeric_df.columns:
            # Oblicz korelacje z target variable
            corr_with_target = numeric_df.corr()[target].abs().sort_values(ascending=False)
            
            # Wybierz zmienne o korelacji powyżej threshold
            significant_vars = corr_with_target[corr_with_target > threshold].drop(target)
            
            correlations[target] = significant_vars
            
            print(f"\nZmienne skorelowane z {target} (|r| > {threshold}):")
            for var, corr in significant_vars.head(10).items():
                print(f"  {var}: {corr:.3f}")
            
            # Mapa ciepła dla top 15 zmiennych
            if len(significant_vars) > 0:
                top_vars = list(significant_vars.head(15).index) + [target]
                corr_matrix = numeric_df[top_vars].corr()
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f')
                plt.title(f'Mapa korelacji dla {target}')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(f'correlation_heatmap_{target.replace(" ", "_").replace("-", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
    
    return correlations

def feature_selection(df: pd.DataFrame, target_col: str, n_features: int = 10) -> Tuple[List[str], pd.DataFrame]:
    """Selekcja najważniejszych cech dla danej zmiennej docelowej."""
    print(f"\n=== SELEKCJA CECH DLA {target_col} ===")
    
    # Przygotuj dane
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        raise ValueError(f"Kolumna {target_col} nie istnieje w danych numerycznych")
    
    # Usuń target z features
    features = numeric_df.drop(columns=[target_col])
    target = numeric_df[target_col]
    
    # Usuń wiersze z NaN
    mask = ~(features.isnull().any(axis=1) | target.isnull())
    features_clean = features[mask]
    target_clean = target[mask]
    
    # Normalizacja danych
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_clean),
        columns=features_clean.columns,
        index=features_clean.index
    )
    
    # 1. Univariate Feature Selection
    selector_univariate = SelectKBest(score_func=f_regression, k=min(n_features, len(features_clean.columns)))
    features_selected_univariate = selector_univariate.fit_transform(features_scaled, target_clean)
    selected_features_univariate = features_clean.columns[selector_univariate.get_support()].tolist()
    
    # 2. Recursive Feature Elimination z Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=min(n_features, len(features_clean.columns)))
    rfe.fit(features_scaled, target_clean)
    selected_features_rfe = features_clean.columns[rfe.support_].tolist()
    
    # 3. Feature Importance z Random Forest
    rf.fit(features_scaled, target_clean)
    feature_importance = pd.DataFrame({
        'feature': features_clean.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selected_features_rf = feature_importance.head(n_features)['feature'].tolist()
    
    print("Top cechy (Univariate):", selected_features_univariate[:5])
    print("Top cechy (RFE):", selected_features_rfe[:5])
    print("Top cechy (RF Importance):", selected_features_rf[:5])
    
    # Wykres ważności cech
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Ważność cechy')
    plt.title(f'Ważność cech dla {target_col}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'feature_importance_{target_col.replace(" ", "_").replace("-", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Połącz wszystkie wybrane cechy
    all_selected = list(set(selected_features_univariate + selected_features_rfe + selected_features_rf))
    
    return all_selected, feature_importance

def build_models(df: pd.DataFrame, target_col: str, selected_features: List[str]) -> dict:
    """Zbuduj i porównaj różne modele regresyjne."""
    print(f"\n=== MODELOWANIE DLA {target_col} ===")
    
    # Przygotuj dane
    numeric_df = df.select_dtypes(include=[np.number])
    features = numeric_df[selected_features]
    target = numeric_df[target_col]
    
    # Usuń wiersze z NaN
    mask = ~(features.isnull().any(axis=1) | target.isnull())
    X = features[mask]
    y = target[mask]
    
    # Podział na zbiory treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizacja
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modele do porównania
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTrening modelu: {name}")
        
        # Trenuj model
        if name in ['Linear Regression', 'Ridge', 'Lasso']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Cross-validation na znormalizowanych danych
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Cross-validation na oryginalnych danych
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Metryki
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'model': model,
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred,
            'y_test': y_test
        }
        
        print(f"  R²: {r2:.3f}")
        print(f"  MSE: {mse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  CV R² (mean±std): {cv_mean:.3f}±{cv_std:.3f}")
    
    # Wykres porównania predykcji vs rzeczywiste wartości
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        ax.scatter(result['y_test'], result['predictions'], alpha=0.6)
        ax.plot([result['y_test'].min(), result['y_test'].max()], 
                [result['y_test'].min(), result['y_test'].max()], 'r--', lw=2)
        ax.set_xlabel('Rzeczywiste wartości')
        ax.set_ylabel('Predykcje')
        ax.set_title(f'{name} (R² = {result["r2"]:.3f})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{target_col.replace(" ", "_").replace("-", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def analyze_variable_relationships(df: pd.DataFrame, target_col: str, key_variables: List[str]) -> None:
    """Szczegółowa analiza związków między kluczowymi zmiennymi a target."""
    print(f"\n=== ANALIZA ZWIĄZKÓW ZMIENNYCH DLA {target_col} ===")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Scatter plots dla najważniejszych zmiennych
    n_vars = min(6, len(key_variables))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, var in enumerate(key_variables[:n_vars]):
        if var in numeric_df.columns:
            ax = axes[idx]
            
            # Usuń NaN dla tej pary zmiennych
            mask = ~(numeric_df[var].isnull() | numeric_df[target_col].isnull())
            x_clean = numeric_df[var][mask]
            y_clean = numeric_df[target_col][mask]
            
            if len(x_clean) > 0:
                ax.scatter(x_clean, y_clean, alpha=0.6)
                
                # Dodaj linię trendu
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                ax.plot(x_clean, p(x_clean), "r--", alpha=0.8)
                
                # Oblicz korelację
                corr, p_value = pearsonr(x_clean, y_clean)
                
                ax.set_xlabel(var)
                ax.set_ylabel(target_col)
                ax.set_title(f'{var}\nr = {corr:.3f}, p = {p_value:.3f}')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'variable_relationships_{target_col.replace(" ", "_").replace("-", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Główna funkcja analizy."""
    file_path = os.path.join('data', 'K-1_MI.xlsx')
    sheet_name = "d5"  # Zmień na odpowiedni arkusz
    
    # Zmienne docelowe (temperatura spalin)
    target_columns = [
        "temperatura wylotowa spalin - strona A",
        "temperatura wylotowa spalin - strona B"
    ]
    
    try:
        print("Ładowanie danych...")
        df = get_full_dataset(file_path, sheet_name)
        
        # 1. Eksploracyjna analiza danych
        exploratory_data_analysis(df, target_columns)
        
        # 2. Analiza korelacji
        correlations = correlation_analysis(df, target_columns, threshold=0.2)
        
        # Dla każdej zmiennej docelowej
        for target_col in target_columns:
            if target_col in df.columns:
                print(f"\n{'='*60}")
                print(f"ANALIZA DLA: {target_col}")
                print(f"{'='*60}")
                
                # 3. Selekcja cech
                selected_features, feature_importance = feature_selection(df, target_col, n_features=15)
                
                # 4. Modelowanie
                model_results = build_models(df, target_col, selected_features)
                
                # 5. Analiza związków
                top_features = feature_importance.head(6)['feature'].tolist()
                analyze_variable_relationships(df, target_col, top_features)
                
                # Podsumowanie najlepszego modelu
                best_model_name = max(model_results.keys(), 
                                    key=lambda x: model_results[x]['r2'])
                best_result = model_results[best_model_name]
                
                print(f"\nNAJLEPSZY MODEL dla {target_col}: {best_model_name}")
                print(f"R² = {best_result['r2']:.3f}")
                print(f"Najważniejsze cechy:")
                for i, feature in enumerate(top_features[:5]):
                    print(f"  {i+1}. {feature}")
        
        print("\n" + "="*60)
        print("ANALIZA ZAKOŃCZONA - sprawdź wygenerowane wykresy!")
        print("="*60)
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()