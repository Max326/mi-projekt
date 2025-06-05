import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

def load_excel_data(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plik {file_path} nie został znaleziony.")
    
    excel_data = pd.read_excel(file_path, sheet_name=None)
    return excel_data

def get_full_dataset(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Załaduj pełny dataset z określonego arkusza."""
    excel_data = load_excel_data(file_path)
    
    if sheet_name not in excel_data:
        raise ValueError(f"Arkusz '{sheet_name}' nie istnieje w pliku. "
                         f"Dostępne arkusze: {list(excel_data.keys())}")
    
    return excel_data[sheet_name]

def data_inspection(df: pd.DataFrame) -> None:
    """Szczegółowa inspekcja danych."""
    print("=== SZCZEGÓŁOWA INSPEKCJA DANYCH ===")
    print(f"Wymiary datasetu: {df.shape}")
    print(f"\nNazwy wszystkich kolumn:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    print(f"\nTypy danych:")
    print(df.dtypes.value_counts())
    
    print(f"\nBrakujące wartości per kolumna:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        for col, count in missing_cols.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("  Brak brakujących wartości!")

def basic_data_info(df: pd.DataFrame, target_columns: List[str]) -> None:
    """Podstawowe informacje o danych."""
    print("\n=== PODSTAWOWE INFORMACJE O ZMIENNYCH DOCELOWYCH ===")
    
    # Informacje o zmiennych docelowych
    for target in target_columns:
        if target in df.columns:
            print(f"\n--- {target} ---")
            print(f"Średnia: {df[target].mean():.2f}°C")
            print(f"Odchylenie std: {df[target].std():.2f}°C")
            print(f"Min: {df[target].min():.2f}°C")
            print(f"Max: {df[target].max():.2f}°C")
            print(f"Zakres: {df[target].max() - df[target].min():.2f}°C")
            print(f"Brakujące wartości: {df[target].isnull().sum()}")
            
            # Sprawdź czy są stałe wartości
            unique_values = df[target].nunique()
            print(f"Liczba unikalnych wartości: {unique_values}")
            if unique_values < 10:
                print(f"Unikalne wartości: {sorted(df[target].unique())}")
        else:
            print(f"UWAGA: Kolumna '{target}' nie została znaleziona!")

def check_all_burner_angles(df: pd.DataFrame) -> None:
    """Sprawdź wszystkie kąty wychylenia palników."""
    print("\n=== SPRAWDZENIE WSZYSTKICH KĄTÓW PALNIKÓW ===")
    
    burner_patterns = ['kąt wychylenia palnika', 'palnik', 'róg']
    burner_columns = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in burner_patterns):
            burner_columns.append(col)
    
    print(f"Znalezione kolumny związane z palnikami ({len(burner_columns)}):")
    for i, col in enumerate(burner_columns, 1):
        if col in df.select_dtypes(include=[np.number]).columns:
            stats = df[col].describe()
            print(f"{i:2d}. {col}")
            print(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, "
                  f"Std: {stats['std']:.2f}, Unikalne: {df[col].nunique()}")
        else:
            print(f"{i:2d}. {col} (nie numeryczne)")

def temperature_distribution_analysis(df: pd.DataFrame, target_columns: List[str]) -> None:
    """Analiza rozkładu temperatury spalin."""
    print("\n=== ANALIZA ROZKŁADU TEMPERATURY SPALIN ===")
    
    for target in target_columns:
        if target in df.columns:
            # Podstawowe statystyki
            print(f"\nStatystyki dla {target}:")
            print(df[target].describe())
            
            # Wykresy rozkładu i serii czasowej
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram
            ax1.hist(df[target].dropna(), bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Temperatura [°C]')
            ax1.set_ylabel('Częstość')
            ax1.set_title(f'Rozkład temperatury - {target}')
            ax1.grid(True, alpha=0.3)
            
            # Seria czasowa
            ax2.plot(df.index, df[target], alpha=0.7, linewidth=0.8)
            ax2.set_xlabel('Numer próbki')
            ax2.set_ylabel('Temperatura [°C]')
            ax2.set_title(f'Przebieg czasowy - {target}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'temperature_analysis_{target.replace(" ", "_").replace("-", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()

def correlation_analysis(df: pd.DataFrame, target_columns: List[str], threshold: float = 0.15) -> Dict:
    """Szczegółowa analiza korelacji z temperaturą spalin."""
    print("\n=== ANALIZA KORELACJI Z TEMPERATURĄ SPALIN ===")
    
    # Wybierz tylko kolumny numeryczne
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"Liczba kolumn numerycznych: {len(numeric_df.columns)}")
    
    correlation_results = {}
    
    for target in target_columns:
        if target not in numeric_df.columns:
            print(f"Brak danych dla {target}")
            continue
            
        print(f"\n{'='*80}")
        print(f"ANALIZA KORELACJI DLA: {target}")
        print(f"{'='*80}")
        
        # Oblicz korelacje z temperaturą spalin
        correlations = []
        
        for column in numeric_df.columns:
            if column != target:
                # Usuń wiersze z NaN dla tej pary zmiennych
                mask = ~(numeric_df[column].isnull() | numeric_df[target].isnull())
                
                if mask.sum() > 10:  # Minimum 10 próbek do analizy
                    x = numeric_df[column][mask]
                    y = numeric_df[target][mask]
                    
                    # Sprawdź czy są jakiekolwiek różnice w danych
                    if x.std() > 0 and y.std() > 0:  # Tylko jeśli są wariacje
                        # Oblicz korelację Pearsona
                        corr_coef, p_value = pearsonr(x, y)
                        
                        # Sprawdź czy korelacja nie jest NaN
                        if not np.isnan(corr_coef):
                            correlations.append({
                                'variable': column,
                                'correlation': corr_coef,
                                'abs_correlation': abs(corr_coef),
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'samples': len(x),
                                'x_std': x.std(),
                                'y_std': y.std()
                            })
        
        # Sortuj według siły korelacji
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        # Filtruj znaczące korelacje
        significant_correlations = [c for c in correlations 
                                  if c['abs_correlation'] > threshold and c['significant']]
        
        correlation_results[target] = significant_correlations
        
        # Wyświetl wszystkie obliczone korelacje (debug)
        print(f"\nObliczono korelacje dla {len(correlations)} zmiennych")
        print(f"Znaczące korelacje (|r| > {threshold}, p < 0.05): {len(significant_correlations)}")
        
        # Sprawdź czy są zmienne z bardzo małą wariancją
        low_variance = [c for c in correlations if c['x_std'] < 0.001]
        if low_variance:
            print(f"Zmienne z bardzo małą wariancją: {len(low_variance)}")
            for var in low_variance[:5]:
                print(f"  - {var['variable']}: std = {var['x_std']:.6f}")
        
        # Wyświetl wyniki
        if significant_correlations:
            print(f"\nTOP {min(25, len(significant_correlations))} ZMIENNYCH o znaczącej korelacji:")
            print(f"{'Zmienna':<50} {'Korelacja':<12} {'|Korelacja|':<12} {'p-value':<12} {'Próbki':<8}")
            print("-" * 94)
            
            for i, corr in enumerate(significant_correlations[:25]):  # Top 25
                print(f"{corr['variable']:<50} {corr['correlation']:<12.3f} "
                      f"{corr['abs_correlation']:<12.3f} {corr['p_value']:<12.3e} {corr['samples']:<8}")
            
            # Kategoryzacja zmiennych według siły wpływu
            very_strong = [c for c in significant_correlations if c['abs_correlation'] > 0.7]
            strong = [c for c in significant_correlations if 0.5 < c['abs_correlation'] <= 0.7]
            moderate = [c for c in significant_correlations if 0.3 < c['abs_correlation'] <= 0.5]
            weak = [c for c in significant_correlations if threshold < c['abs_correlation'] <= 0.3]
            
            print(f"\nKATEGORYZACJA WPŁYWU NA {target}:")
            print(f"• Bardzo silny wpływ (|r| > 0.7): {len(very_strong)} zmiennych")
            print(f"• Silny wpływ (0.5 < |r| ≤ 0.7): {len(strong)} zmiennych")
            print(f"• Umiarkowany wpływ (0.3 < |r| ≤ 0.5): {len(moderate)} zmiennych")
            print(f"• Słaby wpływ ({threshold} < |r| ≤ 0.3): {len(weak)} zmiennych")
            
            # Szczegółowe listy dla każdej kategorii
            categories = [
                ("BARDZO SILNY WPŁYW", very_strong),
                ("SILNY WPŁYW", strong),
                ("UMIARKOWANY WPŁYW", moderate[:10])  # Ogranicz do 10 dla czytelności
            ]
            
            for cat_name, cat_vars in categories:
                if cat_vars:
                    print(f"\n{cat_name}:")
                    for var in cat_vars:
                        direction = "↗ pozytywny" if var['correlation'] > 0 else "↘ negatywny"
                        print(f"  • {var['variable']}: r = {var['correlation']:.3f} ({direction})")
            
            # Mapa ciepła korelacji dla top zmiennych
            create_correlation_heatmap(numeric_df, target, significant_correlations[:15])
        else:
            print("Brak znaczących korelacji przy zadanym progu!")
    
    return correlation_results  # NAPRAWIONE: dodaj return

def create_correlation_heatmap(numeric_df: pd.DataFrame, target: str, correlations: List[Dict]) -> None:
    """Tworzy mapę ciepła korelacji dla najważniejszych zmiennych."""
    
    # Wybierz zmienne do mapy ciepła
    top_variables = [c['variable'] for c in correlations] + [target]
    
    # Oblicz macierz korelacji
    correlation_matrix = numeric_df[top_variables].corr()
    
    # Twórz mapę ciepła
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True, 
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title(f'Mapa korelacji dla najważniejszych zmiennych - {target}', 
              fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f'correlation_heatmap_{target.replace(" ", "_").replace("-", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Mapa korelacji zapisana jako: {filename}")

def analyze_variable_relationships(df: pd.DataFrame, target: str, correlations: List[Dict]) -> None:
    """Szczegółowa analiza związków najważniejszych zmiennych z temperaturą."""
    print(f"\n=== SZCZEGÓŁOWA ANALIZA ZWIĄZKÓW DLA {target} ===")
    
    numeric_df = df.select_dtypes(include=[np.number])
    top_correlations = correlations[:6]  # Top 6 zmiennych
    
    if len(top_correlations) == 0:
        print("Brak zmiennych do analizy.")
        return
    
    # Wykresy scatter dla najważniejszych zmiennych
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, corr_info in enumerate(top_correlations):
        var_name = corr_info['variable']
        correlation = corr_info['correlation']
        
        ax = axes[idx]
        
        # Usuń NaN dla tej pary zmiennych
        mask = ~(numeric_df[var_name].isnull() | numeric_df[target].isnull())
        x = numeric_df[var_name][mask]
        y = numeric_df[target][mask]
        
        if len(x) > 0:
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=20)
            
            # Linia trendu
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Formatowanie
            ax.set_xlabel(var_name, fontsize=10)
            ax.set_ylabel(target, fontsize=10)
            ax.set_title(f'r = {correlation:.3f}\n{var_name}', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Dodaj kierunek zależności
            direction = "↗" if correlation > 0 else "↘"
            ax.text(0.02, 0.98, direction, transform=ax.transAxes, 
                   fontsize=20, verticalalignment='top')
    
    plt.tight_layout()
    filename = f'relationships_{target.replace(" ", "_").replace("-", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Wykresy związków zapisane jako: {filename}")

def summarize_findings(correlation_results: Dict) -> None:
    """Podsumowanie kluczowych ustaleń."""
    print("\n" + "="*100)
    print("PODSUMOWANIE KLUCZOWYCH USTALEŃ")
    print("="*100)
    
    for target, correlations in correlation_results.items():
        print(f"\n--- {target} ---")
        
        if not correlations:
            print("Brak znaczących korelacji.")
            continue
        
        # Top 5 najsilniejszych korelacji
        top_5 = correlations[:5]
        
        print(f"TOP 5 CZYNNIKÓW wpływających na temperaturę:")
        for i, corr in enumerate(top_5, 1):
            effect = "zwiększa" if corr['correlation'] > 0 else "zmniejsza"
            strength = "bardzo silnie" if corr['abs_correlation'] > 0.7 else \
                      "silnie" if corr['abs_correlation'] > 0.5 else \
                      "umiarkowanie" if corr['abs_correlation'] > 0.3 else "słabo"
            
            print(f"  {i}. {corr['variable']}")
            print(f"     → {strength} {effect} temperaturę (r = {corr['correlation']:.3f})")
        
        # Praktyczne wnioski
        print(f"\nPRAKTYCZNE WNIOSKI:")
        positive_strong = [c for c in correlations[:10] if c['correlation'] > 0.3]
        negative_strong = [c for c in correlations[:10] if c['correlation'] < -0.3]
        
        if positive_strong:
            print(f"• Aby ZWIĘKSZYĆ temperaturę spalin, należy zwiększyć:")
            for var in positive_strong[:3]:
                print(f"  - {var['variable']}")
        
        if negative_strong:
            print(f"• Aby ZMNIEJSZYĆ temperaturę spalin, należy zwiększyć:")
            for var in negative_strong[:3]:
                print(f"  - {var['variable']}")

def main():
    """Główna funkcja analizy identyfikacji wpływu zmiennych."""
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
        
        # 1. Szczegółowa inspekcja danych
        data_inspection(df)
        
        # 2. Sprawdź wszystkie palniki
        check_all_burner_angles(df)
        
        # 3. Podstawowe informacje o zmiennych docelowych
        basic_data_info(df, target_columns)
        
        # 4. Analiza rozkładu temperatury
        temperature_distribution_analysis(df, target_columns)
        
        # 5. Szczegółowa analiza korelacji (niższy próg)
        correlation_results = correlation_analysis(df, target_columns, threshold=0.15)
        
        # 6. Analiza związków dla każdej zmiennej docelowej
        for target in target_columns:
            if target in correlation_results and correlation_results[target]:
                analyze_variable_relationships(df, target, correlation_results[target])
        
        # 7. Podsumowanie ustaleń
        summarize_findings(correlation_results)
        
        print("\n" + "="*100)
        print("ANALIZA IDENTYFIKACJI ZAKOŃCZONA!")
        print("Sprawdź wygenerowane wykresy i wyniki powyżej.")
        print("="*100)
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()