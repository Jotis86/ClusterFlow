"""
Funciones estadísticas y de análisis de datos
"""
import pandas as pd
import numpy as np


def calculate_skewness_kurtosis(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Calcula asimetría y curtosis para columnas numéricas
    
    Args:
        df: DataFrame con los datos
        numeric_cols: Lista de columnas numéricas
        
    Returns:
        DataFrame con asimetría, curtosis e interpretación
    """
    skew_kurt_data = []
    for col in numeric_cols:
        skew_kurt_data.append({
            'Variable': col,
            'Asimetría': df[col].skew(),
            'Curtosis': df[col].kurtosis(),
            'Interpretación Asimetría': 'Simétrica' if abs(df[col].skew()) < 0.5 
                                       else ('Asimétrica derecha' if df[col].skew() > 0 
                                            else 'Asimétrica izquierda')
        })
    
    return pd.DataFrame(skew_kurt_data)


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> tuple:
    """
    Detecta outliers usando el método IQR
    
    Args:
        df: DataFrame con los datos
        column: Nombre de la columna a analizar
        
    Returns:
        Tuple con (número de outliers, límite inferior, límite superior, serie de outliers)
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    return len(outliers), lower_bound, upper_bound, outliers


def get_correlation_pairs(correlation_matrix: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Extrae pares de variables con correlación superior al umbral
    
    Args:
        correlation_matrix: Matriz de correlación
        threshold: Umbral de correlación mínimo
        
    Returns:
        DataFrame con pares de variables y sus correlaciones
    """
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                corr_pairs.append({
                    'Variable 1': correlation_matrix.columns[i],
                    'Variable 2': correlation_matrix.columns[j],
                    'Correlación': corr_value
                })
    
    corr_pairs_df = pd.DataFrame(corr_pairs)
    if len(corr_pairs_df) > 0:
        corr_pairs_df = corr_pairs_df.sort_values('Correlación', key=abs, ascending=False)
    
    return corr_pairs_df


def calculate_variance_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Calcula estadísticas de varianza para las columnas especificadas
    
    Args:
        df: DataFrame con los datos
        columns: Lista de columnas a analizar
        
    Returns:
        DataFrame con estadísticas incluyendo CV (coeficiente de variación)
    """
    stats_df = df[columns].describe().T
    
    # Calcular CV con manejo de división por cero
    cv_values = []
    for col in stats_df.index:
        mean_val = stats_df.loc[col, 'mean']
        std_val = stats_df.loc[col, 'std']
        if mean_val != 0 and not pd.isna(mean_val):
            cv_values.append(round((std_val / abs(mean_val)) * 100, 2))
        else:
            cv_values.append(0.0)
    
    stats_df['CV (%)'] = cv_values
    stats_df['Rango'] = (stats_df['max'] - stats_df['min']).round(2)
    
    return stats_df
