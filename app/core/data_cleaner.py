"""
Módulo para limpieza y preprocesamiento de datos
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import Dict, List


def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Analizar calidad de datos
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Dict con información de calidad: shape, nulls, duplicates, dtypes, etc.
    """
    try:
        null_pct = (df.isnull().sum() / len(df) * 100).round(2) if len(df) > 0 else pd.Series()
    except:
        null_pct = pd.Series()
    
    report = {
        'shape': df.shape,
        'nulls': df.isnull().sum(),
        'null_pct': null_pct,
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes,
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist()
    }
    return report


def clean_data(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    fill_nulls_method: str = 'mean',
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Limpieza automática de datos
    
    Args:
        df: DataFrame a limpiar
        remove_duplicates: Si True, elimina filas duplicadas
        fill_nulls_method: Método para rellenar valores nulos ('mean', 'median', 'zero', 'ffill', 'bfill', 'drop', 'none')
        remove_outliers: Si True, elimina outliers
        outlier_threshold: Umbral de Z-score para considerar outliers
        
    Returns:
        DataFrame limpio
    """
    df_clean = df.copy()
    
    # Eliminar duplicados
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    # Rellenar valores nulos en columnas numéricas
    if fill_nulls_method != 'none':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                if fill_nulls_method == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif fill_nulls_method == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif fill_nulls_method == 'zero':
                    df_clean[col].fillna(0, inplace=True)
                elif fill_nulls_method == 'ffill':
                    df_clean[col].fillna(method='ffill', inplace=True)
                    df_clean[col].fillna(method='bfill', inplace=True)
                    df_clean[col].fillna(0, inplace=True)
                elif fill_nulls_method == 'bfill':
                    df_clean[col].fillna(method='bfill', inplace=True)
                    df_clean[col].fillna(method='ffill', inplace=True)
                    df_clean[col].fillna(0, inplace=True)
                elif fill_nulls_method == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
    
    # Eliminar outliers
    if remove_outliers:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() == 0 and df_clean[col].std() > 0:
                try:
                    z_scores = np.abs(zscore(df_clean[col]))
                    df_clean = df_clean[z_scores < outlier_threshold]
                except (ValueError, RuntimeWarning):
                    # Saltar columnas con valores constantes o problemas
                    continue
    
    # VALIDACIÓN FINAL: Asegurar que no queden NaN en columnas numéricas
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dropna().shape[0] > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(0, inplace=True)
    
    return df_clean
