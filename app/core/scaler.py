"""
Módulo para escalado de datos
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Optional, List


def scale_data(
    df: pd.DataFrame,
    scaler_type: str = 'standard',
    columns_to_scale: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, object]:
    """
    Escalar datos numéricos
    
    Args:
        df: DataFrame con datos a escalar
        scaler_type: Tipo de escalador ('standard', 'minmax', 'robust')
        columns_to_scale: Lista de columnas a escalar. Si None, escala todas las numéricas
        
    Returns:
        Tuple[DataFrame escalado, scaler fitted]
    """
    import numpy as np
    
    if columns_to_scale is None:
        columns_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns_to_scale) == 0:
        raise ValueError("No hay columnas numéricas para escalar")
    
    # Verificar que no haya NaN
    if df[columns_to_scale].isnull().any().any():
        raise ValueError("Los datos contienen valores NaN. Por favor, limpia los datos primero.")
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        return df[columns_to_scale], None
    
    try:
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[columns_to_scale]),
            columns=columns_to_scale,
            index=df.index
        )
        return df_scaled, scaler
    except Exception as e:
        raise ValueError(f"Error al escalar los datos: {str(e)}")
