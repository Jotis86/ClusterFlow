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
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        return df[columns_to_scale], None
    
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[columns_to_scale]),
        columns=columns_to_scale,
        index=df.index
    )
    
    return df_scaled, scaler
