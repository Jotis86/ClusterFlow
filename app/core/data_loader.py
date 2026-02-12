"""
Módulo para carga de datos
"""
import pandas as pd
from typing import Tuple, Optional


def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Cargar datos desde archivo CSV
    
    Args:
        uploaded_file: Archivo subido a través de Streamlit
        
    Returns:
        Tuple[DataFrame, error]: DataFrame con los datos o None si hay error, mensaje de error o None
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)
