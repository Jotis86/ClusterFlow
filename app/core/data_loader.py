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
        
        # Validaciones básicas
        if df.empty:
            return None, "El archivo CSV está vacío"
        
        if len(df.columns) == 0:
            return None, "El archivo CSV no tiene columnas"
        
        # Verificar que haya al menos una columna numérica
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return None, "El archivo debe contener al menos una columna numérica"
        
        return df, None
    except pd.errors.EmptyDataError:
        return None, "El archivo CSV está vacío"
    except pd.errors.ParserError:
        return None, "Error al parsear el archivo CSV. Verifica el formato."
    except UnicodeDecodeError:
        return None, "Error de codificación. Intenta guardar el archivo como UTF-8."
    except Exception as e:
        return None, f"Error inesperado: {str(e)}"
