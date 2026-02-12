"""
Configuración general de la aplicación ClusterFlow
"""

# Configuración de Streamlit
PAGE_TITLE = "Cluster APP"
PAGE_ICON = "📊"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Configuración de Clustering
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 10
DEFAULT_OUTLIER_THRESHOLD = 3.0
DEFAULT_VARIANCE_THRESHOLD = 1.0
DEFAULT_CORRELATION_THRESHOLD = 0.90

# Configuración de Escalado
AVAILABLE_SCALERS = {
    'standard': 'StandardScaler (Z-score)',
    'minmax': 'MinMaxScaler (0-1)',
    'robust': 'RobustScaler (resistente a outliers)'
}

# Configuración de Limpieza de Datos
AVAILABLE_FILL_METHODS = {
    'none': 'No hacer nada (⚠️ puede causar errores en clustering)',
    'mean': 'Rellenar con media',
    'median': 'Rellenar con mediana (recomendado)',
    'zero': 'Rellenar con 0',
    'ffill': 'Forward Fill (propagar último valor válido)',
    'bfill': 'Backward Fill (propagar siguiente valor válido)',
    'drop': 'Eliminar filas con NaN'
}

# Configuración de Visualización
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = "husl"

# Límites de Archivo
MAX_FILE_SIZE_MB = 100

# Mensajes
MESSAGES = {
    'no_data': '⚠️ Primero debes cargar un archivo CSV en la sección **Carga de Datos**',
    'no_numeric': '❌ No hay columnas numéricas disponibles',
    'no_scaled': '⚠️ Primero debes escalar los datos en la sección **Escalado de Datos**',
    'clustering_success': '✅ Clustering completado exitosamente',
    'data_loaded': '✅ Archivo cargado exitosamente',
    'data_cleaned': '✅ Datos limpiados exitosamente',
    'data_scaled': '✅ Datos escalados exitosamente'
}
