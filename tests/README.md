# Tests para ClusterFlow

## Estructura de Tests

```
tests/
├── __init__.py
├── test_data_loader.py      # Tests para carga de datos
├── test_data_cleaner.py     # Tests para limpieza
├── test_scaler.py           # Tests para escalado
├── test_clustering.py       # Tests para clustering
├── test_stats.py            # Tests para funciones estadísticas
└── test_integration.py      # Tests de integración
```

## Ejecutar Tests

### Instalar dependencias de testing
```bash
pip install pytest pytest-cov
```

### Ejecutar todos los tests
```bash
pytest
```

### Ejecutar con coverage
```bash
pytest --cov=app --cov-report=html
```

### Ejecutar tests específicos
```bash
# Un archivo
pytest tests/test_data_loader.py

# Una clase
pytest tests/test_clustering.py::TestClustering

# Un test específico
pytest tests/test_clustering.py::TestClustering::test_perform_clustering_kmeans
```

### Ejecutar con verbose
```bash
pytest -v
```

### Ejecutar solo tests unitarios o de integración
```bash
pytest -m unit
pytest -m integration
```

## Cobertura de Tests

### Módulos testeados:
- ✅ `core.data_loader` - Carga de archivos CSV
- ✅ `core.data_cleaner` - Limpieza y calidad de datos
- ✅ `core.scaler` - Escalado de datos (StandardScaler, MinMaxScaler, RobustScaler)
- ✅ `core.clustering` - Algoritmos de clustering y métricas
- ✅ `utils.stats` - Funciones estadísticas

### Tests de integración:
- ✅ Pipeline completo: limpieza → escalado → clustering
- ✅ Comparación de múltiples métodos
- ✅ Validación de dimensiones

## Descripción de Tests

### test_data_loader.py
- Carga de CSV válido
- CSV con separador punto y coma
- Manejo de archivos vacíos
- Manejo de archivos inválidos

### test_data_cleaner.py
- Análisis de calidad de datos
- Eliminación de duplicados
- Relleno de nulos (mean, median, forward fill)
- Detección y eliminación de outliers
- Combinación de operaciones

### test_scaler.py
- StandardScaler (media=0, std=1)
- MinMaxScaler (rango 0-1)
- RobustScaler (robusto a outliers)
- Preservación de columnas no numéricas
- Escalado selectivo de columnas

### test_clustering.py
- Determinación de k óptimo
- KMeans clustering
- Hierarchical clustering
- DBSCAN
- Gaussian Mixture Models
- Selección del mejor método
- Validación de k inválido

### test_stats.py
- Cálculo de skewness y kurtosis
- Detección de outliers con IQR
- Pares de correlación
- Estadísticas de varianza (CV)
- Manejo de DataFrames vacíos

### test_integration.py
- Pipeline completo de procesamiento
- Comparación de múltiples métodos
- Preservación de dimensiones
- Validación de métricas

## Fixtures

Los tests utilizan pytest fixtures para datos de ejemplo:
- `sample_data`: DataFrame limpio para tests básicos
- `raw_data`: DataFrame con problemas (nulls, duplicados, outliers)

## Comandos útiles

```bash
# Ver tests disponibles sin ejecutar
pytest --collect-only

# Ejecutar tests con prints visibles
pytest -s

# Ejecutar tests que fallaron la última vez
pytest --lf

# Detener en el primer fallo
pytest -x

# Generar reporte HTML de coverage
pytest --cov=app --cov-report=html
# Abrir htmlcov/index.html en navegador
```
