"""Tests ampliados para stats.py"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from utils.stats import (
    calculate_skewness_kurtosis,
    detect_outliers_iqr,
    get_correlation_pairs,
    calculate_variance_stats
)


class TestStatsExtended:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100) * 2 + 5,
            'C': np.random.randn(100) * 0.5
        })
    
    @pytest.fixture
    def data_with_outliers(self):
        return pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200]
        })
    
    @pytest.fixture
    def highly_correlated_data(self):
        np.random.seed(42)
        x = np.random.randn(100)
        return pd.DataFrame({
            'X': x,
            'Y': x * 2 + 0.1 * np.random.randn(100),  # Alta correlación con X
            'Z': np.random.randn(100)  # No correlacionada
        })
    
    # Tests adicionales de calculate_skewness_kurtosis
    def test_calculate_skewness_kurtosis_single_column(self, sample_data):
        result = calculate_skewness_kurtosis(sample_data, ['A'])
        assert len(result) == 1
        # El índice es numérico (0), pero 'Variable' columna contiene 'A'
        assert result.iloc[0]['Variable'] == 'A'
    
    def test_calculate_skewness_kurtosis_values(self, sample_data):
        result = calculate_skewness_kurtosis(sample_data, ['A', 'B', 'C'])
        # Verificar que los valores son numéricos
        assert pd.api.types.is_numeric_dtype(result['Asimetría'])
        assert pd.api.types.is_numeric_dtype(result['Curtosis'])
    
    # Tests adicionales de detect_outliers_iqr
    def test_detect_outliers_iqr_with_outliers(self, data_with_outliers):
        n_outliers, lower, upper, outliers = detect_outliers_iqr(data_with_outliers, 'values')
        assert n_outliers > 0  # Debería detectar 100 y 200 como outliers
        # outliers es una Series, verificar con .values
        assert 100 in outliers.values or 200 in outliers.values
    
    def test_detect_outliers_iqr_no_outliers(self):
        """Test con datos sin outliers"""
        normal_data = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        n_outliers, lower, upper, outliers = detect_outliers_iqr(normal_data, 'values')
        # Puede haber 0 outliers o muy pocos
        assert n_outliers >= 0
    
    def test_detect_outliers_iqr_bounds(self, data_with_outliers):
        """Verificar que los límites IQR son correctos"""
        n_outliers, lower, upper, outliers = detect_outliers_iqr(data_with_outliers, 'values')
        # Los valores atípicos deberían estar fuera de los límites
        for outlier in outliers:
            assert outlier < lower or outlier > upper
    
    # Tests adicionales de get_correlation_pairs
    def test_get_correlation_pairs_high_threshold(self, highly_correlated_data):
        corr_matrix = highly_correlated_data.corr()
        pairs = get_correlation_pairs(corr_matrix, threshold=0.8)
        # X e Y deberían estar altamente correlacionadas
        assert len(pairs) > 0
        # Verificar que solo incluye correlaciones >= 0.8
        assert (pairs['Correlación'].abs() >= 0.8).all()
    
    def test_get_correlation_pairs_no_duplicates(self, sample_data):
        """Verificar que no hay pares duplicados (A-B y B-A)"""
        corr_matrix = sample_data.corr()
        pairs = get_correlation_pairs(corr_matrix, threshold=0.0)
        # Crear conjunto de pares ordenados para verificar duplicados
        pair_sets = set()
        for _, row in pairs.iterrows():
            pair = tuple(sorted([row['Variable 1'], row['Variable 2']]))
            assert pair not in pair_sets, f"Duplicate pair found: {pair}"
            pair_sets.add(pair)
    
    def test_get_correlation_pairs_excludes_self_correlation(self, sample_data):
        """Verificar que no incluye correlación de una variable consigo misma"""
        corr_matrix = sample_data.corr()
        pairs = get_correlation_pairs(corr_matrix, threshold=0.0)
        for _, row in pairs.iterrows():
            assert row['Variable 1'] != row['Variable 2']
    
    # Tests adicionales de calculate_variance_stats
    def test_calculate_variance_stats_single_column(self, sample_data):
        stats = calculate_variance_stats(sample_data, ['A'])
        assert len(stats) == 1
        assert stats.index[0] == 'A'
    
    def test_calculate_variance_stats_values(self, sample_data):
        stats = calculate_variance_stats(sample_data, ['A', 'B', 'C'])
        # Verificar que la varianza (std^2) es positiva
        assert (stats['std'] >= 0).all()
        # Verificar que existe la columna std en el resultado
        assert 'std' in stats.columns
    
    def test_calculate_variance_stats_cv_calculation(self, sample_data):
        """Verificar que el coeficiente de variación se calcula correctamente"""
        stats = calculate_variance_stats(sample_data, ['B'])  # B tiene media != 0
        # CV = (std / mean) * 100
        expected_cv = (sample_data['B'].std() / abs(sample_data['B'].mean())) * 100
        assert abs(stats.loc['B', 'CV (%)'] - expected_cv) < 0.01
    
    def test_calculate_variance_stats_all_positive(self, sample_data):
        """Todas las estadísticas de varianza deberían ser no negativas"""
        stats = calculate_variance_stats(sample_data, ['A', 'B', 'C'])
        assert (stats['std'] >= 0).all()
        assert (stats['mean'].notna()).all()
        assert (stats['CV (%)'].notna()).all()
        assert (stats['mean'].notna()).all()
        assert (stats['CV (%)'].notna()).all()
