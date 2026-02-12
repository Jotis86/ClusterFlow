"""Tests ampliados para data_cleaner.py"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.data_cleaner import analyze_data_quality, clean_data


class TestDataCleanerExtended:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'A': [1, 2, 2, np.nan, 5],
            'B': [10, 20, 20, 40, 1000],
            'C': ['x', 'y', 'y', 'z', 'w']
        })
    
    @pytest.fixture
    def data_with_many_nulls(self):
        return pd.DataFrame({
            'num1': [1, np.nan, 3, np.nan, 5, np.nan],
            'num2': [np.nan, 2, np.nan, 4, np.nan, 6],
            'cat': ['a', 'b', 'c', 'd', 'e', 'f']
        })
    
    @pytest.fixture
    def data_all_nulls_column(self):
        return pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [1, 2, 3]
        })
    
    # Tests adicionales de analyze_data_quality
    def test_analyze_data_quality_all_fields(self, sample_data):
        stats = analyze_data_quality(sample_data)
        assert 'shape' in stats
        assert 'nulls' in stats
        assert 'null_pct' in stats
        assert 'duplicates' in stats
        assert 'dtypes' in stats
        assert 'numeric_cols' in stats
        assert 'categorical_cols' in stats
    
    def test_analyze_data_quality_numeric_categorical(self, sample_data):
        stats = analyze_data_quality(sample_data)
        assert 'A' in stats['numeric_cols']
        assert 'B' in stats['numeric_cols']
        assert 'C' in stats['categorical_cols']
    
    def test_analyze_data_quality_null_percentage(self, data_with_many_nulls):
        stats = analyze_data_quality(data_with_many_nulls)
        assert stats['null_pct']['num1'] == 50.0
        assert stats['null_pct']['num2'] == 50.0
    
    # Tests de fill_nulls adicionales
    def test_clean_data_fill_nulls_zero(self, sample_data):
        cleaned = clean_data(sample_data, remove_duplicates=False, fill_nulls_method='zero', remove_outliers=False)
        assert cleaned['A'].isna().sum() == 0
        assert cleaned.loc[3, 'A'] == 0
    
    def test_clean_data_fill_nulls_ffill(self, data_with_many_nulls):
        cleaned = clean_data(data_with_many_nulls, remove_duplicates=False, fill_nulls_method='ffill', remove_outliers=False)
        assert cleaned['num1'].isna().sum() == 0
        assert cleaned['num2'].isna().sum() == 0
    
    def test_clean_data_fill_nulls_bfill(self, data_with_many_nulls):
        cleaned = clean_data(data_with_many_nulls, remove_duplicates=False, fill_nulls_method='bfill', remove_outliers=False)
        assert cleaned['num1'].isna().sum() == 0
        assert cleaned['num2'].isna().sum() == 0
    
    def test_clean_data_fill_nulls_drop(self, sample_data):
        cleaned = clean_data(sample_data, remove_duplicates=False, fill_nulls_method='drop', remove_outliers=False)
        assert cleaned['A'].isna().sum() == 0
        assert len(cleaned) == 4  # Se eliminó la fila con NaN
    
    def test_clean_data_fill_nulls_none(self, sample_data):
        cleaned = clean_data(sample_data, remove_duplicates=False, fill_nulls_method='none', remove_outliers=False)
        # La validación final debería llenar los NaN
        assert cleaned['A'].isna().sum() == 0
    
    def test_clean_data_all_nulls_column_validation(self, data_all_nulls_column):
        """Test validación final con columna completamente nula"""
        cleaned = clean_data(data_all_nulls_column, remove_duplicates=False, fill_nulls_method='mean', remove_outliers=False)
        # La validación final debería rellenar con 0
        assert cleaned['A'].isna().sum() == 0
        assert (cleaned['A'] == 0).all()
    
    # Tests de outliers
    def test_clean_data_remove_outliers_different_thresholds(self, sample_data):
        cleaned_strict = clean_data(sample_data, remove_duplicates=False, fill_nulls_method='mean', remove_outliers=True, outlier_threshold=1.5)
        cleaned_lenient = clean_data(sample_data, remove_duplicates=False, fill_nulls_method='mean', remove_outliers=True, outlier_threshold=5.0)
        assert len(cleaned_strict) <= len(cleaned_lenient)
    
    def test_clean_data_preserves_categorical(self, sample_data):
        """Verificar que las columnas categóricas no se modifican"""
        cleaned = clean_data(sample_data, remove_duplicates=True, fill_nulls_method='median', remove_outliers=True)
        assert 'C' in cleaned.columns
        assert cleaned['C'].dtype == 'object'
    
    def test_clean_data_empty_dataframe(self):
        """Test con DataFrame vacío"""
        empty_df = pd.DataFrame()
        cleaned = clean_data(empty_df, remove_duplicates=True, fill_nulls_method='mean', remove_outliers=True)
        assert len(cleaned) == 0
