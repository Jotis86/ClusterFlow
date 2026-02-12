"""Tests de integración simplificados"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.data_cleaner import clean_data
from core.scaler import scale_data
from core.clustering import perform_clustering


class TestIntegration:
    @pytest.fixture
    def raw_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': [1, 2, 2, np.nan, 5, 6, 7, 8, 9, 100],
            'feature2': [10, 20, 20, 40, 50, 60, 70, 80, 90, 100],
            'feature3': np.random.randn(10) * 5 + 50
        })
    
    def test_complete_pipeline(self, raw_data):
        # Limpieza
        cleaned = clean_data(raw_data, remove_duplicates=True, fill_nulls_method='mean', remove_outliers=True, outlier_threshold=2.0)
        assert cleaned.isna().sum().sum() == 0
        
        # Escalado
        scaled_df, scaler = scale_data(cleaned, 'standard', ['feature1', 'feature2', 'feature3'])
        assert scaler is not None
        
        # Clustering
        result = perform_clustering(scaled_df, 2, 'kmeans')
        assert 'labels' in result
        assert len(result['labels']) == len(scaled_df)
    
    def test_pipeline_with_multiple_methods(self, raw_data):
        cleaned = clean_data(raw_data, remove_duplicates=True, fill_nulls_method='median', remove_outliers=True, outlier_threshold=2.0)
        scaled_df, _ = scale_data(cleaned, 'minmax', ['feature1', 'feature2', 'feature3'])
        
        methods = ['kmeans', 'hierarchical']
        results = {}
        
        for method in methods:
            result = perform_clustering(scaled_df, 2, method)
            results[method] = result
            assert 'labels' in result
    
    def test_pipeline_preserves_shape(self, raw_data):
        initial_cols = len(raw_data.columns)
        
        cleaned = clean_data(raw_data, remove_duplicates=False, fill_nulls_method='mean', remove_outliers=False)
        assert len(cleaned.columns) == initial_cols
        
        scaled_df, _ = scale_data(cleaned, 'standard', cleaned.columns.tolist())
        assert len(scaled_df.columns) == initial_cols
