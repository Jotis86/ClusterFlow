"""Tests corregidos para data_cleaner.py"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.data_cleaner import analyze_data_quality, clean_data


class TestDataCleaner:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'A': [1, 2, 2, np.nan, 5],
            'B': [10, 20, 20, 40, 1000],
            'C': ['x', 'y', 'y', 'z', 'w']
        })
    
    def test_analyze_data_quality(self, sample_data):
        stats = analyze_data_quality(sample_data)
        assert 'shape' in stats
        assert 'duplicates' in stats
        assert stats['shape'] == (5, 3)
    
    def test_clean_data_remove_duplicates(self, sample_data):
        cleaned = clean_data(sample_data, remove_duplicates=True, fill_nulls_method='mean', remove_outliers=False)
        assert len(cleaned) == 4
    
    def test_clean_data_fill_nulls_mean(self, sample_data):
        cleaned = clean_data(sample_data, remove_duplicates=False, fill_nulls_method='mean', remove_outliers=False)
        assert cleaned['A'].isna().sum() == 0
    
    def test_clean_data_fill_nulls_median(self, sample_data):
        cleaned = clean_data(sample_data, remove_duplicates=False, fill_nulls_method='median', remove_outliers=False)
        assert cleaned['A'].isna().sum() == 0
    
    def test_clean_data_remove_outliers(self, sample_data):
        cleaned = clean_data(sample_data, remove_duplicates=False, fill_nulls_method='mean', remove_outliers=True, outlier_threshold=1.5)
        # Con threshold 1.5 debería eliminar el outlier 1000
        assert len(cleaned) < len(sample_data) or cleaned['B'].max() < 1000
    
    def test_clean_data_all_operations(self, sample_data):
        cleaned = clean_data(sample_data, remove_duplicates=True, fill_nulls_method='mean', remove_outliers=True, outlier_threshold=2.0)
        assert cleaned.isna().sum().sum() == 0
        assert cleaned.duplicated().sum() == 0
