"""Tests corregidos para stats.py"""
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


class TestStats:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100) * 2 + 5,
            'C': np.random.randn(100) * 0.5
        })
    
    def test_calculate_skewness_kurtosis(self, sample_data):
        result = calculate_skewness_kurtosis(sample_data, ['A', 'B', 'C'])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_detect_outliers_iqr(self, sample_data):
        n_outliers, lower, upper, outliers = detect_outliers_iqr(sample_data, 'B')
        assert isinstance(n_outliers, int)
        assert n_outliers >= 0
    
    def test_get_correlation_pairs(self, sample_data):
        corr_matrix = sample_data.corr()
        pairs = get_correlation_pairs(corr_matrix, threshold=0.0)
        assert isinstance(pairs, pd.DataFrame)
    
    def test_calculate_variance_stats(self, sample_data):
        stats = calculate_variance_stats(sample_data, ['A', 'B', 'C'])
        assert isinstance(stats, pd.DataFrame)
        assert 'CV (%)' in stats.columns
