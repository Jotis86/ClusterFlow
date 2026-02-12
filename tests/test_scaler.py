"""Tests corregidos para scaler.py"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.scaler import scale_data


class TestScaler:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
    
    def test_scale_standard(self, sample_data):
        scaled_df, scaler = scale_data(sample_data, 'standard', ['A', 'B'])
        assert scaler is not None
        assert abs(scaled_df['A'].mean()) < 0.01
    
    def test_scale_minmax(self, sample_data):
        scaled_df, scaler = scale_data(sample_data, 'minmax', ['A', 'B'])
        assert scaler is not None
        assert scaled_df['A'].min() == 0.0
        assert scaled_df['A'].max() == 1.0
    
    def test_scale_robust(self, sample_data):
        scaled_df, scaler = scale_data(sample_data, 'robust', ['A', 'B'])
        assert scaler is not None
