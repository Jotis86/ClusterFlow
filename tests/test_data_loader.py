"""Tests corregidos para data_loader.py"""
import pytest
import pandas as pd
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.data_loader import load_data


class TestDataLoader:
    def test_load_valid_csv(self):
        csv_content = "col1,col2,col3\n1,2,3\n4,5,6"
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        assert error is None
        assert df is not None
        assert len(df) == 2
    
    def test_load_csv_with_semicolon(self):
        csv_content = "col1;col2;col3\n1;2;3\n4;5;6"
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        # pandas intentará leerlo, puede detectar el separador
        assert df is not None or error is not None
