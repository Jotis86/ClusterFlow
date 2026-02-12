"""Tests ampliados para data_loader.py"""
import pytest
import pandas as pd
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.data_loader import load_data


class TestDataLoaderExtended:
    def test_load_csv_with_header(self):
        csv_content = "name,age,score\nAlice,25,90\nBob,30,85"
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        assert error is None
        assert df is not None
        assert 'name' in df.columns
        assert df.loc[0, 'name'] == 'Alice'
    
    def test_load_csv_with_missing_values(self):
        csv_content = "col1,col2,col3\n1,2,3\n4,,6\n7,8,"
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        assert error is None
        assert df is not None
        assert df.isna().sum().sum() > 0
    
    def test_load_invalid_csv(self):
        """Test con contenido inválido"""
        invalid_content = "This is not a valid CSV file\nRandom text"
        file = io.BytesIO(invalid_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        # Debería cargar pero sin estructura clara
        assert df is not None or error is not None
    
    def test_load_empty_csv(self):
        """Test con archivo vacío"""
        csv_content = ""
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        # Pandas puede lanzar error o devolver DataFrame vacío
        assert df is not None or error is not None
    
    def test_load_csv_single_column(self):
        csv_content = "value\n1\n2\n3"
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        assert error is None
        assert df is not None
        assert len(df.columns) == 1
    
    def test_load_csv_numeric_and_text(self):
        csv_content = "id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300"
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        assert error is None
        assert df is not None
        assert len(df) == 3
        assert df['id'].dtype in ['int64', 'int32']
        assert df['name'].dtype == 'object'
    
    def test_load_csv_with_quotes(self):
        csv_content = 'name,description\n"Alice","A person"\n"Bob","Another person"'
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        assert error is None
        assert df is not None
        assert df.loc[0, 'name'] == 'Alice'
    
    def test_load_large_csv(self):
        """Test con CSV más grande"""
        rows = ["col1,col2,col3"]
        rows.extend([f"{i},{i*2},{i*3}" for i in range(1000)])
        csv_content = "\n".join(rows)
        file = io.BytesIO(csv_content.encode())
        file.name = "test.csv"
        
        df, error = load_data(file)
        
        assert error is None
        assert df is not None
        assert len(df) == 1000
