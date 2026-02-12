"""
Módulo core
"""
from .data_loader import load_data
from .data_cleaner import analyze_data_quality, clean_data
from .scaler import scale_data
from .clustering import determine_optimal_k, perform_clustering, select_best_method
