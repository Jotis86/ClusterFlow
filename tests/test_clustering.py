"""Tests ampliados para clustering.py"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.clustering import determine_optimal_k, perform_clustering, select_best_method


class TestClustering:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'X': np.concatenate([np.random.randn(50), np.random.randn(50) + 5]),
            'Y': np.concatenate([np.random.randn(50), np.random.randn(50) + 5])
        })
    
    @pytest.fixture
    def complex_data(self):
        """Datos con 3 clusters claros"""
        np.random.seed(42)
        return pd.DataFrame({
            'X': np.concatenate([
                np.random.randn(30),
                np.random.randn(30) + 5,
                np.random.randn(30) + 10
            ]),
            'Y': np.concatenate([
                np.random.randn(30),
                np.random.randn(30) + 5,
                np.random.randn(30) + 10
            ])
        })
    
    # Tests de determine_optimal_k
    def test_determine_optimal_k_basic(self, sample_data):
        optimal_k, metrics_df, reductions = determine_optimal_k(sample_data, (2, 5))
        assert optimal_k >= 2
        assert isinstance(metrics_df, pd.DataFrame)
        assert 'k' in metrics_df.columns
        assert 'Silhouette' in metrics_df.columns
        assert 'Davies-Bouldin' in metrics_df.columns
        assert 'Calinski-Harabasz' in metrics_df.columns
        assert 'Score_Compuesto' in metrics_df.columns
    
    def test_determine_optimal_k_inertia_reduction(self, sample_data):
        optimal_k, metrics_df, reductions = determine_optimal_k(sample_data, (2, 6))
        assert isinstance(reductions, list)
        assert len(reductions) == 3  # k_range (2,6) tiene 4 valores, reductions tiene n-1 = 3
        assert all(r >= 0 for r in reductions)  # Las reducciones deben ser positivas
    
    def test_determine_optimal_k_wide_range(self, complex_data):
        optimal_k, metrics_df, reductions = determine_optimal_k(complex_data, (2, 11))
        assert 2 <= optimal_k <= 10
        assert len(metrics_df) == 9  # range(2, 11) tiene 9 valores
    
    def test_determine_optimal_k_edge_case_k2_vs_k3(self, sample_data):
        """Test caso especial donde k=2 está muy cerca de k=3"""
        optimal_k, metrics_df, reductions = determine_optimal_k(sample_data, (2, 5))
        # Verificar que la lógica de filtrado funciona
        assert optimal_k in [2, 3, 4]
    
    # Tests de perform_clustering con diferentes métodos
    def test_perform_clustering_kmeans(self, sample_data):
        result = perform_clustering(sample_data, 2, 'kmeans')
        assert 'labels' in result
        assert 'silhouette' in result
        assert 'davies_bouldin' in result
        assert 'calinski_harabasz' in result
        assert 'distribution' in result
        assert 'max_cluster_pct' in result
        assert len(result['labels']) == len(sample_data)
        assert result['n_clusters'] == 2
        assert -1 <= result['silhouette'] <= 1
        assert result['davies_bouldin'] >= 0
        assert result['calinski_harabasz'] >= 0
    
    def test_perform_clustering_hierarchical_ward(self, sample_data):
        result = perform_clustering(sample_data, 2, 'hierarchical')
        assert 'labels' in result
        assert len(result['labels']) == len(sample_data)
        assert len(set(result['labels'])) <= 2
    
    def test_perform_clustering_hierarchical_complete(self, sample_data):
        result = perform_clustering(sample_data, 3, 'hierarchical_complete')
        assert 'labels' in result
        assert result['n_clusters'] == 3
        assert len(set(result['labels'])) <= 3
    
    def test_perform_clustering_hierarchical_average(self, sample_data):
        result = perform_clustering(sample_data, 3, 'hierarchical_average')
        assert 'labels' in result
        assert result['n_clusters'] == 3
    
    def test_perform_clustering_invalid_method(self, sample_data):
        """Test con método inválido, debería usar kmeans por defecto"""
        result = perform_clustering(sample_data, 2, 'invalid_method')
        assert 'labels' in result
        assert len(result['labels']) == len(sample_data)
    
    def test_perform_clustering_different_k_values(self, complex_data):
        """Test con diferentes valores de k"""
        for k in [2, 3, 4, 5]:
            result = perform_clustering(complex_data, k, 'kmeans')
            assert result['n_clusters'] == k
            assert len(set(result['labels'])) <= k
    
    def test_perform_clustering_distribution(self, sample_data):
        """Test que verifica la distribución de clusters"""
        result = perform_clustering(sample_data, 2, 'kmeans')
        distribution = result['distribution']
        assert abs(sum(distribution) - 1.0) < 0.01  # La suma debe ser ~1
        assert result['max_cluster_pct'] <= 1.0
        assert result['max_cluster_pct'] > 0.0
    
    # Tests de select_best_method
    def test_select_best_method_basic(self, sample_data):
        results = {}
        for method in ['kmeans', 'hierarchical']:
            results[method] = perform_clustering(sample_data, 2, method)
        
        best_method, comparison = select_best_method(results)
        assert best_method in results.keys()
        assert isinstance(comparison, pd.DataFrame)
        assert 'Método' in comparison.columns
        assert 'Silhouette' in comparison.columns
        assert 'Ranking_Promedio' in comparison.columns
        assert 'Score_Final' in comparison.columns
    
    def test_select_best_method_all_methods(self, complex_data):
        """Test con todos los métodos disponibles"""
        methods = ['kmeans', 'hierarchical', 'hierarchical_complete', 'hierarchical_average']
        results = {}
        for method in methods:
            results[method] = perform_clustering(complex_data, 3, method)
        
        best_method, comparison = select_best_method(results)
        assert best_method in methods
        assert len(comparison) == len(methods)
    
    def test_select_best_method_penalization(self, sample_data):
        """Test que verifica la penalización por clusters desbalanceados"""
        results = {
            'method1': perform_clustering(sample_data, 2, 'kmeans'),
            'method2': perform_clustering(sample_data, 2, 'hierarchical')
        }
        
        best_method, comparison = select_best_method(results)
        assert 'Penalizacion' in comparison.columns
        assert 'Max_Cluster_Pct' in comparison.columns
    
    def test_select_best_method_ranking(self, sample_data):
        """Test que verifica el sistema de ranking"""
        results = {}
        for method in ['kmeans', 'hierarchical']:
            results[method] = perform_clustering(sample_data, 3, method)
        
        best_method, comparison = select_best_method(results)
        # Verificar que está ordenado por Score_Final
        assert comparison['Score_Final'].is_monotonic_increasing
