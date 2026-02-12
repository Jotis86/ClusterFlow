"""
Módulo para algoritmos de clustering
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Tuple


def determine_optimal_k(data: pd.DataFrame, k_range: Tuple[int, int] = (2, 11)) -> Tuple[int, pd.DataFrame, list]:
    """
    Determinar número óptimo de clusters usando múltiples métricas
    
    Args:
        data: DataFrame con datos escalados
        k_range: Tuple con (k_min, k_max) para evaluar
        
    Returns:
        Tuple[k óptimo, DataFrame con métricas, lista de reducciones de inercia]
    """
    if len(data) < k_range[0]:
        raise ValueError(f"No hay suficientes datos ({len(data)}) para el número mínimo de clusters ({k_range[0]})")
    
    K_range = range(k_range[0], k_range[1])
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(data)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
        davies_bouldin_scores.append(davies_bouldin_score(data, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(data, labels))
    
    # Calcular score compuesto
    metrics_df = pd.DataFrame({
        'k': list(K_range),
        'Silhouette': silhouette_scores,
        'Davies-Bouldin': davies_bouldin_scores,
        'Calinski-Harabasz': calinski_harabasz_scores,
        'Inercia': inertias
    })
    
    # Normalizar métricas (con manejo de división por cero)
    def safe_normalize(series):
        range_val = series.max() - series.min()
        if range_val == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / range_val
    
    metrics_df['Silhouette_norm'] = safe_normalize(metrics_df['Silhouette'])
    metrics_df['Davies_norm'] = 1 - safe_normalize(metrics_df['Davies-Bouldin'])
    metrics_df['Calinski_norm'] = safe_normalize(metrics_df['Calinski-Harabasz'])
    
    metrics_df['Score_Compuesto'] = (metrics_df['Silhouette_norm'] + 
                                      metrics_df['Davies_norm'] + 
                                      metrics_df['Calinski_norm']) / 3
    
    # Calcular reducción de inercia
    inertia_reduction = [(inertias[i-1] - inertias[i]) / inertias[i-1] * 100 
                         for i in range(1, len(inertias))]
    
    # Filtrar k=2 si está muy cerca de k=3
    metrics_filtered = metrics_df.copy()
    best_k_composite = metrics_df.loc[metrics_df['Score_Compuesto'].idxmax(), 'k']
    
    if best_k_composite == 2:
        k3_score = metrics_df.loc[metrics_df['k'] == 3, 'Score_Compuesto'].values[0]
        best_score = metrics_df.loc[metrics_df['k'] == best_k_composite, 'Score_Compuesto'].values[0]
        score_diff_pct = abs(best_score - k3_score) / best_score * 100
        
        if score_diff_pct < 5:
            metrics_filtered = metrics_df[metrics_df['k'] > 2]
    
    # Seleccionar k óptimo
    optimal_k = int(metrics_filtered.loc[metrics_filtered['Score_Compuesto'].idxmax(), 'k'])
    
    return optimal_k, metrics_df, inertia_reduction


def perform_clustering(data: pd.DataFrame, n_clusters: int, method: str = 'kmeans') -> Dict:
    """
    Realizar clustering con el método especificado
    
    Args:
        data: DataFrame con datos escalados
        n_clusters: Número de clusters a formar
        method: Método de clustering ('kmeans', 'hierarchical', 'hierarchical_complete', 'hierarchical_average')
        
    Returns:
        Dict con modelo, labels, métricas y distribución
    """
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif method == 'hierarchical_complete':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    elif method == 'hierarchical_average':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    
    labels = model.fit_predict(data)
    
    # Calcular métricas
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    
    # Calcular distribución
    distribution = pd.Series(labels).value_counts(normalize=True)
    max_cluster_pct = distribution.max()
    
    return {
        'model': model,
        'labels': labels,
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'distribution': distribution,
        'max_cluster_pct': max_cluster_pct
    }


def select_best_method(results_dict: Dict[str, Dict]) -> Tuple[str, pd.DataFrame]:
    """
    Seleccionar el mejor método de clustering basado en métricas
    
    Args:
        results_dict: Diccionario con resultados de diferentes métodos
        
    Returns:
        Tuple[nombre del mejor método, DataFrame con comparación]
    """
    comparison = []
    
    for method_name, result in results_dict.items():
        comparison.append({
            'Método': method_name,
            'Silhouette': result['silhouette'],
            'Davies-Bouldin': result['davies_bouldin'],
            'Calinski-Harabasz': result['calinski_harabasz'],
            'Max_Cluster_Pct': result['max_cluster_pct']
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Ranking por métricas
    silhouette_ranking = comparison_df['Silhouette'].rank(ascending=False).values
    davies_ranking = comparison_df['Davies-Bouldin'].rank(ascending=True).values
    calinski_ranking = comparison_df['Calinski-Harabasz'].rank(ascending=False).values
    
    comparison_df['Ranking_Promedio'] = (silhouette_ranking + davies_ranking + calinski_ranking) / 3
    
    # Penalizar clusters muy desbalanceados
    comparison_df['Penalizacion'] = comparison_df['Max_Cluster_Pct'].apply(
        lambda x: (x - 0.8) / 0.2 if x > 0.8 else 0
    )
    
    comparison_df['Score_Final'] = comparison_df['Ranking_Promedio'] + (comparison_df['Penalizacion'] * 10)
    
    # Ordenar por score final
    comparison_df = comparison_df.sort_values('Score_Final')
    
    best_method = comparison_df.iloc[0]['Método']
    
    return best_method, comparison_df
