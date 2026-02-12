"""
Página 7: Resultados y Visualización (Simplificada)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from config import settings


def render():
    """Renderizar página de resultados"""
    st.markdown('<h2 class="section-header">📈 Resultados del Clustering</h2>', unsafe_allow_html=True)
    
    # Verificar que hay resultados de clustering
    if st.session_state.cluster_results is None:
        st.warning("⚠️ Primero debes ejecutar el clustering en la sección **Clustering**")
        return
    
    result = st.session_state.cluster_results
    data_scaled = st.session_state.data_scaled
    data_original = st.session_state.data_clean if st.session_state.data_clean is not None else st.session_state.data
    
    # Agregar labels a los datos
    data_with_clusters = data_original.copy()
    data_with_clusters['Cluster'] = result['labels']
    
    # ===================
    # SECCIÓN 1: RESUMEN
    # ===================
    st.markdown("### 📊 Resumen del Análisis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Clusters", result['n_clusters'])
    col2.metric("📊 Observaciones", len(data_with_clusters))
    col3.metric("✅ Silhouette", f"{result['silhouette']:.3f}")
    col4.metric("🔧 Método", st.session_state.get('method_used', 'kmeans').upper())
    
    # Interpretación del Silhouette
    if result['silhouette'] > 0.7:
        st.success("✅ **Excelente separación** de clusters")
    elif result['silhouette'] > 0.5:
        st.info("✔️ **Buena separación** de clusters")
    elif result['silhouette'] > 0.25:
        st.warning("⚠️ **Separación aceptable** - considera ajustar parámetros")
    else:
        st.error("❌ **Separación débil** - se recomienda revisar el número de clusters")
    
    st.markdown("---")
    
    # ===================
    # SECCIÓN 2: VISUALIZACIÓN DE CLUSTERS (PRINCIPAL)
    # ===================
    st.markdown("### 🎨 Visualización de Clusters")
    
    st.info("💡 Visualización automática usando **PCA (Análisis de Componentes Principales)** para proyectar todas las variables en 2D")
    
    # Obtener variables para visualizar
    numeric_cols = data_scaled.columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Se necesitan al menos 2 variables para visualización")
    else:
        # Aplicar PCA automáticamente
        with st.spinner("Calculando proyección PCA..."):
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_scaled)
            
            # Varianza explicada
            explained_var = pca.explained_variance_ratio_
            
            st.success(f"✅ PCA aplicado - Varianza explicada: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}, Total={sum(explained_var):.1%}")
        
        # GRÁFICO PRINCIPAL DE CLUSTERS CON PCA
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Paleta de colores vibrantes
        colors = plt.cm.tab10(np.linspace(0, 1, result['n_clusters']))
        
        for cluster_id in range(result['n_clusters']):
            cluster_mask = result['labels'] == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            ax.scatter(
                data_pca[cluster_mask, 0],
                data_pca[cluster_mask, 1],
                c=[colors[cluster_id]],
                label=f'Cluster {cluster_id} (n={cluster_size})',
                alpha=0.7,
                s=150,
                edgecolors='black',
                linewidth=1.5
            )
            
            # Calcular y mostrar centroide
            centroid_x = data_pca[cluster_mask, 0].mean()
            centroid_y = data_pca[cluster_mask, 1].mean()
            ax.scatter(centroid_x, centroid_y, c=[colors[cluster_id]], 
                      marker='*', s=800, edgecolors='black', linewidth=2,
                      zorder=10)
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} varianza)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} varianza)', fontsize=14, fontweight='bold')
        ax.set_title(f'Distribución General de Clusters (PCA)', 
                    fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_facecolor('#F8F9FA')
        
        # Añadir elipses de confianza para cada cluster
        for cluster_id in range(result['n_clusters']):
            cluster_mask = result['labels'] == cluster_id
            cluster_points = data_pca[cluster_mask]
            
            if len(cluster_points) > 2:
                # Calcular elipse de confianza (2 std)
                mean = cluster_points.mean(axis=0)
                cov = np.cov(cluster_points.T)
                
                # Eigenvalues y eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                
                # Dibujar elipse
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(mean, width=2*np.sqrt(eigenvalues[0])*2, 
                                height=2*np.sqrt(eigenvalues[1])*2,
                                angle=angle, alpha=0.2, 
                                facecolor=colors[cluster_id], 
                                edgecolor=colors[cluster_id], linewidth=2)
                ax.add_patch(ellipse)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # ===================
    # SECCIÓN 3: DISTRIBUCIÓN Y TABLA RESUMEN
    # ===================
    st.markdown("### 📊 Distribución de Clusters")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        # Gráfico de barras de distribución
        fig, ax = plt.subplots(figsize=(8, 6))
        cluster_counts = data_with_clusters['Cluster'].value_counts().sort_index()
        colors_bar = plt.cm.tab10(np.linspace(0, 1, result['n_clusters']))
        
        bars = ax.bar(cluster_counts.index, cluster_counts.values, 
                     color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Añadir valores sobre las barras
        for i, (idx, v) in enumerate(zip(cluster_counts.index, cluster_counts.values)):
            percentage = (v / len(data_with_clusters)) * 100
            ax.text(idx, v + max(cluster_counts.values)*0.02, 
                   f'{v}\n({percentage:.1f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Observaciones', fontsize=12, fontweight='bold')
        ax.set_title('Distribución por Cluster', fontsize=14, fontweight='bold', color='#2C3E50')
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        ax.set_facecolor('#F8F9FA')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col_b:
        # Tabla resumen
        st.markdown("#### 📋 Resumen por Cluster")
        
        summary_data = []
        numeric_original = data_original.select_dtypes(include=[np.number]).columns.tolist()
        
        for cluster_id in range(result['n_clusters']):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
            summary_data.append({
                'Cluster': f'Cluster {cluster_id}',
                'Tamaño': len(cluster_data),
                'Porcentaje': f"{len(cluster_data) / len(data_with_clusters) * 100:.1f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch', height=250)
        
        # Métricas de calidad compactas
        st.markdown("#### 📏 Métricas de Calidad")
        
        metrics_compact = pd.DataFrame({
            'Métrica': ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'],
            'Valor': [
                f"{result['silhouette']:.3f}",
                f"{result['davies_bouldin']:.3f}",
                f"{result['calinski_harabasz']:.1f}"
            ],
            'Status': [
                '✅ Bueno' if result['silhouette'] > 0.5 else '⚠️ Mejorable',
                '✅ Bueno' if result['davies_bouldin'] < 1.0 else '⚠️ Mejorable',
                '✅ Bueno' if result['calinski_harabasz'] > 100 else '⚠️ Mejorable'
            ]
        })
        
        st.dataframe(metrics_compact, width='stretch')
    
    st.markdown("---")
    
    # ===================
    # SECCIÓN 4: EXPORTAR
    # ===================
    st.markdown("### 💾 Exportar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    # Preparar datos
    export_data = data_original.copy()
    export_data['Cluster'] = result['labels']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with col1:
        st.markdown("#### 📊 Datos Completos")
        csv_data = export_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv_data,
            file_name=f"clusters_{timestamp}.csv",
            mime="text/csv",
            width='stretch'
        )
    
    with col2:
        st.markdown("#### 📋 Perfiles")
        
        # Crear perfiles
        profiles = []
        for cluster_id in range(result['n_clusters']):
            cluster_subset = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
            row = {'Cluster': cluster_id, 'Tamaño': len(cluster_subset)}
            
            for col in numeric_original:
                row[f'{col}_mean'] = cluster_subset[col].mean()
            
            profiles.append(row)
        
        profiles_df = pd.DataFrame(profiles)
        csv_profiles = profiles_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv_profiles,
            file_name=f"profiles_{timestamp}.csv",
            mime="text/csv",
            width='stretch'
        )
    
    with col3:
        st.markdown("#### 📏 Métricas")
        
        metrics_export = pd.DataFrame({
            'Métrica': ['Clusters', 'Método', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Observaciones'],
            'Valor': [
                str(result['n_clusters']),
                str(st.session_state.get('method_used', 'kmeans').upper()),
                f"{result['silhouette']:.4f}",
                f"{result['davies_bouldin']:.4f}",
                f"{result['calinski_harabasz']:.2f}",
                str(len(data_with_clusters))
            ]
        })
        
        csv_metrics = metrics_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv_metrics,
            file_name=f"metrics_{timestamp}.csv",
            mime="text/csv",
            width='stretch'
        )
    
    st.markdown("---")
    st.success("✅ Análisis de clustering completado. Puedes exportar los resultados usando los botones de arriba.")
