"""
Página 6: Clustering
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import settings
from core import determine_optimal_k, perform_clustering, select_best_method


def render():
    """Renderizar página de clustering"""
    st.markdown('<h2 class="section-header">🎯 Clustering</h2>', unsafe_allow_html=True)
    
    # Verificar que hay datos escalados
    if st.session_state.data_scaled is None:
        st.warning(settings.MESSAGES['no_scaled'])
        st.info("👉 Primero debes escalar los datos en la sección **Escalado de Datos**")
        return
    
    data_scaled = st.session_state.data_scaled
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3 = st.tabs([
        "🔍 Determinar K Óptimo",
        "⚙️ Ejecutar Clustering",
        "📊 Comparar Métodos"
    ])
    
    # TAB 1: Determinar K Óptimo
    with tab1:
        st.markdown("### 🔍 Determinación de K Óptimo")
        
        st.info("""
        💡 **Métodos para determinar el número óptimo de clusters:**
        - **Método del Codo (Elbow):** Busca el punto donde la reducción de inercia disminuye
        - **Silhouette Score:** Mide qué tan similares son los objetos en su cluster vs otros clusters
        - **Davies-Bouldin:** Mide la separación entre clusters (menor es mejor)
        - **Calinski-Harabasz:** Relación entre dispersión inter-cluster e intra-cluster (mayor es mejor)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            k_min = st.number_input(
                "K mínimo",
                min_value=2,
                max_value=20,
                value=settings.DEFAULT_K_MIN,
                step=1
            )
        
        with col2:
            k_max = st.number_input(
                "K máximo",
                min_value=k_min + 1,
                max_value=20,
                value=min(settings.DEFAULT_K_MAX, 10),
                step=1
            )
        
        if st.button("🔍 Calcular K Óptimo", type="primary", width='stretch'):
            with st.spinner("Calculando métricas para diferentes valores de K..."):
                optimal_k, metrics_df, inertia_reduction = determine_optimal_k(
                    data_scaled, 
                    (k_min, k_max + 1)
                )
                
                # Guardar resultados
                st.session_state.optimal_k = optimal_k
                st.session_state.k_metrics = metrics_df
                st.session_state.inertia_reduction = inertia_reduction
                
                st.success(f"✅ K óptimo determinado: **{optimal_k}** clusters")
                
                # Mostrar métricas
                st.markdown("### 📊 Métricas por Valor de K")
                
                display_metrics = metrics_df[['k', 'Silhouette', 'Davies-Bouldin', 
                                              'Calinski-Harabasz', 'Score_Compuesto']].copy()
                display_metrics = display_metrics.round(4)
                
                # Destacar el K óptimo
                def highlight_optimal(row):
                    if row['k'] == optimal_k:
                        return ['background-color: #90EE90'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    display_metrics.style.apply(highlight_optimal, axis=1),
                    width='stretch'
                )
                
                # Visualizaciones
                st.markdown("### 📈 Visualización de Métricas")
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Silhouette Score
                axes[0, 0].plot(metrics_df['k'], metrics_df['Silhouette'], 'o-', linewidth=2)
                axes[0, 0].axvline(optimal_k, color='red', linestyle='--', label=f'K óptimo={optimal_k}')
                axes[0, 0].set_xlabel('Número de Clusters (k)')
                axes[0, 0].set_ylabel('Silhouette Score')
                axes[0, 0].set_title('Silhouette Score (mayor es mejor)')
                axes[0, 0].grid(alpha=0.3)
                axes[0, 0].legend()
                
                # Davies-Bouldin
                axes[0, 1].plot(metrics_df['k'], metrics_df['Davies-Bouldin'], 'o-', 
                               linewidth=2, color='orange')
                axes[0, 1].axvline(optimal_k, color='red', linestyle='--', label=f'K óptimo={optimal_k}')
                axes[0, 1].set_xlabel('Número de Clusters (k)')
                axes[0, 1].set_ylabel('Davies-Bouldin Index')
                axes[0, 1].set_title('Davies-Bouldin Index (menor es mejor)')
                axes[0, 1].grid(alpha=0.3)
                axes[0, 1].legend()
                
                # Calinski-Harabasz
                axes[1, 0].plot(metrics_df['k'], metrics_df['Calinski-Harabasz'], 'o-',
                               linewidth=2, color='green')
                axes[1, 0].axvline(optimal_k, color='red', linestyle='--', label=f'K óptimo={optimal_k}')
                axes[1, 0].set_xlabel('Número de Clusters (k)')
                axes[1, 0].set_ylabel('Calinski-Harabasz Score')
                axes[1, 0].set_title('Calinski-Harabasz Score (mayor es mejor)')
                axes[1, 0].grid(alpha=0.3)
                axes[1, 0].legend()
                
                # Método del Codo (Inercia)
                axes[1, 1].plot(metrics_df['k'], metrics_df['Inercia'], 'o-',
                               linewidth=2, color='purple')
                axes[1, 1].axvline(optimal_k, color='red', linestyle='--', label=f'K óptimo={optimal_k}')
                axes[1, 1].set_xlabel('Número de Clusters (k)')
                axes[1, 1].set_ylabel('Inercia')
                axes[1, 1].set_title('Método del Codo - Inercia')
                axes[1, 1].grid(alpha=0.3)
                axes[1, 1].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Reducción de inercia
                if len(inertia_reduction) > 0:
                    st.markdown("### 📉 Reducción Porcentual de Inercia")
                    
                    reduction_df = pd.DataFrame({
                        'De k': list(range(k_min, k_max)),
                        'A k': list(range(k_min + 1, k_max + 1)),
                        'Reducción (%)': [f"{r:.2f}%" for r in inertia_reduction]
                    })
                    st.dataframe(reduction_df, width='stretch')
        
        # Mostrar resultados si ya existen
        elif 'optimal_k' in st.session_state:
            st.success(f"ℹ️ K óptimo calculado previamente: **{st.session_state.optimal_k}** clusters")
            
            if st.button("🔄 Recalcular K Óptimo"):
                st.rerun()
    
    # TAB 2: Ejecutar Clustering
    with tab2:
        st.markdown("### ⚙️ Configuración y Ejecución de Clustering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Número de clusters
            default_k = st.session_state.get('optimal_k', 3)
            n_clusters = st.number_input(
                "Número de clusters",
                min_value=2,
                max_value=20,
                value=default_k,
                step=1,
                help="Puedes usar el K óptimo calculado o elegir otro valor"
            )
        
        with col2:
            # Método de clustering
            clustering_method = st.selectbox(
                "Método de clustering",
                ['kmeans', 'hierarchical', 'hierarchical_complete', 'hierarchical_average'],
                format_func=lambda x: {
                    'kmeans': 'K-Means',
                    'hierarchical': 'Hierarchical (Ward)',
                    'hierarchical_complete': 'Hierarchical (Complete)',
                    'hierarchical_average': 'Hierarchical (Average)'
                }[x],
                index=0
            )
        
        if st.button("🎯 Ejecutar Clustering", type="primary", width='stretch'):
            with st.spinner(f"Ejecutando clustering con {n_clusters} clusters..."):
                result = perform_clustering(data_scaled, n_clusters, clustering_method)
                
                # Guardar resultados
                st.session_state.cluster_results = result
                st.session_state.n_clusters_used = n_clusters
                st.session_state.method_used = clustering_method
                
                st.success(settings.MESSAGES['clustering_success'])
                
                # Mostrar métricas
                st.markdown("### 📊 Métricas del Clustering")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Silhouette Score", f"{result['silhouette']:.4f}")
                col_b.metric("Davies-Bouldin", f"{result['davies_bouldin']:.4f}")
                col_c.metric("Calinski-Harabasz", f"{result['calinski_harabasz']:.2f}")
                
                # Distribución de clusters
                st.markdown("### 📊 Distribución de Clusters")
                
                distribution = result['distribution']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tabla de distribución
                    dist_df = pd.DataFrame({
                        'Cluster': distribution.index,
                        'Cantidad': distribution.values,
                        'Porcentaje': [f"{v*100:.1f}%" for v in distribution.values]
                    })
                    st.dataframe(dist_df, width='stretch')
                    
                    max_pct = result['max_cluster_pct']
                    if max_pct > 0.8:
                        st.warning(f"⚠️ El cluster más grande tiene {max_pct*100:.1f}% de los datos. Considera usar más clusters.")
                    else:
                        st.success("✅ Distribución balanceada de clusters")
                
                with col2:
                    # Gráfico de distribución
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(distribution)))
                    ax.pie(distribution.values, labels=[f'Cluster {i}' for i in distribution.index],
                          autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title('Distribución de Clusters')
                    st.pyplot(fig)
                    plt.close()
                
                st.info("👉 Ve a la sección **Resultados** para visualizar y analizar los clusters en detalle")
        
        # Mostrar resultados si ya existen
        elif st.session_state.cluster_results is not None:
            st.info(f"ℹ️ Clustering ejecutado previamente con {st.session_state.n_clusters_used} clusters")
            
            result = st.session_state.cluster_results
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette", f"{result['silhouette']:.4f}")
            col2.metric("Davies-Bouldin", f"{result['davies_bouldin']:.4f}")
            col3.metric("Clusters", st.session_state.n_clusters_used)
            
            if st.button("🔄 Ejecutar Nuevo Clustering"):
                st.rerun()
    
    # TAB 3: Comparar Métodos
    with tab3:
        st.markdown("### 📊 Comparación de Métodos de Clustering")
        
        st.info("""
        💡 **Comparación automática de métodos:**
        - Se ejecutarán múltiples algoritmos de clustering
        - Se compararán sus métricas de calidad
        - Se recomendará el mejor método basado en un score compuesto
        """)
        
        # Configuración
        compare_k = st.number_input(
            "Número de clusters para comparar",
            min_value=2,
            max_value=20,
            value=st.session_state.get('optimal_k', 3),
            step=1,
            key="compare_k"
        )
        
        methods_to_compare = st.multiselect(
            "Métodos a comparar",
            ['kmeans', 'hierarchical', 'hierarchical_complete', 'hierarchical_average'],
            default=['kmeans', 'hierarchical'],
            format_func=lambda x: {
                'kmeans': 'K-Means',
                'hierarchical': 'Hierarchical (Ward)',
                'hierarchical_complete': 'Hierarchical (Complete)',
                'hierarchical_average': 'Hierarchical (Average)'
            }[x]
        )
        
        if st.button("🔬 Comparar Métodos", type="primary", width='stretch'):
            if len(methods_to_compare) < 2:
                st.warning("⚠️ Selecciona al menos 2 métodos para comparar")
            else:
                with st.spinner("Ejecutando y comparando métodos..."):
                    results_dict = {}
                    
                    progress_bar = st.progress(0)
                    for idx, method in enumerate(methods_to_compare):
                        result = perform_clustering(data_scaled, compare_k, method)
                        results_dict[method] = result
                        progress_bar.progress((idx + 1) / len(methods_to_compare))
                    
                    progress_bar.empty()
                    
                    # Seleccionar mejor método
                    best_method, comparison_df = select_best_method(results_dict)
                    
                    st.success(f"✅ Mejor método: **{best_method.upper()}**")
                    
                    # Mostrar comparación
                    st.markdown("### 📊 Tabla Comparativa")
                    
                    display_comparison = comparison_df[[
                        'Método', 'Silhouette', 'Davies-Bouldin', 
                        'Calinski-Harabasz', 'Max_Cluster_Pct', 'Score_Final'
                    ]].copy()
                    
                    # Formatear valores
                    display_comparison['Silhouette'] = display_comparison['Silhouette'].round(4)
                    display_comparison['Davies-Bouldin'] = display_comparison['Davies-Bouldin'].round(4)
                    display_comparison['Calinski-Harabasz'] = display_comparison['Calinski-Harabasz'].round(2)
                    display_comparison['Max_Cluster_Pct'] = display_comparison['Max_Cluster_Pct'].apply(lambda x: f"{x*100:.1f}%")
                    display_comparison['Score_Final'] = display_comparison['Score_Final'].round(2)
                    
                    # Destacar mejor método
                    def highlight_best(row):
                        if row['Método'] == best_method:
                            return ['background-color: #90EE90'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(
                        display_comparison.style.apply(highlight_best, axis=1),
                        width='stretch'
                    )
                    
                    # Gráfico de radar
                    st.markdown("### 📈 Comparación Visual")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    x = np.arange(len(methods_to_compare))
                    width = 0.25
                    
                    silhouettes = [results_dict[m]['silhouette'] for m in methods_to_compare]
                    davies = [results_dict[m]['davies_bouldin'] for m in methods_to_compare]
                    calinski = [results_dict[m]['calinski_harabasz'] / 1000 for m in methods_to_compare]  # Escalar
                    
                    ax.bar(x - width, silhouettes, width, label='Silhouette', alpha=0.8)
                    ax.bar(x, davies, width, label='Davies-Bouldin', alpha=0.8)
                    ax.bar(x + width, calinski, width, label='Calinski/1000', alpha=0.8)
                    
                    ax.set_xlabel('Método')
                    ax.set_ylabel('Valor')
                    ax.set_title('Comparación de Métricas por Método')
                    ax.set_xticks(x)
                    ax.set_xticklabels([m.upper() for m in methods_to_compare], rotation=45)
                    ax.legend()
                    ax.grid(alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Guardar mejor resultado
                    if st.button("✅ Usar Mejor Método", width='stretch'):
                        st.session_state.cluster_results = results_dict[best_method]
                        st.session_state.n_clusters_used = compare_k
                        st.session_state.method_used = best_method
                        st.success(f"✅ Resultado de {best_method.upper()} guardado. Ve a **Resultados** para visualizar.")
