"""
Página 3: Análisis Exploratorio de Datos
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import settings
from utils import (
    calculate_skewness_kurtosis,
    detect_outliers_iqr,
    get_correlation_pairs,
    calculate_variance_stats
)


def render():
    """Renderizar página de análisis exploratorio"""
    st.markdown('<h2 class="section-header">📊 Análisis Exploratorio</h2>', unsafe_allow_html=True)
    
    # Verificar que hay datos limpios
    data = st.session_state.data_clean if st.session_state.data_clean is not None else st.session_state.data
    
    if data is None:
        st.warning(settings.MESSAGES['no_data'])
        return
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error(settings.MESSAGES['no_numeric'])
        return
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Estadísticas", 
        "📊 Distribuciones", 
        "🔍 Outliers", 
        "🔗 Correlaciones",
        "📉 Análisis Bivariado"
    ])
    
    # TAB 1: Estadísticas Descriptivas
    with tab1:
        st.markdown("### 📋 Estadísticas Descriptivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Resumen General")
            st.dataframe(data[numeric_cols].describe(), width='stretch')
        
        with col2:
            st.markdown("#### Asimetría y Curtosis")
            skew_kurt = calculate_skewness_kurtosis(data, numeric_cols)
            st.dataframe(skew_kurt, width='stretch')
        
        st.markdown("#### Estadísticas de Varianza")
        variance_stats = calculate_variance_stats(data, numeric_cols)
        st.dataframe(variance_stats, width='stretch')
    
    # TAB 2: Distribuciones
    with tab2:
        st.markdown("### 📊 Análisis de Distribuciones")
        
        selected_vars = st.multiselect(
            "Selecciona variables para visualizar",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))]
        )
        
        if selected_vars:
            n_vars = len(selected_vars)
            n_cols = 2
            n_rows = (n_vars + 1) // 2
            
            # Colores vibrantes
            colors_hist = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
            axes = axes.flatten() if n_vars > 1 else [axes]
            
            for idx, var in enumerate(selected_vars):
                ax = axes[idx]
                color = colors_hist[idx % len(colors_hist)]
                data[var].hist(bins=30, ax=ax, edgecolor='#2C3E50', 
                             alpha=0.75, color=color, linewidth=1.5)
                ax.set_title(f'Distribución de {var}', fontsize=13, 
                           fontweight='bold', color='#2C3E50')
                ax.set_xlabel(var, fontsize=11, fontweight='bold')
                ax.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
                ax.grid(alpha=0.3, linestyle='--')
                ax.set_facecolor('#F8F9FA')
            
            # Ocultar ejes vacíos
            for idx in range(len(selected_vars), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Boxplots
            st.markdown("#### 📦 Boxplots")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
            axes = axes.flatten() if n_vars > 1 else [axes]
            
            for idx, var in enumerate(selected_vars):
                ax = axes[idx]
                color = colors_hist[idx % len(colors_hist)]
                bp = ax.boxplot(data[var].dropna(), patch_artist=True, widths=0.6)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.7)
                bp['boxes'][0].set_linewidth(2)
                for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                    plt.setp(bp[element], color='#2C3E50', linewidth=2)
                ax.set_ylabel(var, fontsize=11, fontweight='bold')
                ax.set_title(f'Boxplot: {var}', fontsize=13, 
                           fontweight='bold', color='#2C3E50')
                ax.grid(alpha=0.3, linestyle='--', axis='y')
                ax.set_facecolor('#F8F9FA')
            
            for idx in range(len(selected_vars), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # TAB 3: Outliers
    with tab3:
        st.markdown("### 🔍 Análisis de Outliers")
        
        st.info("⚠️ **IMPORTANTE:** Si detectas outliers aquí, debes eliminarlos o imputarlos en la página de **Limpieza de Datos** antes de continuar.")
        
        # Boxplots para todas las variables
        st.markdown("#### 📊 Boxplots por Variable")
        
        # Selector de variables a mostrar
        selected_vars = st.multiselect(
            "Selecciona variables para visualizar (vacío = todas)",
            numeric_cols,
            default=[],
            key="outlier_vars"
        )
        
        vars_to_plot = selected_vars if len(selected_vars) > 0 else numeric_cols
        
        if len(vars_to_plot) > 0:
            # Calcular número de filas necesarias
            n_cols = 3
            n_rows = (len(vars_to_plot) + n_cols - 1) // n_cols
            
            # Crear subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Colores vibrantes
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788']
            
            for idx, col in enumerate(vars_to_plot):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                # Crear boxplot
                bp = ax.boxplot([data[col].dropna()], 
                               patch_artist=True,
                               labels=[col],
                               widths=0.6)
                
                # Colorear el boxplot
                color = colors[idx % len(colors)]
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.7)
                bp['boxes'][0].set_linewidth(2)
                
                # Colorear otros elementos
                for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                    plt.setp(bp[element], color='#2C3E50', linewidth=2)
                
                # Estilo
                ax.set_ylabel('Valor', fontsize=11, fontweight='bold')
                ax.set_title(col, fontsize=13, fontweight='bold', color='#2C3E50')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_facecolor('#F8F9FA')
                
                # Estadísticas
                n_outliers, _, _, _ = detect_outliers_iqr(data, col)
                if n_outliers > 0:
                    ax.text(0.5, 0.95, f'⚠️ {n_outliers} outliers', 
                           transform=ax.transAxes,
                           ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8),
                           fontsize=10, fontweight='bold', color='#E74C3C')
            
            # Ocultar ejes vacíos
            for idx in range(len(vars_to_plot), n_rows * n_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Resumen de outliers
            st.markdown("#### 📋 Resumen de Outliers")
            outlier_summary = []
            for col in numeric_cols:
                n_out, lower, upper, _ = detect_outliers_iqr(data, col)
                outlier_summary.append({
                    'Variable': col,
                    'Outliers': n_out,
                    'Porcentaje': f"{(n_out / len(data) * 100):.2f}%",
                    'Límite Inferior': f"{lower:.2f}",
                    'Límite Superior': f"{upper:.2f}"
                })
            
            outlier_df = pd.DataFrame(outlier_summary)
            outlier_df = outlier_df[outlier_df['Outliers'] > 0] if len(outlier_df[outlier_df['Outliers'] > 0]) > 0 else outlier_df
            
            if len(outlier_df[outlier_df['Outliers'] > 0]) > 0:
                st.warning("⚠️ **Se detectaron outliers.** Ve a **Limpieza de Datos** y activa 'Eliminar outliers' antes de continuar.")
            else:
                st.success("✅ No se detectaron outliers significativos en los datos.")
            
            st.dataframe(outlier_df, width='stretch')
    
    # TAB 4: Correlaciones
    with tab4:
        st.markdown("### 🔗 Análisis de Correlaciones")
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas para calcular correlaciones")
        else:
            corr_matrix = data[numeric_cols].corr()
            
            # Heatmap
            st.markdown("#### 🗺️ Matriz de Correlación")
            fig, ax = plt.subplots(figsize=(14, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdYlBu_r', center=0, square=True, linewidths=2,
                       cbar_kws={"shrink": 0.8, "label": "Correlación"},
                       annot_kws={"size": 10, "weight": "bold"},
                       ax=ax, vmin=-1, vmax=1)
            ax.set_title('Matriz de Correlación', fontsize=16, 
                        fontweight='bold', color='#2C3E50', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Pares de correlación alta
            st.markdown("#### 📋 Pares con Alta Correlación")
            threshold = st.slider(
                "Umbral de correlación",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="corr_threshold"
            )
            
            corr_pairs = get_correlation_pairs(corr_matrix, threshold=threshold)
            
            if len(corr_pairs) > 0:
                st.dataframe(corr_pairs, width='stretch')
                
                if len(corr_pairs) > 0:
                    st.info(f"💡 Encontradas {len(corr_pairs)} pares con correlación > {threshold}")
            else:
                st.success(f"✅ No hay pares con correlación superior a {threshold}")
    
    # TAB 5: Análisis Bivariado
    with tab5:
        st.markdown("### 📉 Análisis Bivariado")
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables para análisis bivariado")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                var_x = st.selectbox("Variable X", numeric_cols, key="bivar_x")
            
            with col2:
                var_y = st.selectbox(
                    "Variable Y", 
                    [col for col in numeric_cols if col != var_x],
                    key="bivar_y"
                )
            
            if var_x and var_y:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(data[var_x], data[var_y], alpha=0.6)
                ax.set_xlabel(var_x)
                ax.set_ylabel(var_y)
                ax.set_title(f'{var_x} vs {var_y}')
                ax.grid(alpha=0.3)
                
                # Línea de tendencia
                z = np.polyfit(data[var_x], data[var_y], 1)
                p = np.poly1d(z)
                ax.plot(data[var_x], p(data[var_x]), "r--", alpha=0.8, 
                       label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.2f}')
                ax.legend()
                
                st.pyplot(fig)
                plt.close()
                
                # Estadísticas
                correlation = data[var_x].corr(data[var_y])
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Correlación", f"{correlation:.4f}")
                col_b.metric("R²", f"{correlation**2:.4f}")
                col_c.metric("Tipo", "Positiva" if correlation > 0 else "Negativa")
