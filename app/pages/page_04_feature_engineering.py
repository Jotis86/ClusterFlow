"""
Página 4: Feature Engineering
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import settings


def render():
    """Renderizar página de feature engineering"""
    st.markdown('<h2 class="section-header">🔧 Feature Engineering</h2>', unsafe_allow_html=True)
    
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
    tab1, tab2, tab3 = st.tabs([
        "📋 Selección de Variables",
        "🔗 Multicolinealidad",
        "📊 Filtrado por Varianza"
    ])
    
    # TAB 1: Selección de Variables
    with tab1:
        st.markdown("### 📋 Selección de Variables para Clustering")
        
        st.info("""
        💡 **Recomendaciones:**
        - Usa **🧠 Selección Inteligente** para una selección automática óptima
        - El algoritmo excluye: IDs, variables constantes, baja varianza, alta correlación
        - Selecciona variables numéricas relevantes para el clustering
        - El clustering funciona mejor con 2-10 variables
        """)
        
        # Variables disponibles
        st.markdown("#### Variables Disponibles")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Información de variables
            var_info = []
            for col in numeric_cols:
                var_info.append({
                    'Variable': col,
                    'Tipo': str(data[col].dtype),
                    'Únicos': data[col].nunique(),
                    'Nulos': data[col].isnull().sum(),
                    'Media': f"{data[col].mean():.2f}",
                    'Std': f"{data[col].std():.2f}",
                    'Min': f"{data[col].min():.2f}",
                    'Max': f"{data[col].max():.2f}"
                })
            
            var_df = pd.DataFrame(var_info)
            st.dataframe(var_df, use_container_width=True)
        
        with col2:
            st.metric("Total Variables Numéricas", len(numeric_cols))
            st.metric("Variables Recomendadas", f"2-{min(10, len(numeric_cols))}")
        
        # Selector de variables
        st.markdown("#### 🎯 Seleccionar Variables para Clustering")
        
        # Opciones de preselección
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("✅ Seleccionar Todas", use_container_width=True):
                st.session_state.selected_features = numeric_cols
        
        with col_b:
            if st.button("🧠 Selección Inteligente", use_container_width=True, type="primary"):
                # SELECCIÓN AUTOMÁTICA INTELIGENTE
                selected = []
                
                # 1. Excluir variables tipo ID (bajo número de valores únicos o secuencial)
                for col in numeric_cols:
                    unique_ratio = data[col].nunique() / len(data)
                    
                    # Excluir si parece un ID
                    if 'id' in col.lower():
                        continue
                    
                    # Excluir si tiene énicamente valores únicos (probablemente ID)
                    if unique_ratio > 0.95:
                        continue
                    
                    # Excluir si tiene muy pocos valores únicos (menos de 5)
                    if data[col].nunique() < 5:
                        continue
                    
                    # 2. Verificar varianza significativa (CV > 10%)
                    if data[col].std() > 0:
                        cv = (data[col].std() / abs(data[col].mean())) * 100 if data[col].mean() != 0 else 0
                        if cv < 10:  # Baja varianza
                            continue
                    
                    selected.append(col)
                
                # 3. Si hay más de 10, seleccionar las de mayor varianza
                if len(selected) > 10:
                    variances = {col: data[col].var() for col in selected}
                    selected = sorted(variances, key=variances.get, reverse=True)[:10]
                
                # 4. Eliminar variables altamente correlacionadas (> 0.9)
                corr_excluded = 0
                if len(selected) > 1:
                    corr_matrix = data[selected].corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
                    corr_excluded = len(to_drop)
                    selected = [col for col in selected if col not in to_drop]
                
                # Guardar información para mostrar después
                excluded_count = len(numeric_cols) - len(selected)
                st.session_state.intelligent_selection_info = {
                    'used': True
                }
                
                st.session_state.selected_features = selected
                st.success(f"✅ {len(selected)} variables seleccionadas automáticamente ({excluded_count} excluidas)")
                st.rerun()
        
        with col_c:
            if st.button("🔄 Limpiar Selección", use_container_width=True):
                st.session_state.selected_features = []
        
        # Inicializar selección
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = numeric_cols[:min(5, len(numeric_cols))]
        
        selected_features = st.multiselect(
            "Variables seleccionadas",
            numeric_cols,
            default=st.session_state.selected_features,
            key="feature_selector",
            help="🧠 Usa 'Selección Inteligente' para una selección automática basada en criterios estadísticos"
        )
        
        st.session_state.selected_features = selected_features
        
        if len(selected_features) > 0:
            # Mostrar badge si se usó selección inteligente
            if st.session_state.get('intelligent_selection_info', {}).get('used'):
                st.success(f"✅ {len(selected_features)} variables seleccionadas (🧠 Selección Inteligente Aplicada)")
            else:
                st.success(f"✅ {len(selected_features)} variables seleccionadas")
            
            # Vista previa de datos seleccionados
            st.markdown("#### 👁️ Vista Previa de Variables Seleccionadas")
            st.dataframe(data[selected_features].head(10), use_container_width=True)
        else:
            st.warning("⚠️ No hay variables seleccionadas")
    
    # TAB 2: Multicolinealidad
    with tab2:
        st.markdown("### 🔗 Análisis de Multicolinealidad")
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables para analizar multicolinealidad")
        else:
            st.info("""
            💡 **Multicolinealidad:** Correlación alta entre variables independientes.
            - **Umbral recomendado:** |correlación| > 0.8 - 0.9
            - **Problema:** Variables redundantes que no aportan información nueva
            - **Solución:** Eliminar una de las variables altamente correlacionadas
            """)
            
            # Calcular correlaciones
            corr_matrix = data[numeric_cols].corr()
            
            # Umbral de correlación
            corr_threshold = st.slider(
                "Umbral de correlación para considerar multicolinealidad",
                min_value=0.5,
                max_value=1.0,
                value=settings.DEFAULT_CORRELATION_THRESHOLD,
                step=0.05,
                help="Variables con correlación superior a este valor se consideran multicolineales"
            )
            
            # Encontrar pares multicolineales
            multicollinear_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > corr_threshold:
                        multicollinear_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlación': corr_value,
                            'Correlación Abs': abs(corr_value)
                        })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Pares Multicolineales", len(multicollinear_pairs))
            
            with col2:
                st.metric("Umbral Actual", f"{corr_threshold:.2f}")
            
            if len(multicollinear_pairs) > 0:
                st.warning(f"⚠️ Se encontraron {len(multicollinear_pairs)} pares de variables multicolineales")
                
                pairs_df = pd.DataFrame(multicollinear_pairs)
                pairs_df = pairs_df.sort_values('Correlación Abs', ascending=False)
                st.dataframe(pairs_df[['Variable 1', 'Variable 2', 'Correlación']], 
                           use_container_width=True)
                
                st.info("""
                💡 **Recomendación:** Considera eliminar una variable de cada par para reducir redundancia.
                Puedes hacerlo en la pestaña "Selección de Variables".
                """)
            else:
                st.success(f"✅ No se detectó multicolinealidad con umbral > {corr_threshold}")
            
            # Heatmap de correlación
            st.markdown("#### 🗺️ Matriz de Correlación")
            
            # Filtrar solo variables seleccionadas si existen
            if 'selected_features' in st.session_state and len(st.session_state.selected_features) > 1:
                show_selected = st.checkbox("Mostrar solo variables seleccionadas", value=True)
                if show_selected:
                    cols_to_show = st.session_state.selected_features
                else:
                    cols_to_show = numeric_cols
            else:
                cols_to_show = numeric_cols
            
            if len(cols_to_show) > 1:
                corr_subset = data[cols_to_show].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                mask = np.triu(np.ones_like(corr_subset, dtype=bool))
                sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', 
                           cmap='coolwarm', center=0, ax=ax, square=True, 
                           linewidths=1, cbar_kws={"shrink": 0.8})
                ax.set_title('Matriz de Correlación', fontsize=16, pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # TAB 3: Filtrado por Varianza
    with tab3:
        st.markdown("### 📊 Filtrado por Varianza")
        
        st.info("""
        💡 **Filtrado por Varianza:**
        - Variables con **baja varianza** tienen valores muy similares y aportan poca información
        - **Coeficiente de Variación (CV)** = (Desviación Estándar / Media) × 100
        - **CV < 10%**: Baja varianza (considerar eliminar)
        - **CV > 30%**: Alta varianza (útil para clustering)
        """)
        
        # Calcular estadísticas de varianza
        variance_data = []
        for col in numeric_cols:
            mean_val = data[col].mean()
            std_val = data[col].std()
            cv = (std_val / abs(mean_val) * 100) if mean_val != 0 else 0
            
            variance_data.append({
                'Variable': col,
                'Media': mean_val,
                'Std': std_val,
                'Varianza': data[col].var(),
                'CV (%)': cv,
                'Rango': data[col].max() - data[col].min()
            })
        
        variance_df = pd.DataFrame(variance_data)
        variance_df = variance_df.sort_values('CV (%)', ascending=False)
        
        # Umbral de varianza
        variance_threshold = st.slider(
            "Coeficiente de Variación mínimo (%)",
            min_value=0.0,
            max_value=50.0,
            value=settings.DEFAULT_VARIANCE_THRESHOLD,
            step=0.5,
            help="Variables con CV inferior a este valor tienen baja varianza"
        )
        
        # Clasificar variables
        high_variance = variance_df[variance_df['CV (%)'] > variance_threshold]
        low_variance = variance_df[variance_df['CV (%)'] <= variance_threshold]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Variables Alta Varianza", len(high_variance), 
                     help=f"CV > {variance_threshold}%")
        
        with col2:
            st.metric("Variables Baja Varianza", len(low_variance),
                     help=f"CV ≤ {variance_threshold}%")
        
        # Mostrar resultados
        st.markdown("#### 📊 Análisis de Varianza por Variable")
        
        # Formatear dataframe para mejor visualización
        display_df = variance_df.copy()
        display_df['Media'] = display_df['Media'].round(2)
        display_df['Std'] = display_df['Std'].round(2)
        display_df['Varianza'] = display_df['Varianza'].round(2)
        display_df['CV (%)'] = display_df['CV (%)'].round(2)
        display_df['Rango'] = display_df['Rango'].round(2)
        
        # Colorear filas según varianza
        def highlight_variance(row):
            if row['CV (%)'] <= variance_threshold:
                return ['background-color: #ffcccc'] * len(row)
            else:
                return ['background-color: #ccffcc'] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_variance, axis=1),
            use_container_width=True
        )
        
        if len(low_variance) > 0:
            st.warning(f"""
            ⚠️ **{len(low_variance)} variables con baja varianza detectadas:**
            {', '.join(low_variance['Variable'].tolist())}
            
            Estas variables aportan poca información y podrían ser excluidas del clustering.
            """)
        else:
            st.success(f"✅ Todas las variables tienen varianza suficiente (CV > {variance_threshold}%)")
        
        # Visualización
        st.markdown("#### 📈 Visualización de Coeficiente de Variación")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if cv > variance_threshold else 'red' 
                 for cv in variance_df['CV (%)']]
        ax.barh(variance_df['Variable'], variance_df['CV (%)'], color=colors, alpha=0.7)
        ax.axvline(x=variance_threshold, color='orange', linestyle='--', 
                  linewidth=2, label=f'Umbral: {variance_threshold}%')
        ax.set_xlabel('Coeficiente de Variación (%)')
        ax.set_title('Coeficiente de Variación por Variable')
        ax.legend()
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Recomendaciones
        st.markdown("#### 💡 Recomendaciones")
        
        if len(high_variance) > 0 and len(low_variance) > 0:
            st.info(f"""
            **Sugerencia de selección:**
            - Mantener: {', '.join(high_variance['Variable'].head(10).tolist())}
            - Considerar excluir: {', '.join(low_variance['Variable'].tolist())}
            """)
        elif len(low_variance) == 0:
            st.success("✅ Todas las variables son adecuadas para clustering basándose en su varianza")
