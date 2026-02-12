"""
Página 2: Limpieza de Datos
"""
import streamlit as st
import pandas as pd
import numpy as np
from config import settings
from core import analyze_data_quality, clean_data


def render():
    """Renderizar página de limpieza de datos"""
    st.markdown('<h2 class="section-header">🧹 Limpieza de Datos</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning(settings.MESSAGES['no_data'])
    else:
        data = st.session_state.data
        
        # Análisis de calidad
        st.markdown("### 🔍 Análisis de Calidad de Datos")
        quality_report = analyze_data_quality(data)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Valores Nulos", quality_report['nulls'].sum())
        col2.metric("Filas Duplicadas", quality_report['duplicates'])
        col3.metric("Columnas Numéricas", len(quality_report['numeric_cols']))
        
        # Detalles de valores nulos
        if quality_report['nulls'].sum() > 0:
            st.markdown("### 📊 Valores Nulos por Columna")
            null_df = pd.DataFrame({
                'Columna': quality_report['nulls'].index,
                'Valores Nulos': quality_report['nulls'].values,
                'Porcentaje': quality_report['null_pct'].values
            })
            null_df = null_df[null_df['Valores Nulos'] > 0]
            st.dataframe(null_df, use_container_width=True)
        
        # Opciones de limpieza
        st.markdown("### ⚙️ Configuración de Limpieza")
        
        col1, col2 = st.columns(2)
        
        with col1:
            remove_duplicates = st.checkbox("Eliminar filas duplicadas", value=True)
            
            fill_nulls_method = st.selectbox(
                "Método para valores nulos",
                list(settings.AVAILABLE_FILL_METHODS.keys()),
                format_func=lambda x: settings.AVAILABLE_FILL_METHODS[x],
                index=2,  # Default: median
                help="⚠️ IMPORTANTE: El clustering requiere que NO haya valores NaN. Se recomienda usar mediana o media."
            )
            
            if fill_nulls_method == 'none' and quality_report['nulls'].sum() > 0:
                st.warning("⚠️ Si dejas valores NaN, el clustering fallará. Se recomienda elegir un método de imputación.")
        
        with col2:
            remove_outliers = st.checkbox("Eliminar outliers", value=True)
            outlier_threshold = st.slider(
                "Umbral de outliers (Z-score)",
                min_value=2.0,
                max_value=4.0,
                value=settings.DEFAULT_OUTLIER_THRESHOLD,
                step=0.5,
                help="Valores con Z-score mayor a este umbral se consideran outliers"
            )
        
        # Botón de limpieza
        if st.button("🧹 Limpiar Datos", type="primary", use_container_width=True):
            with st.spinner("Limpiando datos..."):
                try:
                    data_clean = clean_data(
                        data,
                        remove_duplicates=remove_duplicates,
                        fill_nulls_method=fill_nulls_method,
                        remove_outliers=remove_outliers,
                        outlier_threshold=outlier_threshold
                    )
                    
                    st.session_state.data_clean = data_clean
                    
                    st.success(settings.MESSAGES['data_cleaned'])
                    
                    # VALIDACIÓN: Verificar si quedan NaN en columnas numéricas
                    numeric_cols_clean = data_clean.select_dtypes(include=[np.number]).columns
                    nan_count = data_clean[numeric_cols_clean].isnull().sum().sum()
                    
                    if nan_count > 0:
                        st.error(f"⚠️ ADVERTENCIA: Aún quedan {nan_count} valores NaN en columnas numéricas. El clustering FALLARÁ.")
                        st.info("💡 Solución: Vuelve a limpiar los datos seleccionando un método de imputación diferente (mediana, media, o cero).")
                    else:
                        st.success("✅ No hay valores NaN en columnas numéricas. Los datos están listos para clustering.")
                    
                    # Comparación antes/después
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Filas (Original)", f"{data.shape[0]:,}")
                    col2.metric("Filas (Limpio)", f"{data_clean.shape[0]:,}", 
                               delta=f"{data_clean.shape[0] - data.shape[0]:,}")
                    col3.metric("NaN Restantes", nan_count, delta=int(-quality_report['nulls'].sum() + nan_count))
                    
                    st.markdown("### 👁️ Vista Previa de Datos Limpios")
                    st.dataframe(data_clean.head(10), use_container_width=True)
                    
                except ValueError as e:
                    st.error(f"❌ Error durante la limpieza: {str(e)}")
                    st.info("💡 Sugerencia: Verifica los parámetros de limpieza seleccionados.")
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")
                    st.info("💡 Intenta con diferentes opciones de limpieza.")
