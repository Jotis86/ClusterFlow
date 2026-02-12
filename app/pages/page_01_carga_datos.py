"""
Página 1: Carga de Datos
"""
import streamlit as st
import pandas as pd
from config import settings
from core import load_data


def render():
    """Renderizar página de carga de datos"""
    st.markdown('<h2 class="section-header">📁 Carga de Datos</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Sube tu archivo CSV")
        uploaded_file = st.file_uploader(
            "Arrastra o selecciona un archivo CSV",
            type=['csv'],
            help="El archivo debe estar en formato CSV con separador por comas"
        )
        
        if uploaded_file is not None:
            data, error = load_data(uploaded_file)
            
            if error:
                st.error(f"❌ Error al cargar el archivo: {error}")
            else:
                st.session_state.data = data
                st.markdown(f'<div class="success-box">{settings.MESSAGES["data_loaded"]}</div>', 
                           unsafe_allow_html=True)
                
                # Información básica
                st.markdown("### 📋 Información del Dataset")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Filas", f"{data.shape[0]:,}")
                col_b.metric("Columnas", data.shape[1])
                col_c.metric("Tamaño", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Vista previa
                st.markdown("### 👁️ Vista Previa de Datos")
                st.dataframe(data.head(10), width='stretch')
                
                # Tipos de datos
                st.markdown("### 🔢 Tipos de Datos")
                dtype_df = pd.DataFrame({
                    'Columna': data.columns,
                    'Tipo': data.dtypes.astype(str).values,
                    'Valores Únicos': [data[col].nunique() for col in data.columns],
                    'Valores Nulos': [data[col].isnull().sum() for col in data.columns]
                })
                st.dataframe(dtype_df, width='stretch')
    
    with col2:
        if uploaded_file is not None and st.session_state.data is not None:
            st.markdown("### ℹ️ Información")
            st.info("""
            **Datos cargados correctamente**
            
            Siguiente paso:
            - Ve a **Limpieza de Datos** para preprocesar
            - O salta a **Análisis Exploratorio** si ya están limpios
            """)
        else:
            st.markdown("### 📌 Instrucciones")
            st.info("""
            1. Sube un archivo CSV
            2. Verifica que los datos se cargaron correctamente
            3. Continúa con la limpieza
            
            **Formato recomendado:**
            - Separador: coma (,)
            - Codificación: UTF-8
            - Primera fila: nombres de columnas
            """)
