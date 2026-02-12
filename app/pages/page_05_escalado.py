"""
Página 5: Escalado de Datos
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import settings
from core import scale_data


def render():
    """Renderizar página de escalado de datos"""
    st.markdown('<h2 class="section-header">📏 Escalado de Datos</h2>', unsafe_allow_html=True)
    
    # Verificar que hay datos limpios
    data = st.session_state.data_clean if st.session_state.data_clean is not None else st.session_state.data
    
    if data is None:
        st.warning(settings.MESSAGES['no_data'])
        return
    
    # Obtener variables seleccionadas o todas las numéricas
    if 'selected_features' in st.session_state and len(st.session_state.selected_features) > 0:
        columns_to_scale = st.session_state.selected_features
    else:
        columns_to_scale = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns_to_scale) == 0:
        st.error(settings.MESSAGES['no_numeric'])
        return
    
    st.info("""
    💡 **¿Por qué escalar?**
    - Los algoritmos de clustering son sensibles a la escala de las variables
    - Variables con rangos mayores dominan el cálculo de distancias
    - El escalado normaliza todas las variables a un rango comparable
    """)
    
    # Selección de método de escalado
    st.markdown("### ⚙️ Configuración de Escalado")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scaler_type = st.selectbox(
            "Método de Escalado",
            list(settings.AVAILABLE_SCALERS.keys()),
            format_func=lambda x: settings.AVAILABLE_SCALERS[x],
            index=0,  # StandardScaler por defecto
            help="""
            **StandardScaler:** Media=0, Std=1 (recomendado para distribuciones normales)
            **MinMaxScaler:** Escala a rango [0,1] (recomendado si hay outliers controlados)
            **RobustScaler:** Usa mediana y cuartiles (robusto a outliers)
            """
        )
        
        st.markdown(f"""
        **Método seleccionado:** {settings.AVAILABLE_SCALERS[scaler_type]}
        
        **Características:**
        """)
        
        if scaler_type == 'standard':
            st.markdown("""
            - ✅ Centra datos en 0
            - ✅ Desviación estándar = 1
            - ✅ Mejor para distribuciones normales
            - ⚠️ Sensible a outliers
            """)
        elif scaler_type == 'minmax':
            st.markdown("""
            - ✅ Escala a rango [0, 1]
            - ✅ Mantiene forma de distribución
            - ✅ Útil para redes neuronales
            - ⚠️ Muy sensible a outliers
            """)
        elif scaler_type == 'robust':
            st.markdown("""
            - ✅ Usa mediana en lugar de media
            - ✅ Resistente a outliers
            - ✅ Recomendado si hay valores atípicos
            - ⚠️ Puede no estar en rango [0,1]
            """)
    
    with col2:
        st.markdown("#### 📊 Información")
        st.metric("Variables a Escalar", len(columns_to_scale))
        st.metric("Filas", len(data))
        
        if st.button("🔄 Cambiar Variables", use_container_width=True):
            st.info("Ve a la sección **Feature Engineering** para cambiar la selección de variables")
    
    # Mostrar variables que se escalarán
    with st.expander("📋 Ver variables seleccionadas"):
        st.write(columns_to_scale)
    
    # Vista previa de datos antes de escalar
    st.markdown("### 👁️ Vista Previa - Datos Originales")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### Primeras 5 filas")
        st.dataframe(data[columns_to_scale].head(), use_container_width=True)
    
    with col_b:
        st.markdown("#### Estadísticas")
        stats_original = data[columns_to_scale].describe().loc[['mean', 'std', 'min', 'max']]
        st.dataframe(stats_original, use_container_width=True)
    
    # Botón de escalado
    st.markdown("### 🚀 Ejecutar Escalado")
    
    if st.button("📏 Escalar Datos", type="primary", use_container_width=True):
        with st.spinner(f"Aplicando {settings.AVAILABLE_SCALERS[scaler_type]}..."):
            scaled_df, scaler = scale_data(data, scaler_type, columns_to_scale)
            
            if scaler is None:
                st.error("❌ Error al escalar los datos. Verifica el método seleccionado.")
            else:
                # Guardar en session state
                st.session_state.data_scaled = scaled_df
                st.session_state.scaler = scaler
                st.session_state.scaler_type = scaler_type
                st.session_state.scaled_columns = columns_to_scale
                
                st.success(settings.MESSAGES['data_scaled'])
                
                # Comparación antes/después
                st.markdown("### 📊 Comparación Antes vs Después")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 Datos Escalados")
                    st.dataframe(scaled_df.head(), use_container_width=True)
                
                with col2:
                    st.markdown("#### 📈 Estadísticas Escaladas")
                    stats_scaled = scaled_df.describe().loc[['mean', 'std', 'min', 'max']]
                    st.dataframe(stats_scaled, use_container_width=True)
                
                # Visualización comparativa
                st.markdown("### 📉 Visualización Comparativa")
                
                # Seleccionar variable para comparar
                compare_var = st.selectbox(
                    "Selecciona variable para comparar",
                    columns_to_scale,
                    key="compare_scale_var"
                )
                
                if compare_var:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Gráfico original
                    axes[0].hist(data[compare_var], bins=30, edgecolor='black', alpha=0.7, color='blue')
                    axes[0].set_title(f'Original: {compare_var}')
                    axes[0].set_xlabel('Valor')
                    axes[0].set_ylabel('Frecuencia')
                    axes[0].grid(alpha=0.3)
                    axes[0].axvline(data[compare_var].mean(), color='red', 
                                   linestyle='--', label=f'Media: {data[compare_var].mean():.2f}')
                    axes[0].legend()
                    
                    # Gráfico escalado
                    axes[1].hist(scaled_df[compare_var], bins=30, edgecolor='black', alpha=0.7, color='green')
                    axes[1].set_title(f'Escalado: {compare_var}')
                    axes[1].set_xlabel('Valor')
                    axes[1].set_ylabel('Frecuencia')
                    axes[1].grid(alpha=0.3)
                    axes[1].axvline(scaled_df[compare_var].mean(), color='red',
                                   linestyle='--', label=f'Media: {scaled_df[compare_var].mean():.2f}')
                    axes[1].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Comparación de rangos
                st.markdown("### 📊 Comparación de Rangos")
                
                comparison_data = []
                for col in columns_to_scale:
                    comparison_data.append({
                        'Variable': col,
                        'Min Original': f"{data[col].min():.2f}",
                        'Max Original': f"{data[col].max():.2f}",
                        'Rango Original': f"{data[col].max() - data[col].min():.2f}",
                        'Min Escalado': f"{scaled_df[col].min():.2f}",
                        'Max Escalado': f"{scaled_df[col].max():.2f}",
                        'Rango Escalado': f"{scaled_df[col].max() - scaled_df[col].min():.2f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                st.markdown("""
                <div class="success-box">
                ✅ <b>Datos escalados correctamente</b><br>
                Ahora puedes continuar con la sección de <b>Clustering</b>
                </div>
                """, unsafe_allow_html=True)
    
    # Mostrar datos escalados si ya existen
    elif st.session_state.data_scaled is not None:
        st.info("ℹ️ Ya tienes datos escalados en memoria. Puedes continuar al Clustering o volver a escalar con otro método.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Método Actual", 
                   settings.AVAILABLE_SCALERS.get(st.session_state.get('scaler_type', 'standard'), 'Standard'))
        col2.metric("Variables Escaladas", len(st.session_state.get('scaled_columns', [])))
        col3.metric("Filas", len(st.session_state.data_scaled))
        
        st.markdown("#### 👁️ Vista Previa de Datos Escalados")
        st.dataframe(st.session_state.data_scaled.head(10), use_container_width=True)
