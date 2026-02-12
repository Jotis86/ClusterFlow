"""
ClusterFlow - Aplicación de Clustering Automático
Versión Modular Completa
"""
import streamlit as st
from config import settings
from styles import apply_custom_styles
from pages import (
    page_01_carga_datos,
    page_02_limpieza,
    page_03_exploratorio,
    page_04_feature_engineering,
    page_05_escalado,
    page_06_clustering,
    page_07_resultados
)

# Configuración de la página
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon=settings.PAGE_ICON,
    layout=settings.LAYOUT,
    initial_sidebar_state=settings.INITIAL_SIDEBAR_STATE
)

# Aplicar estilos
apply_custom_styles()

# Inicializar session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_clean' not in st.session_state:
    st.session_state.data_clean = None
if 'data_scaled' not in st.session_state:
    st.session_state.data_scaled = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []

# Banner Hero Visual
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2.5rem; 
            border-radius: 15px; 
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
    <div style='text-align: center;'>
        <h1 style='color: white; 
                   font-size: 3.5rem; 
                   margin: 0; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                   font-weight: 800;'>
            📊 ClusterFlow
        </h1>
        <p style='color: #E0E7FF; 
                  font-size: 1.3rem; 
                  margin: 0.5rem 0 0 0;
                  font-weight: 500;'>
            Plataforma Inteligente de Análisis de Clustering
        </p>
        <div style='margin-top: 1.5rem;'>
            <span style='background: rgba(255,255,255,0.2); 
                        padding: 0.5rem 1rem; 
                        border-radius: 20px; 
                        color: white; 
                        font-size: 0.9rem;
                        margin: 0 0.5rem;
                        display: inline-block;'>
                ✨ K-Means
            </span>
            <span style='background: rgba(255,255,255,0.2); 
                        padding: 0.5rem 1rem; 
                        border-radius: 20px; 
                        color: white; 
                        font-size: 0.9rem;
                        margin: 0 0.5rem;
                        display: inline-block;'>
                🎯 DBSCAN
            </span>
            <span style='background: rgba(255,255,255,0.2); 
                        padding: 0.5rem 1rem; 
                        border-radius: 20px; 
                        color: white; 
                        font-size: 0.9rem;
                        margin: 0 0.5rem;
                        display: inline-block;'>
                🔥 Agglomerative
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Barra lateral de navegación
with st.sidebar:
    # Banner del sidebar
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; 
                border-radius: 10px; 
                margin-bottom: 1.5rem;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                text-align: center;'>
        <h2 style='color: white; 
                   font-size: 1.8rem; 
                   margin: 0; 
                   text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
                   font-weight: 700;'>
            📊 ClusterFlow
        </h2>
        <p style='color: #E0E7FF; 
                  font-size: 0.85rem; 
                  margin: 0.3rem 0 0 0;'>
            Panel de Control
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📋 Navegación")
    
    # Diccionario de páginas
    PAGES = {
        "📁 Carga de Datos": page_01_carga_datos.render,
        "🧹 Limpieza de Datos": page_02_limpieza.render,
        "📊 Análisis Exploratorio": page_03_exploratorio.render,
        "🔧 Feature Engineering": page_04_feature_engineering.render,
        "📏 Escalado de Datos": page_05_escalado.render,
        "🎯 Clustering": page_06_clustering.render,
        "📈 Resultados": page_07_resultados.render
    }
    
    page = st.radio(
        "Selecciona una sección:",
        list(PAGES.keys())
    )
    
    st.markdown("---")
    
    # Estado de los datos con diseño mejorado
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 0.8rem;
                border-radius: 8px;
                margin-bottom: 1rem;'>
        <h3 style='color: white; 
                   font-size: 1.1rem; 
                   margin: 0;
                   text-align: center;
                   font-weight: 600;'>
            📊 Pipeline de Datos
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        st.success("✅ Datos cargados")
        st.caption(f"📦 Filas: {st.session_state.data.shape[0]:,}")
        st.caption(f"📋 Columnas: {st.session_state.data.shape[1]}")
    else:
        st.info("⏳ Sin datos")
    
    if st.session_state.data_clean is not None:
        st.success("✅ Datos limpiados")
    else:
        st.info("⏳ Sin limpiar")
    
    if st.session_state.data_scaled is not None:
        st.success("✅ Datos escalados")
        st.caption(f"⚖️ {st.session_state.get('scaler_type', 'N/A')}")
    else:
        st.info("⏳ Sin escalar")
    
    if st.session_state.cluster_results is not None:
        st.success("✅ Clustering completo")
        st.caption(f"🎯 {st.session_state.cluster_results['n_clusters']} clusters")
        st.caption(f"🔧 {st.session_state.get('method_used', 'N/A').upper()}")
    else:
        st.info("⏳ Sin clustering")
    
    st.markdown("---")
    
    # Footer del sidebar con stats
    st.markdown("""
    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 0.8rem;
                border-radius: 8px;
                margin-top: 1rem;
                text-align: center;'>
        <p style='margin: 0; font-size: 0.75rem; color: #2C3E50; font-weight: 600;'>
            ✨ Análisis Inteligente
        </p>
        <p style='margin: 0.2rem 0 0 0; font-size: 0.65rem; color: #555;'>
            Powered by ML
        </p>
    </div>
    """, unsafe_allow_html=True)

# Renderizar página seleccionada
PAGES[page]()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>ClusterFlow | Desarrollado por Juan</small>
</div>
""", unsafe_allow_html=True)
