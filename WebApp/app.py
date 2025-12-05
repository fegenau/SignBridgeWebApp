"""
Página principal de SignBridge - Home con navegación
"""

import streamlit as st
from pathlib import Path
import config

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

# Ruta al logo
logo_path = Path(__file__).parent / "assets" / "Imagenes" / "Logo" / "IconSignBridge.png"

st.set_page_config(
    page_title="SignBridge - Inicio",
    page_icon=str(logo_path) if logo_path.exists() else "🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# ESTILOS PERSONALIZADOS
# ============================================================================

st.markdown("""
<style>
    /* Ocultar sidebar en home */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Estilos para botones grandes */
    .big-button {
        display: inline-block;
        padding: 2rem 3rem;
        margin: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        border-radius: 15px;
        text-decoration: none;
        font-size: 1.5rem;
        font-weight: bold;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        cursor: pointer;
        border: none;
        width: 100%;
    }
    
    .big-button:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .stats-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stats-number {
        font-size: 3rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stats-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .logo-container {
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONTENIDO PRINCIPAL
# ============================================================================

# Hero Section con mejor diseño
st.markdown("""
<div class="hero-section">
    <h1 style="font-size: 4rem; margin-bottom: 1rem; font-weight: 700; letter-spacing: 2px;">SignBridge</h1>
    <h2 style="font-size: 1.8rem; font-weight: 300; margin-bottom: 0.5rem;">Detección de Lenguaje de Señas en Tiempo Real</h2>
    <p style="font-size: 1.2rem; opacity: 0.9;">Comunicación sin barreras usando inteligencia artificial</p>
</div>
""", unsafe_allow_html=True)

# Estadísticas
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stats-box">
        <div class="stats-number">67</div>
        <div class="stats-label">Señas Detectables</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-box">
        <div class="stats-number">95%</div>
        <div class="stats-label">Precisión del Modelo</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stats-box">
        <div class="stats-number">24</div>
        <div class="stats-label">FPS en Tiempo Real</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Botones de navegación principales
st.markdown("## 🚀 ¿Qué deseas hacer?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="text-align: center; color: #667eea;">📹 Detección en Vivo</h3>
        <p style="text-align: center;">Usa tu cámara para detectar señas en tiempo real</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎥 Ir a Detectar", key="detect", width='stretch', type="primary"):
        st.switch_page("pages/1_📹_Detección.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="text-align: center; color: #667eea;">📚 Diccionario de Señas</h3>
        <p style="text-align: center;">Explora todas las señas disponibles con imágenes</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📖 Ver Diccionario", key="dictionary", width='stretch', type="primary"):
        st.switch_page("pages/2_📚_Diccionario.py")

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3 style="text-align: center; color: #667eea;">⚙️ Configuración</h3>
        <p style="text-align: center;">Ajusta parámetros y preferencias del sistema</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔧 Configuración", key="settings", width='stretch', type="primary"):
        st.switch_page("pages/3_⚙️_Configuración.py")

# Características principales
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("## ✨ Características Principales")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Detección Precisa
    - **67 señas diferentes**: Números, letras y frases comunes
    - **Señas estáticas y dinámicas**: Adaptado a diferentes tipos
    - **Alta precisión**: Modelo entrenado con miles de ejemplos
    
    ### 🚀 Rendimiento Óptimo
    - **Tiempo real**: Procesamiento a 24 FPS
    - **Baja latencia**: Respuesta instantánea
    - **Optimizado**: Uso eficiente de recursos
    """)

with col2:
    st.markdown("""
    ### 🛡️ Robustez
    - **Tolerancia a oclusiones**: Maneja pérdidas momentáneas
    - **Suavizado adaptativo**: Predicciones estables
    - **Control de calidad**: Verifica claridad de las manos
    
    ### 💻 Fácil de Usar
    - **Interfaz intuitiva**: Diseño simple y claro
    - **Sin instalación compleja**: Listo para usar
    - **Multiplataforma**: Funciona en cualquier navegador
    """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p style="font-size: 0.9rem;">
        Desarrollado con ❤️ usando <strong>Streamlit</strong>, <strong>TensorFlow</strong> y <strong>MediaPipe</strong>
    </p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">
        SignBridge © 2025 - Comunicación sin barreras
    </p>
</div>
""", unsafe_allow_html=True)
