"""
Página de Configuración - Ajustes y parámetros del sistema
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="SignBridge - Configuración",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# ESTILOS PERSONALIZADOS
# ============================================================================

st.markdown("""
<style>
    .config-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .setting-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONTENIDO PRINCIPAL
# ============================================================================

def main():
    # Botón de volver al inicio
    if st.button("⬅️ Volver al Inicio"):
        st.switch_page("app.py")
    
    # Título
    st.title("⚙️ Configuración del Sistema")
    st.markdown("Ajusta los parámetros de detección y rendimiento")
    
    # Tabs para organizar configuraciones
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Detección", "🎥 MediaPipe", "📊 Rendimiento", "ℹ️ Información"])
    
    # ============================================================================
    # TAB 1: CONFIGURACIÓN DE DETECCIÓN
    # ============================================================================
    with tab1:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 Parámetros de Detección")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### Secuencia y Buffer")
            
            sequence_length = st.slider(
                "Longitud de Secuencia (frames)",
                min_value=12,
                max_value=48,
                value=config.SEQUENCE_LENGTH,
                help="Número de frames consecutivos para hacer una predicción"
            )
            
            smooth_window = st.slider(
                "Ventana de Suavizado",
                min_value=3,
                max_value=15,
                value=config.SMOOTH_WINDOW,
                help="Número de predicciones para suavizar el resultado"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### Tolerancia")
            
            max_missing = st.slider(
                "Frames de Tolerancia",
                min_value=1,
                max_value=10,
                value=config.MAX_MISSING_FRAMES,
                help="Frames sin manos antes de limpiar buffer"
            )
            
            reset_after = st.slider(
                "Reset después de (frames)",
                min_value=15,
                max_value=60,
                value=config.RESET_PREDICTION_AFTER_NO_HANDS,
                help="Frames sin manos para resetear predicción"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### Umbrales de Confianza")
            
            min_confidence = st.slider(
                "Confianza Mínima",
                min_value=0.3,
                max_value=0.9,
                value=config.MIN_CONFIDENCE,
                step=0.05,
                help="Confianza mínima para considerar una predicción"
            )
            
            confidence_threshold = st.slider(
                "Umbral de Confirmación",
                min_value=0.5,
                max_value=0.95,
                value=config.CONFIDENCE_THRESHOLD,
                step=0.05,
                help="Confianza para confirmar predicción estable"
            )
            
            min_hands_quality = st.slider(
                "Calidad Mínima de Manos",
                min_value=0.1,
                max_value=0.7,
                value=config.MIN_HANDS_QUALITY,
                step=0.05,
                help="Calidad mínima de detección de manos"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### Estabilidad")
            
            stable_frames = st.slider(
                "Frames Estables Requeridos",
                min_value=1,
                max_value=5,
                value=config.REQUIRED_STABLE_FRAMES,
                help="Frames consecutivos iguales para confirmar"
            )
            
            cooldown = st.slider(
                "Cooldown (frames)",
                min_value=1,
                max_value=10,
                value=config.COOLDOWN_FRAMES,
                help="Espera entre predicciones"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **💡 Recomendaciones:**
        - Aumenta la confianza mínima si hay muchas falsas detecciones
        - Reduce la ventana de suavizado para respuesta más rápida
        - Aumenta los frames estables para mayor precisión
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 2: CONFIGURACIÓN DE MEDIAPIPE
    # ============================================================================
    with tab2:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### 🎥 Configuración de MediaPipe")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### Confianza de Detección")
            
            detection_conf = st.slider(
                "Confianza de Detección",
                min_value=0.3,
                max_value=0.9,
                value=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                step=0.05,
                help="Confianza mínima para detectar landmarks"
            )
            
            tracking_conf = st.slider(
                "Confianza de Seguimiento",
                min_value=0.3,
                max_value=0.9,
                value=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
                step=0.05,
                help="Confianza mínima para seguir landmarks"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### Complejidad del Modelo")
            
            model_complexity = st.select_slider(
                "Complejidad",
                options=[0, 1, 2],
                value=config.MEDIAPIPE_MODEL_COMPLEXITY,
                help="0: Rápido, 1: Balanceado, 2: Preciso"
            )
            
            st.markdown("**Características activas:**")
            st.checkbox("Usar Face Landmarks", value=config.USE_FACE, disabled=True)
            st.checkbox("Usar Pose Landmarks", value=config.USE_POSE, disabled=True)
            st.checkbox("Usar Hand Landmarks", value=config.USE_HANDS, disabled=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        **⚠️ Nota:** Los cambios en MediaPipe requieren reiniciar la aplicación para tener efecto.
        Actualmente, solo los landmarks de manos están activos.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 3: CONFIGURACIÓN DE RENDIMIENTO
    # ============================================================================
    with tab3:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Optimización de Rendimiento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### Video")
            
            max_width = st.slider(
                "Ancho Máximo de Video (px)",
                min_value=320,
                max_value=1280,
                value=config.MAX_VIDEO_WIDTH,
                step=80,
                help="Ancho máximo para procesamiento (menor = más rápido)"
            )
            
            target_fps = st.slider(
                "FPS Objetivo",
                min_value=15,
                max_value=60,
                value=config.TARGET_FPS,
                step=5,
                help="Frames por segundo objetivo"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### Sistema")
            
            st.markdown("**Información del Sistema:**")
            st.text(f"Modelo: {config.MODEL_PATH.name}")
            st.text(f"Clases: 67 señas")
            st.text(f"Input Shape: (24, 126)")
            
            st.markdown("**Uso de Recursos:**")
            st.info("El modelo se carga una sola vez y se mantiene en caché")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **💡 Consejos de Rendimiento:**
        - Reduce el ancho máximo si experimentas lag
        - Baja el FPS objetivo en computadoras lentas
        - Cierra otras aplicaciones que usen la cámara
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 4: INFORMACIÓN DEL SISTEMA
    # ============================================================================
    with tab4:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Información del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### 📁 Rutas")
            
            st.text_input("Modelo", str(config.MODEL_PATH), disabled=True)
            st.text_input("Label Encoder", str(config.LABEL_ENCODER_PATH), disabled=True)
            st.text_input("Directorio Base", str(config.BASE_DIR), disabled=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="setting-card">', unsafe_allow_html=True)
            st.markdown("#### 🔧 Versiones")
            
            try:
                import tensorflow as tf
                st.text(f"TensorFlow: {tf.__version__}")
            except:
                st.text("TensorFlow: No disponible")
            
            try:
                import mediapipe as mp
                st.text(f"MediaPipe: {mp.__version__}")
            except:
                st.text("MediaPipe: No disponible")
            
            try:
                import streamlit
                st.text(f"Streamlit: {streamlit.__version__}")
            except:
                st.text("Streamlit: No disponible")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📝 Acerca de SignBridge")
        st.markdown("""
        **SignBridge** es una aplicación de detección de lenguaje de señas en tiempo real 
        desarrollada con Streamlit, TensorFlow y MediaPipe.
        
        **Características:**
        - 67 señas detectables (números, letras y frases)
        - Detección en tiempo real con WebRTC
        - Sistema robusto de predicción con suavizado adaptativo
        - Interfaz moderna y fácil de usar
        
        **Desarrollado con ❤️ para facilitar la comunicación**
        """)
    
    # Botón para guardar configuración (placeholder)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💾 Guardar Configuración", use_container_width=True, type="primary"):
            st.success("✅ Configuración guardada (funcionalidad en desarrollo)")
            st.info("ℹ️ Los cambios se aplicarán en la próxima sesión")

if __name__ == "__main__":
    main()
