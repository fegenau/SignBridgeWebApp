"""
Página de Detección en Vivo - Cámara y reconocimiento de señas
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import pyttsx3
import threading
import time

# Importar configuración y utilidades
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    extract_keypoints,
    has_hands_detected,
    get_hands_quality_score,
    keypoint_dim,
    load_model,
    load_classes,
    get_sign_type
)

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

# Ruta al logo
logo_path = Path(__file__).parent.parent / "assets" / "Imagenes" / "Logo" / "IconSignBridge.png"

st.set_page_config(
    page_title="SignBridge - Detección",
    page_icon=str(logo_path) if logo_path.exists() else "📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INICIALIZACIÓN DE MEDIAPIPE
# ============================================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================================================================
# CARGA DEL MODELO Y CLASES (CON CACHÉ)
# ============================================================================

@st.cache_resource
def initialize_model():
    """Carga el modelo y las clases (con caché de Streamlit)"""
    model = load_model()
    classes = load_classes()
    return model, classes

# ============================================================================
# PROCESADOR DE VIDEO
# ============================================================================

class SignLanguageProcessor(VideoProcessorBase):
    """
    Procesador de video que detecta señas en tiempo real.
    """
    
    def __init__(self):
        # Cargar modelo y clases
        self.model, self.classes = initialize_model()
        
        # Configuración de MediaPipe
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            refine_face_landmarks=False
        )
        
        # Buffer de secuencias
        self.feat_dim = keypoint_dim(config.USE_FACE, config.USE_POSE, config.USE_HANDS)
        self.buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.pred_history = deque(maxlen=config.SMOOTH_WINDOW)
        
        # Sistema de tolerancia
        self.hands_missing_count = 0
        self.last_valid_keypoints = None
        
        # Sistema de predicción
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_sign_type = None
        self.frames_since_last_prediction = 0
        self.prediction_cooldown = 0
        self.stable_prediction_count = 0
        
        # Estado
        self.hands_detected = False
        self.hands_quality = 0.0
        self.status_message = "Iniciando..."
        
        # 🆕 Sistema de acumulación de señas para TTS
        self.sign_buffer = deque(maxlen=config.SIGN_BUFFER_SIZE)
        self.last_added_sign = None
        
        # 🆕 Historial de sesión (todas las señas detectadas)
        self.session_history = []
        
        # 🆕 Control de TTS (no inicializar motor aquí para evitar conflictos)
        self.tts_enabled = config.ENABLE_TTS
        self.is_speaking = False
    
    def recv(self, frame):
        """
        Procesa cada frame del video.
        """
        # Convertir frame de av a numpy
        img = frame.to_ndarray(format="bgr24")
        
        # Procesar con MediaPipe
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Extraer keypoints
        keypoints = extract_keypoints(results, config.USE_FACE, config.USE_POSE, config.USE_HANDS)
        if keypoints.shape[0] != self.feat_dim:
            keypoints = np.zeros((self.feat_dim,), dtype=np.float32)
        
        # Detectar manos y calidad
        self.hands_detected = has_hands_detected(results)
        self.hands_quality = get_hands_quality_score(results)
        
        # Actualizar contador de frames
        self.frames_since_last_prediction += 1
        
        # Sistema de tolerancia
        if self.hands_detected and self.hands_quality >= config.MIN_HANDS_QUALITY:
            self.hands_missing_count = 0
            self.last_valid_keypoints = keypoints.copy()
            self.buffer.append(keypoints)
            self.status_message = f"Manos detectadas (calidad: {self.hands_quality*100:.0f}%)"
        else:
            self.hands_missing_count += 1
            
            # Resetear si se pierden las manos por mucho tiempo
            if self.hands_missing_count >= config.RESET_PREDICTION_AFTER_NO_HANDS:
                self.current_prediction = None
                self.current_confidence = 0.0
                self.current_sign_type = None
                self.buffer.clear()
                self.frames_since_last_prediction = 0
                self.status_message = "Sin manos detectadas - Predicción reseteada"
            elif self.hands_missing_count <= config.MAX_MISSING_FRAMES and self.last_valid_keypoints is not None:
                # Dentro de tolerancia
                self.buffer.append(self.last_valid_keypoints)
                self.status_message = f"Tolerancia: {self.hands_missing_count}/{config.MAX_MISSING_FRAMES}"
            else:
                # Fuera de tolerancia
                if len(self.buffer) > 0:
                    self.buffer.popleft()
                self.status_message = "Calidad de mano insuficiente"
        
        # Hacer predicción si hay suficientes frames
        if len(self.buffer) >= config.SEQUENCE_LENGTH:
            if self.current_prediction is None or self.frames_since_last_prediction >= config.MIN_FRAMES_FOR_NEW_PREDICTION:
                if self.prediction_cooldown > 0:
                    self.prediction_cooldown -= 1
                else:
                    # Hacer predicción
                    inp = np.expand_dims(np.stack(self.buffer, axis=0), axis=0)
                    probs = self.model.predict(inp, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    max_conf = float(np.max(probs))
                    
                    if max_conf >= config.MIN_CONFIDENCE:
                        self.pred_history.append(pred_idx)
                        
                        if len(self.pred_history) == config.SMOOTH_WINDOW:
                            counts = np.bincount(list(self.pred_history), minlength=len(self.classes))
                            final_idx = int(np.argmax(counts))
                            final_conf = float(probs[final_idx])
                            
                            if final_conf >= config.CONFIDENCE_THRESHOLD:
                                recent_predictions = list(self.pred_history)[-config.REQUIRED_STABLE_FRAMES:]
                                if len(set(recent_predictions)) == 1:
                                    self.stable_prediction_count += 1
                                    if self.stable_prediction_count >= config.REQUIRED_STABLE_FRAMES:
                                        new_prediction = self.classes[final_idx]
                                        if self.current_prediction != new_prediction:
                                            self.current_prediction = new_prediction
                                            self.current_confidence = final_conf
                                            self.current_sign_type = get_sign_type(new_prediction)
                                            self.frames_since_last_prediction = 0
                                            self.prediction_cooldown = config.COOLDOWN_FRAMES
                                            self.stable_prediction_count = 0
                                            self.status_message = f"Nueva predicción: {new_prediction}"
                                            
                                            # 🆕 Agregar al buffer de señas (evitar duplicados consecutivos)
                                            if config.ENABLE_TTS and self.last_added_sign != new_prediction:
                                                self.sign_buffer.append({
                                                    'label': new_prediction,
                                                    'confidence': final_conf,
                                                    'timestamp': time.time()
                                                })
                                                self.last_added_sign = new_prediction
                                                
                                                # Agregar al historial de sesión
                                                self.session_history.append({
                                                    'label': new_prediction,
                                                    'confidence': final_conf,
                                                    'timestamp': time.time()
                                                })
                                                
                                                print(f"[TTS DEBUG] Seña agregada al buffer: {new_prediction}, Total: {len(self.sign_buffer)}")
                                else:
                                    self.stable_prediction_count = 0
        else:
            self.status_message = f"Reuniendo secuencia... {len(self.buffer)}/{config.SEQUENCE_LENGTH}"
        
        # Dibujar landmarks
        if config.USE_POSE and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if config.USE_HANDS:
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Dibujar información en el frame
        self._draw_info_overlay(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_info_overlay(self, img):
        """Dibuja información sobre el frame"""
        h, w = img.shape[:2]
        
        # Panel superior oscuro
        cv2.rectangle(img, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Predicción actual
        if self.current_prediction:
            cv2.putText(img, f"SEÑA: {self.current_prediction}", 
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(img, f"Conf: {self.current_confidence*100:.1f}%", 
                       (w-200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Tipo: {self.current_sign_type}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(img, "Detectando...", 
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Indicador de manos
        color = (0, 255, 0) if self.hands_detected else (0, 0, 255)
        cv2.circle(img, (w-30, 30), 12, color, -1)
        cv2.circle(img, (w-30, 30), 15, (255, 255, 255), 2)
    
    # 🆕 Métodos para gestión del buffer de señas y TTS
    def get_accumulated_signs(self):
        """Retorna las señas acumuladas como lista"""
        return list(self.sign_buffer)
    
    def get_session_history(self):
        """Retorna el historial completo de la sesión"""
        return self.session_history
    
    def clear_session_history(self):
        """Limpia el historial de sesión"""
        self.session_history.clear()
    
    def clear_sign_buffer(self):
        """Limpia el buffer de señas"""
        self.sign_buffer.clear()
        self.last_added_sign = None
    
    def remove_last_sign(self):
        """Elimina la última seña del buffer"""
        if len(self.sign_buffer) > 0:
            self.sign_buffer.pop()
            if len(self.sign_buffer) > 0:
                self.last_added_sign = self.sign_buffer[-1]['label']
            else:
                self.last_added_sign = None
    
    def speak_accumulated_signs(self):
        """Reproduce todas las señas acumuladas como una frase"""
        if not self.tts_enabled:
            return
        
        if len(self.sign_buffer) == 0:
            return
        
        if self.is_speaking:
            print("[TTS] Ya hay una reproducción en curso")
            return
        
        def speak():
            self.is_speaking = True
            try:
                # Crear nueva instancia de TTS para cada reproducción
                # Esto evita el error "run loop already started"
                engine = pyttsx3.init()
                engine.setProperty('rate', config.TTS_RATE)
                engine.setProperty('volume', config.TTS_VOLUME)
                
                # Construir frase completa
                phrase = ' '.join([sign['label'].replace('_', ' ') 
                                  for sign in self.sign_buffer])
                
                print(f"[TTS] Reproduciendo: {phrase}")
                engine.say(phrase)
                engine.runAndWait()
                engine.stop()
                del engine  # Liberar recursos
            except Exception as e:
                print(f"Error TTS: {e}")
            finally:
                self.is_speaking = False
        
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

def main():
    # Botón de volver al inicio
    if st.button("⬅️ Volver al Inicio"):
        st.switch_page("app.py")
    
    # Título
    st.title("📹 Detección en Vivo")
    st.markdown("Usa tu cámara para detectar señas en tiempo real")
    
    # Sidebar con configuración
    with st.sidebar:
        st.header("⚙️ Información")
        
        st.markdown("### 📊 Modelo")
        try:
            model, classes = initialize_model()
            st.success(f"✅ Modelo cargado")
            st.info(f"📝 {len(classes)} clases detectables")
            
            with st.expander("Ver clases"):
                # Organizar por categorías
                numbers = [c for c in classes if c.isdigit()]
                letters = [c for c in classes if len(c) == 1 and c.isalpha()]
                phrases = [c for c in classes if c not in numbers and c not in letters]
                
                st.markdown("**Números:**")
                st.write(", ".join(numbers))
                st.markdown("**Letras:**")
                st.write(", ".join(letters))
                st.markdown("**Frases:**")
                st.write(", ".join(phrases))
        except Exception as e:
            st.error(f"❌ Error al cargar modelo: {e}")
        
        st.markdown("---")
        st.markdown("### 📹 Instrucciones")
        st.info("""
        1. Haz clic en **START**
        2. Permite acceso a la cámara
        3. Realiza una seña
        4. Espera la predicción
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Consejos")
        st.warning("""
        - Buena iluminación
        - Manos visibles
        - Mantén la seña estable
        - Para señas dinámicas, completa el movimiento
        """)
    
    # Área principal - Video
    st.header("🎥 Cámara")
    
    # Configuración RTC para WebRTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Streamer de video
    webrtc_ctx = webrtc_streamer(
        key="sign-language-detection",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # 🆕 Panel de control TTS
    if config.ENABLE_TTS:
        st.markdown("---")
        st.header("💬 Señas Acumuladas")
        
        # Mostrar buffer de señas
        if webrtc_ctx.video_processor:
            processor = webrtc_ctx.video_processor
            
            # Usar session_state para controlar actualizaciones
            if 'last_buffer_size' not in st.session_state:
                st.session_state.last_buffer_size = 0
            if 'last_history_size' not in st.session_state:
                st.session_state.last_history_size = 0
            
            accumulated = processor.get_accumulated_signs()
            history = processor.get_session_history()
            
            # Detectar cambios y forzar actualización
            current_buffer_size = len(accumulated)
            current_history_size = len(history)
            
            if (current_buffer_size != st.session_state.last_buffer_size or 
                current_history_size != st.session_state.last_history_size):
                st.session_state.last_buffer_size = current_buffer_size
                st.session_state.last_history_size = current_history_size
            
            # Panel de señas acumuladas (buffer actual)
            if len(accumulated) > 0:
                st.markdown("### 📝 Buffer Actual")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Crear visualización de señas
                    signs_text = ' → '.join([s['label'] for s in accumulated])
                    st.markdown(f"**{signs_text}**")
                    
                    # Mostrar confianza promedio
                    avg_conf = sum([s['confidence'] for s in accumulated]) / len(accumulated)
                    st.caption(f"Confianza promedio: {avg_conf*100:.1f}%")
                
                with col2:
                    st.metric("Señas", f"{len(accumulated)}/{config.SIGN_BUFFER_SIZE}")
                
                # Botones de control
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔊 Reproducir", type="primary", use_container_width=True, key="btn_play"):
                        processor.speak_accumulated_signs()
                        st.success("Reproduciendo...")
                
                with col2:
                    if st.button("⬅️ Borrar Última", use_container_width=True, key="btn_remove"):
                        processor.remove_last_sign()
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Limpiar Buffer", use_container_width=True, key="btn_clear"):
                        processor.clear_sign_buffer()
                        st.rerun()
            else:
                st.info("👆 Realiza señas para comenzar a acumular")
            
            # Historial de sesión
            if len(history) > 0:
                st.markdown("---")
                st.markdown("### 📜 Historial de Sesión")
                st.caption(f"Total de señas detectadas: {len(history)}")
                
                # Mostrar últimas 10 señas del historial
                with st.expander("Ver historial completo", expanded=False):
                    # Mostrar en orden inverso (más reciente primero)
                    for i, sign in enumerate(reversed(history[-20:])):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"{len(history)-i}. {sign['label']}")
                        with col2:
                            st.caption(f"{sign['confidence']*100:.1f}%")
                
                # Botón para limpiar historial
                if st.button("🗑️ Limpiar Historial", key="btn_clear_history"):
                    processor.clear_session_history()
                    st.rerun()
            
            # Auto-refresh cada 1 segundo si hay actividad
            if webrtc_ctx.state.playing:
                import time as time_module
                time_module.sleep(1)
                st.rerun()
        else:
            st.info("▶️ Inicia la cámara para comenzar")
    
    # Información adicional
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Señas Estáticas")
        st.markdown("Números (0-9) y la mayoría de letras requieren mantener la posición")
    
    with col2:
        st.markdown("### 🔄 Señas Dinámicas")
        st.markdown("Frases y algunas letras requieren movimiento")
    
    with col3:
        st.markdown("### 💡 Estado")
        st.markdown("El círculo verde indica manos detectadas")

if __name__ == "__main__":
    main()
