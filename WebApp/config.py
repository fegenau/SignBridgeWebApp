"""
Configuración centralizada para la aplicación web de detección de lenguaje de señas.
"""

from pathlib import Path

# ============================================================================
# RUTAS
# ============================================================================

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent

# Rutas al modelo y configuración
MODEL_DIR = BASE_DIR / "EntrenamientoMovimiento" / "model"
MODEL_PATH = MODEL_DIR / "best_model.keras"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.json"

# ============================================================================
# PARÁMETROS DE INFERENCIA
# ============================================================================

# Longitud de la secuencia (número de frames)
SEQUENCE_LENGTH = 24

# Ventana de suavizado para predicciones
SMOOTH_WINDOW = 8

# Umbral de confianza mínima para mostrar predicción
MIN_CONFIDENCE = 0.65

# Umbral de confianza para confirmar predicción estable
CONFIDENCE_THRESHOLD = 0.75

# Frames requeridos para confirmar predicción estable
REQUIRED_STABLE_FRAMES = 3

# Frames mínimos antes de permitir nueva predicción
MIN_FRAMES_FOR_NEW_PREDICTION = 15

# Frames sin manos antes de resetear predicción
RESET_PREDICTION_AFTER_NO_HANDS = 30

# Frames de tolerancia para pérdida momentánea de manos
MAX_MISSING_FRAMES = 5

# Umbral de calidad mínima de manos
MIN_HANDS_QUALITY = 0.35

# Cooldown entre predicciones (frames)
COOLDOWN_FRAMES = 3

# ============================================================================
# CONFIGURACIÓN DE MEDIAPIPE
# ============================================================================

# Usar landmarks faciales
USE_FACE = False

# Usar landmarks de pose
USE_POSE = False

# Usar landmarks de manos
USE_HANDS = True

# Confianza mínima de detección
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7

# Confianza mínima de seguimiento
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.7

# Complejidad del modelo (0, 1, 2)
MEDIAPIPE_MODEL_COMPLEXITY = 1

# ============================================================================
# CONFIGURACIÓN DE UI
# ============================================================================

# Título de la aplicación
APP_TITLE = "🤟 SignBridge - Detección de Lenguaje de Señas"

# Descripción
APP_DESCRIPTION = """
Aplicación web para detección de lenguaje de señas en tiempo real usando tu cámara web.
Detecta **68 señas** diferentes: números (0-9), letras (A-Z) y frases comunes.
"""

# Instrucciones
APP_INSTRUCTIONS = """
### 📋 Instrucciones:
1. Haz clic en **START** para activar tu cámara
2. Permite el acceso a la cámara cuando se solicite
3. Realiza una seña frente a la cámara
4. Espera a que se complete el buffer (24 frames)
5. La predicción aparecerá con su nivel de confianza

### 💡 Consejos:
- Mantén buena iluminación
- Asegúrate de que tus manos sean visibles
- Mantén la seña estable por unos segundos
- Para señas dinámicas, realiza el movimiento completo
"""

# Colores para la UI
COLOR_SUCCESS = "#00ff00"
COLOR_WARNING = "#ffff00"
COLOR_ERROR = "#ff0000"
COLOR_INFO = "#00ffff"

# ============================================================================
# CONFIGURACIÓN DE VIDEO
# ============================================================================

# Ancho máximo del video para procesamiento
MAX_VIDEO_WIDTH = 640

# FPS objetivo para procesamiento
TARGET_FPS = 30

# ============================================================================
# CONFIGURACIÓN DE TTS (TEXT-TO-SPEECH)
# ============================================================================

# Activar/desactivar TTS
ENABLE_TTS = True

# Tamaño del buffer de señas acumuladas
SIGN_BUFFER_SIZE = 5

# Velocidad de reproducción (palabras por minuto)
TTS_RATE = 150

# Volumen (0.0 a 1.0)
TTS_VOLUME = 0.9

# Índice de voz (0 = predeterminada del sistema)
TTS_VOICE_INDEX = 0

# ============================================================================
# GESTIÓN DE MEMORIA
# ============================================================================

# Tamaño máximo del historial de sesión (evita crecimiento ilimitado)
MAX_SESSION_HISTORY = 100

# Intervalo de limpieza de frames (libera recursos cada N frames)
FRAME_CLEANUP_INTERVAL = 100

# Habilitar salto de frames para reducir carga de procesamiento
# Habilitar salto de frames para reducir carga de procesamiento
ENABLE_FRAME_SKIP = False

# Procesar cada N frames (1 = todos, 2 = saltar uno de cada dos)
FRAME_SKIP_RATE = 1

