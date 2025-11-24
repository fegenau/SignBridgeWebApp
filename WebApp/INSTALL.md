# ============================================================================
# SIGNBRIDGE - INSTALACIÓN EN NUEVO ORDENADOR
# ============================================================================

## Requisitos del Sistema

- **Python**: 3.11.x (recomendado) o 3.10.x
- **Sistema Operativo**: Windows 10/11, macOS, o Linux
- **RAM**: Mínimo 4GB, recomendado 8GB
- **Espacio en Disco**: ~2GB para dependencias
- **Cámara Web**: Necesaria para detección en tiempo real

## Pasos de Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/SignBridgeWebApp.git
cd SignBridgeWebApp/WebApp
```

### 2. Crear Entorno Virtual

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 4. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Nota**: La instalación puede tardar 5-10 minutos dependiendo de tu conexión.

### 5. Verificar Instalación

```bash
python -c "import streamlit; import tensorflow; import mediapipe; print('✅ Todo instalado correctamente')"
```

### 6. Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

## Solución de Problemas Comunes

### Error: "No module named 'tensorflow'"

```bash
pip install tensorflow==2.17.0
```

### Error: "Could not find a version that satisfies mediapipe"

Asegúrate de usar Python 3.11 o 3.10:
```bash
python --version
```

Si tienes Python 3.12+, instala Python 3.11 desde [python.org](https://www.python.org/downloads/)

### Error: "Microsoft Visual C++ 14.0 is required" (Windows)

Instala [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Error con opencv-python

Si `opencv-python-headless` falla, intenta:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.8.1.78
```

### Error con pyttsx3 (TTS)

**Windows**: Debería funcionar sin configuración adicional

**Linux**: Instala espeak
```bash
sudo apt-get install espeak libespeak1 libespeak-dev
```

**macOS**: Usa el motor nativo (sin configuración adicional)

## Verificación de Componentes

### Verificar TensorFlow + GPU (Opcional)

```python
import tensorflow as tf
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
```

### Verificar MediaPipe

```python
import mediapipe as mp
print("MediaPipe versión:", mp.__version__)
```

### Verificar Cámara

```python
import cv2
cap = cv2.VideoCapture(0)
print("Cámara disponible:", cap.isOpened())
cap.release()
```

## Configuración Adicional

### Desactivar TTS (si da problemas)

Edita `config.py`:
```python
ENABLE_TTS = False
```

### Ajustar Rendimiento

En `config.py`:
```python
MAX_VIDEO_WIDTH = 480  # Reducir para mejor rendimiento
TARGET_FPS = 24        # Reducir si hay lag
```

## Dependencias del Sistema (para deployment)

Si vas a desplegar en servidor Linux, necesitas:

```bash
sudo apt-get update
sudo apt-get install -y \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    espeak \
    libespeak1 \
    libespeak-dev \
    ffmpeg
```

## Actualizar Dependencias

Para actualizar a las últimas versiones compatibles:

```bash
pip install --upgrade -r requirements.txt
```

## Desinstalar

```bash
deactivate  # Salir del entorno virtual
rm -rf venv  # Eliminar entorno virtual
```

## Soporte

Si encuentras problemas:
1. Verifica que estás usando Python 3.11 o 3.10
2. Asegúrate de estar en el entorno virtual activado
3. Revisa los logs de error completos
4. Consulta el README.md principal

---

**¡Listo para usar SignBridge! 🤟**
