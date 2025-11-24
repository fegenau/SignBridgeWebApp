# 🤟 SignBridge - Aplicación Web de Detección de Lenguaje de Señas

Aplicación web multi-página en tiempo real para detección de lenguaje de señas usando **Streamlit + WebRTC** y el modelo Keras entrenado.

## 🌟 Características

- ✅ **Aplicación multi-página** con navegación intuitiva
- ✅ **Detección en tiempo real** usando tu cámara web
- ✅ **67 señas detectables**: Números (0-9), letras (A-Z) y 31 frases comunes
- ✅ **Diccionario completo** con todas las señas organizadas
- ✅ **Configuración avanzada** de parámetros de detección
- ✅ **Sin conversión del modelo**: Usa `best_model.keras` directamente
- ✅ **Interfaz moderna** con diseño profesional
- ✅ **WebRTC nativo** para streaming fluido
- ✅ **Predicciones estables** con sistema de suavizado adaptativo

## 📱 Páginas de la Aplicación

### 🏠 Inicio
- Hero section con estadísticas
- Navegación rápida a todas las secciones
- Información sobre características

### 📹 Detección
- Cámara en tiempo real con WebRTC
- Detección de señas con predicción en vivo
- Indicadores visuales de estado
- Información del modelo en sidebar

### 📚 Diccionario
- Visualización de todas las 67 señas
- Filtros por categoría (Números, Letras, Frases)
- Filtros por tipo (Estáticas, Dinámicas)
- Búsqueda de señas
- Grid organizado con cards

### ⚙️ Configuración
- **Detección**: Ajusta secuencia, suavizado, confianza, tolerancia
- **MediaPipe**: Configura detección y seguimiento
- **Rendimiento**: Optimiza video y FPS
- **Información**: Rutas, versiones, acerca de

## 📋 Requisitos

- Python 3.8 o superior
- Cámara web funcional
- GPU recomendada (opcional, pero mejora el rendimiento)

## 🚀 Instalación

### Método 1: Script Automático (Recomendado)

```powershell
cd C:\Users\matia\Documents\SignBridgeKeras\WebApp
.\start.ps1
```

El script automáticamente:
- Crea un entorno virtual si no existe
- Instala todas las dependencias
- Inicia la aplicación

### Método 2: Manual

#### 1. Navegar al directorio del proyecto

```powershell
cd C:\Users\usuario\Documents\SignBridgeKeras\WebApp
```

#### 2. Crear un entorno virtual (recomendado)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

#### 4. Ejecutar la aplicación

```powershell
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🎮 Uso

### Navegación

1. **Página de Inicio**: Elige tu acción con los botones grandes
   - 🎥 **Ir a Detectar**: Abre la cámara para detección en vivo
   - 📖 **Ver Diccionario**: Explora todas las señas disponibles
   - 🔧 **Configuración**: Ajusta parámetros del sistema

2. **Detección en Vivo**:
   - Haz clic en **START** para activar la cámara
   - Permite el acceso a la cámara cuando se solicite
   - Realiza una seña frente a la cámara
   - Espera a que se complete el buffer (24 frames ≈ 1 segundo)
   - Observa la predicción con su nivel de confianza
   - Usa el botón **⬅️ Volver al Inicio** para regresar

3. **Diccionario**:
   - Explora señas por categoría
   - Usa los filtros para encontrar señas específicas
   - Busca señas por nombre
   - Observa el tipo de cada seña (Estática/Dinámica)

4. **Configuración**:
   - Ajusta parámetros de detección en tiempo real
   - Configura MediaPipe según tus necesidades
   - Optimiza el rendimiento
   - Consulta información del sistema

## 📊 Señas Detectables

### Números (10)
`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

### Letras (26)
`A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z`

### Frases Comunes (31)
- **Preguntas**: Por_que, Quien, Como, Cuando, Donde, Cuantos, Que_quieres
- **Respuestas**: Si, No, Tal_vez, No_lo_se, No_lo_recuerdo
- **Direcciones**: A_la_derecha, A_la_izquierda, En_la_entrada, Al_final_del_pasillo, En_el_segundo_piso, En_el_edificio, Por_las_escaleras, Por_el_ascensor
- **Saludos**: Hola, Adios, Como_estas, Como_te_llamas, Nos_vemos, Permiso
- **Cortesía**: Gracias, Por_favor, Cuidate, Repite_por_favor
- **Otros**: Mi_casa, Mi_nombre

## 🏗️ Estructura del Proyecto

```
WebApp/
├── app.py                           # Página de inicio con navegación
├── config.py                        # Configuración centralizada
├── requirements.txt                 # Dependencias de Python
├── README.md                        # Este archivo
├── start.ps1                        # Script de inicio rápido
├── pages/                           # Páginas de la aplicación
│   ├── 1_📹_Detección.py           # Página de detección con cámara
│   ├── 2_📚_Diccionario.py         # Diccionario de señas
│   └── 3_⚙️_Configuración.py       # Configuración del sistema
├── utils/                           # Utilidades compartidas
│   ├── __init__.py                 # Inicialización del paquete
│   ├── keypoint_extractor.py       # Extracción de keypoints con MediaPipe
│   └── model_loader.py             # Carga del modelo y clases
└── assets/                          # Recursos (imágenes, etc.)
```

## ⚙️ Configuración Avanzada

### Parámetros Principales (config.py)

**Detección:**
- `SEQUENCE_LENGTH`: Longitud de la secuencia (default: 24 frames)
- `SMOOTH_WINDOW`: Ventana de suavizado (default: 8)
- `MIN_CONFIDENCE`: Confianza mínima para predicción (default: 0.65)
- `CONFIDENCE_THRESHOLD`: Umbral para confirmar predicción (default: 0.75)
- `REQUIRED_STABLE_FRAMES`: Frames estables requeridos (default: 3)

**Tolerancia:**
- `MAX_MISSING_FRAMES`: Tolerancia sin manos (default: 5 frames)
- `RESET_PREDICTION_AFTER_NO_HANDS`: Reset automático (default: 30 frames)
- `MIN_HANDS_QUALITY`: Calidad mínima de manos (default: 0.35)

**MediaPipe:**
- `MEDIAPIPE_MIN_DETECTION_CONFIDENCE`: Confianza de detección (default: 0.7)
- `MEDIAPIPE_MIN_TRACKING_CONFIDENCE`: Confianza de seguimiento (default: 0.7)
- `MEDIAPIPE_MODEL_COMPLEXITY`: Complejidad del modelo (default: 1)

**Rendimiento:**
- `MAX_VIDEO_WIDTH`: Ancho máximo de video (default: 640px)
- `TARGET_FPS`: FPS objetivo (default: 30)

## 🔧 Troubleshooting

### La cámara no se activa

- Asegúrate de que tu navegador tenga permisos para acceder a la cámara
- Verifica que ninguna otra aplicación esté usando la cámara
- Intenta con otro navegador (Chrome/Edge recomendados)
- Revisa la consola del navegador para errores de WebRTC

### Error al cargar el modelo

- Verifica que `best_model.keras` existe en `../EntrenamientoMovimiento/model/`
- Verifica que `label_encoder.json` existe en el mismo directorio
- Comprueba que las rutas en `config.py` son correctas
- Revisa los logs en la terminal donde ejecutaste Streamlit

### Predicciones inestables

- Mejora la iluminación del ambiente
- Asegúrate de que tus manos sean claramente visibles
- Mantén la seña estable por más tiempo (al menos 1 segundo)
- Aumenta `CONFIDENCE_THRESHOLD` en la página de Configuración
- Aumenta `REQUIRED_STABLE_FRAMES` para mayor estabilidad

### Rendimiento lento

- Cierra otras aplicaciones que usen la cámara
- Reduce `MAX_VIDEO_WIDTH` en Configuración → Rendimiento
- Baja el `TARGET_FPS` si experimentas lag
- Considera usar una GPU si está disponible
- Cierra pestañas innecesarias del navegador

### Navegación no funciona

- Asegúrate de que todos los archivos en `pages/` tienen el prefijo numérico
- Verifica que no hay errores de sintaxis en los archivos de páginas
- Reinicia la aplicación con `Ctrl+C` y vuelve a ejecutar `streamlit run app.py`

## 🌐 Despliegue

### Opción 1: Local (ya configurado)
```powershell
streamlit run app.py
```
O usa el script de inicio:
```powershell
.\start.ps1
```

### Opción 2: Streamlit Cloud (gratis)
1. Sube el proyecto a GitHub
2. Conecta tu repositorio en [share.streamlit.io](https://share.streamlit.io)
3. Configura las rutas del modelo correctamente
4. Asegúrate de incluir el modelo en el repositorio o usar Git LFS

### Opción 3: Hugging Face Spaces
1. Crea un Space en [huggingface.co/spaces](https://huggingface.co/spaces)
2. Sube el código y el modelo
3. Configura como aplicación Streamlit
4. Ajusta las rutas en `config.py` si es necesario

### Opción 4: Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Opción 5: Servidor propio
Despliega directamente en un servidor con Python usando systemd o supervisor para mantener la aplicación corriendo.

## 📝 Notas Técnicas

### Arquitectura
- **Multi-página**: Streamlit native multi-page apps
- **MediaPipe**: Detecta landmarks de manos en tiempo real
- **Modelo**: LSTM/GRU entrenado con secuencias de 24 frames
- **Keypoints**: 126 features (2 manos × 21 landmarks × 3 coordenadas)
- **WebRTC**: Streaming de video sin latencia significativa
- **Suavizado**: Sistema adaptativo para predicciones estables

### Lógica de Detección
La aplicación usa **exactamente la misma lógica** que el script `Inferir.py` original:
- Misma extracción de keypoints
- Mismo sistema de tolerancia (5 frames)
- Mismos umbrales de confianza (0.65 min, 0.75 confirmación)
- Mismo suavizado (ventana de 8 frames)
- Misma configuración de MediaPipe

### Rendimiento
- **Carga del modelo**: Una sola vez al inicio (caché)
- **Procesamiento**: ~24 FPS en hardware moderno
- **Latencia**: <100ms desde detección hasta predicción
- **Memoria**: ~500MB con todas las dependencias

## 🚀 Próximas Mejoras

- [ ] Agregar imágenes reales de señas al diccionario
- [ ] Implementar guardado persistente de configuración
- [ ] Añadir historial de predicciones
- [ ] Exportar resultados a archivo
- [ ] Soporte multi-idioma
- [ ] Text-to-speech para predicciones
- [ ] Grabación de sesiones
- [ ] Estadísticas de uso

## 🤝 Contribuciones

Este proyecto es parte de SignBridge, una iniciativa para facilitar la comunicación mediante lenguaje de señas.

## 📄 Licencia

Proyecto educativo - Uso libre para fines académicos y de investigación.

---

**Desarrollado con ❤️ usando Streamlit, TensorFlow y MediaPipe**

## 📞 Soporte

Si encuentras problemas:
1. Revisa la sección de Troubleshooting
2. Consulta los logs en la terminal
3. Verifica la configuración en la página de Configuración
4. Asegúrate de tener todas las dependencias instaladas

**¡Disfruta usando SignBridge! 🤟**
