"""
Inferencia en vivo usando ./model/best_model.keras y ./model/label_encoder.json

Uso:
  python inferir.py --sequence_length 24 --smooth_window 8 --min_conf 0.5 --use_face 0 --use_pose 1 --use_hands 1 --camera 0

Dependencias:
  pip install tensorflow mediapipe opencv-python numpy
"""
import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque
 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

try:
    import mediapipe as mp
except Exception as e:
    print("[ERROR] No se pudo importar mediapipe. Instala con: pip install mediapipe")
    raise e
 
MODEL_DIR = Path("./model")
SEQ_LEN_DEFAULT = 24
SMOOTH_WIN_DEFAULT = 8
MIN_CONF_DEFAULT = 0.50
USE_FACE_DEFAULT = False
USE_POSE_DEFAULT = False
USE_HANDS_DEFAULT = True
 
POSE_LM = 33 * 4
FACE_LM = 468 * 3
HAND_LM = 21 * 3
 
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results, use_face=False, use_pose=True, use_hands=True):
    pose = []
    if use_pose:
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
        else:
            pose = np.zeros(POSE_LM, dtype=np.float32)
    face = []
    if use_face:
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark], dtype=np.float32).flatten()
        else:
            face = np.zeros(FACE_LM, dtype=np.float32)
    lh = []
    rh = []
    if use_hands:
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            lh = np.zeros(HAND_LM, dtype=np.float32)
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            rh = np.zeros(HAND_LM, dtype=np.float32)
    return np.concatenate([arr for arr in [pose, face, lh, rh] if len(arr) > 0]).astype(np.float32)


def has_hands_detected(results):
    """Verifica si hay al menos una mano detectada en los resultados de MediaPipe"""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def get_hands_quality_score(results):
    """Calcula un puntaje de calidad de las manos detectadas (0-1)"""
    if not has_hands_detected(results):
        return 0.0
    
    quality_score = 0.0
    hands_count = 0
    
    # Evaluar mano izquierda
    if results.left_hand_landmarks:
        hands_count += 1
        # Calcular dispersión de landmarks (manos más abiertas = mejor calidad)
        landmarks = results.left_hand_landmarks.landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        quality_score += min(1.0, (x_range + y_range) * 2)  # Normalizar
    
    # Evaluar mano derecha
    if results.right_hand_landmarks:
        hands_count += 1
        landmarks = results.right_hand_landmarks.landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        quality_score += min(1.0, (x_range + y_range) * 2)  # Normalizar
    
    return quality_score / hands_count if hands_count > 0 else 0.0


def keypoint_dim(use_face, use_pose, use_hands):
    dim = 0
    if use_pose: dim += POSE_LM
    if use_face: dim += FACE_LM
    if use_hands: dim += HAND_LM * 2
    return dim


def load_label_classes():
    enc_path = MODEL_DIR / "label_encoder.json"
    if not enc_path.exists():
        raise FileNotFoundError("No se encontró ./model/label_encoder.json. Entrena o copia ese archivo.")
    with open(enc_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "classes" not in data:
        raise ValueError("label_encoder.json inválido: falta clave 'classes'")
    return data["classes"]

# --- NUEVO: Lista de señas estáticas para clasificación de tipo ---
STATIC_SIGN_LABELS = set([
    # Números
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Letras estáticas
    "A", "B", "C", "D", "E", "F", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "T", "U", "V", "W", "Y"
])

def get_sign_type(label):
    """Devuelve si una seña es 'ESTÁTICA' o 'MOVIMIENTO'."""
    return "ESTÁTICA" if label in STATIC_SIGN_LABELS else "MOVIMIENTO"


def main():
    parser = argparse.ArgumentParser(description="Inferencia en vivo")
    parser.add_argument("--sequence_length", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--smooth_window", type=int, default=SMOOTH_WIN_DEFAULT)
    parser.add_argument("--min_conf", type=float, default=MIN_CONF_DEFAULT)
    parser.add_argument("--use_face", type=int, default=int(USE_FACE_DEFAULT))
    parser.add_argument("--use_pose", type=int, default=int(USE_POSE_DEFAULT))
    parser.add_argument("--use_hands", type=int, default=int(USE_HANDS_DEFAULT))
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
 
    model_path = MODEL_DIR.resolve() / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError("No se encontró ./model/best_model.keras. Entrena o copia ese archivo.")
 
    classes = load_label_classes()
    model = keras.models.load_model(model_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba con --camera 1 (u otro índice).")

    feat_dim = keypoint_dim(bool(args.use_face), bool(args.use_pose), bool(args.use_hands))
    buffer = deque(maxlen=args.sequence_length)
    pred_history = deque(maxlen=args.smooth_window)
    
    # Sistema de tolerancia para interrupciones momentáneas
    hands_missing_count = 0
    max_missing_frames = 5  # Tolerancia de 5 frames sin manos
    last_valid_keypoints = None
    
    # Mejoras de rendimiento y estabilidad
    confidence_threshold = 0.75  # Umbral más estricto para confirmar una predicción
    prediction_cooldown = 0  # Evitar predicciones muy frecuentes
    cooldown_frames = 3  # Frames de espera entre predicciones
    
    # Suavizado adaptativo
    min_confidence_for_prediction = 0.65 # Confianza mínima para considerar una predicción válida
    stable_prediction_count = 0
    required_stable_frames = 3  # Frames consecutivos para confirmar predicción
    
    # Sistema de predicción fija
    current_fixed_prediction = None
    current_fixed_confidence = 0.0
    current_sign_type = None # --- NUEVO: Para guardar el tipo de seña ---
    frames_since_last_prediction = 0
    min_frames_for_new_prediction = 15  # Mínimo de frames antes de nueva predicción
    reset_prediction_after_no_hands = 30  # Resetear después de X frames sin manos

    # --- NUEVO: Sistema de preparación ---
    preparation_countdown = 0
    PREPARATION_FRAMES = 45  # Aprox. 1.5 segundos de preparación

    with mp_holistic.Holistic(
        min_detection_confidence=0.7,  # Aumentar confianza de detección
        min_tracking_confidence=0.7,   # Aumentar confianza de seguimiento
        model_complexity=1,             # Mejor precisión vs velocidad
        refine_face_landmarks=False     # Desactivar refinamiento facial para mejor rendimiento
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame no capturado.")
                break
            # Procesamiento de imagen optimizado
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Redimensionar para mejor rendimiento (opcional)
            h_orig, w_orig = image.shape[:2]
            if w_orig > 640:  # Si la imagen es muy grande, redimensionar
                scale = 640 / w_orig
                new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                image_resized = cv2.resize(image, (new_w, new_h))
                results = holistic.process(image_resized)
            else:
                results = holistic.process(image)
            
            image.flags.writeable = True

            keypoints = extract_keypoints(results, bool(args.use_face), bool(args.use_pose), bool(args.use_hands))
            if keypoints.shape[0] != feat_dim:
                keypoints = np.zeros((feat_dim,), dtype=np.float32)
            
            # Sistema de tolerancia para interrupciones momentáneas con calidad
            hands_detected = has_hands_detected(results)
            hands_quality = get_hands_quality_score(results)
            
            # Actualizar contador de frames desde última predicción
            frames_since_last_prediction += 1
            
            if hands_detected and hands_quality >= 0.35:  # Umbral de calidad de manos ligeramente más estricto
                # Manos detectadas con buena calidad: resetear contador y guardar keypoints válidos
                hands_missing_count = 0
                last_valid_keypoints = keypoints.copy()
                buffer.append(keypoints)
            else:
                # No hay manos o calidad insuficiente: incrementar contador de frames perdidos
                hands_missing_count += 1
                
                # Si no hay manos por mucho tiempo, resetear predicción fija
                if hands_missing_count >= reset_prediction_after_no_hands:
                    current_fixed_prediction = None
                    current_fixed_confidence = 0.0
                    current_sign_type = None
                    preparation_countdown = 0 # Resetear preparación
                    buffer.clear() # Limpiar buffer
                    frames_since_last_prediction = 0
                
                if hands_missing_count <= max_missing_frames and last_valid_keypoints is not None:
                    # Dentro de la tolerancia: usar los últimos keypoints válidos
                    buffer.append(last_valid_keypoints)
                else:
                    # Fuera de tolerancia: limpiar buffer gradualmente
                    if len(buffer) > 0:
                        buffer.popleft()  # Remover el frame más antiguo

            if bool(args.use_pose):
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if bool(args.use_hands):
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if bool(args.use_face):
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)

            # --- LÓGICA DE VISUALIZACIÓN MEJORADA ---
            status_text = "" # Texto secundario para diagnóstico
            conf_show = ""
            
            # Mostrar predicción fija si existe
            if current_fixed_prediction is not None:
                # El área principal solo muestra la predicción fija
                conf_show = f"{current_fixed_confidence*100:.1f}%"
            
            # Determinar el texto de estado secundario
            # Lógica de nuevas predicciones
            if not hands_detected and hands_missing_count > max_missing_frames:
                status_text = "(Sin manos detectadas)"
                current_fixed_prediction = None # Limpiar predicción si se pierden las manos
                current_sign_type = None
                preparation_countdown = 0
                buffer.clear()
            elif hands_missing_count > 0:
                status_text = f"(Tolerancia: {hands_missing_count}/{max_missing_frames})"
            elif hands_quality < 0.35:
                status_text = f"(Calidad de mano baja: {hands_quality*100:.0f}%)"
            elif preparation_countdown > 0:
                # --- NUEVO: Lógica de cuenta atrás ---
                status_text = f"¡Prepárate! ... {int(preparation_countdown / 15)}" # 15 frames ~ 0.5s
                preparation_countdown -= 1
            elif len(buffer) < args.sequence_length:
                if len(buffer) == 0 and current_fixed_prediction is None: # Iniciar cuenta atrás
                    preparation_countdown = PREPARATION_FRAMES
                status_text = f"(Reuniendo secuencia... {len(buffer)}/{args.sequence_length})"
            else:
                # Solo buscar nueva predicción si ha pasado tiempo suficiente o no hay predicción fija
                if current_fixed_prediction is None or frames_since_last_prediction >= min_frames_for_new_prediction:
                    # Cooldown para evitar predicciones muy frecuentes
                    if prediction_cooldown > 0:
                        prediction_cooldown -= 1
                        if current_fixed_prediction is None:
                            status_text = f"(Procesando... espera {prediction_cooldown})"
                    else:
                        # Hacer predicción con mejoras de estabilidad
                        inp = np.expand_dims(np.stack(buffer, axis=0), axis=0)
                        probs = model.predict(inp, verbose=0)[0]
                        pred_idx = int(np.argmax(probs))
                        max_conf = float(np.max(probs))
                        
                        # Solo agregar a historial si la confianza es suficiente
                        if max_conf >= min_confidence_for_prediction:
                            pred_history.append(pred_idx)
                            
                            if len(pred_history) == args.smooth_window:
                                counts = np.bincount(list(pred_history), minlength=len(classes))
                                final_idx = int(np.argmax(counts))
                                final_conf = float(probs[final_idx])
                                
                                # Verificar estabilidad de la predicción
                                if final_conf >= confidence_threshold:
                                    # Verificar si es una predicción estable
                                    recent_predictions = list(pred_history)[-required_stable_frames:]
                                    if len(set(recent_predictions)) == 1:  # Todas iguales
                                        stable_prediction_count += 1
                                        if stable_prediction_count >= required_stable_frames:
                                            new_prediction = classes[final_idx]
                                            
                                            # Solo actualizar si es diferente a la predicción actual
                                            if current_fixed_prediction != new_prediction:
                                                current_fixed_prediction = new_prediction
                                                current_fixed_confidence = final_conf
                                                current_sign_type = get_sign_type(new_prediction) # --- NUEVO: Clasificar tipo ---
                                                frames_since_last_prediction = 0
                                                prediction_cooldown = cooldown_frames
                                                stable_prediction_count = 0
                                            else:
                                                # Misma predicción, solo resetear cooldown
                                                prediction_cooldown = cooldown_frames
                                                stable_prediction_count = 0
                                        else:
                                            if current_fixed_prediction is None:
                                                status_text = f"(Confirmando... {stable_prediction_count}/{required_stable_frames})"
                                    else:
                                        stable_prediction_count = 0
                                        if current_fixed_prediction is None:
                                            status_text = "(Predicción inestable)"
                                else:
                                    if current_fixed_prediction is None:
                                        status_text = f"(Baja confianza: {final_conf*100:.1f}%)"
                                    stable_prediction_count = 0
                        else:
                            if current_fixed_prediction is None:
                                status_text = f"(Confianza insuficiente: {max_conf*100:.1f}%)"

            h, w = frame.shape[:2]
            
            # Panel de información mejorado
            cv2.rectangle(frame, (0,0), (w, 120), (0,0,0), -1)
            
            # Predicción principal con estilo mejorado para predicciones fijas
            if current_fixed_prediction:
                # --- PANTALLA DE PREDICCIÓN FIJA ---
                pred_color = (0, 255, 255)  # Amarillo para predicción fija
                cv2.putText(frame, f"SEÑA: {current_fixed_prediction}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
                cv2.putText(frame, f"Conf: {conf_show}", (w-200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                # --- NUEVO: Mostrar tipo de seña ---
                cv2.putText(frame, f"TIPO: {current_sign_type}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                # --- PANTALLA DE ESPERA ---
                cv2.putText(frame, "(Detectando...)", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                
            # Mostrar texto de estado secundario si no hay una predicción fija
            if not current_fixed_prediction and status_text:
                cv2.putText(frame, f"Estado: {status_text}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
            
            # Indicador de detección de manos con tolerancia mejorado
            if hands_detected:
                hands_color = (0, 255, 0)  # Verde: manos detectadas
                hands_text = "MANOS: SI"
                status_color = (0, 255, 0)
            elif hands_missing_count <= max_missing_frames:
                hands_color = (0, 255, 255)  # Amarillo: en tolerancia
                hands_text = f"TOLERANCIA: {hands_missing_count}/{max_missing_frames}"
                status_color = (0, 255, 255)
            else:
                hands_color = (0, 0, 255)  # Rojo: sin manos
                hands_text = "MANOS: NO"
                status_color = (0, 0, 255)
                
            # Indicadores visuales mejorados
            cv2.circle(frame, (w-30, 30), 12, hands_color, -1)
            cv2.circle(frame, (w-30, 30), 15, (255, 255, 255), 2)
            cv2.putText(frame, hands_text, (w-220, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Instrucciones para el usuario
            cv2.putText(frame, "ESC: Salir | SPACE: Pausar | R: Reset prediccion", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

            cv2.imshow("Inferencia SeñAS (MVP) - Mejorado", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC para salir
                break
            elif key == 32:  # SPACE para pausar
                cv2.waitKey(0)
            elif key == ord('r') or key == ord('R'):  # R para resetear predicción
                current_fixed_prediction = None
                current_fixed_confidence = 0.0
                current_sign_type = None
                preparation_countdown = 0
                buffer.clear()
                frames_since_last_prediction = 0
                print("[INFO] Predicción fija reseteada")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
