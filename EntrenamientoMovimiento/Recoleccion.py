"""
Recolección de secuencias desde la webcam para cada clase/seña.
Guarda arrays .npy en ./sign_data/<CLASE>/

Uso:
  # Recolectar todo (números + abecedario + palabras/frases):
  python recoleccion.py --sequence_length 24 --capture_fps 24 --samples 30
  
  # Recolectar solo números:
  python recoleccion.py --clases 0 1 2 3 4 5 6 7 8 9 --samples 30
  
  # Recolectar solo letras específicas:
  python recoleccion.py --clases A B C D E --samples 30
  
  # Recolectar solo preguntas básicas:
  python recoleccion.py --clases Por_que Quien Como Cuando Donde --samples 30
  
  # Recolectar solo direcciones:
  python recoleccion.py --clases A_la_derecha A_la_izquierda En_la_entrada --samples 30

Dependencias:
  pip install mediapipe opencv-python numpy pyyaml
"""
import os
import cv2
import time
import yaml
import argparse
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import mediapipe as mp
except Exception as e:
    print("[ERROR] No se pudo importar mediapipe. Instala con: pip install mediapipe")
    raise e

# Paths y defaults
BASE_DIR = Path("./sign_data")
CFG_PATH = Path("sign_config.yaml")

DEFAULT_CFG = {
    "sequence_length": 24,
    "capture_fps": 24,
    "classes": [
        # Números (10 dígitos)
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        # Abecedario (26 letras)
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        # Palabras/frases comunes (32 expresiones)
        "Por_que", "Quien", "Como", "Cuando", "Donde", "Cuantos", "Que_quieres", 
        "No_lo_se", "Si", "No", "Tal_vez", "No_lo_recuerdo", "Repite_por_favor",
        "A_la_derecha", "A_la_izquierda", "En_la_entrada", "Al_final_del_pasillo",
        "En_el_segundo_piso", "En_el_edificio", "Por_las_escaleras", "Por_el_ascensor",
        "Hola", "Adios", "Como_te_llamas", "Permiso", "Nos_vemos", "Mi_casa",
        "Como_estas", "Gracias", "Por_favor", "Cuidate", "Mi_nombre",
    ],
    "samples_per_class": 30,
    "use_face": False,
    "use_pose": False,
    "use_hands": True,
}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

POSE_LM = 33 * 4
FACE_LM = 468 * 3
HAND_LM = 21 * 3


def load_config():
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = {**DEFAULT_CFG, **user_cfg}
        return cfg
    else:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CFG, f, sort_keys=False, allow_unicode=True)
        return DEFAULT_CFG


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


def main():
    parser = argparse.ArgumentParser(description="Recolección de secuencias para señas")
    parser.add_argument("--clases", nargs="*", help="Lista de clases a capturar")
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--capture_fps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None, help="Número de ejemplos por clase")
    args = parser.parse_args()

    cfg = load_config()
    if args.sequence_length is not None:
        cfg["sequence_length"] = int(args.sequence_length)
    if args.capture_fps is not None:
        cfg["capture_fps"] = int(args.capture_fps)
    if args.samples is not None:
        cfg["samples_per_class"] = int(args.samples)

    classes = args.clases if args.clases else cfg["classes"]
    samples_per_class = cfg["samples_per_class"]

    seq_len = cfg["sequence_length"]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    target_fps = cfg["capture_fps"]
    delay = max(1, int(1000 / target_fps))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for label in classes:
            print(f"\n[INFO] Clase: {label}")
            class_dir = BASE_DIR / label
            class_dir.mkdir(parents=True, exist_ok=True)

            input("Posiciónate para '{0}'. Pulsa ENTER para comenzar la captura de {1} ejemplos...".format(label, samples_per_class))
            for sample_idx in range(samples_per_class):
                frames = []
                print(f"  - Grabando ejemplo {sample_idx+1}/{samples_per_class}...")
                while len(frames) < seq_len:
                    ok, frame = cap.read()
                    if not ok:
                        print("[WARN] Frame no capturado.")
                        break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True

                    keypoints = extract_keypoints(results, cfg["use_face"], cfg["use_pose"], cfg["use_hands"])
                    frames.append(keypoints)

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    cv2.putText(frame, f"Clase: {label}  Ej:{sample_idx+1}/{samples_per_class}  F:{len(frames)}/{seq_len}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow("Recoleccion", frame)
                    if cv2.waitKey(delay) & 0xFF == 27:
                        print("[INFO] Recolección interrumpida por el usuario.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                arr = np.stack(frames, axis=0)
                out_path = class_dir / f"{int(time.time())}_{sample_idx:03d}.npy"
                np.save(out_path, arr)
                print(f"  ✓ Guardado {out_path.name}")
                
                # Pequeña pausa entre ejemplos para evitar sobrecargar
                if sample_idx < samples_per_class - 1:
                    time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] Recolección finalizada. Archivos en:", BASE_DIR.resolve())


if __name__ == "__main__":
    main()
