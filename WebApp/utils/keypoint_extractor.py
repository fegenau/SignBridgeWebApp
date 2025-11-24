"""
Módulo para extracción de keypoints usando MediaPipe.
Reutiliza la lógica exacta del script Inferir.py.
"""

import numpy as np

# Constantes para dimensiones de landmarks
POSE_LM = 33 * 4  # 33 landmarks × 4 valores (x, y, z, visibility)
FACE_LM = 468 * 3  # 468 landmarks × 3 valores (x, y, z)
HAND_LM = 21 * 3   # 21 landmarks × 3 valores (x, y, z)


def extract_keypoints(results, use_face=False, use_pose=False, use_hands=True):
    """
    Extrae keypoints de los resultados de MediaPipe Holistic.
    
    Args:
        results: Resultados de MediaPipe Holistic
        use_face: Si se deben incluir landmarks faciales
        use_pose: Si se deben incluir landmarks de pose
        use_hands: Si se deben incluir landmarks de manos
        
    Returns:
        np.ndarray: Array concatenado de keypoints (float32)
    """
    pose = []
    if use_pose:
        if results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                dtype=np.float32
            ).flatten()
        else:
            pose = np.zeros(POSE_LM, dtype=np.float32)
    
    face = []
    if use_face:
        if results.face_landmarks:
            face = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
                dtype=np.float32
            ).flatten()
        else:
            face = np.zeros(FACE_LM, dtype=np.float32)
    
    lh = []
    rh = []
    if use_hands:
        if results.left_hand_landmarks:
            lh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32
            ).flatten()
        else:
            lh = np.zeros(HAND_LM, dtype=np.float32)
        
        if results.right_hand_landmarks:
            rh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32
            ).flatten()
        else:
            rh = np.zeros(HAND_LM, dtype=np.float32)
    
    return np.concatenate([arr for arr in [pose, face, lh, rh] if len(arr) > 0]).astype(np.float32)


def has_hands_detected(results):
    """
    Verifica si hay al menos una mano detectada en los resultados de MediaPipe.
    
    Args:
        results: Resultados de MediaPipe Holistic
        
    Returns:
        bool: True si hay al menos una mano detectada
    """
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def get_hands_quality_score(results):
    """
    Calcula un puntaje de calidad de las manos detectadas (0-1).
    Manos más abiertas/dispersas tienen mejor calidad.
    
    Args:
        results: Resultados de MediaPipe Holistic
        
    Returns:
        float: Puntaje de calidad entre 0 y 1
    """
    if not has_hands_detected(results):
        return 0.0
    
    quality_score = 0.0
    hands_count = 0
    
    # Evaluar mano izquierda
    if results.left_hand_landmarks:
        hands_count += 1
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
    """
    Calcula la dimensión total de keypoints según las opciones activadas.
    
    Args:
        use_face: Si se incluyen landmarks faciales
        use_pose: Si se incluyen landmarks de pose
        use_hands: Si se incluyen landmarks de manos
        
    Returns:
        int: Dimensión total de keypoints
    """
    dim = 0
    if use_pose:
        dim += POSE_LM
    if use_face:
        dim += FACE_LM
    if use_hands:
        dim += HAND_LM * 2  # Ambas manos
    return dim
