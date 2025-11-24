"""
Módulo para carga del modelo Keras y gestión de clases.
"""

import json
import os
from pathlib import Path
from functools import lru_cache

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

# Lista de señas estáticas para clasificación de tipo
STATIC_SIGN_LABELS = set([
    # Números
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Letras estáticas
    "A", "B", "C", "D", "E", "F", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "T", "U", "V", "W", "Y"
])


def get_sign_type(label):
    """
    Devuelve si una seña es 'ESTÁTICA' o 'MOVIMIENTO'.
    
    Args:
        label: Etiqueta de la seña
        
    Returns:
        str: "ESTÁTICA" o "MOVIMIENTO"
    """
    return "ESTÁTICA" if label in STATIC_SIGN_LABELS else "MOVIMIENTO"


@lru_cache(maxsize=1)
def load_model(model_path=None):
    """
    Carga el modelo Keras. Usa caché para evitar recargas.
    
    Args:
        model_path: Ruta al archivo .keras (opcional)
        
    Returns:
        keras.Model: Modelo cargado
    """
    if model_path is None:
        # Ruta por defecto relativa al directorio del proyecto
        base_dir = Path(__file__).parent.parent.parent
        model_path = base_dir / "EntrenamientoMovimiento" / "model" / "best_model.keras"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {model_path}\n"
            f"Asegúrate de que el archivo best_model.keras existe."
        )
    
    print(f"[INFO] Cargando modelo desde: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"[INFO] Modelo cargado exitosamente")
    return model


@lru_cache(maxsize=1)
def load_classes(encoder_path=None):
    """
    Carga las clases desde label_encoder.json. Usa caché para evitar recargas.
    
    Args:
        encoder_path: Ruta al archivo label_encoder.json (opcional)
        
    Returns:
        list: Lista de clases/etiquetas
    """
    if encoder_path is None:
        # Ruta por defecto relativa al directorio del proyecto
        base_dir = Path(__file__).parent.parent.parent
        encoder_path = base_dir / "EntrenamientoMovimiento" / "model" / "label_encoder.json"
    else:
        encoder_path = Path(encoder_path)
    
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"No se encontró label_encoder.json en: {encoder_path}\n"
            f"Asegúrate de que el archivo existe."
        )
    
    with open(encoder_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "classes" not in data:
        raise ValueError("label_encoder.json inválido: falta clave 'classes'")
    
    classes = data["classes"]
    print(f"[INFO] Cargadas {len(classes)} clases")
    return classes
