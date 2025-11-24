"""
Entrenamiento del clasificador secuencial sobre keypoints (Keras 3.x).
Lee ./sign_data/<CLASE>/*.npy y guarda modelo en ./model/best_model.keras

Uso:
  python entrenamiento.py --epochs 40 --batch_size 64 --val_split 0.15

Dependencias:
  pip install tensorflow scikit-learn numpy pyyaml
"""
import os
import glob
import math
import json
import yaml
import argparse
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths y defaults
BASE_DIR = Path("./sign_data")
MODEL_DIR = Path("./model"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
CFG_PATH = Path("sign_config.yaml")

DEFAULT_CFG = {
    "sequence_length": 24,
    "use_face": False,
    "use_pose": True,
    "use_hands": True,
    "train": {
        "batch_size": 64,
        "epochs": 40,
        "val_split": 0.15,
        "learning_rate": 1e-3,
        "patience": 8,
        "l2": 1e-5,
        "dropout": 0.3,
        "augment": True,
    }
}

POSE_LM = 33 * 4
FACE_LM = 468 * 3
HAND_LM = 21 * 3


def load_config():
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # merge shallow
        cfg = {**DEFAULT_CFG, **user_cfg}
        if "train" in user_cfg:
            cfg["train"] = {**DEFAULT_CFG["train"], **user_cfg["train"]}
        return cfg
    else:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CFG, f, sort_keys=False, allow_unicode=True)
        return DEFAULT_CFG


def keypoint_dim(cfg):
    dim = 0
    if cfg["use_pose"]: dim += POSE_LM
    if cfg["use_face"]: dim += FACE_LM
    if cfg["use_hands"]: dim += HAND_LM * 2
    return dim


def augment_static_signs(X, y, le, cfg):
    """
    Aumenta el dataset para señas estáticas. Por cada secuencia original,
    genera múltiples secuencias "perfectamente estáticas" muestreando
    varios frames representativos.
    """
    print("[INFO] Aplicando aumentación intensiva para señas estáticas...")
    X_aug, y_aug = [], []
    # Lista precisa de señas estáticas (números y letras específicas)
    static_sign_labels = set([
        # Números
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        # Letras estáticas
        "A", "B", "C", "D", "E", "F", "H", "I", "K", "L", "M",
        "N", "O", "P", "Q", "R", "T", "U", "V", "W", "Y"
    ])
    
    N_FRAMES_TO_SAMPLE = 5  # Número de frames a muestrear de cada secuencia

    for i, seq in enumerate(X):
        label_name = le.classes_[y[i]]
        if label_name in static_sign_labels:
            # Muestrear N frames de la parte central de la secuencia para evitar ruido al inicio/final
            sample_indices = np.linspace(4, cfg["sequence_length"] - 5, num=N_FRAMES_TO_SAMPLE, dtype=int)
            for frame_idx in sample_indices:
                frame = seq[frame_idx]
                # Crear una secuencia perfectamente estática replicando el frame
                static_seq = np.tile(frame, (cfg["sequence_length"], 1))
                X_aug.append(static_seq)
                y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)


def load_dataset(cfg, classes=None):
    X, y = [], []
    seq_len = cfg["sequence_length"]
    feat_dim = keypoint_dim(cfg)

    # Si el usuario definió clases en config, úsalas. Sino, deduce por carpetas.
    if classes is None:
        if CFG_PATH.exists():
            with open(CFG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            classes = data.get("classes")
    if not classes:
        # Detectar carpetas dentro de sign_data
        classes = [p.name for p in BASE_DIR.glob("*") if p.is_dir()]
    
    # Filtrar solo strings válidos
    classes = [str(label) for label in classes if label and isinstance(label, (str, int, float))]

    for label in classes:
        label_dir = BASE_DIR / label
        for f in label_dir.glob("*.npy"):
            arr = np.load(str(f))
            if arr.ndim != 2:
                continue
            # Normalizar longitud
            if arr.shape[0] != seq_len:
                if arr.shape[0] > seq_len:
                    arr = arr[:seq_len]
                else:
                    pad = np.zeros((seq_len - arr.shape[0], feat_dim), dtype=np.float32)
                    arr = np.vstack([arr, pad])
            X.append(arr)
            y.append(label)

    if not X:
        raise RuntimeError("No se encontraron ejemplos en ./sign_data. Ejecuta la recolección primero.")

    X = np.stack(X).astype(np.float32)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    with open(MODEL_DIR / "label_encoder.json", "w", encoding="utf-8") as f:
        json.dump({"classes": le.classes_.tolist()}, f, ensure_ascii=False)

    return X, y_enc, le


def augment_batch(X, prob=0.5):
    X_aug = []
    for seq in X:
        s = seq.copy()
        if np.random.rand() < prob:
            s += np.random.normal(0, 0.01, size=s.shape).astype(np.float32)
        if np.random.rand() < prob:
            L = s.shape[0]; w = max(1, L//8); i = np.random.randint(0, L-w)
            s[i:i+w] = 0.0
        if np.random.rand() < prob:
            mask = np.random.binomial(1, 0.98, size=s.shape).astype(np.float32)
            s *= mask
        X_aug.append(s)
    return np.stack(X_aug)


def build_model(cfg, n_classes):
    seq_len = cfg["sequence_length"]
    feat_dim = keypoint_dim(cfg)

    inputs = keras.Input(shape=(seq_len, feat_dim), name="keypoints")
    
    # --- Ruta Temporal (LSTM) para gestos ---
    # La misma que ya tenías
    masked_input = layers.Masking(mask_value=0.0)(inputs)
    lstm_out = layers.Bidirectional(layers.LSTM(160, return_sequences=True, kernel_regularizer=keras.regularizers.l2(cfg["train"]["l2"])))(masked_input)
    lstm_out = layers.Dropout(cfg["train"]["dropout"])(lstm_out)
    lstm_out = layers.Bidirectional(layers.LSTM(160))(lstm_out)
    
    # --- Ruta Espacial (MLP) para poses estáticas ---
    # Tomamos el promedio de todos los frames para obtener una representación de la pose
    spatial_input = layers.GlobalAveragePooling1D()(masked_input)
    mlp_out = layers.Dense(128, activation="relu")(spatial_input)
    mlp_out = layers.Dropout(cfg["train"]["dropout"])(mlp_out)

    # --- Combinación de ambas rutas ---
    combined = layers.Concatenate()([lstm_out, mlp_out])
    
    # --- Clasificador Final ---
    final_dense = layers.Dense(128, activation="relu")(combined)
    final_dense = layers.Dropout(cfg["train"]["dropout"])(final_dense)
    outputs = layers.Dense(n_classes, activation="softmax")(final_dense)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["train"]["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo de señas")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.epochs is not None: cfg["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None: cfg["train"]["batch_size"] = int(args.batch_size)
    if args.val_split is not None: cfg["train"]["val_split"] = float(args.val_split)

    print("[INFO] Cargando dataset...")
    X, y, le = load_dataset(cfg)
    n_classes = len(le.classes_)

    # Aumentación para señas estáticas
    X_static_aug, y_static_aug = augment_static_signs(X, y, le, cfg)
    if len(X_static_aug) > 0:
        X = np.concatenate([X, X_static_aug])
        y = np.concatenate([y, y_static_aug])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg["train"]["val_split"], stratify=y, random_state=42)

    if cfg["train"]["augment"]:
        def gen():
            bs = cfg["train"]["batch_size"]
            idx = np.arange(len(X_train))
            while True:
                np.random.shuffle(idx)
                for i in range(0, len(idx), bs):
                    batch_idx = idx[i:i+bs]
                    xb = X_train[batch_idx]
                    xb = augment_batch(xb, prob=0.6)
                    yb = y_train[batch_idx]
                    yield xb, yb
        steps = math.ceil(len(X_train) / cfg["train"]["batch_size"])
        train_ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, X.shape[1], X.shape[2]), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int64)
            )
        ).prefetch(tf.data.AUTOTUNE)
    else:
        steps = None
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(cfg["train"]["batch_size"]).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(cfg["train"]["batch_size"]).prefetch(tf.data.AUTOTUNE)

    model = build_model(cfg, n_classes)
    model.summary()

    ckpt_path = MODEL_DIR / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=cfg["train"]["patience"], mode="max", restore_best_weights=True)
    ]

    print("[INFO] Entrenando...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg["train"]["epochs"], steps_per_epoch=steps, callbacks=callbacks)

    with open(MODEL_DIR / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False)

    with open(MODEL_DIR / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print("[OK] Modelo guardado en:", ckpt_path.resolve())


if __name__ == "__main__":
    main()
