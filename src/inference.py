from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf


@dataclass(frozen=True)
class TFLiteModel:
    interpreter: tf.lite.Interpreter
    input_index: int
    output_index: int
    input_shape: Tuple[int, int, int, int]  # (1, H, W, 3)
    labels: List[str]


def load_labels(labels_path: Path) -> List[str]:
    labels = [ln.strip() for ln in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [l for l in labels if l]
    if not labels:
        raise ValueError(f"No labels found in: {labels_path}")
    return labels


def load_tflite_model(model_path: Path, labels_path: Path) -> TFLiteModel:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Typical image model has a single input and output tensor.
    input_index = int(input_details[0]["index"])
    output_index = int(output_details[0]["index"])
    input_shape = tuple(int(x) for x in input_details[0]["shape"])  # e.g. (1, 224, 224, 3)

    labels = load_labels(labels_path)

    return TFLiteModel(
        interpreter=interpreter,
        input_index=input_index,
        output_index=output_index,
        input_shape=input_shape,  # (1, H, W, C)
        labels=labels,
    )


def preprocess_image(img: Image.Image, input_shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Resize & format image for the TFLite model.

    The exported model includes an internal Rescaling layer, so we keep pixel range 0..255
    (float32), matching the training notebook.
    """
    if len(input_shape) != 4:
        raise ValueError(f"Unexpected input shape: {input_shape}")

    _, h, w, c = input_shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB), got: {c}")

    img = img.convert("RGB")
    img = img.resize((w, h))

    x = np.asarray(img, dtype=np.float32)  # 0..255
    x = np.expand_dims(x, axis=0)          # (1, H, W, 3)
    return x


def predict_proba(model: TFLiteModel, img: Image.Image) -> np.ndarray:
    x = preprocess_image(img, model.input_shape)

    model.interpreter.set_tensor(model.input_index, x)
    model.interpreter.invoke()
    y = model.interpreter.get_tensor(model.output_index)

    # y shape usually (1, num_classes)
    y = np.asarray(y).squeeze()
    return y


def top_k(pred: np.ndarray, labels: List[str], k: int = 3) -> List[Tuple[str, float]]:
    if pred.ndim != 1:
        pred = pred.reshape(-1)
    k = max(1, min(int(k), pred.shape[0]))

    idx = np.argsort(pred)[::-1][:k]
    return [(labels[int(i)], float(pred[int(i)])) for i in idx]
