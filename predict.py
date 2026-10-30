"""Predict the traffic sign in an image.  Usage: python predict.py path/to/image.jpg"""
import json, sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/traffic_sign_model.keras"
CLASS_NAMES_PATH = "results/class_names.json"


def predict(image_path, top_k=3):
    model = tf.keras.models.load_model(MODEL_PATH)
    names = json.load(open(CLASS_NAMES_PATH))
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    x = np.asarray(img, dtype="float32")[None] / 255.0
    probs = model.predict(x, verbose=0)[0]
    return [(names[i], float(probs[i])) for i in np.argsort(probs)[::-1][:top_k]]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for name, p in predict(sys.argv[1]):
        print(f"{name}: {p * 100:.2f}%")
