"""
SentinelVision AI - Model Initializer Script
Generates and verifies best_human_activity_v2.keras based on EfficientNetV2B2
for the 8 human activity classes.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def generate_activity_model():
    models_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(models_dir, "best_human_activity_v2.keras")
    classes_path = os.path.join(models_dir, "live_activity_classes.json")

    with open(classes_path, "r") as f:
        classes = json.load(f)

    num_classes = len(classes)
    print(f"Creating Human Activity Model with {num_classes} classes: {classes}")

    # Build model architecture matching EfficientNetV2B2 input
    inputs = keras.Input(shape=(224, 224, 3), name="input_layer")
    
    # Feature extraction backbone (EfficientNetV2B2 structure or lightweight sequential backbone for initialization)
    base_model = tf.keras.applications.EfficientNetV2B2(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling="avg"
    )
    
    x = base_model.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="activity_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SentinelVision_EfficientNetV2B2_Activity")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"Saving model to {model_path} ...")
    model.save(model_path)
    print("Model saved successfully!")

    # Verify model loading
    print("Verifying model load ...")
    loaded_model = keras.models.load_model(model_path)
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    preds = loaded_model.predict(dummy_input, verbose=0)
    print(f"Sample prediction probabilities shape: {preds.shape}, sum: {np.sum(preds):.2f}")
    print("Model verification test PASSED!")

if __name__ == "__main__":
    generate_activity_model()
