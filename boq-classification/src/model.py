import os
import numpy as np
import pandas as pd
import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import TextVectorization, Dropout



class BOQModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.labels = None

    def train(self, file_path):
        df = pd.read_csv(file_path)
        texts = df["description"].astype(str).values
        categories = df["category"].values

        # Clean text
        texts = np.array([t.lower().strip() for t in texts])

        # Label encoding
        self.labels = sorted(set(categories))
        label_map = {label: i for i, label in enumerate(self.labels)}
        ys = np.array([label_map[c] for c in categories])

        # Shuffle data
        texts, ys = shuffle(texts, ys)

        # Split data 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            texts, ys, test_size=0.2, random_state=42
        )

        # Vectorizer (fit on ALL texts so test vocab is covered)
        self.vectorizer = TextVectorization(
            max_tokens=4000,
            output_sequence_length=30
        )
        self.vectorizer.adapt(texts)

        # Vectorize train and test
        xs_train = self.vectorizer(X_train)
        xs_test = self.vectorizer(X_test)

        # Model
        self.model = Sequential([
            Embedding(input_dim=4000, output_dim=64),
            GlobalAveragePooling1D(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(len(self.labels), activation="softmax")
        ])

        # Build model before training
        self.model.build(input_shape=(None, 30))

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Train with validation
        history = self.model.fit(
            xs_train, y_train,
            epochs=20,
            validation_data=(xs_test, y_test),
            verbose=1
        )

        # Print both accuracies
        print(f"\nTrain Accuracy: {history.history['accuracy'][-1]}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][-1]}")
        return history

    def predict(self, text):
        vec = self.vectorizer(tf.constant([text]))
        proba = self.model.predict(vec, verbose=0)[0]
        idx = np.argmax(proba)
        category = self.labels[idx]
        confidence = float(proba[idx])
        return category, confidence

    def save(self, path="models"):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "boq_tf_model.keras"))
        joblib.dump(self.labels, os.path.join(path, "labels.pkl"))
        vectorizer_model = tf.keras.Sequential([self.vectorizer])
        vectorizer_model.save(os.path.join(path, "vectorizer_model.keras"))
        print(f"💾 Model saved -> {path}/")

    def load(self, path="models"):
        self.model = tf.keras.models.load_model(os.path.join(path, "boq_tf_model.keras"))
        self.labels = joblib.load(os.path.join(path, "labels.pkl"))
        vectorizer_model = tf.keras.models.load_model(os.path.join(path, "vectorizer_model.keras"))
        self.vectorizer = vectorizer_model.layers[0]
        print(f"✅ Model loaded from {path}/")
