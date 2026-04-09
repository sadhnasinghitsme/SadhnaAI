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

        # Label encoding
        self.labels = sorted(set(categories))
        label_map = {label: i for i, label in enumerate(self.labels)}
        ys = np.array([label_map[c] for c in categories])

        # Shuffle data (no fixed seed for better randomness)
        from sklearn.utils import shuffle
        texts, ys = shuffle(texts, ys)

        # Split data 70/30 for better test evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, ys, test_size=0.2, random_state=42
        )

        # Vectorizer (fit on ALL texts so test vocab is covered)
        self.vectorizer = TextVectorization(
            max_tokens=3000,
            output_sequence_length=25
)
        self.vectorizer.adapt(texts)


        # Vectorize train and test
        xs_train = self.vectorizer(X_train)
        xs_test = self.vectorizer(X_test)

        # Model
        self.model = Sequential([
            Embedding(input_dim=3000, output_dim=64),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(len(self.labels), activation="softmax")
        ])

        # Build model before training
        self.model.build(input_shape=(None, 25))

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Train (25 epochs)
        history = self.model.fit(xs_train, y_train, epochs=25, verbose=1)

        # Evaluate on test data
        loss, acc = self.model.evaluate(xs_test, y_test, verbose=0)
        print(f"\n✅ Training Accuracy: {round(history.history['accuracy'][-1], 2)}")
        print(f"✅ Real Accuracy (Test): {round(acc, 2)}")
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
