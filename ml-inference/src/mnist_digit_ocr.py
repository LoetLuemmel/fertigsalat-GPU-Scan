#!/usr/bin/env python3
"""
MNIST-based digit recognition using a pre-trained CNN.
Optimized for Jetson Orin Nano with GPU support.
"""
import os
import cv2
import numpy as np

# Lazy-load TensorFlow to avoid startup delay
_model = None
_model_path = '/app/models/mnist_cnn.h5'


def _build_model():
    """Build a simple CNN for MNIST digit recognition."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def _train_model():
    """Train the model on MNIST dataset."""
    from tensorflow import keras

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("Building model...")
    model = _build_model()

    print("Training model (this may take a few minutes)...")
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs(os.path.dirname(_model_path), exist_ok=True)
    model.save(_model_path)
    print(f"Model saved to {_model_path}")

    return model


def _load_model():
    """Load or train the MNIST model."""
    global _model

    if _model is not None:
        return _model

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf

    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"MNIST OCR: Using GPU ({len(gpus)} device(s))")
        except RuntimeError as e:
            print(f"GPU config error: {e}")

    if os.path.exists(_model_path):
        print(f"Loading MNIST model from {_model_path}...")
        _model = tf.keras.models.load_model(_model_path)
    else:
        print("No pre-trained model found. Training new model...")
        _model = _train_model()

    return _model


def preprocess_for_mnist(image):
    """
    Preprocess an image for MNIST model input.

    Args:
        image: BGR or grayscale image (any size)

    Returns:
        28x28 normalized array ready for model input
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binarize with OTSU (white digit on black background for MNIST)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find bounding box of content
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    # Extract digit region with padding
    pad = max(w, h) // 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + w + pad)
    y2 = min(binary.shape[0], y + h + pad)

    digit = binary[y1:y2, x1:x2]

    if digit.size == 0:
        return None

    # Make square by padding
    h, w = digit.shape
    if h > w:
        diff = h - w
        left = diff // 2
        right = diff - left
        digit = cv2.copyMakeBorder(digit, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        digit = cv2.copyMakeBorder(digit, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Resize to 20x20 (MNIST digits are centered in 28x28 with 4px border)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Center in 28x28
    result = np.zeros((28, 28), dtype=np.float32)
    result[4:24, 4:24] = digit.astype(np.float32) / 255.0

    return result


def recognize_digit_mnist(image, debug=False):
    """
    Recognize a digit using the MNIST-trained CNN.

    Args:
        image: BGR or grayscale image containing a single digit
        debug: If True, print debug info

    Returns:
        (digit_string, confidence) or (None, 0)
    """
    # Preprocess
    processed = preprocess_for_mnist(image)
    if processed is None:
        if debug:
            print("  MNIST: Could not preprocess image")
        return None, 0

    # Load model (lazy)
    model = _load_model()

    # Predict
    input_data = processed.reshape(1, 28, 28, 1)
    predictions = model.predict(input_data, verbose=0)[0]

    digit = int(np.argmax(predictions))
    confidence = float(predictions[digit])

    if debug:
        print(f"  MNIST: digit={digit}, confidence={confidence:.3f}")
        print(f"  All predictions: {[f'{i}:{p:.2f}' for i, p in enumerate(predictions)]}")

    return str(digit), confidence


def recognize_digit(image, debug=False):
    """
    Main entry point - alias for recognize_digit_mnist.
    """
    return recognize_digit_mnist(image, debug)


# Pre-train model if run directly
if __name__ == '__main__':
    print("=" * 50)
    print("MNIST Digit Recognition - Model Training")
    print("=" * 50)

    # Force training
    if os.path.exists(_model_path):
        os.remove(_model_path)

    model = _load_model()

    print("\nModel ready for inference!")
    print(f"Model size: {os.path.getsize(_model_path) / 1024:.1f} KB")
