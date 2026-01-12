#!/usr/bin/env python3
"""
Fine-tune the EMNIST model with collected training data.
Run this script after collecting training samples via the web interface.
"""
import os
import sys
import cv2
import numpy as np
import glob

# Paths
MODEL_PATH = '/app/models/emnist_cnn.h5'
TRAINING_DIR = '/app/training_data'
BACKUP_PATH = '/app/models/emnist_cnn_backup.h5'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_training_data():
    """Load training samples from the training_data directory."""
    images = []
    labels = []

    for digit in range(10):
        digit_dir = os.path.join(TRAINING_DIR, str(digit))
        if not os.path.exists(digit_dir):
            continue

        samples = glob.glob(os.path.join(digit_dir, '*.png'))
        print(f"Digit {digit}: {len(samples)} samples")

        for sample_path in samples:
            img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Preprocess: binarize with OTSU
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find bounding box
            coords = cv2.findNonZero(binary)
            if coords is None:
                continue

            x, y, w, h = cv2.boundingRect(coords)

            # Extract with padding
            pad = max(w, h) // 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(binary.shape[1], x + w + pad)
            y2 = min(binary.shape[0], y + h + pad)

            digit_img = binary[y1:y2, x1:x2]
            if digit_img.size == 0:
                continue

            # Make square
            dh, dw = digit_img.shape
            if dh > dw:
                diff = dh - dw
                digit_img = cv2.copyMakeBorder(digit_img, 0, 0, diff//2, diff-diff//2,
                                                cv2.BORDER_CONSTANT, value=0)
            elif dw > dh:
                diff = dw - dh
                digit_img = cv2.copyMakeBorder(digit_img, diff//2, diff-diff//2, 0, 0,
                                                cv2.BORDER_CONSTANT, value=0)

            # Resize to 20x20 and center in 28x28
            digit_img = cv2.resize(digit_img, (20, 20), interpolation=cv2.INTER_AREA)
            result = np.zeros((28, 28), dtype=np.float32)
            result[4:24, 4:24] = digit_img.astype(np.float32) / 255.0

            images.append(result)
            labels.append(digit)

    if len(images) == 0:
        return None, None

    return np.array(images), np.array(labels)


def finetune_model():
    """Fine-tune the existing model with collected training data."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {len(gpus)} device(s)")

    # Load training data
    print("\nLoading training data...")
    x_train, y_train = load_training_data()

    if x_train is None or len(x_train) == 0:
        print("ERROR: No training data found!")
        print(f"Please add training samples to {TRAINING_DIR}/[0-9]/")
        return False

    print(f"\nTotal training samples: {len(x_train)}")

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))

    print("\nClass distribution:")
    missing_digits = []
    low_samples = []
    MIN_SAMPLES_PER_DIGIT = 10

    for digit in range(10):
        count = class_counts.get(digit, 0)
        status = ""
        if count == 0:
            missing_digits.append(digit)
            status = " <- FEHLT!"
        elif count < MIN_SAMPLES_PER_DIGIT:
            low_samples.append((digit, count))
            status = f" <- zu wenig (min. {MIN_SAMPLES_PER_DIGIT})"
        print(f"  Ziffer {digit}: {count}{status}")

    # Abort if data is insufficient
    if missing_digits:
        print(f"\n{'='*60}")
        print("ABBRUCH: Folgende Ziffern haben KEINE Trainingsbeispiele:")
        print(f"  {missing_digits}")
        print(f"\nBitte zuerst Samples für alle Ziffern 0-9 sammeln!")
        print(f"{'='*60}")
        return False

    if low_samples:
        print(f"\n{'='*60}")
        print("ABBRUCH: Folgende Ziffern haben zu wenige Samples:")
        for digit, count in low_samples:
            print(f"  Ziffer {digit}: nur {count} (mindestens {MIN_SAMPLES_PER_DIGIT} nötig)")
        print(f"\nBitte mehr Samples sammeln!")
        print(f"{'='*60}")
        return False

    # Check for imbalance
    max_count = max(counts)
    min_count = min(counts)
    if max_count > min_count * 5:
        print(f"\n{'='*60}")
        print("WARNUNG: Stark unausgewogene Verteilung!")
        print(f"  Max: {max_count}, Min: {min_count}")
        print("  Empfehlung: Mehr Samples für unterrepräsentierte Ziffern sammeln")
        print(f"{'='*60}")

    # Reshape for model
    x_train = x_train.reshape(-1, 28, 28, 1)

    # Load existing model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return False

    print(f"\nLoading model from {MODEL_PATH}...")

    # Backup existing model
    if os.path.exists(MODEL_PATH):
        import shutil
        shutil.copy(MODEL_PATH, BACKUP_PATH)
        print(f"Backup saved to {BACKUP_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1
    )

    # Fine-tune with more epochs on small dataset
    print("\nFine-tuning model...")

    # If we have very few samples, use more epochs
    epochs = 20 if len(x_train) < 100 else 10
    batch_size = min(32, len(x_train))

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=max(1, len(x_train) // batch_size * 3),  # 3x augmentation
        verbose=1
    )

    # Save fine-tuned model
    model.save(MODEL_PATH)
    print(f"\nFine-tuned model saved to {MODEL_PATH}")

    # Test on training data (just to verify)
    predictions = model.predict(x_train, verbose=0)
    correct = np.sum(np.argmax(predictions, axis=1) == y_train)
    print(f"Training accuracy: {correct}/{len(y_train)} ({100*correct/len(y_train):.1f}%)")

    return True


def test_recognition():
    """Test the fine-tuned model on some samples."""
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nTesting recognition on training samples...")

    for digit in range(10):
        digit_dir = os.path.join(TRAINING_DIR, str(digit))
        if not os.path.exists(digit_dir):
            continue

        samples = glob.glob(os.path.join(digit_dir, '*.png'))[:3]  # Test first 3

        for sample_path in samples:
            img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Preprocess
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = cv2.findNonZero(binary)
            if coords is None:
                continue

            x, y, w, h = cv2.boundingRect(coords)
            pad = max(w, h) // 4
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(binary.shape[1], x + w + pad), min(binary.shape[0], y + h + pad)

            digit_img = binary[y1:y2, x1:x2]
            if digit_img.size == 0:
                continue

            dh, dw = digit_img.shape
            if dh > dw:
                diff = dh - dw
                digit_img = cv2.copyMakeBorder(digit_img, 0, 0, diff//2, diff-diff//2,
                                                cv2.BORDER_CONSTANT, value=0)
            elif dw > dh:
                diff = dw - dh
                digit_img = cv2.copyMakeBorder(digit_img, diff//2, diff-diff//2, 0, 0,
                                                cv2.BORDER_CONSTANT, value=0)

            digit_img = cv2.resize(digit_img, (20, 20), interpolation=cv2.INTER_AREA)
            result = np.zeros((28, 28), dtype=np.float32)
            result[4:24, 4:24] = digit_img.astype(np.float32) / 255.0

            # Predict
            pred = model.predict(result.reshape(1, 28, 28, 1), verbose=0)[0]
            predicted = np.argmax(pred)
            conf = pred[predicted]

            status = "OK" if predicted == digit else "WRONG"
            print(f"  {os.path.basename(sample_path)}: expected={digit}, got={predicted} ({conf:.2f}) {status}")


if __name__ == '__main__':
    print("=" * 60)
    print("EMNIST Model Fine-Tuning")
    print("=" * 60)

    if '--test' in sys.argv:
        test_recognition()
    else:
        success = finetune_model()
        if success:
            print("\n" + "=" * 60)
            test_recognition()
            print("\nDone! Restart the ml-inference container to use the updated model:")
            print("  docker compose restart ml-inference")
