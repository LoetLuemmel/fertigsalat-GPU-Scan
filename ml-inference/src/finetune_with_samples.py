#!/usr/bin/env python3
"""
Finetune the EMNIST digit model with collected training samples.

Combines MNIST base data with custom samples to prevent catastrophic forgetting.
After finetuning, regenerates the TensorRT engine.

Usage:
    python3 finetune_with_samples.py [--epochs 5] [--lr 0.0001]
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2

# Paths
MODEL_PATH = '/app/models/emnist_cnn.h5'
MODEL_BACKUP = '/app/models/emnist_cnn_backup.h5'
MODEL_FINETUNED = '/app/models/emnist_cnn_finetuned.h5'
TRAINING_DATA_DIR = '/app/training_data'
ONNX_PATH = '/app/models/emnist_cnn.onnx'
TRT_PATH = '/app/models/emnist_cnn.trt'


def load_custom_samples():
    """Load custom training samples from training_data directory."""
    images = []
    labels = []

    print(f"\nLoading custom samples from {TRAINING_DATA_DIR}")

    for digit in range(10):
        digit_dir = os.path.join(TRAINING_DATA_DIR, str(digit))
        if not os.path.exists(digit_dir):
            print(f"  Digit {digit}: 0 samples")
            continue

        samples = glob.glob(os.path.join(digit_dir, '*.png'))
        print(f"  Digit {digit}: {len(samples)} samples")

        for sample_path in samples:
            img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Preprocess to 28x28 (same as training)
            processed = preprocess_sample(img)
            if processed is not None:
                images.append(processed)
                labels.append(digit)

    if len(images) == 0:
        return None, None

    return np.array(images), np.array(labels)


def preprocess_sample(image):
    """Preprocess a sample image to 28x28 normalized format."""
    # Binarize with OTSU (white digit on black background)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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

    # Resize to 20x20 (digits are centered in 28x28 with 4px border)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Center in 28x28
    result = np.zeros((28, 28), dtype=np.float32)
    result[4:24, 4:24] = digit.astype(np.float32) / 255.0

    return result


def load_mnist_subset(samples_per_class=500):
    """Load a subset of MNIST to mix with custom samples."""
    from tensorflow import keras

    print(f"\nLoading MNIST subset ({samples_per_class} per class)")
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255.0

    # Sample subset per class for balanced training
    selected_images = []
    selected_labels = []

    for digit in range(10):
        indices = np.where(y_train == digit)[0]
        if len(indices) > samples_per_class:
            indices = np.random.choice(indices, samples_per_class, replace=False)

        selected_images.extend(x_train[indices])
        selected_labels.extend(y_train[indices])
        print(f"  Digit {digit}: {len(indices)} samples")

    return np.array(selected_images), np.array(selected_labels)


def augment_custom_samples(images, labels, augment_factor=10):
    """Augment custom samples to balance against MNIST."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if len(images) == 0:
        return images, labels

    print(f"\nAugmenting {len(images)} custom samples (factor: {augment_factor})")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1
    )

    augmented_images = list(images)
    augmented_labels = list(labels)

    for img, label in zip(images, labels):
        img_reshaped = img.reshape(1, 28, 28, 1)
        aug_iter = datagen.flow(img_reshaped, batch_size=1)

        for _ in range(augment_factor - 1):
            aug_img = next(aug_iter)[0].reshape(28, 28)
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    print(f"  Total custom samples after augmentation: {len(augmented_images)}")

    return np.array(augmented_images), np.array(augmented_labels)


def finetune_model(epochs=5, learning_rate=0.0001, mnist_samples=500, augment_factor=10):
    """Finetune the model with combined data."""
    import tensorflow as tf
    from tensorflow import keras

    # Suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {len(gpus)} device(s)")

    # Load custom samples
    custom_images, custom_labels = load_custom_samples()

    if custom_images is None or len(custom_images) == 0:
        print("\nERROR: No custom training samples found!")
        print(f"Please add samples to {TRAINING_DATA_DIR}/0/, {TRAINING_DATA_DIR}/1/, etc.")
        return False

    # Check which digits have samples
    unique_digits = np.unique(custom_labels)
    print(f"\nDigits with custom samples: {sorted(unique_digits)}")

    if len(unique_digits) < 3:
        print("WARNING: Less than 3 digits have samples. Consider collecting more data.")

    # Augment custom samples
    custom_images, custom_labels = augment_custom_samples(
        custom_images, custom_labels, augment_factor
    )

    # Load MNIST subset
    mnist_images, mnist_labels = load_mnist_subset(mnist_samples)

    # Combine datasets
    print(f"\nCombining datasets:")
    print(f"  MNIST samples: {len(mnist_images)}")
    print(f"  Custom samples: {len(custom_images)}")

    all_images = np.concatenate([mnist_images, custom_images])
    all_labels = np.concatenate([mnist_labels, custom_labels])

    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]

    # Reshape for model
    all_images = all_images.reshape(-1, 28, 28, 1)

    print(f"  Total training samples: {len(all_images)}")

    # Load existing model
    print(f"\nLoading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return False

    # Backup original model
    if not os.path.exists(MODEL_BACKUP):
        import shutil
        shutil.copy(MODEL_PATH, MODEL_BACKUP)
        print(f"Backup created: {MODEL_BACKUP}")

    model = keras.models.load_model(MODEL_PATH)

    # Compile with low learning rate to prevent catastrophic forgetting
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nFinetuning with lr={learning_rate} for {epochs} epochs...")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]

    # Train
    history = model.fit(
        all_images, all_labels,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Save finetuned model
    model.save(MODEL_FINETUNED)
    print(f"\nFinetuned model saved: {MODEL_FINETUNED}")

    # Replace original with finetuned
    model.save(MODEL_PATH)
    print(f"Updated main model: {MODEL_PATH}")

    # Show final accuracy
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history.get('val_accuracy', [0])[-1]
    print(f"\nFinal accuracy: {final_acc:.4f}")
    print(f"Final val_accuracy: {final_val_acc:.4f}")

    return True


def regenerate_tensorrt():
    """Regenerate TensorRT engine from updated model."""
    print("\n" + "=" * 50)
    print("Regenerating TensorRT engine...")
    print("=" * 50)

    # Remove old ONNX and TRT files
    if os.path.exists(ONNX_PATH):
        os.remove(ONNX_PATH)
        print(f"Removed old ONNX: {ONNX_PATH}")

    if os.path.exists(TRT_PATH):
        os.remove(TRT_PATH)
        print(f"Removed old TRT: {TRT_PATH}")

    # Convert to ONNX
    try:
        import tensorflow as tf
        import tf2onnx

        print("\nConverting to ONNX...")
        model = tf.keras.models.load_model(MODEL_PATH)

        input_signature = [tf.TensorSpec(shape=(1, 28, 28, 1), dtype=tf.float32, name='input')]

        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13,
            output_path=ONNX_PATH
        )
        print(f"ONNX saved: {ONNX_PATH}")

    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        return False

    # Convert to TensorRT
    try:
        import tensorrt as trt

        print("\nConverting to TensorRT...")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(ONNX_PATH, 'rb') as f:
            if not parser.parse(f.read()):
                print("ONNX parsing failed")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024)

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 mode")

        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("TensorRT build failed")
            return False

        with open(TRT_PATH, 'wb') as f:
            f.write(serialized_engine)

        print(f"TensorRT saved: {TRT_PATH}")
        print(f"Size: {os.path.getsize(TRT_PATH) / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"TensorRT conversion failed: {e}")
        return False


def show_sample_stats():
    """Show statistics about collected samples."""
    print("\n" + "=" * 50)
    print("Training Sample Statistics")
    print("=" * 50)

    total = 0
    for digit in range(10):
        digit_dir = os.path.join(TRAINING_DATA_DIR, str(digit))
        if os.path.exists(digit_dir):
            count = len(glob.glob(os.path.join(digit_dir, '*.png')))
        else:
            count = 0

        bar = '█' * count + '░' * (20 - min(count, 20))
        print(f"  {digit}: {bar} {count}")
        total += count

    print(f"\n  Total: {total} samples")

    if total < 30:
        print("\n  Recommendation: Collect more samples (aim for 10+ per digit)")

    return total


def main():
    parser = argparse.ArgumentParser(description='Finetune digit recognition model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--mnist-samples', type=int, default=500, help='MNIST samples per digit (default: 500)')
    parser.add_argument('--augment', type=int, default=10, help='Augmentation factor (default: 10)')
    parser.add_argument('--stats-only', action='store_true', help='Only show sample statistics')
    parser.add_argument('--skip-tensorrt', action='store_true', help='Skip TensorRT regeneration')

    args = parser.parse_args()

    print("=" * 50)
    print("EMNIST Digit Model Finetuning")
    print("=" * 50)

    # Show stats
    total_samples = show_sample_stats()

    if args.stats_only:
        return

    if total_samples == 0:
        print("\nNo samples found. Please collect training data first.")
        print(f"Add images to: {TRAINING_DATA_DIR}/0/, {TRAINING_DATA_DIR}/1/, etc.")
        sys.exit(1)

    # Finetune
    success = finetune_model(
        epochs=args.epochs,
        learning_rate=args.lr,
        mnist_samples=args.mnist_samples,
        augment_factor=args.augment
    )

    if not success:
        sys.exit(1)

    # Regenerate TensorRT
    if not args.skip_tensorrt:
        regenerate_tensorrt()

    print("\n" + "=" * 50)
    print("Finetuning complete!")
    print("=" * 50)
    print("\nRestart the scanner to use the updated model.")


if __name__ == '__main__':
    main()
