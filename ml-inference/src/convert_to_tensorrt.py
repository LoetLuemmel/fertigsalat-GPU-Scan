#!/usr/bin/env python3
"""
Convert Keras CNN model to TensorRT for faster inference on Jetson.

Usage:
    python3 convert_to_tensorrt.py

Pipeline:
    Keras (.h5) → ONNX → TensorRT Engine (.trt)
"""
import os
import sys
import subprocess

# Paths
MODEL_DIR = '/app/models'
KERAS_MODEL = os.path.join(MODEL_DIR, 'emnist_cnn.h5')
ONNX_MODEL = os.path.join(MODEL_DIR, 'emnist_cnn.onnx')
TRT_ENGINE = os.path.join(MODEL_DIR, 'emnist_cnn.trt')


def convert_keras_to_onnx():
    """Convert Keras model to ONNX format."""
    print("=" * 60)
    print("Step 1: Keras → ONNX")
    print("=" * 60)

    if not os.path.exists(KERAS_MODEL):
        print(f"ERROR: Keras model not found: {KERAS_MODEL}")
        return False

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    import tf2onnx

    print(f"Loading Keras model: {KERAS_MODEL}")
    model = tf.keras.models.load_model(KERAS_MODEL)
    model.summary()

    # Define input signature
    input_signature = [tf.TensorSpec(shape=(1, 28, 28, 1), dtype=tf.float32, name='input')]

    print(f"\nConverting to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=ONNX_MODEL
    )

    print(f"ONNX model saved: {ONNX_MODEL}")
    print(f"Size: {os.path.getsize(ONNX_MODEL) / 1024 / 1024:.2f} MB")
    return True


def convert_onnx_to_tensorrt():
    """Convert ONNX model to TensorRT engine."""
    print("\n" + "=" * 60)
    print("Step 2: ONNX → TensorRT")
    print("=" * 60)

    if not os.path.exists(ONNX_MODEL):
        print(f"ERROR: ONNX model not found: {ONNX_MODEL}")
        return False

    # Use trtexec for conversion (full path for Jetson L4T container)
    trtexec_paths = [
        '/usr/src/tensorrt/bin/trtexec',
        '/usr/local/bin/trtexec',
        'trtexec'
    ]
    trtexec = None
    for path in trtexec_paths:
        if os.path.exists(path):
            trtexec = path
            break
    if trtexec is None:
        trtexec = trtexec_paths[0]  # Try default path

    cmd = [
        trtexec,
        f'--onnx={ONNX_MODEL}',
        f'--saveEngine={TRT_ENGINE}',
        '--fp16',  # Use FP16 for Jetson (faster, minimal accuracy loss)
        '--workspace=256',  # 256 MB workspace
        '--verbose'
    ]

    print(f"Running: {' '.join(cmd)}")
    print("\nThis may take a few minutes...")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("ERROR: trtexec failed")
        return False

    if os.path.exists(TRT_ENGINE):
        print(f"\nTensorRT engine saved: {TRT_ENGINE}")
        print(f"Size: {os.path.getsize(TRT_ENGINE) / 1024 / 1024:.2f} MB")
        return True
    else:
        print("ERROR: TensorRT engine not created")
        return False


def verify_tensorrt_engine():
    """Verify the TensorRT engine works."""
    print("\n" + "=" * 60)
    print("Step 3: Verify TensorRT Engine")
    print("=" * 60)

    if not os.path.exists(TRT_ENGINE):
        print(f"ERROR: TensorRT engine not found: {TRT_ENGINE}")
        return False

    try:
        import tensorrt as trt
        import numpy as np

        # Create logger and runtime
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        # Load engine
        print(f"Loading TensorRT engine: {TRT_ENGINE}")
        with open(TRT_ENGINE, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            print("ERROR: Failed to load engine")
            return False

        print(f"Engine loaded successfully!")
        print(f"  - Bindings: {engine.num_bindings}")

        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            print(f"  - {'Input' if is_input else 'Output'} '{name}': {shape} ({dtype})")

        # Quick inference test
        print("\nRunning test inference...")

        import pycuda.driver as cuda
        import pycuda.autoinit

        context = engine.create_execution_context()

        # Create dummy input (28x28 image)
        input_data = np.random.rand(1, 28, 28, 1).astype(np.float32)
        output_data = np.zeros(10, dtype=np.float32)

        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_output = cuda.mem_alloc(output_data.nbytes)

        # Copy input to GPU
        cuda.memcpy_htod(d_input, input_data)

        # Run inference
        context.execute_v2([int(d_input), int(d_output)])

        # Copy output from GPU
        cuda.memcpy_dtoh(output_data, d_output)

        print(f"Output shape: {output_data.shape}")
        print(f"Predicted digit: {np.argmax(output_data)}")
        print(f"Confidence: {np.max(output_data):.4f}")

        print("\n✓ TensorRT engine verified successfully!")
        return True

    except ImportError as e:
        print(f"WARNING: Could not verify engine (missing module: {e})")
        print("Engine was created, but verification requires pycuda")
        return True
    except Exception as e:
        print(f"ERROR during verification: {e}")
        return False


def main():
    print("=" * 60)
    print("Keras → TensorRT Converter for EMNIST Digit Recognition")
    print("=" * 60)
    print(f"Input:  {KERAS_MODEL}")
    print(f"Output: {TRT_ENGINE}")
    print()

    # Step 1: Keras → ONNX
    if not convert_keras_to_onnx():
        sys.exit(1)

    # Step 2: ONNX → TensorRT
    if not convert_onnx_to_tensorrt():
        sys.exit(1)

    # Step 3: Verify
    verify_tensorrt_engine()

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"TensorRT engine: {TRT_ENGINE}")
    print("\nTo use in inference, import tensorrt_digit_ocr instead of emnist_digit_ocr")


if __name__ == '__main__':
    main()
