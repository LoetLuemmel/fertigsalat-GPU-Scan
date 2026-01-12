#!/usr/bin/env python3
"""
TensorRT-accelerated digit recognition for Jetson.
2-5x faster than Keras/TensorFlow inference.

Usage:
    from tensorrt_digit_ocr import recognize_digit
    digit, confidence = recognize_digit(image)
"""
import os
import cv2
import numpy as np

# Lazy-load TensorRT components
_engine = None
_context = None
_d_input = None
_d_output = None
_stream = None
_output_buffer = None

TRT_ENGINE_PATH = '/app/models/emnist_cnn.trt'
KERAS_FALLBACK_PATH = '/app/models/emnist_cnn.h5'


def _load_engine():
    """Load TensorRT engine and allocate GPU buffers."""
    global _engine, _context, _d_input, _d_output, _stream, _output_buffer

    if _engine is not None:
        return True

    if not os.path.exists(TRT_ENGINE_PATH):
        print(f"TensorRT engine not found: {TRT_ENGINE_PATH}")
        return False

    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        print(f"Loading TensorRT engine: {TRT_ENGINE_PATH}")
        with open(TRT_ENGINE_PATH, 'rb') as f:
            _engine = runtime.deserialize_cuda_engine(f.read())

        if _engine is None:
            print("ERROR: Failed to deserialize engine")
            return False

        _context = _engine.create_execution_context()

        # Allocate GPU buffers
        input_size = 1 * 28 * 28 * 1 * 4  # float32
        output_size = 10 * 4  # 10 classes, float32

        _d_input = cuda.mem_alloc(input_size)
        _d_output = cuda.mem_alloc(output_size)
        _output_buffer = np.zeros(10, dtype=np.float32)
        _stream = cuda.Stream()

        print("TensorRT engine loaded successfully!")
        return True

    except ImportError as e:
        print(f"TensorRT not available: {e}")
        return False
    except Exception as e:
        print(f"Error loading TensorRT engine: {e}")
        return False


def preprocess_for_inference(image):
    """
    Preprocess an image for model input.
    Same preprocessing as emnist_digit_ocr.py.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binarize with OTSU (white digit on black background)
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

    # Resize to 20x20 (digits are centered in 28x28 with 4px border)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Center in 28x28
    result = np.zeros((28, 28), dtype=np.float32)
    result[4:24, 4:24] = digit.astype(np.float32) / 255.0

    return result


def recognize_digit_tensorrt(image, debug=False):
    """
    Recognize a digit using TensorRT-accelerated inference.

    Args:
        image: BGR or grayscale image containing a single digit
        debug: If True, print debug info

    Returns:
        (digit_string, confidence) or (None, 0)
    """
    import pycuda.driver as cuda

    # Preprocess
    processed = preprocess_for_inference(image)
    if processed is None:
        if debug:
            print("  TRT: Could not preprocess image")
        return None, 0

    # Load engine if needed
    if not _load_engine():
        if debug:
            print("  TRT: Engine not available, falling back to Keras")
        return _keras_fallback(image, debug)

    # Prepare input (NHWC format: 1x28x28x1)
    input_data = processed.reshape(1, 28, 28, 1).astype(np.float32)

    # Copy to GPU
    cuda.memcpy_htod_async(_d_input, input_data, _stream)

    # Run inference
    _context.execute_async_v2(
        bindings=[int(_d_input), int(_d_output)],
        stream_handle=_stream.handle
    )

    # Copy result from GPU
    cuda.memcpy_dtoh_async(_output_buffer, _d_output, _stream)
    _stream.synchronize()

    # Get prediction
    digit = int(np.argmax(_output_buffer))
    confidence = float(_output_buffer[digit])

    if debug:
        print(f"  TRT: digit={digit}, confidence={confidence:.3f}")

    return str(digit), confidence


def _keras_fallback(image, debug=False):
    """Fall back to Keras inference if TensorRT is not available."""
    try:
        from emnist_digit_ocr import recognize_digit_emnist
        return recognize_digit_emnist(image, debug)
    except ImportError:
        if debug:
            print("  Keras fallback not available")
        return None, 0


def recognize_digit(image, debug=False):
    """
    Main entry point for digit recognition.
    Uses TensorRT if available, falls back to Keras.
    """
    if os.path.exists(TRT_ENGINE_PATH):
        return recognize_digit_tensorrt(image, debug)
    else:
        return _keras_fallback(image, debug)


def benchmark(num_iterations=100):
    """Benchmark TensorRT vs Keras inference speed."""
    import time

    # Create test image
    test_img = np.random.randint(0, 255, (40, 30), dtype=np.uint8)

    # Warm up
    recognize_digit(test_img)

    # TensorRT timing
    if os.path.exists(TRT_ENGINE_PATH):
        start = time.time()
        for _ in range(num_iterations):
            recognize_digit_tensorrt(test_img)
        trt_time = (time.time() - start) / num_iterations * 1000
        print(f"TensorRT: {trt_time:.2f} ms/inference")
    else:
        print("TensorRT engine not found")
        trt_time = None

    # Keras timing
    try:
        from emnist_digit_ocr import recognize_digit_emnist

        # Warm up
        recognize_digit_emnist(test_img)

        start = time.time()
        for _ in range(num_iterations):
            recognize_digit_emnist(test_img)
        keras_time = (time.time() - start) / num_iterations * 1000
        print(f"Keras:    {keras_time:.2f} ms/inference")

        if trt_time:
            speedup = keras_time / trt_time
            print(f"Speedup:  {speedup:.1f}x")
    except ImportError:
        print("Keras model not available for comparison")


if __name__ == '__main__':
    print("=" * 60)
    print("TensorRT Digit Recognition - Benchmark")
    print("=" * 60)
    benchmark()
