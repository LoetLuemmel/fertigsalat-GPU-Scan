#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT using Python API.
More memory-efficient than trtexec for Jetson devices.
"""
import os
import tensorrt as trt

ONNX_MODEL = '/app/models/emnist_cnn.onnx'
TRT_ENGINE = '/app/models/emnist_cnn.trt'

def build_engine():
    """Build TensorRT engine from ONNX model."""
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    print(f"Loading ONNX: {ONNX_MODEL}")
    print(f"Output: {TRT_ENGINE}")

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_MODEL, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return False

    print(f"Network inputs: {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024)  # 256 MB

    # Enable FP16 for Jetson (faster inference)
    if builder.platform_has_fast_fp16:
        print("Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    print("Building TensorRT engine (this may take a minute)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False

    # Save engine
    with open(TRT_ENGINE, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine saved: {TRT_ENGINE}")
    print(f"Size: {os.path.getsize(TRT_ENGINE) / 1024 / 1024:.2f} MB")
    return True


def verify_engine():
    """Quick verification of the engine."""
    import numpy as np

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(TRT_ENGINE, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("ERROR: Could not load engine")
        return False

    print(f"\nEngine verification:")
    print(f"  Bindings: {engine.num_bindings}")
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        print(f"  {'Input' if is_input else 'Output'} '{name}': {shape}")

    print("\n✓ Engine verified successfully!")
    return True


if __name__ == '__main__':
    print("=" * 50)
    print("ONNX → TensorRT Converter")
    print("=" * 50)

    if not os.path.exists(ONNX_MODEL):
        print(f"ERROR: ONNX model not found: {ONNX_MODEL}")
        exit(1)

    if build_engine():
        verify_engine()
    else:
        exit(1)
