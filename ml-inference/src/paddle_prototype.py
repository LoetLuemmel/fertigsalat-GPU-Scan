#!/usr/bin/env python3
"""
Prototype: Test PaddleOCR for handwriting recognition on order forms.
"""

import sys
import os

# First, try to install paddleocr if not present
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Installing PaddleOCR (CPU-only)...")
    os.system("pip install paddlepaddle paddleocr -q")
    from paddleocr import PaddleOCR

import cv2
import numpy as np
from pathlib import Path


def test_paddleocr(image_path: str):
    """Test PaddleOCR on an image."""
    print(f"\n{'='*60}")
    print(f"PaddleOCR Prototype - Handwriting Recognition Test")
    print(f"{'='*60}")

    # Initialize PaddleOCR (German)
    print("\nInitializing PaddleOCR...")
    ocr = PaddleOCR(lang='german')
    print("PaddleOCR initialized.")

    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")

    # Run OCR on full image
    print("\nRunning OCR on full image...")
    result = ocr.ocr(image_path, cls=True)

    if not result or not result[0]:
        print("No text detected!")
        return

    print(f"\nDetected {len(result[0])} text regions:")
    print("-" * 60)

    for i, line in enumerate(result[0][:30]):  # Show first 30
        bbox = line[0]
        text = line[1][0]
        confidence = line[1][1]

        # Calculate position
        x = int(bbox[0][0])
        y = int(bbox[0][1])

        print(f"  [{i+1:2}] ({x:4},{y:4}) conf={confidence:.2f}: '{text}'")

    if len(result[0]) > 30:
        print(f"  ... and {len(result[0]) - 30} more regions")

    # Focus on specific areas (order quantity columns)
    print(f"\n{'='*60}")
    print("Scanning order quantity columns (right side of form)...")
    print(f"{'='*60}")

    # Define column regions based on template (scaled for 2350px width)
    # Original template: 1241px, Current image: ~2350px
    scale = w / 1241.0

    columns = {
        'Mo': (int(468 * scale), int(577 * scale)),
        'Di': (int(577 * scale), int(686 * scale)),
        'Mi': (int(686 * scale), int(795 * scale)),
        'Do': (int(795 * scale), int(904 * scale)),
        'Fr': (int(904 * scale), int(1013 * scale)),
        'Sa': (int(1013 * scale), int(1086 * scale)),
    }

    print(f"\nScale factor: {scale:.2f}")
    print("Column positions:")
    for name, (x_start, x_end) in columns.items():
        print(f"  {name}: x={x_start}-{x_end}")

    # Find text in each column
    print("\nText found in columns:")
    for col_name, (x_start, x_end) in columns.items():
        col_texts = []
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]

            # Check if text center is in this column
            text_x = (bbox[0][0] + bbox[2][0]) / 2
            if x_start <= text_x <= x_end:
                text_y = (bbox[0][1] + bbox[2][1]) / 2
                col_texts.append((text_y, text, conf))

        if col_texts:
            col_texts.sort(key=lambda x: x[0])  # Sort by Y position
            print(f"\n  {col_name}:")
            for y, text, conf in col_texts[:10]:
                print(f"    y={int(y):4}: '{text}' (conf={conf:.2f})")


def test_region(image_path: str, x: int, y: int, w: int, h: int):
    """Test OCR on a specific region."""
    print(f"\n{'='*60}")
    print(f"Testing specific region: x={x}, y={y}, w={w}, h={h}")
    print(f"{'='*60}")

    # Initialize PaddleOCR
    ocr = PaddleOCR(lang='german')

    # Load and crop image
    image = cv2.imread(image_path)
    region = image[y:y+h, x:x+w]

    # Save temp file
    temp_path = "/tmp/test_region.png"
    cv2.imwrite(temp_path, region)

    # Run OCR
    result = ocr.ocr(temp_path, cls=True)

    if result and result[0]:
        print(f"Found {len(result[0])} text regions:")
        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            print(f"  '{text}' (conf={conf:.2f})")
    else:
        print("No text detected in region.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default: use latest processed image
        import glob
        images = glob.glob('/app/intermediate/*/page_000_normalized.png')
        if images:
            images.sort(key=os.path.getmtime, reverse=True)
            image_path = images[0]
        else:
            print("Usage: python paddle_prototype.py <image_path>")
            sys.exit(1)
    else:
        image_path = sys.argv[1]

    test_paddleocr(image_path)
