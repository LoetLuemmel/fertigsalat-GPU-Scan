#!/usr/bin/env python3
"""
Backend server for interactive order scanner.
Provides APIs for scanning, image serving, and training data collection.
"""
import http.server
import json
import os
import sys
import glob
import urllib.parse
import base64
import re
import email
from email import policy
from pathlib import Path
import subprocess
import tempfile
import cgi
import cv2
import numpy as np

# Configuration
PORT = 8080
TEMPLATES_DIR = '/app/templates'
INTERMEDIATE_DIR = '/app/intermediate'
TRAINING_DIR = '/app/training_data'
SAVED_ZONES_FILE = '/app/templates/saved_zones.json'

# Ensure directories exist
os.makedirs(TRAINING_DIR, exist_ok=True)

# Lazy-loaded OCR engine for orientation detection
_ocr_engine = None

def get_ocr_engine():
    """Get or initialize the OCR engine (lazy loading)."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from ocr_engine import OCREngine
            _ocr_engine = OCREngine(langs=['de', 'en'], use_gpu=True)
            print("OCR engine initialized for orientation detection")
        except Exception as e:
            print(f"Warning: Could not initialize OCR engine: {e}")
    return _ocr_engine


# Lazy-loaded EMNIST model for digit recognition
_emnist_model = None

def get_emnist_model():
    """Get or initialize the EMNIST digit recognition model (lazy loading)."""
    global _emnist_model
    if _emnist_model is None:
        try:
            # Suppress TensorFlow warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf

            # Configure GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"EMNIST: Using GPU ({len(gpus)} device(s))")

            model_path = '/app/models/emnist_cnn.h5'
            if os.path.exists(model_path):
                _emnist_model = tf.keras.models.load_model(model_path)
                print(f"EMNIST model loaded from {model_path}")
            else:
                print(f"Warning: EMNIST model not found at {model_path}")
        except Exception as e:
            print(f"Warning: Could not load EMNIST model: {e}")
    return _emnist_model


def recognize_digit_emnist(image):
    """
    Recognize a digit using the EMNIST-trained CNN.
    Runs in the main process to avoid TensorFlow subprocess issues.

    Args:
        image: BGR or grayscale image containing a single digit

    Returns:
        (digit_string, confidence) or (None, 0)
    """
    model = get_emnist_model()
    if model is None:
        return None, 0

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
        return None, 0

    x, y, w, h = cv2.boundingRect(coords)

    # Extract digit region with padding
    pad = max(w, h) // 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + w + pad)
    y2 = min(binary.shape[0], y + h + pad)

    digit = binary[y1:y2, x1:x2]

    if digit.size == 0:
        return None, 0

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

    # Predict
    input_data = result.reshape(1, 28, 28, 1)
    predictions = model.predict(input_data, verbose=0)[0]

    digit_val = int(np.argmax(predictions))
    confidence = float(predictions[digit_val])

    return str(digit_val), confidence


def deskew(image: np.ndarray, max_angle: float = 10.0):
    """
    Correct small skew/rotation in document image using Hough line detection.

    Args:
        image: Input image (BGR or grayscale)
        max_angle: Maximum rotation angle to apply (default 10°)

    Returns:
        Tuple of (deskewed image, rotation angle in degrees)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        print("Deskew skipped: no lines detected")
        return image, 0.0

    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # Only consider near-horizontal lines (within ±45° of horizontal)
        if abs(angle) < 45:
            angles.append(angle)
        # Also check near-vertical lines (should be close to ±90°)
        elif abs(abs(angle) - 90) < 45:
            # Convert to deviation from vertical
            angles.append(angle - 90 if angle > 0 else angle + 90)

    if len(angles) == 0:
        print("Deskew skipped: no suitable lines found")
        return image, 0.0

    # Use median angle to be robust against outliers
    median_angle = np.median(angles)

    # Skip very small angles
    if abs(median_angle) < 0.3:
        print(f"Deskew skipped: angle {median_angle:.2f}° too small")
        return image, 0.0

    # Skip large angles - likely false detections
    if abs(median_angle) > max_angle:
        print(f"Deskew skipped: angle {median_angle:.2f}° exceeds max {max_angle}°")
        return image, 0.0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Calculate new image size to avoid cutting corners
    cos_a = abs(np.cos(np.radians(median_angle)))
    sin_a = abs(np.sin(np.radians(median_angle)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust rotation matrix for new size
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    print(f"Deskew applied: {median_angle:.2f}° (detected {len(angles)} lines)")
    return rotated, median_angle


def parse_eml_attachments(eml_content):
    """
    Parse EML content and extract attachments.
    Returns list of attachments with metadata and base64 data.
    """
    msg = email.message_from_bytes(eml_content, policy=policy.default)

    attachments = []
    metadata = {
        'subject': msg.get('subject', ''),
        'from': msg.get('from', ''),
        'date': msg.get('date', '')
    }

    for part in msg.walk():
        content_type = part.get_content_type()
        filename = part.get_filename()

        if filename is None:
            continue

        print(f"Found attachment: '{filename}' (type: {content_type})")

        payload = part.get_payload(decode=True)
        if payload is None:
            print(f"  -> Skipped: no payload")
            continue

        # Determine if it's an image or PDF
        is_image = content_type.startswith('image/')
        is_pdf = content_type == 'application/pdf' or filename.lower().endswith('.pdf')

        print(f"  -> is_image={is_image}, is_pdf={is_pdf}")

        if is_image or is_pdf:
            attachment = {
                'filename': filename,
                'content_type': content_type,
                'size': len(payload),
                'is_pdf': is_pdf
            }

            if is_image:
                # Save for scanner and get corrected bytes
                corrected_bytes = save_image_for_scanner(payload, filename)
                attachment['data'] = f"data:image/png;base64,{base64.b64encode(corrected_bytes).decode('utf-8')}"
            elif is_pdf:
                # Convert PDF to image using pdftoppm (also saves for scanner)
                attachment['data'] = convert_pdf_to_image(payload, filename)

            # Add processing info to attachment
            attachment['processing'] = get_last_processing_info()
            attachments.append(attachment)

    return {'metadata': metadata, 'attachments': attachments}


def convert_pdf_to_image(pdf_bytes, filename='attachment'):
    """Convert PDF bytes to PNG image using pdftoppm.
    Saves the image to intermediate directory for scanner access.
    Returns base64 data URL for browser display.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, 'input.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)

        # Convert PDF to PNG using pdftoppm (300 DPI)
        output_prefix = os.path.join(tmpdir, 'page')
        try:
            subprocess.run(
                ['pdftoppm', '-png', '-r', '300', pdf_path, output_prefix],
                capture_output=True,
                check=True,
                timeout=30
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Fallback: try using ImageMagick convert
            try:
                output_path = os.path.join(tmpdir, 'page.png')
                subprocess.run(
                    ['convert', '-density', '300', f'{pdf_path}[0]', output_path],
                    capture_output=True,
                    check=True,
                    timeout=30
                )
                with open(output_path, 'rb') as f:
                    img_bytes = f.read()
                # Save to intermediate directory and get corrected bytes
                corrected_bytes = save_image_for_scanner(img_bytes, filename)
                return f"data:image/png;base64,{base64.b64encode(corrected_bytes).decode('utf-8')}"
            except Exception:
                return None

        # Find generated PNG file (first page)
        png_files = glob.glob(os.path.join(tmpdir, 'page*.png'))
        if png_files:
            png_files.sort()
            with open(png_files[0], 'rb') as f:
                img_bytes = f.read()
            # Save to intermediate directory and get corrected bytes
            corrected_bytes = save_image_for_scanner(img_bytes, filename)
            return f"data:image/png;base64,{base64.b64encode(corrected_bytes).decode('utf-8')}"

    return None


# Global variable to store last processing info for status updates
_last_processing_info = {
    'orientation_angle': 0,
    'deskew_angle': 0,
    'orientation_corrected': False,
    'deskew_applied': False
}

def get_last_processing_info():
    """Get info about the last image processing."""
    return _last_processing_info.copy()

def save_image_for_scanner(img_bytes, filename='attachment'):
    """Save image to intermediate directory so scanner can find it.
    Automatically detects and corrects document orientation using OCR.
    Returns the corrected image bytes for browser display.
    """
    global _last_processing_info
    _last_processing_info = {
        'orientation_angle': 0,
        'deskew_angle': 0,
        'orientation_corrected': False,
        'deskew_applied': False
    }

    import time
    # Create a subdirectory with timestamp
    timestamp = int(time.time() * 1000)
    safe_filename = re.sub(r'[^\w\-.]', '_', filename)
    subdir = os.path.join(INTERMEDIATE_DIR, f'eml_{safe_filename}_{timestamp}')
    os.makedirs(subdir, exist_ok=True)

    output_path = os.path.join(subdir, 'page_000_normalized.png')

    # Convert bytes to numpy array for processing
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    corrected_bytes = img_bytes  # Default: return original

    if image is not None:
        # Detect and correct orientation using OCR
        # DISABLED: OCR orientation detection uses too much memory on Jetson
        # TODO: Re-enable with memory optimization or lower resolution
        # ocr = get_ocr_engine()
        # if ocr is not None:
        #     print("Detecting document orientation...")
        #     corrected, angle = ocr.correct_orientation(image)
        #     if angle != 0:
        #         print(f"Orientation corrected by {angle}°")
        #         image = corrected
        #         _last_processing_info['orientation_angle'] = angle
        #         _last_processing_info['orientation_corrected'] = True
        #     else:
        #         print("Document orientation is correct")
        print("Skipping OCR orientation detection (memory optimization)")

        # Apply deskew to correct small rotations
        image, deskew_angle = deskew(image)
        if deskew_angle != 0:
            _last_processing_info['deskew_angle'] = round(deskew_angle, 2)
            _last_processing_info['deskew_applied'] = True

        # Save the (possibly corrected) image
        cv2.imwrite(output_path, image)
        # Encode corrected image back to PNG bytes
        _, corrected_bytes = cv2.imencode('.png', image)
        corrected_bytes = corrected_bytes.tobytes()
    else:
        # Fallback: save raw bytes if decoding failed
        with open(output_path, 'wb') as f:
            f.write(img_bytes)

    print(f"Saved image for scanner: {output_path}")
    return corrected_bytes


def find_latest_image():
    """Find the most recent normalized image."""
    patterns = [
        f'{INTERMEDIATE_DIR}/*/page_000_normalized.png',
        f'{INTERMEDIATE_DIR}/*/page_*.png'
    ]
    for pattern in patterns:
        images = glob.glob(pattern)
        if images:
            images.sort(key=os.path.getmtime, reverse=True)
            return images[0]
    return None


def read_product_codes_from_image(zone_codes, num_rows, image_width, image_height, zone_orders=None):
    """Read product codes from one or more zones in the latest image.

    Args:
        zone_codes: Single zone dict or array of zone dicts
    """
    image_path = find_latest_image()
    if not image_path:
        return {'error': 'No image found', 'codes': {}}

    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Could not load image', 'codes': {}}

    h, w = img.shape[:2]
    scale_x = w / image_width
    scale_y = h / image_height

    # Normalize zone_codes to list
    if isinstance(zone_codes, list):
        zones_list = zone_codes
    else:
        zones_list = [zone_codes]

    # Run OCR
    ocr = get_ocr_engine()
    if ocr is None or ocr.reader is None:
        return {'error': 'OCR not available', 'codes': {}}

    codes = {}
    os.makedirs(f'{TEMPLATES_DIR}/debug_cells', exist_ok=True)

    # Process each codes zone
    for zone_idx, zone in enumerate(zones_list):
        # Scale zone coordinates
        codes_x = int(zone['x'] * scale_x)
        codes_y = int(zone['y'] * scale_y)
        codes_w = int(zone['width'] * scale_x)
        codes_h = int(zone['height'] * scale_y)

        # Use orders zone for row calculation reference (if provided)
        if zone_orders:
            orders_y = int(zone_orders['y'] * scale_y)
            orders_h = int(zone_orders['height'] * scale_y)
            row_height = orders_h / num_rows
            y_offset = codes_y - orders_y  # Offset between zones
        else:
            row_height = codes_h / num_rows
            y_offset = 0

        # Extract zone
        codes_img = img[codes_y:codes_y+codes_h, codes_x:codes_x+codes_w]
        if codes_img.size == 0:
            print(f"Zone {zone_idx+1}: Empty, skipping")
            continue

        # Save for OCR
        codes_img_path = f'{TEMPLATES_DIR}/debug_cells/product_codes_zone_{zone_idx+1}.png'
        cv2.imwrite(codes_img_path, codes_img)

        try:
            print(f"Reading product codes from zone {zone_idx+1}: {codes_w}x{codes_h}, y_offset={y_offset}...")
            results = ocr.reader.readtext(codes_img)

            for bbox, text, conf in results:
                # Y center relative to codes zone, then adjust for offset to orders zone
                y_center_in_zone = (bbox[0][1] + bbox[2][1]) / 2
                y_center_in_orders = y_center_in_zone + y_offset
                row = int(y_center_in_orders / row_height)
                if 0 <= row < num_rows:
                    # Keep digits, hyphens, parentheses
                    clean_text = re.sub(r'[^0-9\-\(\)]', '', text.strip())
                    if clean_text and row not in codes:
                        codes[row] = clean_text
                        print(f"  Row {row}: '{clean_text}' (zone {zone_idx+1}, y={y_center_in_zone:.0f})")

        except Exception as e:
            print(f"OCR error in zone {zone_idx+1}: {e}")

    print(f"Found {len(codes)} product codes total from {len(zones_list)} zone(s)")
    return {'codes': codes}


def read_product_names_from_image(zone_names, num_rows, image_width, image_height, zone_orders=None):
    """Read product names from a zone in the latest image."""
    image_path = find_latest_image()
    if not image_path:
        return {'error': 'No image found', 'names': {}}

    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Could not load image', 'names': {}}

    h, w = img.shape[:2]
    scale_x = w / image_width
    scale_y = h / image_height

    # Scale zone coordinates
    names_x = int(zone_names['x'] * scale_x)
    names_y = int(zone_names['y'] * scale_y)
    names_w = int(zone_names['width'] * scale_x)
    names_h = int(zone_names['height'] * scale_y)

    # Use orders zone for row calculation reference (if provided)
    if zone_orders:
        orders_y = int(zone_orders['y'] * scale_y)
        orders_h = int(zone_orders['height'] * scale_y)
        row_height = orders_h / num_rows
        y_offset = names_y - orders_y  # Offset between zones
    else:
        row_height = names_h / num_rows
        y_offset = 0

    # Extract zone
    names_img = img[names_y:names_y+names_h, names_x:names_x+names_w]
    if names_img.size == 0:
        return {'error': 'Empty zone', 'names': {}}

    # Save for OCR
    names_img_path = f'{TEMPLATES_DIR}/debug_cells/product_names_zone.png'
    os.makedirs(os.path.dirname(names_img_path), exist_ok=True)
    cv2.imwrite(names_img_path, names_img)

    # Run OCR
    ocr = get_ocr_engine()
    if ocr is None or ocr.reader is None:
        return {'error': 'OCR not available', 'names': {}}

    try:
        print(f"Reading product names from zone {names_w}x{names_h}, y_offset={y_offset}...")
        results = ocr.reader.readtext(names_img)

        names = {}

        for bbox, text, conf in results:
            # Y center relative to names zone, then adjust for offset to orders zone
            y_center_in_zone = (bbox[0][1] + bbox[2][1]) / 2
            y_center_in_orders = y_center_in_zone + y_offset
            row = int(y_center_in_orders / row_height)
            if 0 <= row < num_rows:
                clean_text = ' '.join(text.strip().split())
                if clean_text and row not in names:
                    names[row] = clean_text
                    print(f"  Row {row}: '{clean_text}' (y_center={y_center_in_zone:.0f})")

        print(f"Found {len(names)} product names")
        return {'names': names}
    except Exception as e:
        print(f"OCR error: {e}")
        return {'error': str(e), 'names': {}}


def detect_order_zone(image_path=None):
    """
    Automatically detect the order quantity zone in a form using OCR anchors.

    Uses OCR to find specific text anchors:
    - Top-left: "Monatssalat" - zone starts right after this cell
    - Bottom-right: "(210-212)" - zone ends at left edge of this column

    Returns:
        Dictionary with zone coordinates and detected info, or None if detection fails
    """
    if image_path is None:
        image_path = find_latest_image()

    if image_path is None or not os.path.exists(image_path):
        return {'error': 'No image found'}

    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Could not load image'}

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Get OCR engine for text detection
    ocr = get_ocr_engine()
    if ocr is None or ocr.reader is None:
        print("OCR not available, using fallback")
        return _fallback_zone_detection(img, w, h)

    # Run OCR on the image
    print("Running OCR for anchor detection...")
    try:
        result = ocr.reader.readtext(img)
    except Exception as e:
        print(f"OCR error: {e}")
        return _fallback_zone_detection(img, w, h)

    if not result:
        print("No text detected, using fallback")
        return _fallback_zone_detection(img, w, h)

    # Search for anchor texts
    monatssalat_box = None
    code_210_212_box = None

    for detection in result:
        bbox, text, confidence = detection
        text_lower = text.lower().strip()

        # Find "Monatssalat" (top-left anchor)
        if 'monatssalat' in text_lower or 'monats' in text_lower:
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            monatssalat_box = {
                'x': int(min(x_coords)),
                'y': int(min(y_coords)),
                'x2': int(max(x_coords)),
                'y2': int(max(y_coords)),
                'text': text
            }
            print(f"Found anchor 'Monatssalat': {monatssalat_box}")

        # Find "(210-212)" or similar code pattern (bottom-right anchor)
        if '210' in text and '212' in text:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            code_210_212_box = {
                'x': int(min(x_coords)),
                'y': int(min(y_coords)),
                'x2': int(max(x_coords)),
                'y2': int(max(y_coords)),
                'text': text
            }
            print(f"Found anchor '(210-212)': {code_210_212_box}")

    # Detect vertical lines to find cell boundaries
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    v_contours, _ = cv2.findContours(v_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    v_lines = []
    for cnt in v_contours:
        x, y, vw, vh = cv2.boundingRect(cnt)
        if vh > h * 0.15:  # Line must be at least 15% of image height
            v_lines.append(x)
    v_lines.sort()
    print(f"Found {len(v_lines)} vertical lines for cell boundary detection")

    # Calculate zone from anchors
    if monatssalat_box and code_210_212_box:
        # Find the vertical line that is the RIGHT EDGE of the Monatssalat cell
        # This is the first vertical line to the right of the text
        zone_x = monatssalat_box['x2'] + 5  # Default: right of text
        for vline_x in v_lines:
            if vline_x > monatssalat_box['x2']:
                zone_x = vline_x + 2  # Start just after the cell boundary line
                print(f"Found cell boundary at x={vline_x}")
                break

        zone_y = monatssalat_box['y']
        zone_y2 = code_210_212_box['y2']
        zone_height = zone_y2 - zone_y

        # Find the right edge of orders zone using column width calculation
        # Look for vertical lines after zone_x to determine column width
        order_columns_raw = []
        for vline_x in v_lines:
            if vline_x > zone_x and vline_x < w * 0.95:  # Right of zone_x, but not at edge
                order_columns_raw.append(vline_x)
        order_columns_raw.sort()

        # Remove duplicate lines (lines within 15px of each other)
        order_columns = []
        for x in order_columns_raw:
            if not order_columns or x - order_columns[-1] > 15:
                order_columns.append(x)

        print(f"Found {len(order_columns)} vertical lines after zone_x={zone_x} (deduped from {len(order_columns_raw)}): {order_columns[:10]}")

        # Calculate column width from zone_x to first line, then use 5 columns (Mo-Fr)
        NUM_ORDER_COLUMNS = 5  # Mo, Di, Mi, Do, Fr (ohne Sa)
        if len(order_columns) >= 2:
            # Column width = distance between first two lines (Mo-Di gap, more reliable than Sa-Mo)
            # The first column after Monatssalat is Sa, we want to skip it
            sa_column_width = order_columns[0] - zone_x  # Width of Sa column
            column_width = order_columns[1] - order_columns[0]  # Width of Mo column (more representative)
            print(f"Sa column width: {sa_column_width}px, Mo column width: {column_width}px")

            # Save original zone_x for names zone boundary (before Sa column)
            names_zone_right_boundary = zone_x - 5

            # Skip Sa column - shift zone_x to start of Mo
            zone_x = order_columns[0] + 2  # Start right after the Sa-Mo boundary line
            print(f"Skipped Sa column, new zone_x={zone_x} (after line at {order_columns[0]})")

            zone_x2 = zone_x + (NUM_ORDER_COLUMNS * column_width)
            print(f"Orders zone: x={zone_x} to x2={zone_x2} ({NUM_ORDER_COLUMNS} columns)")
        elif len(order_columns) >= 1:
            # Fallback: use first column width
            column_width = order_columns[0] - zone_x
            print(f"Calculated column width (fallback): {column_width}px")

            names_zone_right_boundary = zone_x - 5
            zone_x = zone_x + column_width
            print(f"Skipped Sa column, new zone_x={zone_x}")

            zone_x2 = zone_x + (NUM_ORDER_COLUMNS * column_width)
            print(f"Orders zone: x={zone_x} to x2={zone_x2} ({NUM_ORDER_COLUMNS} columns)")
        else:
            # Fallback: estimate based on typical form width
            names_zone_right_boundary = zone_x - 5
            zone_x2 = zone_x + int(w * 0.45)
            print(f"No vertical lines found, using estimated zone_x2={zone_x2}")

        zone_width = zone_x2 - zone_x

        # Product codes zone: LEFT side of form (first column with product numbers)
        # Find leftmost vertical line as the left edge of the codes zone
        codes_zone_x = 50  # Default: near left edge
        codes_zone_width = 120  # Width for product codes column

        # Find the first two vertical lines to determine codes zone boundaries
        left_lines_raw = [x for x in v_lines if x < monatssalat_box['x'] - 50]
        # Deduplicate lines that are too close together (within 15px)
        left_lines = []
        for x in left_lines_raw:
            if not left_lines or x - left_lines[-1] > 15:
                left_lines.append(x)

        if len(left_lines) >= 2:
            codes_zone_x = left_lines[0] + 2
            codes_zone_width = max(80, left_lines[1] - left_lines[0] - 4)  # Minimum 80px width
            print(f"Found codes zone boundaries: left={left_lines[0]}, right={left_lines[1]}")
        elif len(left_lines) >= 1:
            codes_zone_x = left_lines[0] + 2
            print(f"Found codes zone left boundary: {left_lines[0]}")

        codes_zone = {
            'x': codes_zone_x,
            'y': zone_y,
            'width': codes_zone_width,
            'height': zone_height
        }
        print(f"Product codes zone (LEFT): x={codes_zone['x']}, y={codes_zone['y']}, w={codes_zone['width']}, h={codes_zone['height']}")

        # Product names zone: between codes zone and orders zone
        # Left boundary = right edge of codes zone
        # Right boundary = start of orders zone (names_zone_right_boundary)
        names_zone_x = codes_zone_x + codes_zone_width + 5
        names_zone_x2 = names_zone_right_boundary

        # Ensure names zone starts at or after the Monatssalat text position
        if names_zone_x > monatssalat_box['x']:
            names_zone_x = monatssalat_box['x'] - 5

        print(f"Product names zone: left={names_zone_x}, right={names_zone_x2} (Monatssalat at x={monatssalat_box['x']})")

        names_zone = {
            'x': names_zone_x,
            'y': zone_y,
            'width': names_zone_x2 - names_zone_x,
            'height': zone_height
        }
        print(f"Product names zone: x={names_zone['x']}, y={names_zone['y']}, w={names_zone['width']}, h={names_zone['height']}")

        # Count rows based on typical row height (approximately 50-60 pixels)
        avg_row_height = 53  # Typical for these forms
        num_rows = max(1, int(zone_height / avg_row_height))

        print(f"OCR-detected zone: x={zone_x}, y={zone_y}, w={zone_width}, h={zone_height}")
        print(f"Estimated {num_rows} rows")

        # Saturday zone: right of Mo-Fr, only for products 600-608 (Sandwiches section)
        # Find "Sandwich" or "600" in OCR results to determine Y position
        sandwich_box = None
        for detection in result:
            bbox, text, confidence = detection
            text_lower = text.lower().strip()
            # Look for "Sandwich" which marks the start of 600-series products
            if 'sandwich' in text_lower and 'crustino' not in text_lower:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                sandwich_box = {
                    'x': int(min(x_coords)),
                    'y': int(min(y_coords)),
                    'text': text
                }
                print(f"Found 'Sandwich' section at y={sandwich_box['y']}")
                break

        # Calculate Saturday zone
        sa_zone = None
        if sandwich_box:
            # Sa column is right after Fr (zone_x2 is end of Fr)
            sa_zone_x = zone_x2 + 2
            sa_zone_width = int((zone_x2 - zone_x) / 5)  # Same width as one Mo-Fr column

            # Y range: from "Sandwich" row to end of 600-series (approx 9 rows: 600-608)
            sa_zone_y = sandwich_box['y'] - 5
            sa_zone_height = avg_row_height * 9  # 9 products (600-608)

            sa_zone = {
                'x': sa_zone_x,
                'y': sa_zone_y,
                'width': sa_zone_width,
                'height': sa_zone_height
            }
            print(f"Saturday zone: x={sa_zone['x']}, y={sa_zone['y']}, w={sa_zone['width']}, h={sa_zone['height']}")

        return {
            'zone': {
                'x': zone_x,
                'y': zone_y,
                'width': zone_width,
                'height': zone_height
            },
            'zoneSaturday': sa_zone,  # Second orders zone for Saturday
            'zoneCodes': codes_zone,
            'zoneNames': names_zone,
            'numRows': num_rows,
            'numCols': 5,
            'imageWidth': w,
            'imageHeight': h,
            'debug': {
                'method': 'ocr_anchors',
                'anchor_top': monatssalat_box['text'],
                'anchor_bottom': code_210_212_box['text'],
                'cell_boundary_x': zone_x,
                'sandwich_found': sandwich_box is not None
            }
        }

    # Partial anchor detection - use what we found
    if monatssalat_box:
        print("Only found Monatssalat anchor, using partial detection")
        # Find cell boundary
        zone_x = monatssalat_box['x2'] + 5
        for vline_x in v_lines:
            if vline_x > monatssalat_box['x2']:
                zone_x = vline_x + 2
                break

        zone_y = monatssalat_box['y']
        zone_width = int(w * 0.35)  # Estimate
        zone_height = int(h * 0.85) - zone_y

        # Product codes zone: LEFT side of form
        left_lines = [x for x in v_lines if x < monatssalat_box['x'] - 50]
        codes_zone_x = 50
        codes_zone_width = 120
        if len(left_lines) >= 2:
            codes_zone_x = left_lines[0] + 2
            codes_zone_width = left_lines[1] - left_lines[0] - 4
        elif len(left_lines) >= 1:
            codes_zone_x = left_lines[0] + 2

        codes_zone = {
            'x': codes_zone_x,
            'y': zone_y,
            'width': codes_zone_width,
            'height': zone_height
        }

        # Product names zone: between codes and orders
        names_zone_x = codes_zone_x + codes_zone_width + 5
        if names_zone_x > monatssalat_box['x']:
            names_zone_x = monatssalat_box['x'] - 5

        names_zone = {
            'x': names_zone_x,
            'y': zone_y,
            'width': zone_x - names_zone_x - 5,
            'height': zone_height
        }

        return {
            'zone': {
                'x': zone_x,
                'y': zone_y,
                'width': zone_width,
                'height': zone_height
            },
            'zoneCodes': codes_zone,
            'zoneNames': names_zone,
            'numRows': 52,
            'numCols': 5,
            'imageWidth': w,
            'imageHeight': h,
            'debug': {
                'method': 'partial_ocr',
                'anchor_found': 'monatssalat'
            }
        }

    print("Anchors not found, using fallback line detection")
    return _fallback_zone_detection(img, w, h)


def _fallback_zone_detection(img, w, h):
    """Fallback zone detection using line detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Find vertical line positions
    v_contours, _ = cv2.findContours(v_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_lines = []
    for cnt in v_contours:
        x, y, vw, vh = cv2.boundingRect(cnt)
        if vh > h * 0.15:
            v_lines.append({'x': x, 'y': y, 'height': vh})
    v_lines.sort(key=lambda l: l['x'])

    # Find horizontal line positions
    h_contours, _ = cv2.findContours(h_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_lines = []
    for cnt in h_contours:
        x, y, hw, hh = cv2.boundingRect(cnt)
        if hw > w * 0.15:
            h_lines.append({'x': x, 'y': y, 'width': hw})
    h_lines.sort(key=lambda l: l['y'])

    print(f"Fallback line detection: {len(v_lines)} vertical, {len(h_lines)} horizontal")

    # Use right portion of detected lines
    right_threshold = w * 0.4
    right_v_lines = [l for l in v_lines if l['x'] > right_threshold]
    if len(right_v_lines) < 2:
        right_v_lines = v_lines[-6:] if len(v_lines) >= 6 else v_lines

    zone_x = right_v_lines[0]['x'] if right_v_lines else int(w * 0.55)
    zone_width = w - zone_x - 10
    zone_y = h_lines[0]['y'] if h_lines else int(h * 0.08)
    zone_height = (h_lines[-1]['y'] - h_lines[0]['y']) if len(h_lines) > 1 else int(h * 0.85)

    num_rows = len(h_lines) - 1 if len(h_lines) > 1 else 52
    num_cols = len(right_v_lines) - 1 if len(right_v_lines) > 1 else 5
    num_rows = max(10, min(100, num_rows))
    num_cols = max(1, min(10, num_cols))

    # Product codes zone: LEFT side of form
    left_v_lines = [l for l in v_lines if l['x'] < w * 0.15]
    codes_zone_x = left_v_lines[0]['x'] + 2 if left_v_lines else 50
    codes_zone_width = 120
    if len(left_v_lines) >= 2:
        codes_zone_width = left_v_lines[1]['x'] - left_v_lines[0]['x'] - 4

    codes_zone = {
        'x': codes_zone_x,
        'y': zone_y,
        'width': codes_zone_width,
        'height': zone_height
    }

    # Product names zone: between codes and orders
    names_zone_x = codes_zone_x + codes_zone_width + 5
    names_zone = {
        'x': names_zone_x,
        'y': zone_y,
        'width': zone_x - names_zone_x - 5,
        'height': zone_height
    }

    return {
        'zone': {
            'x': zone_x,
            'y': zone_y,
            'width': zone_width,
            'height': zone_height
        },
        'zoneCodes': codes_zone,
        'zoneNames': names_zone,
        'numRows': num_rows,
        'numCols': num_cols,
        'imageWidth': w,
        'imageHeight': h,
        'debug': {
            'method': 'line_detection',
            'totalVerticalLines': len(v_lines),
            'totalHorizontalLines': len(h_lines)
        }
    }


def clean_product_name(name):
    """Clean up product name by removing OCR artifacts from form edges."""
    if not name:
        return name
    # Remove leading "[" (form edge artifact) and any space after it
    if name.startswith('['):
        name = name[1:].lstrip()
    # Remove leading "]" as well
    if name.startswith(']'):
        name = name[1:].lstrip()
    return name


def scan_row_based(zone_orders, zone_names, zone_codes, image_width, image_height):
    """
    Row-based scanning strategy:
    1. Find order quantities (handwritten digits) in orders zone(s)
    2. For each found order: scan horizontally left for product name
    3. For each found order: scan horizontally right for product code

    This ensures correct row alignment based on the actual position of orders.
    zone_orders can be a single zone dict or an array of zone dicts (Mo-Fr + Sa)
    """
    image_path = find_latest_image()
    if not image_path:
        return {'error': 'No image found'}

    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Could not load image'}

    h, w = img.shape[:2]
    scale_x = w / image_width
    scale_y = h / image_height

    # Handle zone_orders as array (multiple zones) or single object (legacy)
    orders_zones = []
    if isinstance(zone_orders, list):
        # New format: array of zones
        for i, zo in enumerate(zone_orders):
            # First zone (Mo-Fr) has 5 columns, second zone (Sa) has 1 column
            num_cols = 5 if i == 0 else 1
            col_names = ['Mo', 'Di', 'Mi', 'Do', 'Fr'] if i == 0 else ['Sa']
            orders_zones.append({
                'x': int(zo['x'] * scale_x),
                'y': int(zo['y'] * scale_y),
                'w': int(zo['width'] * scale_x),
                'h': int(zo['height'] * scale_y),
                'num_cols': num_cols,
                'col_names': col_names,
                'zone_idx': i
            })
    else:
        # Legacy format: single object
        orders_zones.append({
            'x': int(zone_orders['x'] * scale_x),
            'y': int(zone_orders['y'] * scale_y),
            'w': int(zone_orders['width'] * scale_x),
            'h': int(zone_orders['height'] * scale_y),
            'num_cols': 5,
            'col_names': ['Mo', 'Di', 'Mi', 'Do', 'Fr'],
            'zone_idx': 0
        })

    # Use first orders zone for reference coordinates
    orders_x = orders_zones[0]['x']
    orders_y = orders_zones[0]['y']
    orders_w = orders_zones[0]['w']
    orders_h = orders_zones[0]['h']

    names_x = int(zone_names['x'] * scale_x) if zone_names else 0
    names_w = int(zone_names['width'] * scale_x) if zone_names else orders_x

    # Handle zone_codes as array (multiple zones) or single object (legacy)
    codes_zones = []
    if zone_codes:
        if isinstance(zone_codes, list):
            # New format: array of zones
            for zc in zone_codes:
                codes_zones.append({
                    'x': int(zc['x'] * scale_x),
                    'y': int(zc['y'] * scale_y),
                    'w': int(zc['width'] * scale_x),
                    'h': int(zc['height'] * scale_y)
                })
        else:
            # Legacy format: single object
            codes_zones.append({
                'x': int(zone_codes['x'] * scale_x),
                'y': int(zone_codes['y'] * scale_y),
                'w': int(zone_codes['width'] * scale_x),
                'h': int(zone_codes['height'] * scale_y)
            })

    # Fallback if no codes zones defined
    if not codes_zones:
        codes_zones.append({
            'x': orders_x + orders_w,
            'y': orders_y,
            'w': 100,
            'h': orders_h
        })

    print(f"Row-based scan: {len(orders_zones)} orders zones")
    for oz in orders_zones:
        print(f"  Zone {oz['zone_idx']}: ({oz['x']},{oz['y']},{oz['w']},{oz['h']}) cols={oz['col_names']}")
    print(f"  names_x={names_x}, names_w={names_w}")
    print(f"  codes_zones={len(codes_zones)} zones: {codes_zones}")

    # Find blobs (handwritten digits) in all orders zones
    blobs = []

    for oz in orders_zones:
        oz_x, oz_y, oz_w, oz_h = oz['x'], oz['y'], oz['w'], oz['h']
        num_cols = oz['num_cols']
        col_names = oz['col_names']
        zone_idx = oz['zone_idx']

        # Extract this orders zone
        orders_img = img[oz_y:oz_y+oz_h, oz_x:oz_x+oz_w]
        if orders_img.size == 0:
            continue

        # Find blobs in this zone
        gray = cv2.cvtColor(orders_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove grid lines using morphological operations
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine and dilate lines slightly to ensure complete removal
        grid_lines = cv2.add(horizontal_lines, vertical_lines)
        grid_lines = cv2.dilate(grid_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        # Remove grid lines from binary image
        binary = cv2.subtract(binary, grid_lines)

        # Morphological operations to clean up remaining noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Debug: save binary image
        debug_path = f'/app/templates/debug_zone_{zone_idx}_binary.png'
        cv2.imwrite(debug_path, binary)
        print(f"  Debug: Saved binary image to {debug_path}")

        # Find contours (blobs)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  Zone {zone_idx}: Found {len(contours)} raw contours")

        cell_w = oz_w / num_cols

        filtered_count = {'too_small': 0, 'too_large': 0, 'low_area': 0, 'passed': 0}
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Filter: reasonable size for a digit
            if bw < 8 or bh < 10:
                filtered_count['too_small'] += 1
                continue
            if bw > 80 or bh > 80:
                filtered_count['too_large'] += 1
                continue
            if area < 50:
                filtered_count['low_area'] += 1
                continue
            filtered_count['passed'] += 1

            # Calculate column from x position
            col = int(x / cell_w)
            if col >= num_cols:
                col = num_cols - 1

            # Y center in image coordinates
            y_center_in_zone = y + bh / 2
            y_center_abs = oz_y + y_center_in_zone

            # Extract blob image for EMNIST
            padding = 4
            bx1 = max(0, x - padding)
            by1 = max(0, y - padding)
            bx2 = min(oz_w, x + bw + padding)
            by2 = min(oz_h, y + bh + padding)

            blob_img = orders_img[by1:by2, bx1:bx2]

            blobs.append({
                'x': x,
                'y': y,
                'w': bw,
                'h': bh,
                'col': col,
                'column': col_names[col],
                'y_center_abs': y_center_abs,
                'blob_img': blob_img,
                'zone_idx': zone_idx,
                'zone_x': oz_x,
                'zone_y': oz_y
            })

        print(f"  Zone {zone_idx} filter stats: {filtered_count}")

    print(f"Found {len(blobs)} blobs in {len(orders_zones)} orders zone(s)")

    # Helper function to find matching codes zone for a Y position
    def find_codes_zone_for_y(y_pos):
        """Find the codes zone that covers the given Y position."""
        for cz in codes_zones:
            if cz['y'] <= y_pos <= cz['y'] + cz['h']:
                return cz
        # Fallback: return first zone
        return codes_zones[0] if codes_zones else None

    # Phase 1: Extract all strips for OCR (before using EMNIST/TensorFlow)
    row_height = 53  # Approximate row height
    blob_data = []

    for blob in blobs:
        y_center = blob['y_center_abs']

        # Calculate row strip boundaries (centered on blob)
        strip_y1 = int(y_center - row_height / 2)
        strip_y2 = int(y_center + row_height / 2)
        strip_y1 = max(0, strip_y1)
        strip_y2 = min(h, strip_y2)

        # Extract horizontal strip for product name (left)
        name_strip = img[strip_y1:strip_y2, names_x:names_x+names_w]

        # Find matching codes zone for this Y position and extract code strip
        codes_zone = find_codes_zone_for_y(y_center)
        if codes_zone:
            code_strip = img[strip_y1:strip_y2, codes_zone['x']:codes_zone['x']+codes_zone['w']]
        else:
            code_strip = np.zeros((strip_y2-strip_y1, 100, 3), dtype=np.uint8)

        # Save strips for OCR
        strip_id = f"{int(blob['y_center_abs'])}_{blob['col']}"
        name_path = f'/app/templates/debug_cells/name_strip_{strip_id}.png'
        code_path = f'/app/templates/debug_cells/code_strip_{strip_id}.png'

        cv2.imwrite(name_path, name_strip)
        cv2.imwrite(code_path, code_strip)

        # Save blob image for EMNIST
        blob_path = f'/app/templates/debug_cells/blob_{strip_id}.png'
        cv2.imwrite(blob_path, blob['blob_img'])

        blob_data.append({
            'blob': blob,
            'blob_path': blob_path,
            'name_strip_path': name_path,
            'code_strip_path': code_path,
            'y_center': int(y_center)
        })

    # Phase 2: Run OCR FIRST (before TensorFlow/EMNIST claims GPU)
    print(f"Running OCR on {len(blob_data)} strip pairs...")
    ocr_results = _run_ocr_batch([
        {'path': b['name_strip_path'], 'type': 'name', 'y': b['y_center']} for b in blob_data
    ] + [
        {'path': b['code_strip_path'], 'type': 'code', 'y': b['y_center']} for b in blob_data
    ])

    # Phase 3: Now run EMNIST recognition (TensorFlow will claim GPU)
    print("Running EMNIST digit recognition...")
    results = []

    for bd in blob_data:
        blob = bd['blob']
        y_center = bd['y_center']

        # EMNIST recognition
        digit, confidence = recognize_digit_emnist(blob['blob_img'])

        if digit is None or confidence < 0.4:
            continue

        results.append({
            'digit': digit,
            'confidence': round(confidence, 3),
            'column': blob['column'],
            'y_center': y_center,
            'x': int((blob['zone_x'] + blob['x']) / scale_x),
            'y': int((blob['zone_y'] + blob['y']) / scale_y),
            'width': int(blob['w'] / scale_x),
            'height': int(blob['h'] / scale_y),
            'product_name': clean_product_name(ocr_results.get(f"{y_center}_name", '')),
            'product_code': ocr_results.get(f"{y_center}_code", '')
        })

    return {'results': results}


def scan_row_info(y_center, zone_codes, zone_names, image_width, image_height):
    """Scan product name and code for a specific Y position."""
    print(f"Scanning row info at y_center={y_center}")

    image_path = find_latest_image()
    if not image_path:
        return {'error': 'No image found'}

    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Could not load image'}

    h, w = img.shape[:2]
    scale_x = w / image_width
    scale_y = h / image_height

    # Scale y_center to image coordinates
    y_center_scaled = int(y_center * scale_y)
    row_height = 53  # Approximate row height

    product_code = ''
    product_name = ''

    # Extract and OCR product code from codes zone
    if zone_codes and len(zone_codes) > 0:
        # Use first codes zone
        zc = zone_codes[0] if isinstance(zone_codes, list) else zone_codes
        zc_x = int(zc['x'] * scale_x)
        zc_y = int(zc['y'] * scale_y)
        zc_w = int(zc['width'] * scale_x)
        zc_h = int(zc['height'] * scale_y)

        # Extract horizontal strip at y_center
        strip_y1 = max(0, y_center_scaled - row_height // 2)
        strip_y2 = min(h, y_center_scaled + row_height // 2)
        strip_x1 = max(0, zc_x)
        strip_x2 = min(w, zc_x + zc_w)

        if strip_x2 > strip_x1 and strip_y2 > strip_y1:
            code_strip = img[strip_y1:strip_y2, strip_x1:strip_x2]
            # Run OCR on the strip
            try:
                ocr = get_ocr_engine()
                if ocr and ocr.reader:
                    result = ocr.reader.readtext(code_strip, detail=0)
                    if result:
                        # Join all detected text, filter for numbers
                        text = ' '.join(result).strip()
                        # Extract numbers only for product code
                        numbers = ''.join(c for c in text if c.isdigit())
                        if numbers:
                            product_code = numbers
                            print(f"  Code OCR: '{text}' -> '{product_code}'")
            except Exception as e:
                print(f"  Code OCR error: {e}")

    # Extract and OCR product name from names zone
    if zone_names:
        zn = zone_names
        zn_x = int(zn['x'] * scale_x)
        zn_y = int(zn['y'] * scale_y)
        zn_w = int(zn['width'] * scale_x)
        zn_h = int(zn['height'] * scale_y)

        # Extract horizontal strip at y_center
        strip_y1 = max(0, y_center_scaled - row_height // 2)
        strip_y2 = min(h, y_center_scaled + row_height // 2)
        strip_x1 = max(0, zn_x)
        strip_x2 = min(w, zn_x + zn_w)

        if strip_x2 > strip_x1 and strip_y2 > strip_y1:
            name_strip = img[strip_y1:strip_y2, strip_x1:strip_x2]
            # Run OCR on the strip
            try:
                ocr = get_ocr_engine()
                if ocr and ocr.reader:
                    result = ocr.reader.readtext(name_strip, detail=0)
                    if result:
                        product_name = clean_product_name(' '.join(result).strip())
                        print(f"  Name OCR: '{product_name}'")
            except Exception as e:
                print(f"  Name OCR error: {e}")

    print(f"  Result: code='{product_code}', name='{product_name}'")
    return {
        'product_code': product_code,
        'product_name': product_name
    }


def _run_ocr_batch(strips_to_ocr):
    """Run OCR on all strips in a single subprocess. Must be called BEFORE TensorFlow loads."""
    if not strips_to_ocr:
        return {}

    # Run batch OCR in subprocess
    ocr_script = '''
import sys
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import easyocr
    import re
    reader = easyocr.Reader(["de"], gpu=True, verbose=False)

    strips = ''' + json.dumps(strips_to_ocr) + '''
    results = {}

    for strip in strips:
        path = strip['path']
        strip_type = strip['type']
        y = strip['y']
        key = f"{y}_{strip_type}"

        if not os.path.exists(path):
            results[key] = ''
            continue

        ocr_results = reader.readtext(path)
        if not ocr_results:
            results[key] = ''
            continue

        texts = [r[1] for r in ocr_results]
        full_text = ' '.join(texts).strip()

        if strip_type == 'code':
            # Extract only digits and hyphens for product codes
            clean = re.sub(r'[^0-9\\-()]', '', full_text)
            results[key] = clean
        else:
            # Product name - keep text, remove leading numbers
            clean = re.sub(r'^[0-9\\-]+\\s*', '', full_text)
            results[key] = clean

    print("OCR_RESULT:" + json.dumps(results))
except Exception as e:
    print("OCR_ERROR:" + str(e), file=sys.stderr)
    import traceback
    traceback.print_exc()
'''

    try:
        result = subprocess.run(
            ['python3', '-c', ocr_script],
            capture_output=True, text=True, timeout=180
        )

        # Parse OCR results
        ocr_data = {}
        for line in result.stdout.strip().split('\n'):
            if line.startswith('OCR_RESULT:'):
                ocr_data = json.loads(line[11:])
                break

        if result.stderr:
            print(f"OCR stderr: {result.stderr[:500]}")

        print(f"OCR completed: {len(ocr_data)} strips processed")
        return ocr_data

    except Exception as e:
        print(f"Batch OCR error: {e}")
        return {}


def scan_zone_in_docker(zone, num_rows, num_cols, image_width, image_height, fixed_column='', zone_codes=None, zone_names=None):
    """Run zone scanning in Docker container."""

    # Prepare zone_codes for the script
    zone_codes_str = 'None'
    if zone_codes:
        zone_codes_str = f"{{'x': {zone_codes['x']}, 'y': {zone_codes['y']}, 'width': {zone_codes['width']}, 'height': {zone_codes['height']}}}"
        print(f"Zone codes prepared: {zone_codes_str}")
    else:
        print("No zone_codes provided to scan_zone_in_docker")

    # Prepare zone_names for the script
    zone_names_str = 'None'
    if zone_names:
        zone_names_str = f"{{'x': {zone_names['x']}, 'y': {zone_names['y']}, 'width': {zone_names['width']}, 'height': {zone_names['height']}}}"
        print(f"Zone names prepared: {zone_names_str}")
    else:
        print("No zone_names provided to scan_zone_in_docker")

    # Create the Python script to run in Docker
    script = f'''
import sys
sys.path.insert(0, '/app/intermediate')
sys.path.insert(0, '/app/src')
import json
import os
import glob
import re
import cv2
import numpy as np
import subprocess
# Product codes zone (for reading product numbers)
zone_codes = {zone_codes_str}
# Product names zone (for reading product names)
zone_names = {zone_names_str}

# Cache for product codes per row
product_code_cache = {{}}
# Cache for product names per row
product_name_cache = {{}}

def read_product_codes_from_zone():
    """Read all product codes from the product codes zone using EasyOCR in subprocess."""
    global product_code_cache
    # Clear cache for new scan
    product_code_cache = {{}}

    if zone_codes is None:
        return

    # Scale zone_codes coordinates
    codes_x = int(zone_codes['x'] * scale_x)
    codes_y = int(zone_codes['y'] * scale_y)
    codes_w = int(zone_codes['width'] * scale_x)
    codes_h = int(zone_codes['height'] * scale_y)

    # Extract the entire product codes zone
    codes_img = img[codes_y:codes_y+codes_h, codes_x:codes_x+codes_w]
    if codes_img.size == 0:
        return

    # Save for OCR subprocess
    codes_img_path = '/app/templates/debug_cells/product_codes_zone.png'
    cv2.imwrite(codes_img_path, codes_img)

    # Run EasyOCR in subprocess to avoid memory corruption
    ocr_script = """
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import easyocr
    import json
    reader = easyocr.Reader(["de"], gpu=True, verbose=False)
    results = reader.readtext("/app/templates/debug_cells/product_codes_zone.png")
    output = []
    for (bbox, text, conf) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        output.append({{"y": y_center, "text": text, "conf": conf}})
    print("OCR_RESULT:" + json.dumps(output))
except Exception as e:
    print("OCR_ERROR:" + str(e), file=sys.stderr)
    import traceback
    traceback.print_exc()
"""
    try:
        result = subprocess.run(
            ['python3', '-c', ocr_script],
            capture_output=True, text=True, timeout=120
        )

        # Debug output
        print(f"Product codes zone: {{codes_w}}x{{codes_h}}")
        if result.stderr:
            print(f"OCR stderr: {{result.stderr[:500]}}")

        # Parse output
        row_height = codes_h / num_rows
        ocr_found = False
        for line in result.stdout.strip().split('\\n'):
            if line.startswith('OCR_RESULT:'):
                ocr_found = True
                json_str = line[11:]  # Remove prefix
                data = json.loads(json_str)
                print(f"OCR found {{len(data)}} text items")
                for item in data:
                    y_center = item['y']
                    text = item['text'].strip()
                    row = int(y_center / row_height)
                    if 0 <= row < num_rows:
                        # Keep digits, hyphens, parentheses (for codes like "1-12", "(210-212)")
                        clean_text = re.sub(r'[^0-9\\-\\(\\)]', '', text)
                        if clean_text and row not in product_code_cache:
                            product_code_cache[row] = clean_text
                            print(f"  Row {{row}}: '{{clean_text}}'")
                break

        if not ocr_found:
            print(f"No OCR_RESULT found. stdout: {{result.stdout[:300]}}")

        print(f"Total product codes found: {{len(product_code_cache)}}")
    except Exception as e:
        print("Product code OCR error: " + str(e))
        import traceback
        traceback.print_exc()

def get_product_code(row_num):
    """Get product code for a specific row."""
    return product_code_cache.get(row_num, '')

def get_product_name(row_num):
    """Get product name for a specific row."""
    return product_name_cache.get(row_num, '')

def read_product_names_from_zone():
    """Read all product names from the product names zone using EasyOCR in subprocess."""
    global product_name_cache
    # Clear cache for new scan
    product_name_cache = {{}}

    if zone_names is None:
        return

    # Scale zone_names coordinates
    names_x = int(zone_names['x'] * scale_x)
    names_y = int(zone_names['y'] * scale_y)
    names_w = int(zone_names['width'] * scale_x)
    names_h = int(zone_names['height'] * scale_y)

    # Extract the entire product names zone
    names_img = img[names_y:names_y+names_h, names_x:names_x+names_w]
    if names_img.size == 0:
        return

    # Save for OCR subprocess
    names_img_path = '/app/templates/debug_cells/product_names_zone.png'
    cv2.imwrite(names_img_path, names_img)

    # Run EasyOCR in subprocess to avoid memory corruption
    ocr_script = """
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import easyocr
    import json
    reader = easyocr.Reader(["de"], gpu=True, verbose=False)
    results = reader.readtext("/app/templates/debug_cells/product_names_zone.png")
    output = []
    for (bbox, text, conf) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        output.append({{"y": y_center, "text": text, "conf": conf}})
    print("OCR_RESULT:" + json.dumps(output))
except Exception as e:
    print("OCR_ERROR:" + str(e), file=sys.stderr)
    import traceback
    traceback.print_exc()
"""
    try:
        result = subprocess.run(
            ['python3', '-c', ocr_script],
            capture_output=True, text=True, timeout=120
        )

        # Debug output
        print(f"Product names zone: {{names_w}}x{{names_h}}")
        if result.stderr:
            print(f"Names OCR stderr: {{result.stderr[:500]}}")

        # Parse output
        row_height = names_h / num_rows
        ocr_found = False
        for line in result.stdout.strip().split('\\n'):
            if line.startswith('OCR_RESULT:'):
                ocr_found = True
                json_str = line[11:]  # Remove prefix
                data = json.loads(json_str)
                print(f"Names OCR found {{len(data)}} text items")
                for item in data:
                    y_center = item['y']
                    text = item['text'].strip()
                    row = int(y_center / row_height)
                    if 0 <= row < num_rows:
                        # Keep the product name as-is (just clean whitespace)
                        clean_text = ' '.join(text.split())
                        if clean_text and row not in product_name_cache:
                            product_name_cache[row] = clean_text
                            print(f"  Row {{row}}: '{{clean_text}}'")
                break

        if not ocr_found:
            print(f"No names OCR_RESULT found. stdout: {{result.stdout[:300]}}")

        print(f"Total product names found: {{len(product_name_cache)}}")
    except Exception as e:
        print("Product name OCR error: " + str(e))
        import traceback
        traceback.print_exc()

def run_easyocr_subprocess(image_path):
    """Run EasyOCR in a separate subprocess to avoid memory corruption issues."""
    ocr_script = """
import easyocr
import cv2
import json

reader = easyocr.Reader(['de'], gpu=True, verbose=False)
img = cv2.imread("{{image_path}}")
if img is not None:
    results = reader.readtext(img)
    texts = [r[1] for r in results]
    print(json.dumps({{{{"text": " ".join(texts)}}}}))
else:
    print(json.dumps({{{{"text": ""}}}}))
""".format(image_path=image_path)
    try:
        result = subprocess.run(
            ['python3', '-c', ocr_script],
            capture_output=True, text=True, timeout=30
        )
        # Parse output even if process crashed after printing
        for line in result.stdout.strip().split('\\n'):
            if line.startswith('{{'):
                data = json.loads(line)
                return data.get('text', '')
    except Exception as e:
        print(f"Subprocess OCR error: {{e}}")
    return ""

# NO digit recognition in subprocess - will be done in main process

# Find image
images = glob.glob('/app/intermediate/*/page_000_normalized.png')
if not images:
    print(json.dumps({{"error": "No image found"}}))
    sys.exit(0)

images.sort(key=os.path.getmtime, reverse=True)
image_path = images[0]

# Load image
img = cv2.imread(image_path)
if img is None:
    print(json.dumps({{"error": "Could not load image"}}))
    sys.exit(0)

h, w = img.shape[:2]

# Zone from frontend (in image pixel coordinates)
zone_x = {zone['x']}
zone_y = {zone['y']}
zone_w = {zone['width']}
zone_h = {zone['height']}
num_rows = {num_rows}
num_cols = {num_cols}
frontend_w = {image_width}
frontend_h = {image_height}
fixed_column = '{fixed_column}'  # Empty string means auto-detect

# Scale zone coordinates if frontend image size differs from actual image
scale_x = w / frontend_w
scale_y = h / frontend_h

zone_x = int(zone_x * scale_x)
zone_y = int(zone_y * scale_y)
zone_w = int(zone_w * scale_x)
zone_h = int(zone_h * scale_y)

# Calculate cell dimensions
cell_w = zone_w / num_cols
cell_h = zone_h / num_rows

COLUMNS = ['Mo', 'Di', 'Mi', 'Do', 'Fr']
results = []

# Cache for product info per row (to avoid repeated OCR)
product_info_cache = {{}}

def get_product_info(row_num, blob_y_in_zone):
    """Extract product code and name from the left side of the row using OCR."""
    if row_num in product_info_cache:
        return product_info_cache[row_num]

    # If OCR is disabled, just return empty values
    if not ENABLE_PRODUCT_OCR:
        product_info_cache[row_num] = {{'code': '', 'name': ''}}
        return product_info_cache[row_num]

    # Calculate the Y position in the full image
    row_y_start = zone_y + int(row_num * cell_h)
    row_y_end = zone_y + int((row_num + 1) * cell_h)

    # Add some padding
    row_y_start = max(0, row_y_start - 2)
    row_y_end = min(h, row_y_end + 2)

    # Extract the left portion of the image (product code and name area)
    # From x=0 to where the zone starts
    left_margin = 10
    product_area_width = zone_x - left_margin

    if product_area_width < 50:
        product_info_cache[row_num] = {{'code': '', 'name': ''}}
        return product_info_cache[row_num]

    row_strip = img[row_y_start:row_y_end, left_margin:zone_x-10]

    if row_strip.size == 0:
        product_info_cache[row_num] = {{'code': '', 'name': ''}}
        return product_info_cache[row_num]

    # Save debug image of the row strip
    debug_path = f'/app/templates/debug_cells/row_{{row_num:02d}}_strip.png'
    cv2.imwrite(debug_path, row_strip)

    # Run OCR on the strip using EasyOCR (in subprocess to avoid memory issues)
    try:
        full_text = run_easyocr_subprocess(debug_path)  # Uses the saved debug image

        # Try to extract product code (usually at the start, like "1-12", "30", "600", etc.)
        code_match = re.match(r'^([\\d]{{1,3}}(?:-[\\d]{{1,3}})?)', full_text.strip())
        product_code = code_match.group(1) if code_match else ''

        # Product name is the rest of the text
        if product_code and full_text.startswith(product_code):
            product_name = full_text[len(product_code):].strip()
        else:
            product_name = full_text.strip()

        # Clean up product name
        product_name = re.sub(r'^[\\s\\-\\.]+', '', product_name)

        product_info_cache[row_num] = {{'code': product_code, 'name': product_name}}
    except Exception as e:
        print(f"OCR error for row {{row_num}}: {{e}}")
        product_info_cache[row_num] = {{'code': '', 'name': ''}}

    return product_info_cache[row_num]

# Clear debug directory before scan
debug_dir = '/app/templates/debug_cells'
import shutil
if os.path.exists(debug_dir):
    shutil.rmtree(debug_dir)
os.makedirs(debug_dir, exist_ok=True)

# Read product codes from the product codes zone (if defined)
if zone_codes is not None:
    print("Reading product codes from zone...")
    read_product_codes_from_zone()
    print(f"Found {{len(product_code_cache)}} product codes")

# Read product names from the product names zone (if defined)
if zone_names is not None:
    print("Reading product names from zone...")
    read_product_names_from_zone()
    print(f"Found {{len(product_name_cache)}} product names")

# Alternative approach: Find handwriting blobs first, then assign to cells
# Extract the entire zone with padding to capture digits extending beyond cell boundaries
zone_pad = 15  # Padding to capture digits that extend beyond zone boundaries
zone_x_padded = max(0, zone_x - zone_pad)
zone_y_padded = max(0, zone_y - zone_pad)
zone_x2_padded = min(w, zone_x + zone_w + zone_pad)
zone_y2_padded = min(h, zone_y + zone_h + zone_pad)
zone_w_padded = zone_x2_padded - zone_x_padded
zone_h_padded = zone_y2_padded - zone_y_padded

# Offset to convert padded coordinates back to original zone coordinates
pad_offset_x = zone_x - zone_x_padded
pad_offset_y = zone_y - zone_y_padded

zone_img = img[zone_y_padded:zone_y2_padded, zone_x_padded:zone_x2_padded]
zone_gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)

# Binarize to find dark content
_, zone_binary = cv2.threshold(zone_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove horizontal lines (table structure) - use larger kernel to preserve digit strokes
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
h_lines = cv2.morphologyEx(zone_binary, cv2.MORPH_OPEN, h_kernel)
zone_binary = cv2.subtract(zone_binary, h_lines)

# Remove vertical lines (table structure) - use larger kernel to preserve digit strokes
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
v_lines = cv2.morphologyEx(zone_binary, cv2.MORPH_OPEN, v_kernel)
zone_binary = cv2.subtract(zone_binary, v_lines)

# Morphological closing to connect fragmented strokes
# Closing = dilate then erode (fills gaps while preserving shape)
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
zone_binary = cv2.morphologyEx(zone_binary, cv2.MORPH_CLOSE, close_kernel)

# Find contours (potential handwritten digits)
contours, _ = cv2.findContours(zone_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Process each contour
blob_id = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 80:  # Minimum area for a digit
        continue
    if area > 4000:  # Maximum area (avoid large blobs)
        continue

    bx, by, bw, bh = cv2.boundingRect(cnt)
    aspect = bw / bh if bh > 0 else 0

    # Filter by aspect ratio (allow narrow "1" digits but not too extreme)
    if aspect > 3.0 or aspect < 0.15:
        continue

    # Filter by size
    if bw < 6 or bh < 12:
        continue
    if bw > 80 or bh > 70:
        continue

    # Extract the blob with some padding
    pad = 5
    bx1 = max(0, bx - pad)
    by1 = max(0, by - pad)
    bx2 = min(zone_w_padded, bx + bw + pad)
    by2 = min(zone_h_padded, by + bh + pad)

    blob_img = zone_img[by1:by2, bx1:bx2]

    if blob_img.size == 0:
        continue

    # Determine which grid cell this blob belongs to
    # Adjust for padding offset to get position relative to original zone
    blob_cx = bx - pad_offset_x + bw / 2
    blob_cy = by - pad_offset_y + bh / 2

    col = int(blob_cx / cell_w)
    row = int(blob_cy / cell_h)

    # Skip blobs outside the original zone
    if col < 0 or row < 0:
        continue
    if col >= num_cols:
        col = num_cols - 1
    if row >= num_rows:
        row = num_rows - 1

    # Save blob image for main process to recognize
    blob_path = f'{{debug_dir}}/blob_{{blob_id:03d}}_r{{row:02d}}_c{{col}}.png'
    cv2.imwrite(blob_path, blob_img)

    # Get product code and name from the zones (if defined)
    product_code = get_product_code(row)
    product_name = get_product_name(row)

    # Convert coordinates back to frontend scale (using padded zone coordinates)
    abs_x = zone_x_padded + bx1
    abs_y = zone_y_padded + by1

    # Use fixed column if specified, otherwise auto-detect
    if fixed_column and fixed_column in COLUMNS:
        detected_column = fixed_column
    else:
        detected_column = COLUMNS[col] if col < len(COLUMNS) else f'Col{{col+1}}'

    # Store blob metadata (recognition will happen in main process)
    results.append({{
        'blob_path': blob_path,
        'row': row + 1,
        'column': detected_column,
        'product_code': product_code,
        'product_name': product_name,
        'x': int(abs_x / scale_x),
        'y': int(abs_y / scale_y),
        'width': int((bx2 - bx1) / scale_x),
        'height': int((by2 - by1) / scale_y)
    }})
    blob_id += 1

print(json.dumps({{'blobs': results}}))
'''

    # Run script directly (we're already in the container)
    result = subprocess.run(
        ['python3', '-c', script],
        capture_output=True,
        text=True,
        timeout=180
    )

    # Print subprocess output for debugging
    if result.stderr:
        print(f"Subprocess stderr: {result.stderr[:1000]}")
    # Print non-JSON lines from stdout (debug messages)
    for line in result.stdout.split('\n'):
        if line and not line.startswith('{'):
            print(f"Subprocess: {line}")

    if result.returncode != 0:
        return {'error': f'Scanner error: {result.stderr[:500]}'}

    # Parse output (blob metadata)
    try:
        lines = result.stdout.strip().split('\n')
        blob_data = None
        for line in reversed(lines):
            if line.startswith('{'):
                blob_data = json.loads(line)
                break

        if blob_data is None:
            return {'error': 'No JSON output', 'stdout': result.stdout[:500]}

        # Now run EMNIST recognition on each blob in the main process
        blobs = blob_data.get('blobs', [])
        results = []

        for blob in blobs:
            blob_path = blob.get('blob_path')
            if not blob_path or not os.path.exists(blob_path):
                continue

            # Load blob image
            blob_img = cv2.imread(blob_path)
            if blob_img is None:
                continue

            # Run EMNIST recognition in main process
            digit, confidence = recognize_digit_emnist(blob_img)

            if digit is not None and confidence >= 0.4:
                results.append({
                    'digit': digit,
                    'confidence': confidence,
                    'row': blob['row'],
                    'column': blob['column'],
                    'product_code': blob.get('product_code', ''),
                    'product_name': blob.get('product_name', ''),
                    'x': blob['x'],
                    'y': blob['y'],
                    'width': blob['width'],
                    'height': blob['height']
                })

        return {'results': results}

    except json.JSONDecodeError as e:
        return {'error': f'JSON parse error: {e}', 'stdout': result.stdout[:500]}


def save_training_image(digit, image_data, row, column):
    """Save a base64-encoded image as a training sample."""
    digit_dir = os.path.join(TRAINING_DIR, str(digit))
    os.makedirs(digit_dir, exist_ok=True)

    # Decode base64 image
    match = re.match(r'data:image/\w+;base64,(.+)', image_data)
    if not match:
        return False

    img_bytes = base64.b64decode(match.group(1))

    # Generate filename
    import time
    timestamp = int(time.time() * 1000)
    filename = f'sample_{timestamp}_{row}_{column}.png'
    filepath = os.path.join(digit_dir, filename)

    with open(filepath, 'wb') as f:
        f.write(img_bytes)

    return True


class ScannerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=TEMPLATES_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Serve main page
        if path == '/' or path == '/index.html':
            self.path = '/interactive_scanner.html'
            return super().do_GET()

        # API: Get document image
        if path == '/api/image':
            image_path = find_latest_image()
            if image_path and os.path.exists(image_path):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                with open(image_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, 'Image not found')
            return

        # API: Get training stats
        if path == '/api/training-stats':
            stats = {}
            if os.path.exists(TRAINING_DIR):
                for digit in os.listdir(TRAINING_DIR):
                    digit_dir = os.path.join(TRAINING_DIR, digit)
                    if os.path.isdir(digit_dir):
                        stats[digit] = len(os.listdir(digit_dir))
            self.send_json({'stats': stats})
            return

        # API: Load saved zones
        if path == '/api/zones':
            if os.path.exists(SAVED_ZONES_FILE):
                try:
                    with open(SAVED_ZONES_FILE, 'r') as f:
                        zones = json.load(f)
                    self.send_json({'zones': zones, 'saved': True})
                except Exception as e:
                    print(f"Error loading saved zones: {e}")
                    self.send_json({'zones': None, 'saved': False})
            else:
                self.send_json({'zones': None, 'saved': False})
            return

        # Serve static files
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_type = self.headers.get('Content-Type', '')

        # API: Parse EML file (handle before reading body as JSON)
        if path == '/api/parse-eml':
            try:
                content_length = int(self.headers.get('Content-Length', 0))

                if 'multipart/form-data' in content_type:
                    # Parse form data using cgi.FieldStorage
                    import io
                    body = self.rfile.read(content_length)
                    environ = {
                        'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': content_type,
                        'CONTENT_LENGTH': content_length
                    }
                    form = cgi.FieldStorage(
                        fp=io.BytesIO(body),
                        headers=self.headers,
                        environ=environ
                    )

                    if 'emlFile' not in form:
                        self.send_json({'error': 'No EML file provided'})
                        return

                    eml_bytes = form['emlFile'].file.read()
                else:
                    # Fallback to JSON with base64
                    body = self.rfile.read(content_length)
                    data = json.loads(body) if body else {}
                    eml_data = data.get('emlData')
                    if not eml_data:
                        self.send_json({'error': 'No EML data provided'})
                        return
                    eml_bytes = base64.b64decode(eml_data)

                result = parse_eml_attachments(eml_bytes)
                self.send_json(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json({'error': f'Failed to parse EML: {str(e)}'})
            return

        # Read body as JSON for other endpoints
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return

        # API: Auto-detect zones
        if path == '/api/detect-zones':
            result = detect_order_zone()
            self.send_json(result)
            return

        # API: Save zones
        if path == '/api/zones':
            zones = data.get('zones')
            if zones:
                try:
                    with open(SAVED_ZONES_FILE, 'w') as f:
                        json.dump(zones, f, indent=2)
                    print(f"Zones saved to {SAVED_ZONES_FILE}")
                    self.send_json({'success': True, 'message': 'Zones saved'})
                except Exception as e:
                    print(f"Error saving zones: {e}")
                    self.send_json({'success': False, 'error': str(e)})
            else:
                self.send_json({'success': False, 'error': 'No zones provided'})
            return

        # API: Scan row info (product name and code for a specific Y position)
        if path == '/api/scan-row-info':
            y_center = data.get('y_center')
            zone_codes = data.get('zoneCodes')
            zone_names = data.get('zoneNames')
            image_width = data.get('imageWidth', 2480)
            image_height = data.get('imageHeight', 3508)

            if y_center is None:
                self.send_json({'error': 'No y_center specified'})
                return

            result = scan_row_info(y_center, zone_codes, zone_names, image_width, image_height)
            self.send_json(result)
            return

        # API: Row-based scanning (new strategy)
        if path == '/api/scan-row-based':
            zone_orders = data.get('zone')
            zone_names = data.get('zoneNames')
            zone_codes = data.get('zoneCodes')
            image_width = data.get('imageWidth', 2480)
            image_height = data.get('imageHeight', 3508)

            if not zone_orders:
                self.send_json({'error': 'No zone specified'})
                return

            print(f"Row-based scan request: orders={zone_orders}")
            result = scan_row_based(zone_orders, zone_names, zone_codes, image_width, image_height)
            self.send_json(result)
            return

        # API: Scan zone (legacy)
        if path == '/api/scan-zone':
            zone = data.get('zone')
            zone_names = data.get('zoneNames')  # Optional: Produktnamen-Zone
            zone_codes = data.get('zoneCodes')  # Optional: Produktnummern-Zone
            num_rows = data.get('numRows', 52)
            num_cols = data.get('numCols', 5)
            image_width = data.get('imageWidth', 2350)
            image_height = data.get('imageHeight', 3346)
            fixed_column = data.get('fixedColumn', '')

            print(f"Scan request: zone_codes={zone_codes}, zone_names={zone_names}")

            if not zone:
                self.send_json({'error': 'No zone specified'})
                return

            # Skip product info reading in initial scan for faster response
            result = scan_zone_in_docker(zone, num_rows, num_cols, image_width, image_height, fixed_column, None, None)
            self.send_json(result)
            return

        # API: Read product codes from zone
        if path == '/api/read-product-codes':
            zone_codes = data.get('zoneCodes')
            zone_orders = data.get('zoneOrders')  # Reference zone for row alignment
            num_rows = data.get('numRows', 52)
            image_width = data.get('imageWidth', 2350)
            image_height = data.get('imageHeight', 3346)

            if not zone_codes:
                self.send_json({'error': 'No zoneCodes specified'})
                return

            result = read_product_codes_from_image(zone_codes, num_rows, image_width, image_height, zone_orders)
            self.send_json(result)
            return

        # API: Read product names from zone
        if path == '/api/read-product-names':
            zone_names = data.get('zoneNames')
            zone_orders = data.get('zoneOrders')  # Reference zone for row alignment
            num_rows = data.get('numRows', 52)
            image_width = data.get('imageWidth', 2350)
            image_height = data.get('imageHeight', 3346)

            if not zone_names:
                self.send_json({'error': 'No zoneNames specified'})
                return

            result = read_product_names_from_image(zone_names, num_rows, image_width, image_height, zone_orders)
            self.send_json(result)
            return

        # API: Save training sample
        if path == '/api/save-training-sample':
            digit = data.get('digit')
            image_data = data.get('imageData')
            row = data.get('row', 0)
            column = data.get('column', '')

            if digit and image_data:
                success = save_training_image(digit, image_data, row, column)
                self.send_json({'success': success})
            else:
                self.send_json({'success': False, 'error': 'Missing parameters'})
            return

        self.send_error(404, 'Not found')

    def send_json(self, data):
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    os.chdir(TEMPLATES_DIR)
    print(f"=" * 50)
    print(f"Interaktiver Bestellungs-Scanner")
    print(f"=" * 50)
    print(f"Server: http://192.168.8.100:{PORT}/")
    print(f"Templates: {TEMPLATES_DIR}")
    print(f"Training: {TRAINING_DIR}")
    print(f"=" * 50)

    server = http.server.HTTPServer(('0.0.0.0', PORT), ScannerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.shutdown()


if __name__ == '__main__':
    main()
