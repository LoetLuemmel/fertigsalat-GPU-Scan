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

# Configuration
PORT = 8080
TEMPLATES_DIR = '/home/test/email-processor/shared/templates'
INTERMEDIATE_DIR = '/home/test/email-processor/shared/intermediate'
TRAINING_DIR = '/home/test/email-processor/shared/training_data'

# Ensure directories exist
os.makedirs(TRAINING_DIR, exist_ok=True)


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

        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        # Determine if it's an image or PDF
        is_image = content_type.startswith('image/')
        is_pdf = content_type == 'application/pdf' or filename.lower().endswith('.pdf')

        if is_image or is_pdf:
            attachment = {
                'filename': filename,
                'content_type': content_type,
                'size': len(payload),
                'is_pdf': is_pdf
            }

            if is_image:
                # Directly encode image as base64
                attachment['data'] = f"data:{content_type};base64,{base64.b64encode(payload).decode('utf-8')}"
            elif is_pdf:
                # Convert PDF to image using pdftoppm
                attachment['data'] = convert_pdf_to_image(payload)

            attachments.append(attachment)

    return {'metadata': metadata, 'attachments': attachments}


def convert_pdf_to_image(pdf_bytes):
    """Convert PDF bytes to PNG image using pdftoppm."""
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
                return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            except Exception:
                return None

        # Find generated PNG file (first page)
        png_files = glob.glob(os.path.join(tmpdir, 'page*.png'))
        if png_files:
            png_files.sort()
            with open(png_files[0], 'rb') as f:
                img_bytes = f.read()
            return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

    return None


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


def scan_zone_in_docker(zone, num_rows, num_cols, image_width, image_height, fixed_column=''):
    """Run zone scanning in Docker container."""

    # Create the Python script to run in Docker
    script = f'''
import sys
sys.path.insert(0, '/app/intermediate')
import json
import os
import glob
import re
import cv2
import numpy as np
import subprocess

# Product OCR enabled (uses EasyOCR in subprocess)
ENABLE_PRODUCT_OCR = True

def run_easyocr_subprocess(image_path):
    """Run EasyOCR in a separate subprocess to avoid memory corruption issues."""
    ocr_script = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import easyocr
import cv2
import json

reader = easyocr.Reader(['de'], gpu=False, verbose=False)
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

# Try trained OCR first, fall back to simple
try:
    from trained_digit_ocr import recognize_digit_trained as recognize_digit
    print("Using TRAINED digit recognition")
except ImportError:
    from simple_digit_ocr import recognize_digit
    print("Using simple digit recognition")

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

# Alternative approach: Find handwriting blobs first, then assign to cells
# Extract the entire zone
zone_img = img[zone_y:zone_y+zone_h, zone_x:zone_x+zone_w]
zone_gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)

# Binarize to find dark content
_, zone_binary = cv2.threshold(zone_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove horizontal lines (table structure)
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
h_lines = cv2.morphologyEx(zone_binary, cv2.MORPH_OPEN, h_kernel)
zone_binary = cv2.subtract(zone_binary, h_lines)

# Remove vertical lines (table structure)
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
v_lines = cv2.morphologyEx(zone_binary, cv2.MORPH_OPEN, v_kernel)
zone_binary = cv2.subtract(zone_binary, v_lines)

# Find contours (potential handwritten digits)
contours, _ = cv2.findContours(zone_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Process each contour
blob_id = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 40:  # Minimum area for a digit (lowered for thin strokes)
        continue
    if area > 5000:  # Maximum area (avoid large blobs)
        continue

    bx, by, bw, bh = cv2.boundingRect(cnt)
    aspect = bw / bh if bh > 0 else 0

    # Filter by aspect ratio (allow narrow "1" digits)
    if aspect > 4.0 or aspect < 0.1:
        continue

    # Filter by size (lowered for small/thin digits)
    if bw < 4 or bh < 8:
        continue
    if bw > 80 or bh > 80:
        continue

    # Extract the blob with some padding
    pad = 5
    bx1 = max(0, bx - pad)
    by1 = max(0, by - pad)
    bx2 = min(zone_w, bx + bw + pad)
    by2 = min(zone_h, by + bh + pad)

    blob_img = zone_img[by1:by2, bx1:bx2]

    if blob_img.size == 0:
        continue

    # Determine which grid cell this blob belongs to
    blob_cx = bx + bw / 2
    blob_cy = by + bh / 2

    col = int(blob_cx / cell_w)
    row = int(blob_cy / cell_h)

    if col >= num_cols:
        col = num_cols - 1
    if row >= num_rows:
        row = num_rows - 1

    # Run digit recognition
    digit, confidence = recognize_digit(blob_img)

    # Save debug image
    debug_path = f'{{debug_dir}}/blob_{{blob_id:03d}}_r{{row:02d}}_c{{col}}_{{digit or "none"}}_{{confidence:.2f}}.png'
    cv2.imwrite(debug_path, blob_img)
    blob_id += 1

    if digit and confidence >= 0.4:
        # Get product info from the left side of this row
        product_info = get_product_info(row, by)

        # Convert coordinates back to frontend scale
        # Blob position is relative to zone, need to add zone offset
        abs_x = zone_x + bx1
        abs_y = zone_y + by1

        # Use fixed column if specified, otherwise auto-detect
        if fixed_column and fixed_column in COLUMNS:
            detected_column = fixed_column
        else:
            detected_column = COLUMNS[col] if col < len(COLUMNS) else f'Col{{col+1}}'

        results.append({{
            'digit': digit,
            'confidence': confidence,
            'row': row + 1,
            'column': detected_column,
            'product_code': product_info['code'],
            'product_name': product_info['name'],
            'x': int(abs_x / scale_x),
            'y': int(abs_y / scale_y),
            'width': int((bx2 - bx1) / scale_x),
            'height': int((by2 - by1) / scale_y)
        }})

print(json.dumps({{'results': results}}))
'''

    # Run in Docker
    result = subprocess.run(
        ['docker', 'exec', 'email-ml-inference', 'python3', '-c', script],
        capture_output=True,
        text=True,
        timeout=180
    )

    if result.returncode != 0:
        return {'error': f'Scanner error: {result.stderr[:500]}'}

    # Parse output
    try:
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            if line.startswith('{'):
                return json.loads(line)
        return {'error': 'No JSON output', 'stdout': result.stdout[:500]}
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

        # API: Scan zone
        if path == '/api/scan-zone':
            zone = data.get('zone')
            num_rows = data.get('numRows', 52)
            num_cols = data.get('numCols', 5)
            image_width = data.get('imageWidth', 2350)
            image_height = data.get('imageHeight', 3346)
            fixed_column = data.get('fixedColumn', '')

            if not zone:
                self.send_json({'error': 'No zone specified'})
                return

            result = scan_zone_in_docker(zone, num_rows, num_cols, image_width, image_height, fixed_column)
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
    print(f"Server: http://192.168.3.241:{PORT}/")
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
