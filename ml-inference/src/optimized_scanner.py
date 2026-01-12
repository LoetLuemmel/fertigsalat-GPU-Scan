"""
Optimized Form Scanner for Order Forms (Bestellblätter).

Instead of full-page OCR, this scanner:
1. Loads template definitions with known grid structure
2. Extracts only input column cells
3. Checks each cell for content (empty vs filled)
4. OCRs only non-empty cells
5. Maps results to product info
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedFormScanner:
    """
    Scan order forms using template definitions for targeted OCR.
    """

    def __init__(self, definitions_dir: str, ocr_engine=None):
        """
        Initialize optimized scanner.

        Args:
            definitions_dir: Directory containing template definition JSON files
            ocr_engine: OCREngine instance for text extraction
        """
        self.definitions_dir = Path(definitions_dir)
        self.ocr_engine = ocr_engine
        self.definitions: Dict[str, dict] = {}

        # Thresholds for empty cell detection
        self.variance_threshold = float(os.getenv('EMPTY_CELL_VARIANCE_THRESHOLD', '50'))
        self.white_ratio_threshold = float(os.getenv('EMPTY_CELL_WHITE_RATIO', '0.95'))

        self._load_definitions()

    def _load_definitions(self):
        """Load all template definitions from the definitions directory."""
        if not self.definitions_dir.exists():
            logger.warning(f"Definitions directory not found: {self.definitions_dir}")
            return

        for json_file in self.definitions_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    definition = json.load(f)

                form_type = definition.get('form_type')
                if form_type:
                    self.definitions[form_type] = definition
                    logger.info(f"Loaded definition for: {form_type}")

            except Exception as e:
                logger.error(f"Error loading definition {json_file}: {e}")

        logger.info(f"Loaded {len(self.definitions)} template definitions")

    def get_available_form_types(self) -> List[str]:
        """Return list of form types with definitions available."""
        return list(self.definitions.keys())

    def has_definition(self, form_type: str) -> bool:
        """Check if a definition exists for the given form type."""
        # Normalize form type name for matching
        normalized = self._normalize_form_type(form_type)
        return normalized in self.definitions

    def _normalize_form_type(self, form_type: str) -> str:
        """
        Normalize form type name for matching.

        Handles variations like:
        - "Bestellblatt 1" -> "Bestellblatt-1"
        - "Bestellblatt-1" -> "Bestellblatt-1"
        """
        if not form_type:
            return ''

        # Replace spaces with dashes
        normalized = form_type.replace(' ', '-')

        # Try exact match first
        if normalized in self.definitions:
            return normalized

        # Try without dash
        no_dash = normalized.replace('-', '')
        for key in self.definitions.keys():
            if key.replace('-', '') == no_dash:
                return key

        return normalized

    def scan_form(self, image_path: str, form_type: str) -> Dict:
        """
        Scan form using optimized column-by-column approach.

        Args:
            image_path: Path to the preprocessed form image
            form_type: Detected form type (e.g., "Bestellblatt-1")

        Returns:
            Structured order data with product mappings
        """
        start_time = datetime.now()

        # Normalize and get definition
        normalized_type = self._normalize_form_type(form_type)
        definition = self.definitions.get(normalized_type)

        if not definition:
            logger.warning(f"No definition for form type: {form_type}")
            return {
                'form_type': form_type,
                'error': 'No template definition available',
                'orders': [],
                'statistics': {}
            }

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'form_type': form_type,
                'error': f'Could not read image: {image_path}',
                'orders': [],
                'statistics': {}
            }

        logger.info(f"Scanning form: {form_type} from {image_path}")

        # Calculate scale factors (template vs actual image)
        img_h, img_w = image.shape[:2]
        template_dims = definition.get('image_dimensions', {})
        template_w = template_dims.get('width', img_w)
        template_h = template_dims.get('height', img_h)

        scale_x = img_w / template_w
        scale_y = img_h / template_h
        logger.info(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f} (template: {template_w}x{template_h}, image: {img_w}x{img_h})")

        # Align image to template if needed
        aligned_image, alignment_score = self._align_to_template(image, definition)

        # Get input columns
        input_columns = definition.get('columns', {}).get('input_columns', [])
        products = definition.get('products', [])

        if not input_columns or not products:
            logger.warning("Definition missing input_columns or products")
            return {
                'form_type': form_type,
                'error': 'Invalid template definition',
                'orders': [],
                'statistics': {}
            }

        # Statistics
        stats = {
            'total_cells': 0,
            'empty_cells': 0,
            'ocr_cells': 0,
            'columns_scanned': len(input_columns),
            'products_in_template': len(products)
        }

        orders = []

        # Scan each input column
        for column in input_columns:
            column_name = column['name']
            logger.debug(f"Scanning column: {column_name}")

            # Scan each product row in this column
            for product in products:
                cell_bounds = product.get('cells', {}).get(column_name)
                if not cell_bounds:
                    continue

                stats['total_cells'] += 1

                # Scale cell bounds to match actual image size
                scaled_bounds = {
                    'x': int(cell_bounds.get('x', 0) * scale_x),
                    'y': int(cell_bounds.get('y', 0) * scale_y),
                    'width': int(cell_bounds.get('width', 0) * scale_x),
                    'height': int(cell_bounds.get('height', 0) * scale_y)
                }

                # Extract cell region
                cell_image = self._extract_cell(aligned_image, scaled_bounds)
                if cell_image is None:
                    continue

                # Check if cell is empty
                if self._is_cell_empty(cell_image):
                    stats['empty_cells'] += 1
                    continue

                # OCR non-empty cell
                stats['ocr_cells'] += 1
                ocr_result = self._ocr_cell(cell_image)

                if ocr_result.get('text'):
                    orders.append({
                        'product_number': product.get('product_number', ''),
                        'product_name': product.get('product_name', ''),
                        'column': column_name,
                        'row': product.get('row', 0),
                        'value': ocr_result['text'],
                        'confidence': ocr_result.get('confidence', 0.0),
                        'cell_bounds': scaled_bounds
                    })

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        stats['processing_time_ms'] = round(processing_time, 1)

        result = {
            'form_type': form_type,
            'image_path': str(image_path),
            'scan_timestamp': datetime.now().isoformat(),
            'alignment_score': alignment_score,
            'orders': orders,
            'statistics': stats
        }

        logger.info(f"Scan completed: {stats['ocr_cells']}/{stats['total_cells']} cells with content, "
                   f"{len(orders)} orders found")

        return result

    def _align_to_template(self, image: np.ndarray,
                           definition: Dict) -> Tuple[np.ndarray, float]:
        """
        Align scanned image to template grid.

        For now, assumes images are already preprocessed and aligned.
        Returns the image with an alignment score of 1.0.

        TODO: Implement proper alignment using grid detection and homography.
        """
        # Check image dimensions vs template
        template_dims = definition.get('image_dimensions', {})
        template_w = template_dims.get('width', image.shape[1])
        template_h = template_dims.get('height', image.shape[0])

        # If dimensions differ significantly, resize
        if abs(image.shape[1] - template_w) > 50 or abs(image.shape[0] - template_h) > 50:
            logger.info(f"Resizing image from {image.shape[1]}x{image.shape[0]} "
                       f"to {template_w}x{template_h}")
            image = cv2.resize(image, (template_w, template_h))

        return image, 1.0

    def _extract_cell(self, image: np.ndarray, bounds: Dict) -> Optional[np.ndarray]:
        """
        Extract a single cell region from the image.

        Args:
            image: Full form image
            bounds: Cell bounds dictionary with x, y, width, height

        Returns:
            Cropped cell image or None if bounds are invalid
        """
        x = bounds.get('x', 0)
        y = bounds.get('y', 0)
        w = bounds.get('width', 0)
        h = bounds.get('height', 0)

        # Validate bounds
        if w <= 0 or h <= 0:
            return None

        # Ensure bounds are within image
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        # Add small padding to exclude cell borders
        padding = 3
        x += padding
        y += padding
        w -= 2 * padding
        h -= 2 * padding

        if w <= 0 or h <= 0:
            return None

        return image[y:y+h, x:x+w].copy()

    def _is_cell_empty(self, cell_image: np.ndarray) -> bool:
        """
        Fast check if a cell contains handwritten content.

        Empty cells characteristics:
        - Low pixel variance (uniform white/light gray)
        - High ratio of white/near-white pixels

        Args:
            cell_image: Cropped cell region

        Returns:
            True if cell appears empty
        """
        # Convert to grayscale
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image

        # Check variance - low variance means uniform (empty)
        variance = np.var(gray)
        if variance < self.variance_threshold:
            return True

        # Check white pixel ratio
        white_pixels = np.sum(gray > 200)
        total_pixels = gray.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 1.0

        if white_ratio > self.white_ratio_threshold:
            return True

        # Additional check: look for ink (dark pixels)
        dark_pixels = np.sum(gray < 100)
        dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

        # If very few dark pixels, consider empty
        if dark_ratio < 0.01:
            return True

        return False

    def _ocr_cell(self, cell_image: np.ndarray) -> Dict:
        """
        OCR a single cell image.

        Args:
            cell_image: Cropped cell region

        Returns:
            OCR result dictionary with text and confidence
        """
        if self.ocr_engine is None:
            return {'text': '', 'confidence': 0.0, 'error': 'No OCR engine'}

        try:
            # Save cell to temp file for OCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
                cv2.imwrite(temp_path, cell_image)

            # Run OCR
            result = self.ocr_engine.extract_text(temp_path)

            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

            # Extract and clean text
            text = result.get('full_text', '').strip()

            # Clean common OCR artifacts
            text = self._clean_ocr_text(text)

            return {
                'text': text,
                'confidence': result.get('confidence', 0.0),
                'raw_blocks': result.get('text_blocks', [])
            }

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {'text': '', 'confidence': 0.0, 'error': str(e)}

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text for numeric values.

        Order forms typically contain numbers, so we:
        - Remove whitespace
        - Keep only digits and common separators
        """
        if not text:
            return ''

        # Remove extra whitespace
        text = ' '.join(text.split())

        # For order quantities, extract just the number
        # Handle cases like "3", "12", "5 x", etc.
        import re

        # Try to extract number
        match = re.search(r'(\d+)', text)
        if match:
            return match.group(1)

        return text.strip()

    def scan_form_with_fallback(self, image_path: str, form_type: str,
                                ocr_engine=None) -> Dict:
        """
        Scan form with fallback to full-page OCR if no definition exists.

        Args:
            image_path: Path to form image
            form_type: Detected form type
            ocr_engine: OCR engine for fallback

        Returns:
            Scan result dictionary
        """
        # Try optimized scanning first
        if self.has_definition(form_type):
            return self.scan_form(image_path, form_type)

        # Fallback: indicate that full OCR should be used
        logger.info(f"No definition for {form_type}, using full OCR")
        return {
            'form_type': form_type,
            'use_full_ocr': True,
            'orders': [],
            'statistics': {'reason': 'no_template_definition'}
        }


def scan_with_definition(image_path: str, form_type: str,
                         definitions_dir: str, ocr_engine=None) -> Dict:
    """
    Convenience function to scan a form with template definition.

    Args:
        image_path: Path to form image
        form_type: Form type identifier
        definitions_dir: Path to definitions directory
        ocr_engine: OCR engine instance

    Returns:
        Scan result dictionary
    """
    scanner = OptimizedFormScanner(definitions_dir, ocr_engine)
    return scanner.scan_form(image_path, form_type)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Scan order form with optimized OCR')
    parser.add_argument('image_path', help='Path to form image')
    parser.add_argument('--form-type', '-t', required=True, help='Form type (e.g., Bestellblatt-1)')
    parser.add_argument('--definitions-dir', '-d', default='/app/templates/definitions',
                        help='Path to definitions directory')
    parser.add_argument('--output', '-o', help='Output JSON file')

    args = parser.parse_args()

    # Initialize scanner (without OCR for testing)
    scanner = OptimizedFormScanner(args.definitions_dir)

    print(f"Available form types: {scanner.get_available_form_types()}")

    if scanner.has_definition(args.form_type):
        result = scanner.scan_form(args.image_path, args.form_type)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"No definition found for: {args.form_type}")
