#!/usr/bin/env python3
"""
Populate product numbers and names in template definitions using OCR.

This script reads the template images and extracts product information
from the static columns (product_number, product_name) using OCR.

Usage:
    python populate_product_data.py --definitions-dir /app/templates/definitions --templates-dir /app/templates
"""

import argparse
import json
import sys
from pathlib import Path
import logging
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available")


def extract_cell_text(image: np.ndarray, cell: dict, reader) -> str:
    """Extract text from a cell region using OCR."""
    if not cell:
        return ''

    x = cell.get('x', 0)
    y = cell.get('y', 0)
    w = cell.get('width', 0)
    h = cell.get('height', 0)

    if w <= 0 or h <= 0:
        return ''

    # Ensure bounds are within image
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    # Add small padding to exclude borders
    padding = 2
    x += padding
    y += padding
    w -= 2 * padding
    h -= 2 * padding

    if w <= 0 or h <= 0:
        return ''

    # Extract cell region
    cell_img = image[y:y+h, x:x+w]

    # Run OCR
    try:
        result = reader.readtext(cell_img)
        if result:
            # Concatenate all detected text
            text = ' '.join([r[1] for r in result])
            return text.strip()
    except Exception as e:
        logger.debug(f"OCR error for cell: {e}")

    return ''


def populate_definition(definition_path: Path, templates_dir: Path, reader) -> bool:
    """
    Populate product data in a single definition file.

    Args:
        definition_path: Path to definition JSON
        templates_dir: Directory containing template images
        reader: EasyOCR reader instance

    Returns:
        True if successful
    """
    try:
        # Load definition
        with open(definition_path, 'r', encoding='utf-8') as f:
            definition = json.load(f)

        form_type = definition.get('form_type', '')
        logger.info(f"Processing: {form_type}")

        # Find template image
        template_image = templates_dir / f"{form_type}.png"
        if not template_image.exists():
            logger.error(f"Template image not found: {template_image}")
            return False

        # Load image
        image = cv2.imread(str(template_image))
        if image is None:
            logger.error(f"Could not read image: {template_image}")
            return False

        # Process each product
        products = definition.get('products', [])
        updated_count = 0

        for i, product in enumerate(products):
            # Extract product number
            product_number_cell = product.get('product_number_cell')
            if product_number_cell and not product.get('product_number'):
                text = extract_cell_text(image, product_number_cell, reader)
                if text:
                    product['product_number'] = text
                    updated_count += 1

            # Extract product name
            product_name_cell = product.get('product_name_cell')
            if product_name_cell and not product.get('product_name'):
                text = extract_cell_text(image, product_name_cell, reader)
                if text:
                    product['product_name'] = text
                    updated_count += 1

            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(products)} products")

        # Save updated definition
        with open(definition_path, 'w', encoding='utf-8') as f:
            json.dump(definition, f, indent=2, ensure_ascii=False)

        logger.info(f"Updated {updated_count} fields in {form_type}")
        return True

    except Exception as e:
        logger.error(f"Error processing {definition_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Populate product data in template definitions using OCR'
    )
    parser.add_argument(
        '--definitions-dir', '-d',
        default='/app/templates/definitions',
        help='Directory containing definition JSON files'
    )
    parser.add_argument(
        '--templates-dir', '-t',
        default='/app/templates',
        help='Directory containing template PNG images'
    )
    parser.add_argument(
        '--single', '-s',
        help='Process only a single definition file'
    )

    args = parser.parse_args()

    if not EASYOCR_AVAILABLE:
        logger.error("EasyOCR is required but not available")
        sys.exit(1)

    definitions_dir = Path(args.definitions_dir)
    templates_dir = Path(args.templates_dir)

    if not definitions_dir.exists():
        logger.error(f"Definitions directory not found: {definitions_dir}")
        sys.exit(1)

    if not templates_dir.exists():
        logger.error(f"Templates directory not found: {templates_dir}")
        sys.exit(1)

    # Initialize OCR reader (only German, CPU to save memory)
    logger.info("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['de'], gpu=False)

    if args.single:
        # Process single file
        definition_path = Path(args.single)
        if not definition_path.exists():
            definition_path = definitions_dir / args.single
        if not definition_path.exists():
            logger.error(f"Definition file not found: {args.single}")
            sys.exit(1)

        success = populate_definition(definition_path, templates_dir, reader)
        sys.exit(0 if success else 1)
    else:
        # Process all definitions
        definition_files = list(definitions_dir.glob('*.json'))
        logger.info(f"Found {len(definition_files)} definition files")

        successful = 0
        failed = 0

        for definition_path in definition_files:
            if populate_definition(definition_path, templates_dir, reader):
                successful += 1
            else:
                failed += 1

        logger.info(f"\nSummary: {successful} successful, {failed} failed")


if __name__ == '__main__':
    main()
