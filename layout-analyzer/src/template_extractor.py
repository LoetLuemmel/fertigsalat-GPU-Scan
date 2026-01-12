"""
Template Extractor for Order Forms (Bestellblätter).

Automatically extracts grid structure and product definitions from template forms.

Process:
1. Detect horizontal/vertical lines to build grid
2. Identify row/column intersections as cells
3. Classify columns (product_number, product_name, input columns)
4. Extract header row to identify weekday columns
5. Save as JSON definition
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateExtractor:
    """
    Extract grid structure and product definitions from template forms.
    """

    # Expected column headers for input columns
    WEEKDAY_HEADERS = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'Montag', 'Dienstag',
                       'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag']

    # Minimum dimensions for valid cells
    MIN_CELL_WIDTH = 30
    MIN_CELL_HEIGHT = 20

    # Line clustering tolerance (pixels)
    LINE_CLUSTER_TOLERANCE = 10

    def __init__(self, min_line_length: int = 50):
        """
        Initialize template extractor.

        Args:
            min_line_length: Minimum length for detected lines
        """
        self.min_line_length = min_line_length

    def extract_from_image(self, image_path: str) -> Dict:
        """
        Extract template definition from an image.

        Args:
            image_path: Path to template image (PNG)

        Returns:
            Template definition dictionary
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        logger.info(f"Extracting template from: {image_path}")
        logger.info(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")

        # Detect grid lines
        h_lines, v_lines = self._detect_grid_lines(image)
        logger.info(f"Detected {len(h_lines)} horizontal, {len(v_lines)} vertical grid lines")

        # Cluster lines to get grid positions
        h_positions = self._cluster_lines(h_lines, axis='horizontal')
        v_positions = self._cluster_lines(v_lines, axis='vertical')
        logger.info(f"Clustered to {len(h_positions)} rows, {len(v_positions)} columns")

        # Build cell grid
        cells = self._build_cell_grid(h_positions, v_positions)
        logger.info(f"Built grid with {len(cells)} rows")

        # Identify column types
        columns = self._identify_columns(cells, image)

        # Extract product rows (skip header)
        products = self._extract_products(cells, columns, image)
        logger.info(f"Extracted {len(products)} product rows")

        # Build definition
        form_type = Path(image_path).stem
        definition = {
            'form_type': form_type,
            'version': '1.0',
            'extracted_from': str(image_path),
            'extraction_date': datetime.now().isoformat(),
            'image_dimensions': {
                'width': image.shape[1],
                'height': image.shape[0]
            },
            'grid': {
                'horizontal_lines': h_positions,
                'vertical_lines': v_positions,
                'row_count': len(h_positions) - 1,
                'column_count': len(v_positions) - 1
            },
            'columns': columns,
            'products': products,
            'empty_cell_threshold': {
                'variance': 50,
                'white_ratio': 0.95
            }
        }

        return definition

    def _detect_grid_lines(self, image: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect horizontal and vertical grid lines.

        Returns:
            Tuple of (horizontal_lines, vertical_lines)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Binarize with adaptive threshold for better line detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )

        # Horizontal lines - use longer kernel for table lines
        h_kernel_size = max(image.shape[1] // 30, 40)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # Vertical lines
        v_kernel_size = max(image.shape[0] // 30, 40)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        # Find horizontal line contours
        h_contours, _ = cv2.findContours(h_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_lines = []
        for cnt in h_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > self.min_line_length:
                h_lines.append({'x': x, 'y': y + h // 2, 'width': w, 'height': h})

        # Find vertical line contours
        v_contours, _ = cv2.findContours(v_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_lines = []
        for cnt in v_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > self.min_line_length:
                v_lines.append({'x': x + w // 2, 'y': y, 'width': w, 'height': h})

        return h_lines, v_lines

    def _cluster_lines(self, lines: List[Dict], axis: str,
                       tolerance: int = None) -> List[int]:
        """
        Cluster nearby lines into grid positions.

        Args:
            lines: List of line dictionaries
            axis: 'horizontal' or 'vertical'
            tolerance: Clustering tolerance in pixels

        Returns:
            Sorted list of unique grid positions
        """
        if tolerance is None:
            tolerance = self.LINE_CLUSTER_TOLERANCE

        # Extract relevant coordinate
        if axis == 'horizontal':
            positions = [line['y'] for line in lines]
        else:
            positions = [line['x'] for line in lines]

        if not positions:
            return []

        # Sort positions
        positions = sorted(positions)

        # Cluster nearby positions
        clusters = []
        current_cluster = [positions[0]]

        for pos in positions[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                # Save cluster center
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [pos]

        # Don't forget last cluster
        clusters.append(int(np.mean(current_cluster)))

        return clusters

    def _build_cell_grid(self, h_positions: List[int],
                         v_positions: List[int]) -> List[List[Dict]]:
        """
        Build 2D grid of cells from line positions.

        Returns:
            List of rows, each containing cell definitions
        """
        cells = []

        for row_idx in range(len(h_positions) - 1):
            row_cells = []
            y1 = h_positions[row_idx]
            y2 = h_positions[row_idx + 1]

            for col_idx in range(len(v_positions) - 1):
                x1 = v_positions[col_idx]
                x2 = v_positions[col_idx + 1]

                width = x2 - x1
                height = y2 - y1

                # Skip very small cells
                if width < self.MIN_CELL_WIDTH or height < self.MIN_CELL_HEIGHT:
                    continue

                cell = {
                    'row': row_idx,
                    'col': col_idx,
                    'x': x1,
                    'y': y1,
                    'width': width,
                    'height': height
                }
                row_cells.append(cell)

            if row_cells:
                cells.append(row_cells)

        return cells

    def _identify_columns(self, cells: List[List[Dict]],
                         image: np.ndarray) -> Dict:
        """
        Identify column types based on position and width.

        Returns:
            Column definitions dictionary
        """
        if not cells or not cells[0]:
            return {}

        # Analyze first data row to understand structure
        first_row = cells[0] if len(cells) > 0 else []

        columns = {
            'product_number': None,
            'product_name': None,
            'input_columns': []
        }

        # Sort cells by x position
        sorted_cells = sorted(first_row, key=lambda c: c['x'])

        if len(sorted_cells) < 3:
            logger.warning("Not enough columns detected for a valid form")
            return columns

        # First column is typically product number (narrow)
        columns['product_number'] = {
            'index': 0,
            'x_start': sorted_cells[0]['x'],
            'x_end': sorted_cells[0]['x'] + sorted_cells[0]['width'],
            'width': sorted_cells[0]['width'],
            'type': 'static'
        }

        # Second column is typically product name (wider)
        columns['product_name'] = {
            'index': 1,
            'x_start': sorted_cells[1]['x'],
            'x_end': sorted_cells[1]['x'] + sorted_cells[1]['width'],
            'width': sorted_cells[1]['width'],
            'type': 'static'
        }

        # Remaining columns are input columns
        # Map to weekday names based on position
        weekday_names = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa']

        for i, cell in enumerate(sorted_cells[2:]):
            col_name = weekday_names[i] if i < len(weekday_names) else f'Col{i+3}'
            columns['input_columns'].append({
                'index': i + 2,
                'name': col_name,
                'x_start': cell['x'],
                'x_end': cell['x'] + cell['width'],
                'width': cell['width'],
                'type': 'input'
            })

        logger.info(f"Identified columns: product_number, product_name, "
                   f"{len(columns['input_columns'])} input columns")

        return columns

    def _extract_products(self, cells: List[List[Dict]],
                         columns: Dict, image: np.ndarray) -> List[Dict]:
        """
        Extract product definitions from grid.

        Note: Product numbers and names will be populated by OCR later.
        This method sets up the structure with cell coordinates.

        Returns:
            List of product definitions
        """
        products = []

        if not columns.get('input_columns'):
            return products

        input_cols = columns['input_columns']

        # Skip header rows (typically first 3-5 rows)
        # We'll detect header by looking for rows with different structure
        header_rows = self._detect_header_rows(cells, image)
        logger.info(f"Detected {header_rows} header rows")

        for row_idx, row in enumerate(cells[header_rows:], start=header_rows):
            if not row:
                continue

            # Sort row cells by x position
            sorted_row = sorted(row, key=lambda c: c['x'])

            if len(sorted_row) < 3:
                continue

            # Build cell mappings for input columns
            input_cells = {}
            for i, col_def in enumerate(input_cols):
                col_idx = i + 2  # Skip product_number and product_name columns
                if col_idx < len(sorted_row):
                    cell = sorted_row[col_idx]
                    input_cells[col_def['name']] = {
                        'x': cell['x'],
                        'y': cell['y'],
                        'width': cell['width'],
                        'height': cell['height']
                    }

            # Product number and name cells
            product_number_cell = sorted_row[0] if len(sorted_row) > 0 else None
            product_name_cell = sorted_row[1] if len(sorted_row) > 1 else None

            product = {
                'row': row_idx,
                'product_number': '',  # To be filled by OCR
                'product_name': '',    # To be filled by OCR
                'product_number_cell': {
                    'x': product_number_cell['x'],
                    'y': product_number_cell['y'],
                    'width': product_number_cell['width'],
                    'height': product_number_cell['height']
                } if product_number_cell else None,
                'product_name_cell': {
                    'x': product_name_cell['x'],
                    'y': product_name_cell['y'],
                    'width': product_name_cell['width'],
                    'height': product_name_cell['height']
                } if product_name_cell else None,
                'cells': input_cells
            }

            products.append(product)

        return products

    def _detect_header_rows(self, cells: List[List[Dict]],
                           image: np.ndarray) -> int:
        """
        Detect number of header rows by analyzing cell content/structure.

        Returns:
            Number of header rows to skip
        """
        # Simple heuristic: header rows often have different height
        # or are in the top ~10% of the image

        if not cells:
            return 0

        image_height = image.shape[0]
        header_threshold = image_height * 0.15  # Top 15%

        header_rows = 0
        for row in cells:
            if row and row[0]['y'] < header_threshold:
                header_rows += 1
            else:
                break

        # Minimum 1 header row for column labels
        return max(header_rows, 1)

    def save_definition(self, definition: Dict, output_path: str):
        """
        Save template definition to JSON file.

        Args:
            definition: Template definition dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(definition, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved template definition to: {output_path}")

    def visualize_grid(self, image_path: str, definition: Dict,
                       output_path: str):
        """
        Visualize extracted grid on the template image.

        Args:
            image_path: Path to template image
            definition: Template definition
            output_path: Path to save visualization
        """
        image = cv2.imread(image_path)
        if image is None:
            return

        # Draw horizontal lines
        for y in definition['grid']['horizontal_lines']:
            cv2.line(image, (0, y), (image.shape[1], y), (0, 255, 0), 1)

        # Draw vertical lines
        for x in definition['grid']['vertical_lines']:
            cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 0), 1)

        # Draw input cells in blue
        for product in definition['products']:
            for col_name, cell in product['cells'].items():
                x, y = cell['x'], cell['y']
                w, h = cell['width'], cell['height']
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imwrite(output_path, image)
        logger.info(f"Saved grid visualization to: {output_path}")


def extract_template(image_path: str, output_dir: str = None) -> Dict:
    """
    Convenience function to extract template definition.

    Args:
        image_path: Path to template image
        output_dir: Directory to save JSON definition (optional)

    Returns:
        Template definition dictionary
    """
    extractor = TemplateExtractor()
    definition = extractor.extract_from_image(image_path)

    if output_dir:
        output_path = Path(output_dir) / f"{definition['form_type']}.json"
        extractor.save_definition(definition, str(output_path))

    return definition


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract template definition from form image')
    parser.add_argument('image_path', help='Path to template image')
    parser.add_argument('--output-dir', '-o', help='Output directory for JSON definition')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate grid visualization')

    args = parser.parse_args()

    extractor = TemplateExtractor()
    definition = extractor.extract_from_image(args.image_path)

    if args.output_dir:
        output_path = Path(args.output_dir) / f"{definition['form_type']}.json"
        extractor.save_definition(definition, str(output_path))
    else:
        print(json.dumps(definition, indent=2, ensure_ascii=False))

    if args.visualize:
        vis_path = Path(args.image_path).with_name(
            f"{definition['form_type']}_grid.png"
        )
        extractor.visualize_grid(args.image_path, definition, str(vis_path))
