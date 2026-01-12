import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import logging
from template_matcher import TemplateMatcher, FormTypeRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """Analyze document layout to detect form fields and structure."""

    def __init__(self, templates_dir: Optional[str] = None, registry_path: Optional[str] = None):
        """
        Initialize layout analyzer.

        Args:
            templates_dir: Directory containing form template images
            registry_path: Path to form type registry JSON
        """
        self.min_line_length = 50
        self.min_box_area = 500

        # Initialize template matching
        self.template_matcher = TemplateMatcher(templates_dir)
        self.form_registry = FormTypeRegistry(registry_path)

    def detect_lines(self, image: np.ndarray) -> Dict[str, List]:
        """
        Detect horizontal and vertical lines in the document.

        Returns:
            Dictionary with 'horizontal' and 'vertical' line lists
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Find line contours
        h_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_lines = []
        for cnt in h_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > self.min_line_length:
                h_lines.append({'x': x, 'y': y, 'width': w, 'height': h})

        v_lines = []
        for cnt in v_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > self.min_line_length:
                v_lines.append({'x': x, 'y': y, 'width': w, 'height': h})

        logger.info(f"Detected {len(h_lines)} horizontal, {len(v_lines)} vertical lines")
        return {'horizontal': h_lines, 'vertical': v_lines}

    def detect_boxes(self, image: np.ndarray) -> List[Dict]:
        """
        Detect rectangular boxes/cells in the document.

        Returns:
            List of box coordinates
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_box_area:
                continue

            # Approximate polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0

                # Filter out very thin or very wide boxes
                if 0.1 < aspect_ratio < 10:
                    boxes.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'area': area,
                        'aspect_ratio': round(aspect_ratio, 2)
                    })

        # Sort by position (top-to-bottom, left-to-right)
        boxes.sort(key=lambda b: (b['y'], b['x']))

        logger.info(f"Detected {len(boxes)} boxes")
        return boxes

    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect potential text regions using MSER.

        Returns:
            List of text region bounding boxes
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        # Group nearby regions
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w > 10 and h > 5:
                text_regions.append({'x': x, 'y': y, 'width': w, 'height': h})

        # Merge overlapping regions
        merged = self._merge_overlapping_boxes(text_regions)

        logger.info(f"Detected {len(merged)} text regions")
        return merged

    def _merge_overlapping_boxes(self, boxes: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """Merge overlapping bounding boxes."""
        if not boxes:
            return []

        # Convert to numpy array for processing
        rects = np.array([[b['x'], b['y'], b['x'] + b['width'], b['y'] + b['height']] for b in boxes])

        # Simple merge: group boxes that overlap significantly
        merged = []
        used = set()

        for i, rect1 in enumerate(rects):
            if i in used:
                continue

            x1, y1, x2, y2 = rect1
            for j, rect2 in enumerate(rects[i+1:], i+1):
                if j in used:
                    continue

                # Check overlap
                xx1, yy1, xx2, yy2 = rect2
                if not (x2 < xx1 or xx2 < x1 or y2 < yy1 or yy2 < y1):
                    # Merge
                    x1 = min(x1, xx1)
                    y1 = min(y1, yy1)
                    x2 = max(x2, xx2)
                    y2 = max(y2, yy2)
                    used.add(j)

            merged.append({
                'x': int(x1), 'y': int(y1),
                'width': int(x2 - x1), 'height': int(y2 - y1)
            })
            used.add(i)

        return merged

    def identify_form_fields(self, boxes: List[Dict], text_regions: List[Dict]) -> List[Dict]:
        """
        Identify potential form fields by analyzing boxes and nearby text.

        Returns:
            List of identified form fields with coordinates
        """
        fields = []

        for i, box in enumerate(boxes):
            # Look for text regions near this box (potential labels)
            nearby_text = []
            for text in text_regions:
                # Check if text is to the left or above the box
                if (text['x'] < box['x'] and
                    abs(text['y'] - box['y']) < box['height']):
                    nearby_text.append(text)
                elif (text['y'] < box['y'] and
                      abs(text['x'] - box['x']) < box['width']):
                    nearby_text.append(text)

            field = {
                'field_id': f"field_{i:03d}",
                'bounds': {
                    'x': box['x'],
                    'y': box['y'],
                    'width': box['width'],
                    'height': box['height']
                },
                'type': self._classify_field_type(box),
                'label_regions': nearby_text
            }
            fields.append(field)

        return fields

    def _classify_field_type(self, box: Dict) -> str:
        """Classify field type based on dimensions."""
        aspect_ratio = box['aspect_ratio']
        area = box['area']

        if aspect_ratio > 5:
            return 'text_line'
        elif aspect_ratio > 2:
            return 'text_field'
        elif 0.8 < aspect_ratio < 1.2 and area < 2000:
            return 'checkbox'
        elif area > 10000:
            return 'text_area'
        else:
            return 'unknown'

    def analyze(self, image_path: str) -> Dict:
        """
        Perform full layout analysis on an image.

        Args:
            image_path: Path to the preprocessed image

        Returns:
            Complete layout analysis results
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        logger.info(f"Analyzing layout: {image_path}")

        # Identify form type first
        form_info = self.template_matcher.identify_form_type(image_path)
        logger.info(f"Form identification: matched={form_info['template_match']['matched']}, "
                   f"type={form_info.get('form_type', 'unknown')}")

        # Detect elements
        lines = self.detect_lines(image)
        boxes = self.detect_boxes(image)
        text_regions = self.detect_text_regions(image)

        # Identify form fields
        fields = self.identify_form_fields(boxes, text_regions)

        # If form type is known, get predefined field definitions
        form_definition = None
        if form_info.get('form_type'):
            form_definition = self.form_registry.get_form_definition(form_info['form_type'])

        result = {
            'image_path': str(image_path),
            'image_size': {'width': image.shape[1], 'height': image.shape[0]},
            'form_identification': {
                'form_type': form_info.get('form_type'),
                'header_image': form_info.get('header_image'),
                'header_bounds': form_info.get('header_bounds'),
                'template_matched': form_info['template_match']['matched'],
                'match_confidence': form_info['template_match'].get('best_match', {}).get('similarity'),
                'needs_ocr_for_type': form_info.get('needs_ocr', True)
            },
            'form_definition': form_definition,
            'lines': lines,
            'boxes': boxes,
            'text_regions': text_regions,
            'fields': fields,
            'summary': {
                'total_lines': len(lines['horizontal']) + len(lines['vertical']),
                'total_boxes': len(boxes),
                'total_text_regions': len(text_regions),
                'total_fields': len(fields)
            }
        }

        return result
