import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available, using fallback")


class OCREngine:
    """OCR engine using EasyOCR with GPU support."""

    def __init__(self, langs: List[str] = None, use_gpu: bool = True):
        """
        Initialize OCR engine.

        Args:
            langs: Language codes (e.g., ['de', 'en'] for German and English)
            use_gpu: Whether to use GPU acceleration
        """
        self.langs = langs or ['de', 'en']
        self.use_gpu = use_gpu
        self.reader = None

        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(
                    self.langs,
                    gpu=use_gpu
                )
                logger.info(f"EasyOCR initialized (langs={self.langs}, GPU={use_gpu})")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.reader = None

    def extract_text(self, image_path: str, max_dimension: int = 2000) -> Dict:
        """
        Extract text from an image using OCR.

        Args:
            image_path: Path to the image file
            max_dimension: Maximum width/height before scaling down

        Returns:
            Dictionary with OCR results
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        logger.info(f"Running OCR on: {image_path} (size: {image.shape[1]}x{image.shape[0]})")

        # Scale down large images to reduce memory usage
        h, w = image.shape[:2]
        scale_factor = 1.0
        if max(h, w) > max_dimension:
            scale_factor = max_dimension / max(h, w)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Scaled image to {new_w}x{new_h} (factor: {scale_factor:.2f})")

        if self.reader is None:
            return self._fallback_result(image_path)

        try:
            # EasyOCR returns list of (bbox, text, confidence)
            # Pass numpy array instead of path to use scaled image
            result = self.reader.readtext(image)

            if result is None or len(result) == 0:
                return {
                    'image_path': str(image_path),
                    'text_blocks': [],
                    'full_text': '',
                    'confidence': 0.0
                }

            text_blocks = []
            full_text_parts = []
            total_confidence = 0

            for detection in result:
                bbox, text, confidence = detection

                # Convert bbox to x,y,w,h format
                # EasyOCR bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                # Scale coordinates back to original image size
                inv_scale = 1.0 / scale_factor
                x_coords = [p[0] * inv_scale for p in bbox]
                y_coords = [p[1] * inv_scale for p in bbox]
                x, y = int(min(x_coords)), int(min(y_coords))
                w = int(max(x_coords) - x)
                h = int(max(y_coords) - y)

                text_blocks.append({
                    'text': text,
                    'confidence': round(float(confidence), 4),
                    'bounds': {'x': x, 'y': y, 'width': w, 'height': h},
                    'bbox': [[int(p[0] * inv_scale), int(p[1] * inv_scale)] for p in bbox]
                })

                full_text_parts.append(text)
                total_confidence += confidence

            avg_confidence = total_confidence / len(result) if result else 0

            return {
                'image_path': str(image_path),
                'text_blocks': text_blocks,
                'full_text': '\n'.join(full_text_parts),
                'confidence': round(avg_confidence, 4),
                'block_count': len(text_blocks)
            }

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return self._fallback_result(image_path)

    def _fallback_result(self, image_path: str) -> Dict:
        """Return a placeholder result when OCR is not available."""
        return {
            'image_path': str(image_path),
            'text_blocks': [],
            'full_text': '[OCR not available]',
            'confidence': 0.0,
            'error': 'EasyOCR not initialized'
        }

    def detect_orientation(self, image: np.ndarray, sample_height: int = 400) -> int:
        """
        Detect document orientation using OCR confidence.

        Tests all 4 orientations (0°, 90°, 180°, 270°) on a sample region
        and returns the rotation angle that produces the best OCR results.

        Args:
            image: Input image as numpy array
            sample_height: Height of sample region to test (from top of image)

        Returns:
            Rotation angle needed to correct orientation (0, 90, 180, or -90)
        """
        if self.reader is None:
            logger.warning("EasyOCR not available, cannot detect orientation")
            return 0

        h, w = image.shape[:2]

        # Take a sample from the top portion of the image for faster processing
        sample_h = min(sample_height, h // 3)

        orientations = [
            (0, None),                              # Original
            (90, cv2.ROTATE_90_CLOCKWISE),          # 90° clockwise
            (180, cv2.ROTATE_180),                  # 180°
            (-90, cv2.ROTATE_90_COUNTERCLOCKWISE),  # 90° counter-clockwise
        ]

        best_orientation = 0
        best_score = -1

        for angle, rotate_code in orientations:
            # Rotate full image
            if rotate_code is not None:
                rotated = cv2.rotate(image, rotate_code)
            else:
                rotated = image

            # Take sample from top of rotated image
            rh, rw = rotated.shape[:2]
            sample = rotated[0:min(sample_h, rh), 0:rw]

            # Scale down for faster OCR
            max_dim = 800
            sh, sw = sample.shape[:2]
            if max(sh, sw) > max_dim:
                scale = max_dim / max(sh, sw)
                sample = cv2.resize(sample, (int(sw * scale), int(sh * scale)))

            try:
                # Run OCR on sample
                result = self.reader.readtext(sample)

                if result:
                    # Score based on number of detections and average confidence
                    num_detections = len(result)
                    avg_confidence = sum(r[2] for r in result) / num_detections
                    # Combined score: more detections with higher confidence is better
                    score = num_detections * avg_confidence

                    logger.info(f"Orientation {angle}°: {num_detections} detections, "
                               f"avg confidence {avg_confidence:.3f}, score {score:.2f}")

                    if score > best_score:
                        best_score = score
                        best_orientation = angle
                else:
                    logger.info(f"Orientation {angle}°: no text detected")

            except Exception as e:
                logger.warning(f"OCR failed for orientation {angle}°: {e}")

        logger.info(f"Best orientation: {best_orientation}° (score: {best_score:.2f})")
        return best_orientation

    def correct_orientation(self, image: np.ndarray) -> tuple:
        """
        Detect and correct document orientation.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (corrected image, rotation angle applied)
        """
        angle = self.detect_orientation(image)

        if angle == 0:
            return image, 0

        if angle == 90:
            corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == -90:
            corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            corrected = cv2.rotate(image, cv2.ROTATE_180)
        else:
            corrected = image
            angle = 0

        logger.info(f"Orientation corrected by {angle}°")
        return corrected, angle

    def extract_from_regions(self, image_path: str, regions: List[Dict]) -> List[Dict]:
        """
        Extract text from specific regions of an image.

        Args:
            image_path: Path to the image file
            regions: List of region dictionaries with x, y, width, height

        Returns:
            List of OCR results for each region
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        results = []
        for i, region in enumerate(regions):
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('width', image.shape[1])
            h = region.get('height', image.shape[0])

            # Extract region
            roi = image[y:y+h, x:x+w]

            # Save temporary file for OCR
            temp_path = f"/tmp/region_{i}.png"
            cv2.imwrite(temp_path, roi)

            # Run OCR
            ocr_result = self.extract_text(temp_path)
            ocr_result['region'] = region
            results.append(ocr_result)

        return results


class AnomalyDetector:
    """Simple anomaly detection for document changes."""

    def __init__(self, reference_fields: Optional[Dict] = None):
        """
        Initialize anomaly detector.

        Args:
            reference_fields: Expected field values for comparison
        """
        self.reference_fields = reference_fields or {}

    def compare_fields(self, extracted_fields: Dict) -> Dict:
        """
        Compare extracted fields against reference.

        Args:
            extracted_fields: Dictionary of field_name -> extracted_value

        Returns:
            Anomaly detection results
        """
        if not self.reference_fields:
            return {
                'anomaly_score': 0.0,
                'changed_fields': [],
                'missing_fields': [],
                'new_fields': list(extracted_fields.keys())
            }

        changed = []
        missing = []

        for field, expected in self.reference_fields.items():
            if field not in extracted_fields:
                missing.append(field)
            elif extracted_fields[field] != expected:
                changed.append({
                    'field': field,
                    'expected': expected,
                    'actual': extracted_fields[field]
                })

        new_fields = [f for f in extracted_fields if f not in self.reference_fields]

        # Simple anomaly score
        total_fields = len(self.reference_fields)
        anomaly_count = len(changed) + len(missing)
        anomaly_score = anomaly_count / total_fields if total_fields > 0 else 0

        return {
            'anomaly_score': round(anomaly_score, 4),
            'changed_fields': changed,
            'missing_fields': missing,
            'new_fields': new_fields
        }
