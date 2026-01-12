import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from typing import Tuple, Optional
import json
import logging
from pyzbar import pyzbar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """OpenCV-based document image processor."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: str, output_dir: str) -> list[str]:
        """Convert PDF pages to images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=self.dpi)

        image_paths = []
        for i, image in enumerate(images):
            img_path = output_dir / f"page_{i:03d}.png"
            image.save(img_path, 'PNG')
            image_paths.append(str(img_path))
            logger.info(f"Saved page {i}: {img_path}")

        return image_paths

    def deskew(self, image: np.ndarray, max_angle: float = 15.0) -> Tuple[np.ndarray, float]:
        """
        Correct small skew/rotation in document image.

        Only corrects small angles (default ±15°). Large rotations like 90°
        are not applied as they likely indicate incorrect detection.

        Args:
            image: Input image
            max_angle: Maximum rotation angle to apply (default 15°)

        Returns:
            Tuple of (deskewed image, rotation angle in degrees)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.bitwise_not(gray)

        coords = np.column_stack(np.where(gray > 0))

        if len(coords) < 5:
            return image, 0.0

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Skip very small angles
        if abs(angle) < 0.5:
            logger.info(f"Deskew skipped: angle {angle:.2f}° too small")
            return image, 0.0

        # Skip large angles - these are likely false detections
        # Documents rarely need more than ±15° correction
        if abs(angle) > max_angle:
            logger.warning(f"Deskew skipped: angle {angle:.2f}° exceeds max {max_angle}° (likely incorrect detection)")
            return image, 0.0

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        logger.info(f"Deskew applied: {angle:.2f} degrees")
        return rotated, angle

    def detect_orientation_by_qr(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect document orientation using QR code position.

        The QR code should be in the top-left corner of a correctly oriented document.
        Tests all 4 orientations to find the QR code, then determines correct orientation.
        Uses pyzbar for robust QR code detection.

        Args:
            image: Input image

        Returns:
            Tuple of (corrected image, rotation angle applied in degrees)
        """
        # Test all 4 orientations to find the QR code
        orientations = [
            (0, None, 0),                              # Original
            (90, cv2.ROTATE_90_CLOCKWISE, -90),        # Rotate 90° CW to test
            (180, cv2.ROTATE_180, 180),                # Rotate 180° to test
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE, 90), # Rotate 270° CW to test
        ]

        for test_angle, rotate_code, correction in orientations:
            # Rotate image for testing
            if rotate_code is not None:
                test_image = cv2.rotate(image, rotate_code)
            else:
                test_image = image

            h, w = test_image.shape[:2]

            # Convert to grayscale for pyzbar
            if len(test_image.shape) == 3:
                gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = test_image

            # Try to detect QR code with pyzbar
            decoded_objects = pyzbar.decode(gray)
            qr_codes = [obj for obj in decoded_objects if obj.type == 'QRCODE']

            if qr_codes:
                # QR code found! Get its position
                qr = qr_codes[0]
                # pyzbar returns rect as (left, top, width, height)
                qr_center_x = qr.rect.left + qr.rect.width / 2
                qr_center_y = qr.rect.top + qr.rect.height / 2

                in_left_half = qr_center_x < w / 2
                in_top_half = qr_center_y < h / 2

                logger.info(f"QR code found at test rotation {test_angle}°, "
                           f"position: ({qr_center_x:.0f}, {qr_center_y:.0f}), "
                           f"left={in_left_half}, top={in_top_half}")

                if in_left_half and in_top_half:
                    # QR is in top-left after this rotation - this is correct orientation
                    if correction == 0:
                        logger.info("Document already correctly oriented")
                        return image, 0
                    else:
                        logger.info(f"Orientation correction applied: {correction}°")
                        return test_image, correction

        logger.info("No QR code detected in any orientation, skipping correction")
        return image, 0

    def auto_crop(self, image: np.ndarray, padding: int = 10) -> Tuple[np.ndarray, dict]:
        """
        Automatically crop to content area.

        Returns:
            Tuple of (cropped image, crop coordinates dict)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = cv2.findNonZero(thresh)

        if coords is None:
            return image, {'x': 0, 'y': 0, 'w': image.shape[1], 'h': image.shape[0]}

        x, y, w, h = cv2.boundingRect(coords)

        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        cropped = image[y:y+h, x:x+w]
        crop_coords = {'x': x, 'y': y, 'w': w, 'h': h}

        logger.info(f"Auto-crop: {crop_coords}")
        return cropped, crop_coords

    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """Apply denoising filter."""
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)

        logger.info("Applied denoising filter")
        return denoised

    def adaptive_threshold(self, image: np.ndarray, block_size: int = 11,
                          c: int = 2) -> np.ndarray:
        """
        Apply adaptive thresholding for better contrast.

        Args:
            image: Input image
            block_size: Size of pixel neighborhood (must be odd)
            c: Constant subtracted from mean

        Returns:
            Binary thresholded image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c
        )

        logger.info("Applied adaptive threshold")
        return thresh

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        logger.info("Applied CLAHE contrast enhancement")
        return enhanced

    def process_document(self, image_path: str, output_dir: str,
                        apply_threshold: bool = False) -> dict:
        """
        Full document processing pipeline.

        Args:
            image_path: Path to input image
            output_dir: Directory for output files
            apply_threshold: Whether to apply adaptive thresholding

        Returns:
            Dictionary with processing metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        original_shape = image.shape
        logger.info(f"Processing image: {image_path} ({original_shape})")

        meta = {
            'input_file': str(image_path),
            'original_shape': list(original_shape),
            'steps': []
        }

        # Step 1: Orientation correction (90°/180°/270° via QR code detection)
        image, orientation_angle = self.detect_orientation_by_qr(image)
        meta['steps'].append({'orientation': {'angle': orientation_angle}})

        # Step 2: Deskew (fine adjustment for small angles)
        image, deskew_angle = self.deskew(image)
        meta['steps'].append({'deskew': {'angle': deskew_angle}})

        # Step 3: Denoise
        image = self.denoise(image)
        meta['steps'].append({'denoise': True})

        # Step 4: Enhance contrast
        image = self.enhance_contrast(image)
        meta['steps'].append({'contrast_enhancement': 'CLAHE'})

        # Step 5: Auto-crop
        image, crop_coords = self.auto_crop(image)
        meta['steps'].append({'auto_crop': crop_coords})

        # Step 6: Optional threshold
        if apply_threshold:
            image = self.adaptive_threshold(image)
            meta['steps'].append({'threshold': 'adaptive_gaussian'})

        # Save normalized image
        input_name = Path(image_path).stem
        output_path = output_dir / f"{input_name}_normalized.png"
        cv2.imwrite(str(output_path), image)
        meta['output_file'] = str(output_path)
        meta['output_shape'] = list(image.shape)

        # Save metadata
        meta_path = output_dir / f"{input_name}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved normalized image: {output_path}")
        logger.info(f"Saved metadata: {meta_path}")

        return meta
