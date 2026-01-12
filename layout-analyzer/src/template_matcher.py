import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateMatcher:
    """
    Template matching for form type identification.

    Identifies form types by:
    1. Extracting the header region (top-left area with form identifier)
    2. Matching against known template signatures
    3. Providing the header region for OCR extraction
    """

    # Region for form type identifier (relative to image dimensions)
    # Default is top-left for non-rotated documents
    HEADER_REGION = {
        'x_start': 0.0,    # Start at left edge
        'x_end': 0.4,      # 40% of width
        'y_start': 0.0,    # Start at top
        'y_end': 0.12      # 12% of height
    }

    # Header regions for different rotation angles
    # After rotation, the original top-left header moves to a different position
    HEADER_REGIONS_BY_ROTATION = {
        0: {'x_start': 0.0, 'x_end': 0.4, 'y_start': 0.0, 'y_end': 0.12},      # Original: top-left
        -90: {'x_start': 0.0, 'x_end': 0.12, 'y_start': 0.6, 'y_end': 1.0},    # After -90°: bottom-left
        90: {'x_start': 0.88, 'x_end': 1.0, 'y_start': 0.0, 'y_end': 0.4},     # After +90°: top-right
        180: {'x_start': 0.6, 'x_end': 1.0, 'y_start': 0.88, 'y_end': 1.0},    # After 180°: bottom-right
    }

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize template matcher.

        Args:
            templates_dir: Directory containing form template images
        """
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.templates: Dict[str, dict] = {}
        self.template_signatures: Dict[str, np.ndarray] = {}

        if self.templates_dir and self.templates_dir.exists():
            self._load_templates()

    def _load_templates(self):
        """Load all template images from templates directory."""
        if not self.templates_dir:
            return

        supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.pdf']

        for pattern in supported_formats:
            for template_path in self.templates_dir.glob(pattern):
                self._load_single_template(template_path)

        logger.info(f"Loaded {len(self.templates)} templates from {self.templates_dir}")

    def _load_single_template(self, template_path: Path):
        """Load a single template and extract its signature."""
        try:
            image = cv2.imread(str(template_path))
            if image is None:
                logger.warning(f"Could not load template: {template_path}")
                return

            # Extract header region
            header = self._extract_header_region(image)

            # Create signature (normalized grayscale histogram + structural features)
            signature = self._create_signature(header)

            template_name = template_path.stem
            self.templates[template_name] = {
                'path': str(template_path),
                'size': {'width': image.shape[1], 'height': image.shape[0]},
                'header_size': {'width': header.shape[1], 'height': header.shape[0]}
            }
            self.template_signatures[template_name] = signature

            logger.info(f"Loaded template: {template_name}")

        except Exception as e:
            logger.error(f"Error loading template {template_path}: {e}")

    def _extract_header_region(self, image: np.ndarray, rotation: float = 0) -> np.ndarray:
        """
        Extract the header region where form type is located.
        Adjusts for rotation applied during preprocessing.

        Args:
            image: Full document image
            rotation: Rotation angle applied during preprocessing (e.g., -90, 0, 90, 180)

        Returns:
            Cropped header region (rotated back to readable orientation)
        """
        h, w = image.shape[:2]

        # Get the appropriate header region based on rotation
        rotation_key = int(round(rotation / 90) * 90)  # Normalize to nearest 90°
        if rotation_key not in self.HEADER_REGIONS_BY_ROTATION:
            rotation_key = 0

        region = self.HEADER_REGIONS_BY_ROTATION.get(rotation_key, self.HEADER_REGION)
        logger.info(f"Using header region for rotation {rotation_key}°: {region}")

        x_start = int(w * region['x_start'])
        x_end = int(w * region['x_end'])
        y_start = int(h * region['y_start'])
        y_end = int(h * region['y_end'])

        header = image[y_start:y_end, x_start:x_end]

        # Rotate header back to readable orientation
        if rotation_key == -90:
            header = cv2.rotate(header, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_key == 90:
            header = cv2.rotate(header, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_key == 180:
            header = cv2.rotate(header, cv2.ROTATE_180)

        return header

    def _create_signature(self, region: np.ndarray) -> np.ndarray:
        """
        Create a signature for template matching.

        Combines:
        - Normalized histogram
        - Edge density
        - Structural hash
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

        # Resize to standard size for comparison
        standard_size = (200, 60)
        resized = cv2.resize(gray, standard_size)

        # Histogram
        hist = cv2.calcHist([resized], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Edge features
        edges = cv2.Canny(resized, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Simple structural features (divide into grid and get mean values)
        grid_h, grid_w = 3, 6
        cell_h, cell_w = standard_size[1] // grid_h, standard_size[0] // grid_w
        grid_features = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = resized[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                grid_features.append(np.mean(cell) / 255.0)

        # Combine features
        signature = np.concatenate([
            hist,
            [edge_density],
            grid_features
        ])

        return signature

    def get_header_bounds(self, image: np.ndarray, rotation: float = 0) -> Dict:
        """
        Get the bounds of the header region for a given image.

        Args:
            image: Document image
            rotation: Rotation angle applied during preprocessing

        Returns:
            Dictionary with x, y, width, height of header region
        """
        h, w = image.shape[:2]

        # Get the appropriate header region based on rotation
        rotation_key = int(round(rotation / 90) * 90)
        if rotation_key not in self.HEADER_REGIONS_BY_ROTATION:
            rotation_key = 0
        region = self.HEADER_REGIONS_BY_ROTATION.get(rotation_key, self.HEADER_REGION)

        x = int(w * region['x_start'])
        y = int(h * region['y_start'])
        width = int(w * region['x_end']) - x
        height = int(h * region['y_end']) - y

        return {'x': x, 'y': y, 'width': width, 'height': height}

    def extract_header_for_ocr(self, image_path: str, output_path: str) -> Dict:
        """
        Extract header region and save for OCR processing.

        Args:
            image_path: Path to document image
            output_path: Path to save extracted header

        Returns:
            Dictionary with header bounds and output path
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        header = self._extract_header_region(image)
        bounds = self.get_header_bounds(image)

        # Enhance for OCR
        gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        cv2.imwrite(output_path, enhanced)

        return {
            'header_image': output_path,
            'bounds': bounds,
            'original_image': image_path
        }

    def match_template(self, image: np.ndarray, rotation: float = 0) -> Dict:
        """
        Match document against known templates.

        Args:
            image: Document image
            rotation: Rotation angle applied during preprocessing

        Returns:
            Match results with confidence scores
        """
        if not self.template_signatures:
            return {
                'matched': False,
                'reason': 'No templates loaded',
                'header_bounds': self.get_header_bounds(image, rotation)
            }

        # Extract header and create signature (with rotation adjustment)
        header = self._extract_header_region(image, rotation)
        doc_signature = self._create_signature(header)

        # Compare with all templates
        matches = []
        for name, template_sig in self.template_signatures.items():
            # Cosine similarity
            similarity = np.dot(doc_signature, template_sig) / (
                np.linalg.norm(doc_signature) * np.linalg.norm(template_sig) + 1e-10
            )
            matches.append({
                'template': name,
                'similarity': round(float(similarity), 4)
            })

        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)

        best_match = matches[0] if matches else None
        confidence_threshold = 0.85

        return {
            'matched': best_match and best_match['similarity'] >= confidence_threshold,
            'best_match': best_match,
            'all_matches': matches[:5],  # Top 5
            'header_bounds': self.get_header_bounds(image, rotation),
            'confidence_threshold': confidence_threshold
        }

    def _read_rotation_from_metadata(self, image_path: str) -> float:
        """
        Read rotation angle from metadata file.

        Args:
            image_path: Path to the normalized image

        Returns:
            Rotation angle in degrees (0 if not found)
        """
        # Look for metadata file (e.g., page_000_normalized.png -> page_000_meta.json)
        meta_path = Path(image_path).with_name(
            Path(image_path).stem.replace('_normalized', '_meta') + '.json'
        )

        if not meta_path.exists():
            logger.warning(f"Metadata file not found: {meta_path}")
            return 0.0

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Look for deskew step in processing steps
            for step in meta.get('steps', []):
                if 'deskew' in step:
                    angle = step['deskew'].get('angle', 0)
                    logger.info(f"Found rotation angle from metadata: {angle}°")
                    return float(angle)

            return 0.0
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
            return 0.0

    def identify_form_type(self, image_path: str) -> Dict:
        """
        Identify the form type of a document.

        This method:
        1. Loads the image
        2. Reads rotation from metadata
        3. Extracts the header region (adjusted for rotation)
        4. Attempts template matching
        5. Returns header region for OCR if no match found

        Args:
            image_path: Path to document image

        Returns:
            Form identification results
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        logger.info(f"Identifying form type: {image_path}")

        # Read rotation from metadata
        rotation = self._read_rotation_from_metadata(image_path)
        logger.info(f"Using rotation: {rotation}°")

        # Try template matching with rotation adjustment
        match_result = self.match_template(image, rotation)

        # Extract header region path for OCR (with rotation adjustment)
        header_path = str(Path(image_path).with_name(
            Path(image_path).stem + '_header.png'
        ))
        header = self._extract_header_region(image, rotation)
        cv2.imwrite(header_path, header)

        # Always use OCR for form type detection since template signatures
        # are too similar to reliably distinguish between form types.
        # Template matching is kept for reference/verification only.
        return {
            'image_path': str(image_path),
            'header_image': header_path,
            'header_bounds': match_result['header_bounds'],
            'rotation_detected': rotation,
            'template_match': {
                'matched': match_result['matched'],
                'best_match': match_result.get('best_match'),
                'all_matches': match_result.get('all_matches', [])
            },
            'needs_ocr': True,  # Always use OCR for accurate form type detection
            'form_type': None   # Let OCR determine the form type
        }


class FormTypeRegistry:
    """Registry for known form types and their field definitions."""

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize form type registry.

        Args:
            registry_path: Path to JSON file with form definitions
        """
        self.forms: Dict[str, dict] = {}
        self.registry_path = Path(registry_path) if registry_path else None

        if self.registry_path and self.registry_path.exists():
            self._load_registry()

    def _load_registry(self):
        """Load form definitions from JSON file."""
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                self.forms = json.load(f)
            logger.info(f"Loaded {len(self.forms)} form definitions")
        except Exception as e:
            logger.error(f"Error loading registry: {e}")

    def register_form(self, form_type: str, definition: dict):
        """
        Register a new form type.

        Args:
            form_type: Form type identifier (e.g., "Bestellblatt 1")
            definition: Form field definitions
        """
        self.forms[form_type] = definition
        logger.info(f"Registered form type: {form_type}")

    def get_form_definition(self, form_type: str) -> Optional[dict]:
        """Get field definitions for a form type."""
        return self.forms.get(form_type)

    def save_registry(self, path: Optional[str] = None):
        """Save registry to JSON file."""
        save_path = Path(path) if path else self.registry_path
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.forms, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved registry to {save_path}")
