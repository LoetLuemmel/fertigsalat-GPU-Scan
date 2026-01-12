import os
import re
import time
import json
from pathlib import Path
import logging
from ocr_engine import OCREngine, AnomalyDetector
from optimized_scanner import OptimizedFormScanner


def extract_form_type(text: str) -> str:
    """
    Extract form type identifier from header text.

    Looks for patterns like:
    - "Bestellblatt 1"
    - "Bestellblatt 2"
    - "Formular A"
    - etc.

    Handles common OCR errors like:
    - "Beste ulatt" instead of "Bestellblatt"
    - Missing characters

    Args:
        text: OCR text from header region

    Returns:
        Identified form type or 'unknown'
    """
    if not text:
        return 'unknown'

    # Normalize text
    text = text.strip()

    # Pattern for "Bestellblatt X" (X can be number or letter)
    # Also handles OCR errors like "Beste llblatt", "Bestel lblatt", "Beste ulatt"
    bestellblatt_patterns = [
        r'[Bb]estellblatt\s*(\d+|[A-Za-z])',           # Exact match
        r'[Bb]estell?\s*blatt\s*(\d+|[A-Za-z])',       # Space in middle
        r'[Bb]este\s*l+\s*b?l?att\s*(\d+|[A-Za-z])',   # Various OCR errors
        r'[Bb]este\s*[iu]l?att\s*(\d+|[A-Za-z])',      # "Beste ulatt" error
    ]

    for pattern in bestellblatt_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"Bestellblatt {match.group(1)}"

    # Pattern for "Formular X"
    formular_pattern = r'[Ff]ormular\s*(\d+|[A-Za-z])'
    match = re.search(formular_pattern, text)
    if match:
        return f"Formular {match.group(1)}"

    # If no specific pattern found, return cleaned first line
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) < 50:
        return first_line

    return 'unknown'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path(os.getenv('INPUT_DIR', '/app/intermediate'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/app/output'))
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))
USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'
OCR_LANGS = os.getenv('OCR_LANGS', 'de,en').split(',')

# Optimized scanning settings
DEFINITIONS_DIR = Path(os.getenv('DEFINITIONS_DIR', '/app/templates/definitions'))
USE_OPTIMIZED_SCANNING = os.getenv('USE_OPTIMIZED_SCANNING', 'true').lower() == 'true'


def process_task(task_file: Path, ocr_engine: OCREngine,
                 optimized_scanner: OptimizedFormScanner = None) -> bool:
    """
    Process an ML inference task.

    Task file format (JSON):
    {
        "task_id": "unique_id",
        "input_dir": "path/to/preprocessed/images"
    }

    If optimized_scanner is provided and a template definition exists for the
    detected form type, uses optimized scanning (only input cells).
    Otherwise falls back to full-page OCR.
    """
    try:
        with open(task_file, 'r') as f:
            task = json.load(f)

        task_id = task.get('task_id')
        input_dir = Path(task.get('input_dir', ''))

        logger.info(f"Processing ML task: {task_id}")

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return False

        # Find normalized images
        image_files = list(input_dir.glob('*_normalized.png'))
        if not image_files:
            logger.warning(f"No normalized images found in {input_dir}")
            return False

        # Load layout info if available
        layout_info = {}
        for layout_file in input_dir.glob('*_layout.json'):
            with open(layout_file, 'r') as f:
                layout_info[layout_file.stem.replace('_layout', '')] = json.load(f)

        ocr_results = []
        form_types_detected = []

        for img_path in image_files:
            try:
                # Full page OCR
                ocr_result = ocr_engine.extract_text(str(img_path))

                # Region-based OCR if layout info available
                img_key = img_path.stem.replace('_normalized', '')
                if img_key in layout_info:
                    layout = layout_info[img_key]

                    # Extract form type from header if needed
                    form_id = layout.get('form_identification', {})
                    if form_id.get('needs_ocr_for_type') and form_id.get('header_image'):
                        header_path = form_id['header_image']
                        if Path(header_path).exists():
                            header_ocr = ocr_engine.extract_text(header_path)
                            form_type_text = header_ocr.get('full_text', '').strip()

                            # Look for "Bestellblatt X" pattern
                            form_type = extract_form_type(form_type_text)
                            ocr_result['form_type_detected'] = form_type
                            ocr_result['header_text'] = form_type_text
                            form_types_detected.append(form_type)
                            logger.info(f"Form type detected: {form_type}")

                            # Try optimized scanning if scanner available and definition exists
                            if optimized_scanner and optimized_scanner.has_definition(form_type):
                                logger.info(f"Using optimized scanning for {form_type}")
                                optimized_result = optimized_scanner.scan_form(
                                    str(img_path), form_type
                                )
                                ocr_result['optimized_scan'] = optimized_result
                                ocr_result['orders'] = optimized_result.get('orders', [])
                                ocr_result['scan_statistics'] = optimized_result.get('statistics', {})
                                logger.info(f"Optimized scan found {len(ocr_result['orders'])} orders")
                            else:
                                # Fallback: Extract text from identified fields
                                fields = layout.get('fields', [])
                                if fields:
                                    field_regions = [f['bounds'] for f in fields[:20]]
                                    field_results = ocr_engine.extract_from_regions(str(img_path), field_regions)
                                    ocr_result['field_ocr'] = field_results
                    else:
                        # No form type detection needed, use field extraction
                        fields = layout.get('fields', [])
                        if fields:
                            field_regions = [f['bounds'] for f in fields[:20]]
                            field_results = ocr_engine.extract_from_regions(str(img_path), field_regions)
                            ocr_result['field_ocr'] = field_results

                ocr_results.append(ocr_result)

                # Save individual OCR result
                ocr_file = img_path.with_name(img_path.stem.replace('_normalized', '_ocr') + '.json')
                with open(ocr_file, 'w') as f:
                    json.dump(ocr_result, f, indent=2, ensure_ascii=False)

                logger.info(f"OCR result saved: {ocr_file}")

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        # Run anomaly detection (placeholder - needs reference data)
        detector = AnomalyDetector()
        anomaly_result = detector.compare_fields({})

        # Determine primary form type
        primary_form_type = None
        if form_types_detected:
            # Use most common form type
            from collections import Counter
            type_counts = Counter(form_types_detected)
            primary_form_type = type_counts.most_common(1)[0][0]

        # Save task result
        result = {
            'task_id': task_id,
            'status': 'completed',
            'form_type': primary_form_type,
            'form_types_detected': form_types_detected,
            'images_processed': len(ocr_results),
            'ocr_results': ocr_results,
            'anomaly_detection': anomaly_result
        }

        # Save to intermediate
        result_file = INPUT_DIR / f"{task_id}_ml_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Also save to output
        output_task_dir = OUTPUT_DIR / task_id
        output_task_dir.mkdir(parents=True, exist_ok=True)
        final_result = output_task_dir / 'ocr_result.json'
        with open(final_result, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Task {task_id} completed. Result: {result_file}")

        # Mark task as done
        done_file = task_file.with_suffix('.done')
        task_file.rename(done_file)

        return True

    except Exception as e:
        logger.exception(f"Error processing task {task_file}: {e}")
        return False


def watch_for_tasks():
    """Watch for new ML inference tasks."""
    tasks_dir = INPUT_DIR / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ML Inference Container started")
    logger.info("=" * 60)
    logger.info(f"GPU enabled: {USE_GPU}")
    logger.info(f"OCR languages: {OCR_LANGS}")
    logger.info(f"Optimized scanning: {USE_OPTIMIZED_SCANNING}")
    logger.info(f"Definitions directory: {DEFINITIONS_DIR}")
    logger.info(f"Watching for tasks in: {tasks_dir}")

    # Initialize OCR engine
    logger.info("Initializing OCR engine...")
    ocr_engine = OCREngine(langs=OCR_LANGS, use_gpu=USE_GPU)

    # Initialize optimized scanner if enabled
    optimized_scanner = None
    if USE_OPTIMIZED_SCANNING:
        logger.info("Initializing optimized scanner...")
        optimized_scanner = OptimizedFormScanner(
            str(DEFINITIONS_DIR),
            ocr_engine=ocr_engine
        )
        available_types = optimized_scanner.get_available_form_types()
        logger.info(f"Available form types: {available_types}")

    while True:
        # Look for ML task files
        task_files = list(tasks_dir.glob('ml_task_*.json'))

        for task_file in task_files:
            if task_file.suffix == '.json':
                process_task(task_file, ocr_engine, optimized_scanner)

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    watch_for_tasks()
