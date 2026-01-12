import os
import time
import json
from pathlib import Path
import logging
from email_parser import extract_pdf_from_eml, get_email_metadata, list_pdf_attachments, extract_selected_pdfs
from image_processor import ImageProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path(os.getenv('INPUT_DIR', '/app/input'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/app/intermediate'))
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))


def process_preview_task(task_file: Path, task: dict) -> bool:
    """
    Phase 1: List PDF attachments without processing.
    Creates a preview file for user selection.
    """
    task_id = task.get('task_id', task_file.stem)
    eml_file = task.get('eml_file')

    eml_path = Path(eml_file)
    if not eml_path.is_absolute():
        eml_path = INPUT_DIR / eml_path

    if not eml_path.exists():
        logger.error(f"EML file not found: {eml_path}")
        return False

    # Get email metadata
    email_meta = get_email_metadata(str(eml_path))

    # List all PDF attachments (without extracting)
    attachments = list_pdf_attachments(str(eml_path))

    if not attachments:
        logger.warning(f"No PDF attachments found in {eml_path}")

    # Save preview result
    preview = {
        'task_id': task_id,
        'status': 'preview',
        'eml_file': str(eml_path),
        'email_metadata': email_meta,
        'attachments': attachments,
        'message': f"Gefunden: {len(attachments)} PDF-Anhang/Anhänge. Bitte wählen Sie die zu verarbeitenden Anhänge."
    }

    preview_file = OUTPUT_DIR / 'tasks' / f"{task_id}_preview.json"
    with open(preview_file, 'w') as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    logger.info(f"Preview created: {preview_file}")

    # Mark task as done
    task_file.rename(task_file.with_suffix('.done'))
    return True


def process_extract_task(task_file: Path, task: dict) -> bool:
    """
    Phase 2: Extract and process selected PDF attachments.
    """
    task_id = task.get('task_id', task_file.stem)
    eml_file = task.get('eml_file')
    apply_threshold = task.get('apply_threshold', False)
    selected_indices = task.get('selected_attachments')  # None = all

    eml_path = Path(eml_file)
    if not eml_path.is_absolute():
        eml_path = INPUT_DIR / eml_path

    if not eml_path.exists():
        logger.error(f"EML file not found: {eml_path}")
        return False

    # Create task output directory
    task_output_dir = OUTPUT_DIR / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract email metadata
    email_meta = get_email_metadata(str(eml_path))

    # Extract selected PDF attachments
    pdf_paths = extract_selected_pdfs(str(eml_path), str(task_output_dir), selected_indices)
    if not pdf_paths:
        logger.error(f"No PDF attachment extracted from {eml_path}")
        return False

    logger.info(f"Extracted {len(pdf_paths)} PDF(s)")

    # Convert all PDFs to images and process
    processor = ImageProcessor(dpi=300)
    results = []

    for pdf_idx, pdf_path in enumerate(pdf_paths):
        logger.info(f"Processing PDF {pdf_idx + 1}/{len(pdf_paths)}: {pdf_path}")
        image_paths = processor.pdf_to_images(pdf_path, str(task_output_dir))

        # Process each page
        for img_path in image_paths:
            meta = processor.process_document(
                img_path,
                str(task_output_dir),
                apply_threshold=apply_threshold
            )
            meta['source_pdf'] = pdf_path
            results.append(meta)

    # Save task result
    result = {
        'task_id': task_id,
        'status': 'completed',
        'email_metadata': email_meta,
        'selected_attachments': selected_indices,
        'pdfs_extracted': len(pdf_paths),
        'pages_processed': len(results),
        'results': results
    }

    result_file = OUTPUT_DIR / f"{task_id}_opencv_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Task {task_id} completed. Result: {result_file}")

    # Mark task as done
    task_file.rename(task_file.with_suffix('.done'))
    return True


def process_task(task_file: Path) -> bool:
    """
    Process a single task file.

    Task file format (JSON):
    {
        "task_id": "unique_id",
        "eml_file": "path/to/file.eml",
        "mode": "preview" | "process",
        "selected_attachments": [0, 1],  // optional, only for mode=process
        "apply_threshold": false
    }

    Mode:
    - "preview": List attachments, create preview file for selection
    - "process": Extract and process selected (or all) attachments
    """
    try:
        with open(task_file, 'r') as f:
            task = json.load(f)

        task_id = task.get('task_id', task_file.stem)
        eml_file = task.get('eml_file')
        mode = task.get('mode', 'process')  # Default to process for backwards compatibility

        logger.info(f"Processing task: {task_id} (mode: {mode})")

        if not eml_file:
            logger.error(f"No eml_file specified in task {task_id}")
            return False

        if mode == 'preview':
            return process_preview_task(task_file, task)
        else:
            return process_extract_task(task_file, task)

    except Exception as e:
        logger.exception(f"Error processing task {task_file}: {e}")
        return False


def watch_for_tasks():
    """Watch for new task files and process them."""
    tasks_dir = OUTPUT_DIR / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"OpenCV Prep Container started")
    logger.info(f"Watching for tasks in: {tasks_dir}")
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    while True:
        # Look for opencv task files
        task_files = list(tasks_dir.glob('opencv_task_*.json'))

        for task_file in task_files:
            if task_file.suffix == '.json':
                process_task(task_file)

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    watch_for_tasks()
