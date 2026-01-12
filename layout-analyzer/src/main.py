import os
import time
import json
from pathlib import Path
import logging
from analyzer import LayoutAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path(os.getenv('INPUT_DIR', '/app/intermediate'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/app/intermediate'))
TEMPLATES_DIR = Path(os.getenv('TEMPLATES_DIR', '/app/templates'))
REGISTRY_PATH = os.getenv('REGISTRY_PATH', '/app/templates/form_registry.json')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))


def process_task(task_file: Path) -> bool:
    """
    Process a layout analysis task.

    Task file format (JSON):
    {
        "task_id": "unique_id",
        "input_dir": "path/to/preprocessed/images"
    }
    """
    try:
        with open(task_file, 'r') as f:
            task = json.load(f)

        task_id = task.get('task_id')
        input_dir = Path(task.get('input_dir', ''))

        logger.info(f"Processing layout task: {task_id}")

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return False

        # Initialize analyzer with templates
        templates_path = str(TEMPLATES_DIR) if TEMPLATES_DIR.exists() else None
        registry_path = REGISTRY_PATH if Path(REGISTRY_PATH).exists() else None
        analyzer = LayoutAnalyzer(templates_dir=templates_path, registry_path=registry_path)

        # Find normalized images
        image_files = list(input_dir.glob('*_normalized.png'))
        if not image_files:
            logger.warning(f"No normalized images found in {input_dir}")
            return False

        results = []
        for img_path in image_files:
            try:
                layout = analyzer.analyze(str(img_path))
                results.append(layout)

                # Save individual layout file
                layout_file = img_path.with_name(img_path.stem.replace('_normalized', '_layout') + '.json')
                with open(layout_file, 'w') as f:
                    json.dump(layout, f, indent=2)

                logger.info(f"Layout analysis saved: {layout_file}")
            except Exception as e:
                logger.error(f"Error analyzing {img_path}: {e}")

        # Save task result
        result = {
            'task_id': task_id,
            'status': 'completed',
            'images_analyzed': len(results),
            'results': results
        }

        result_file = OUTPUT_DIR / f"{task_id}_layout_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Task {task_id} completed. Result: {result_file}")

        # Mark task as done
        done_file = task_file.with_suffix('.done')
        task_file.rename(done_file)

        return True

    except Exception as e:
        logger.exception(f"Error processing task {task_file}: {e}")
        return False


def watch_for_tasks():
    """Watch for new layout analysis tasks."""
    tasks_dir = INPUT_DIR / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Layout Analyzer Container started")
    logger.info("=" * 60)
    logger.info(f"Watching for tasks in: {tasks_dir}")
    logger.info(f"Templates directory: {TEMPLATES_DIR}")
    logger.info(f"Templates available: {TEMPLATES_DIR.exists()}")

    # Count available templates
    if TEMPLATES_DIR.exists():
        templates = list(TEMPLATES_DIR.glob('*.png')) + list(TEMPLATES_DIR.glob('*.jpg'))
        logger.info(f"Found {len(templates)} template files")

    while True:
        # Look for layout task files
        task_files = list(tasks_dir.glob('layout_task_*.json'))

        for task_file in task_files:
            if task_file.suffix == '.json':
                process_task(task_file)

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    watch_for_tasks()
