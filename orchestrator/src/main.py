import os
import time
import json
import uuid
from pathlib import Path
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path(os.getenv('INPUT_DIR', '/app/input'))
INTERMEDIATE_DIR = Path(os.getenv('INTERMEDIATE_DIR', '/app/intermediate'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/app/output'))
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))


class Pipeline:
    """Document processing pipeline orchestrator."""

    STAGES = ['opencv', 'layout', 'ml']

    def __init__(self):
        self.tasks_dir = INTERMEDIATE_DIR / 'tasks'
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def create_task(self, eml_file: Path) -> str:
        """Create a new processing task for an EML file."""
        task_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        task = {
            'task_id': task_id,
            'eml_file': str(eml_file),
            'created_at': datetime.now().isoformat(),
            'status': 'pending',
            'current_stage': 'opencv',
            'apply_threshold': False
        }

        # Save master task file
        task_file = self.tasks_dir / f"task_{task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(task, f, indent=2)

        # Create task for OpenCV container
        opencv_task = {
            'task_id': task_id,
            'eml_file': str(eml_file),
            'apply_threshold': False
        }
        opencv_task_file = self.tasks_dir / f"opencv_task_{task_id}.json"
        with open(opencv_task_file, 'w') as f:
            json.dump(opencv_task, f, indent=2)

        logger.info(f"Created task {task_id} for {eml_file}")
        return task_id

    def check_stage_completion(self, task_id: str, stage: str) -> bool:
        """Check if a processing stage has completed."""
        result_file = INTERMEDIATE_DIR / f"{task_id}_{stage}_result.json"
        return result_file.exists()

    def advance_pipeline(self, task_id: str):
        """Advance task to next pipeline stage."""
        task_file = self.tasks_dir / f"task_{task_id}.json"
        if not task_file.exists():
            return

        with open(task_file, 'r') as f:
            task = json.load(f)

        current_stage = task.get('current_stage', 'opencv')
        current_idx = self.STAGES.index(current_stage)

        # Check if current stage is complete
        if self.check_stage_completion(task_id, current_stage):
            if current_idx < len(self.STAGES) - 1:
                # Advance to next stage
                next_stage = self.STAGES[current_idx + 1]
                task['current_stage'] = next_stage
                task['status'] = 'processing'

                # Create task file for next container
                stage_task = {
                    'task_id': task_id,
                    'input_dir': str(INTERMEDIATE_DIR / task_id)
                }
                stage_task_file = self.tasks_dir / f"{next_stage}_task_{task_id}.json"
                with open(stage_task_file, 'w') as f:
                    json.dump(stage_task, f, indent=2)

                logger.info(f"Task {task_id}: {current_stage} -> {next_stage}")
            else:
                # Pipeline complete
                task['status'] = 'completed'
                task['completed_at'] = datetime.now().isoformat()
                self.finalize_task(task_id)
                logger.info(f"Task {task_id}: Pipeline completed")

            with open(task_file, 'w') as f:
                json.dump(task, f, indent=2)

    def finalize_task(self, task_id: str):
        """Copy final results to output directory."""
        task_output = INTERMEDIATE_DIR / task_id
        final_output = OUTPUT_DIR / task_id
        final_output.mkdir(parents=True, exist_ok=True)

        # Copy relevant result files
        for result_file in task_output.glob('*'):
            if result_file.is_file():
                dest = final_output / result_file.name
                dest.write_bytes(result_file.read_bytes())

        # Create summary file
        summary = {
            'task_id': task_id,
            'completed_at': datetime.now().isoformat(),
            'output_files': [f.name for f in final_output.glob('*')]
        }
        summary_file = final_output / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Task {task_id} finalized to {final_output}")


class EMLHandler(FileSystemEventHandler):
    """Handle new EML files in input directory."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.processed_files = set()

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() == '.eml' and file_path not in self.processed_files:
            logger.info(f"New EML file detected: {file_path}")
            self.processed_files.add(file_path)
            time.sleep(1)  # Wait for file to be fully written
            self.pipeline.create_task(file_path)


def scan_existing_files(pipeline: Pipeline, handler: EMLHandler):
    """Process any existing EML files in input directory."""
    for eml_file in INPUT_DIR.glob('*.eml'):
        if eml_file not in handler.processed_files:
            logger.info(f"Found existing EML file: {eml_file}")
            handler.processed_files.add(eml_file)
            pipeline.create_task(eml_file)


def main():
    logger.info("=" * 60)
    logger.info("Email Processing Orchestrator Started")
    logger.info("=" * 60)
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Intermediate directory: {INTERMEDIATE_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    pipeline = Pipeline()
    handler = EMLHandler(pipeline)

    # Set up file watcher
    observer = Observer()
    observer.schedule(handler, str(INPUT_DIR), recursive=False)
    observer.start()

    # Scan for existing files
    scan_existing_files(pipeline, handler)

    try:
        while True:
            # Check and advance active tasks
            for task_file in pipeline.tasks_dir.glob('task_*.json'):
                with open(task_file, 'r') as f:
                    task = json.load(f)

                if task.get('status') not in ['completed', 'failed']:
                    pipeline.advance_pipeline(task['task_id'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        observer.stop()
        logger.info("Orchestrator stopped")

    observer.join()


if __name__ == '__main__':
    main()
