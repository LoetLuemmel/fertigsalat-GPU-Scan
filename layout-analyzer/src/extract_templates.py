#!/usr/bin/env python3
"""
CLI Tool to extract template definitions from all Bestellblatt templates.

Usage:
    python extract_templates.py --templates-dir /app/templates --output-dir /app/templates/definitions

This tool:
1. Finds all PNG template files in the templates directory
2. Extracts grid structure using TemplateExtractor
3. Saves JSON definitions to the output directory
4. Optionally generates visualization images
"""

import argparse
import sys
from pathlib import Path
import logging
import json

from template_extractor import TemplateExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_templates(templates_dir: Path) -> list:
    """
    Find all template image files.

    Args:
        templates_dir: Directory to search

    Returns:
        List of template file paths
    """
    templates = []

    for ext in ['*.png', '*.PNG']:
        templates.extend(templates_dir.glob(ext))

    # Filter to only Bestellblatt files
    templates = [t for t in templates if 'Bestellblatt' in t.name]

    return sorted(templates)


def extract_all_templates(templates_dir: str, output_dir: str,
                         visualize: bool = False) -> dict:
    """
    Extract definitions from all templates.

    Args:
        templates_dir: Directory containing template images
        output_dir: Directory to save JSON definitions
        visualize: Whether to generate visualization images

    Returns:
        Summary of extraction results
    """
    templates_path = Path(templates_dir)
    output_path = Path(output_dir)

    if not templates_path.exists():
        logger.error(f"Templates directory not found: {templates_path}")
        return {'error': 'Templates directory not found'}

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Find templates
    templates = find_templates(templates_path)
    logger.info(f"Found {len(templates)} template files")

    if not templates:
        logger.warning("No template files found")
        return {'error': 'No templates found', 'searched': str(templates_path)}

    # Extract each template
    extractor = TemplateExtractor()
    results = {
        'total': len(templates),
        'successful': 0,
        'failed': 0,
        'definitions': []
    }

    for template_file in templates:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {template_file.name}")

        try:
            # Extract definition
            definition = extractor.extract_from_image(str(template_file))

            # Save JSON
            json_path = output_path / f"{template_file.stem}.json"
            extractor.save_definition(definition, str(json_path))

            # Generate visualization if requested
            if visualize:
                vis_path = output_path / f"{template_file.stem}_grid.png"
                extractor.visualize_grid(str(template_file), definition, str(vis_path))

            results['successful'] += 1
            results['definitions'].append({
                'form_type': definition['form_type'],
                'json_file': str(json_path),
                'products': len(definition['products']),
                'input_columns': len(definition['columns'].get('input_columns', []))
            })

            logger.info(f"Successfully extracted: {template_file.name}")
            logger.info(f"  - Products: {len(definition['products'])}")
            logger.info(f"  - Input columns: {len(definition['columns'].get('input_columns', []))}")

        except Exception as e:
            logger.error(f"Failed to extract {template_file.name}: {e}")
            results['failed'] += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EXTRACTION SUMMARY")
    logger.info(f"  Total templates: {results['total']}")
    logger.info(f"  Successful: {results['successful']}")
    logger.info(f"  Failed: {results['failed']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract template definitions from Bestellblatt form images'
    )
    parser.add_argument(
        '--templates-dir', '-t',
        default='/app/templates',
        help='Directory containing template images (default: /app/templates)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='/app/templates/definitions',
        help='Output directory for JSON definitions (default: /app/templates/definitions)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate grid visualization images'
    )
    parser.add_argument(
        '--single', '-s',
        help='Process only a single template file'
    )

    args = parser.parse_args()

    if args.single:
        # Process single file
        extractor = TemplateExtractor()
        definition = extractor.extract_from_image(args.single)

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        json_path = output_path / f"{definition['form_type']}.json"
        extractor.save_definition(definition, str(json_path))

        if args.visualize:
            vis_path = output_path / f"{definition['form_type']}_grid.png"
            extractor.visualize_grid(args.single, definition, str(vis_path))

        print(f"Extracted definition saved to: {json_path}")
    else:
        # Process all templates
        results = extract_all_templates(
            args.templates_dir,
            args.output_dir,
            args.visualize
        )

        if results.get('error'):
            sys.exit(1)

        # Print JSON summary
        print("\n" + json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
