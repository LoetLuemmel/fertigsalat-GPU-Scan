import email
import os
from email import policy
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_pdf_from_eml(eml_path: str, output_dir: str) -> list[str]:
    """
    Extract all PDF attachments from an EML file.

    Args:
        eml_path: Path to the .eml file
        output_dir: Directory to save the extracted PDFs

    Returns:
        List of paths to extracted PDFs (empty if none found)
    """
    eml_path = Path(eml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing EML file: {eml_path}")

    with open(eml_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    pdf_paths = []
    attachment_count = 0

    for part in msg.walk():
        content_type = part.get_content_type()
        filename = part.get_filename()

        if content_type == 'application/pdf' or (filename and filename.lower().endswith('.pdf')):
            attachment_count += 1
            if filename is None:
                filename = f"{eml_path.stem}_attachment_{attachment_count}.pdf"

            pdf_path = output_dir / filename

            payload = part.get_payload(decode=True)
            if payload:
                with open(pdf_path, 'wb') as pdf_file:
                    pdf_file.write(payload)
                logger.info(f"Extracted PDF {attachment_count}: {pdf_path}")
                pdf_paths.append(str(pdf_path))

    if not pdf_paths:
        logger.warning(f"No PDF attachment found in {eml_path}")
    else:
        logger.info(f"Extracted {len(pdf_paths)} PDF(s) from {eml_path}")

    return pdf_paths


def list_pdf_attachments(eml_path: str) -> list[dict]:
    """
    List all PDF attachments in an EML file without extracting them.

    Args:
        eml_path: Path to the .eml file

    Returns:
        List of dicts with attachment info: index, filename, size
    """
    eml_path = Path(eml_path)

    with open(eml_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    attachments = []
    index = 0

    for part in msg.walk():
        content_type = part.get_content_type()
        filename = part.get_filename()

        if content_type == 'application/pdf' or (filename and filename.lower().endswith('.pdf')):
            if filename is None:
                filename = f"{eml_path.stem}_attachment_{index + 1}.pdf"

            payload = part.get_payload(decode=True)
            size = len(payload) if payload else 0

            attachments.append({
                'index': index,
                'filename': filename,
                'size_bytes': size,
                'size_readable': f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            })
            index += 1

    logger.info(f"Found {len(attachments)} PDF attachment(s) in {eml_path}")
    return attachments


def extract_selected_pdfs(eml_path: str, output_dir: str, selected_indices: list[int] = None) -> list[str]:
    """
    Extract selected PDF attachments from an EML file.

    Args:
        eml_path: Path to the .eml file
        output_dir: Directory to save the extracted PDFs
        selected_indices: List of attachment indices to extract (None = all)

    Returns:
        List of paths to extracted PDFs
    """
    eml_path = Path(eml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(eml_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    pdf_paths = []
    index = 0

    for part in msg.walk():
        content_type = part.get_content_type()
        filename = part.get_filename()

        if content_type == 'application/pdf' or (filename and filename.lower().endswith('.pdf')):
            # Check if this index should be extracted
            if selected_indices is None or index in selected_indices:
                if filename is None:
                    filename = f"{eml_path.stem}_attachment_{index + 1}.pdf"

                pdf_path = output_dir / filename
                payload = part.get_payload(decode=True)

                if payload:
                    with open(pdf_path, 'wb') as pdf_file:
                        pdf_file.write(payload)
                    logger.info(f"Extracted PDF [{index}]: {pdf_path}")
                    pdf_paths.append(str(pdf_path))

            index += 1

    logger.info(f"Extracted {len(pdf_paths)} of {index} PDF(s)")
    return pdf_paths


def get_email_metadata(eml_path: str) -> dict:
    """
    Extract metadata from an EML file.

    Args:
        eml_path: Path to the .eml file

    Returns:
        Dictionary with email metadata
    """
    with open(eml_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    return {
        'subject': msg.get('subject', ''),
        'from': msg.get('from', ''),
        'to': msg.get('to', ''),
        'date': msg.get('date', ''),
        'message_id': msg.get('message-id', '')
    }
