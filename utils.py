import os
import uuid
import logging
from fastapi import UploadFile

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurox")


class InvalidFileError(Exception):
    pass


class EmptyFileError(Exception):
    pass


class CorruptedFileError(Exception):
    pass


def validate_upload(file: UploadFile) -> None:
    if file is None or file.filename is None or file.filename.strip() == "":
        raise InvalidFileError("No file uploaded.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidFileError(
            f"Unsupported file type '{ext}'. Only PDF files are allowed."
        )


def save_upload_file(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(settings.UPLOAD_DIR, unique_name)

    contents = file.file.read()

    if not contents or len(contents) == 0:
        raise EmptyFileError("Uploaded file is empty.")

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise InvalidFileError(
            f"File too large ({size_mb:.2f} MB). Max allowed is {settings.MAX_FILE_SIZE_MB} MB."
        )

    with open(dest_path, "wb") as f:
        f.write(contents)

    return dest_path


def cleanup_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {path}: {e}")
