import io
import re
import unicodedata
import fitz
import easyocr
import numpy as np
from app.utils import (
    CorruptedFileError,
    EmptyFileError,
    logger,
)

# ==========================================================
# OCR CONFIGURATION
# ==========================================================

OCR_ZOOM = 4.0
OCR_MIN_CONFIDENCE = 0.45

try:
    logger.info("Loading EasyOCR...")

    OCR_READER = easyocr.Reader(
        ["en"],
        gpu=False
    )

    logger.info("EasyOCR loaded successfully.")

except Exception as e:

    logger.exception("EasyOCR initialization failed.")

    raise RuntimeError(
        f"Unable to initialize EasyOCR: {e}"
    )


# ==========================================================
# OCR HELPER
# ==========================================================

def perform_ocr_on_page(page: fitz.Page) -> str:
    """
    Perform OCR on a single PDF page.
    """

    try:

        matrix = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        # Convert Pixmap directly to NumPy array
        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            pix.n
        )

        # NOTE: when paragraph=True, EasyOCR merges boxes and does NOT
        # return a confidence score, so results come back as (bbox, text)
        # pairs, not (bbox, text, confidence) triples. The old code
        # filtered on `len(result) != 3`, which is always true here —
        # so every single result was being skipped and no text was
        # ever extracted. Handle both shapes below.
        results = OCR_READER.readtext(
            image,
            paragraph=True,
            detail=1
        )

        extracted_text = []

        for result in results:

            if len(result) == 3:
                _, text, confidence = result

                if confidence < OCR_MIN_CONFIDENCE:
                    continue

            elif len(result) == 2:
                _, text = result

            else:
                continue

            text = text.strip()

            if text:
                extracted_text.append(text)

        logger.info(
            f"Page {page.number + 1}: "
            f"{len(extracted_text)} text blocks extracted."
        )

        return "\n".join(extracted_text)

    except Exception as e:

        logger.exception(
            f"OCR failed on page {page.number + 1}: {e}"
        )

        # Continue processing remaining pages
        return ""

# ==========================================================
# OCR DOCUMENT
# ==========================================================

def extract_text_using_ocr(doc: fitz.Document) -> str:
    """
    Perform OCR on every page of the PDF.
    """

    logger.info("Starting OCR extraction...")

    pages = []

    success = 0
    failed = 0

    for page_number in range(doc.page_count):

        page = doc.load_page(page_number)

        page_text = perform_ocr_on_page(page)

        if page_text.strip():

            success += 1
            pages.append(page_text)

        else:

            failed += 1

    final_text = "\n\n".join(pages)

    logger.info(
        f"OCR Complete | "
        f"Success={success} | "
        f"Failed={failed} | "
        f"Characters={len(final_text)}"
    )

    return final_text

# ==========================================================
# PDF TEXT EXTRACTION
# ==========================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF.

    Workflow
    --------
    1. Open PDF
    2. Extract selectable text using PyMuPDF
    3. If no selectable text exists, perform OCR
    4. Return extracted text
    """

    logger.info(f"Opening PDF: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)

    except Exception as e:

        logger.exception("Unable to open PDF.")

        raise CorruptedFileError(
            f"Unable to open PDF. It may be corrupted.\n{e}"
        )

    try:

        if doc.page_count == 0:
            raise EmptyFileError(
                "PDF contains no pages."
            )

        logger.info(
            f"PDF contains {doc.page_count} pages."
        )

        extracted_pages = []

        # --------------------------------------------------
        # STEP 1
        # Native PDF Text Extraction
        # --------------------------------------------------

        for page_number in range(doc.page_count):

            page = doc.load_page(page_number)

            try:

                page_text = page.get_text("text")

                if page_text and page_text.strip():

                    extracted_pages.append(
                        page_text.strip()
                    )

                    logger.info(
                        f"Page {page_number + 1}: "
                        "Native text extracted."
                    )

                else:

                    logger.info(
                        f"Page {page_number + 1}: "
                        "No selectable text found."
                    )

            except Exception as e:

                logger.warning(
                    f"Failed reading page "
                    f"{page_number + 1}: {e}"
                )

        full_text = "\n\n".join(extracted_pages).strip()

        # --------------------------------------------------
        # STEP 2
        # OCR Fallback
        # --------------------------------------------------

        if not full_text:

            logger.warning(
                "No selectable text found. "
                "Switching to OCR..."
            )

            full_text = extract_text_using_ocr(doc)

        # --------------------------------------------------
        # STEP 3
        # Validation
        # --------------------------------------------------

        if not full_text.strip():

            raise EmptyFileError(
                "No readable text found. "
                "Both PyMuPDF extraction and OCR failed."
            )

        logger.info(
            f"Extraction completed successfully. "
            f"Characters extracted: {len(full_text)}"
        )

        return full_text

    finally:

        doc.close()

        logger.info(
            "PDF closed successfully."
        )


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(raw_text: str) -> str:
    """
    Clean extracted PDF/OCR text while preserving
    important medical terminology.
    """

    if not raw_text:
        return ""

    logger.info("Cleaning extracted text...")

    # Unicode normalization
    text = unicodedata.normalize("NFKC", raw_text)

    # Standardize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove invisible control characters
    text = re.sub(
        r"[\x00-\x08\x0B-\x1F\x7F]",
        "",
        text
    )

    # OCR ligature fixes
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "•": "-",
        "–": "-",
        "—": "-",
        "\u00A0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []

    for line in text.split("\n"):

        line = line.strip()

        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([,.;:])",
        r"\1",
        text
    )

    # Collapse spaces again
    text = re.sub(
        r"\s{2,}",
        " ",
        text
    )

    logger.info(
        f"Cleaned text length: {len(text)} characters."
    )

    return text.strip()


# ==========================================================
# PREPROCESSING PIPELINE
# ==========================================================

def preprocess_pdf(pdf_path: str) -> str:
    """
    Complete preprocessing pipeline.

    PDF
      │
      ▼
    Extract Text
      │
      ▼
    OCR (if needed)
      │
      ▼
    Clean Text
      │
      ▼
    Return ClinicalBERT Ready Text
    """

    logger.info(
        "Starting PDF preprocessing..."
    )

    raw_text = extract_text_from_pdf(
        pdf_path
    )

    cleaned_text = clean_text(
        raw_text
    )

    if not cleaned_text:

        logger.error(
            "No usable text after preprocessing."
        )

        raise EmptyFileError(
            "PDF contains no readable text."
        )

    logger.info(
        "PDF preprocessing completed successfully."
    )

    logger.info(
        f"Final text length: {len(cleaned_text)} characters."
    )

    return cleaned_text      