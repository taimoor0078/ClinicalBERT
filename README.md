# NeuroX — ClinicalBERT Radiology Report API

AI-powered FastAPI backend that accepts a radiology report PDF and returns
structured information extracted from the report, alongside a
Bio_ClinicalBERT document embedding.

> **Disclaimer:** This tool performs text extraction/NLP on report documents
> only. It does not perform image-based diagnosis and is not a substitute
> for professional medical judgment.

## Folder Structure

```
ClinicalBERT_API/
│── app/
│   ├── main.py            # FastAPI app, routes, error handling
│   ├── config.py          # Settings (paths, model name, limits)
│   ├── model_loader.py     # Bio_ClinicalBERT loading + embedding
│   ├── preprocessing.py    # PDF text extraction + cleaning
│   ├── predictor.py         # Rule-based structured field extraction
│   ├── schemas.py          # Pydantic response schemas
│   ├── utils.py            # Upload validation, file helpers, logging
│   └── __init__.py
│── uploads/                # Temporary storage for uploaded PDFs
│── outputs/                 # Reserved for future output artifacts
│── requirements.txt
│── README.md
```

## 1. Create a Virtual Environment

```bash
python3.11 -m venv venv
```

Activate it:

- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```
- Windows:
  ```bash
  venv\Scripts\activate
  ```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The first run will download `emilyalsentzer/Bio_ClinicalBERT` from
Hugging Face (requires internet access on first launch; cached afterwards).

## 3. Run the Server

```bash
uvicorn app.main:app --reload
```

The server starts at: `http://127.0.0.1:8000`

## 4. Swagger UI

Interactive API docs (auto-generated):

```
http://127.0.0.1:8000/docs
```

Alternative ReDoc UI:

```
http://127.0.0.1:8000/redoc
```

## 5. Endpoints

| Method | Path       | Description                          |
|--------|-----------|---------------------------------------|
| GET    | `/`       | Health/info check                     |
| GET    | `/health` | Simple health check                   |
| POST   | `/predict`| Upload a radiology report PDF         |

## 6. Postman Example

**Request**

- Method: `POST`
- URL: `http://127.0.0.1:8000/predict`
- Body type: `form-data`
- Key: `file` (type: File) → select a `.pdf` radiology report

**cURL equivalent**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@report.pdf"
```

**Sample Response**

```json
{
  "success": true,
  "report_text": "...",
  "clinicalbert_embedding_dimension": 768,
  "extracted_features": {
    "Tumor Type": "glioblastoma multiforme",
    "Tumor Size": "4.2 x 3.5 cm",
    "Tumor Location": "left frontal lobe",
    "Edema": "moderate perilesional edema",
    "Midline Shift": "midline shift of 5 mm",
    "Contrast Enhancement": "heterogeneous enhancement",
    "Necrosis": "central necrosis",
    "Mass Effect": "moderate mass effect",
    "Radiologist Impression": "Findings consistent with high-grade glioma..."
  }
}
```

## 7. Error Handling

| Scenario            | HTTP Status | Response                                             |
|----------------------|-------------|-------------------------------------------------------|
| No file uploaded     | 422/400     | `{"success": false, "error": "..."}`                  |
| Non-PDF file         | 400         | `{"success": false, "error": "Unsupported file type"}`|
| Empty PDF            | 422         | `{"success": false, "error": "PDF has no pages / empty content"}` |
| Corrupted PDF        | 422         | `{"success": false, "error": "Unable to open PDF..."}`|
| Unexpected error     | 500         | `{"success": false, "error": "Internal server error..."}` |

## Notes on Extraction Logic

- `clinicalbert_embedding_dimension` reflects the actual embedding size
  returned by the loaded ClinicalBERT model (768 for `Bio_ClinicalBERT`).
- Structured fields are extracted using regex/keyword rules applied to the
  cleaned report text. ClinicalBERT embeddings are computed over the full
  document (chunked and averaged for long reports) and returned for
  downstream use (e.g. similarity search, classification models you train
  separately). Fields not found in the text are returned as `null` —
  no values are fabricated.
