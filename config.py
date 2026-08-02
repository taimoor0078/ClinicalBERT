import os

class Settings:
    APP_NAME: str = "NeuroX ClinicalBERT API"
    APP_VERSION: str = "1.0.0"

    MODEL_NAME: str = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LENGTH: int = 512

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")

    ALLOWED_EXTENSIONS: tuple = (".pdf",)
    MAX_FILE_SIZE_MB: int = 50

    DEVICE: str = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
