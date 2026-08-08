import threading

import torch
from transformers import AutoTokenizer, AutoModel

from config import settings
from utils import logger


class ClinicalBERTModel:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        logger.info(f"Loading tokenizer and model: {settings.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.MODEL_NAME)
        self.device = torch.device(settings.DEVICE)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on device: {self.device}")

    @classmethod
    def get_instance(cls) -> "ClinicalBERTModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_cls_embedding(self, text: str):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=settings.MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        cls_embedding = cls_embedding.squeeze(0).cpu().numpy()

        return cls_embedding


def get_model() -> ClinicalBERTModel:
    return ClinicalBERTModel.get_instance()
