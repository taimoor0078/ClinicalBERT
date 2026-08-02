from typing import Optional
from pydantic import BaseModel, Field


class ExtractedFeatures(BaseModel):
    tumor_type: Optional[str] = Field(default=None, alias="Tumor Type")
    tumor_size: Optional[str] = Field(default=None, alias="Tumor Size")
    tumor_location: Optional[str] = Field(default=None, alias="Tumor Location")
    edema: Optional[str] = Field(default=None, alias="Edema")
    midline_shift: Optional[str] = Field(default=None, alias="Midline Shift")
    contrast_enhancement: Optional[str] = Field(default=None, alias="Contrast Enhancement")
    necrosis: Optional[str] = Field(default=None, alias="Necrosis")
    mass_effect: Optional[str] = Field(default=None, alias="Mass Effect")
    radiologist_impression: Optional[str] = Field(default=None, alias="Radiologist Impression")

    class Config:
        populate_by_name = True
        validate_by_name = True


class PredictionResponse(BaseModel):
    success: bool
    report_text: str
    clinicalbert_embedding_dimension: int
    extracted_features: dict

    class Config:
        populate_by_name = True


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
