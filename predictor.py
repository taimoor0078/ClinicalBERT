import re
from typing import Optional, Dict

import numpy as np

from app.model_loader import get_model
from app.config import settings
from app.utils import logger

FIELD_KEYS = [
    "Tumor Type",
    "Tumor Size",
    "Tumor Location",
    "Edema",
    "Midline Shift",
    "Contrast Enhancement",
    "Necrosis",
    "Mass Effect",
    "Radiologist Impression",
]

TUMOR_TYPE_PATTERNS = [
    r"(glioblastoma(?:\s+multiforme)?(?:,?\s*(?:IDH[- ]?wildtype|IDH[- ]?mutant))?)",
    r"(anaplastic astrocytoma)",
    r"(diffuse astrocytoma)",
    r"(pilocytic astrocytoma)",
    r"(astrocytoma(?:,?\s*grade\s*[I|II|III|IV1-4]+)?)",
    r"(oligodendroglioma(?:,?\s*IDH[- ]?mutant)?(?:,?\s*1p/19q[- ]?codeleted)?)",
    r"(meningioma(?:,?\s*(?:atypical|malignant|benign))?)",
    r"(medulloblastoma)",
    r"(ependymoma)",
    r"(schwannoma|vestibular schwannoma|acoustic neuroma)",
    r"(pituitary adenoma)",
    r"(craniopharyngioma)",
    r"(metastatic (?:brain )?tumor|metastasis|metastases|brain metastasis)",
    r"(hemangioblastoma)",
    r"(lymphoma(?:,?\s*primary CNS)?)",
    r"(DNET|dysembryoplastic neuroepithelial tumor)",
    r"(ganglioglioma)",
]

TUMOR_SIZE_PATTERNS = [
    r"(?:measur(?:ing|es)?|size[d]?(?:\s*of)?|approximately|approx\.?)\s*"
    r"(\d+(?:\.\d+)?\s*(?:x|×)\s*\d+(?:\.\d+)?(?:\s*(?:x|×)\s*\d+(?:\.\d+)?)?\s*(?:cm|mm))",
    r"(\d+(?:\.\d+)?\s*(?:x|×)\s*\d+(?:\.\d+)?(?:\s*(?:x|×)\s*\d+(?:\.\d+)?)?\s*(?:cm|mm))",
    r"(\d+(?:\.\d+)?\s*(?:cm|mm)\s*(?:in\s*(?:maximum\s*)?diameter|diameter|in size))",
]

TUMOR_LOCATION_PATTERNS = [
    r"(?:located in|location[:\-]?|arising (?:from|in)|involving|within|centered in)\s+"
    r"(the\s+)?((?:left|right|bilateral)?\s*"
    r"(?:frontal|parietal|temporal|occipital|cerebellar|cerebellum|brainstem|"
    r"thalamus|thalamic|basal ganglia|corpus callosum|pineal region|sellar|"
    r"suprasellar|cerebellopontine angle|posterior fossa|intraventricular|"
    r"parasagittal|falcine|convexity)\s*(?:lobe|region|area)?)",
]

EDEMA_PATTERNS = [
    r"((?:mild|moderate|severe|significant|extensive|marked|no|minimal)\s+"
    r"(?:perilesional|peritumoral|vasogenic)?\s*edema)",
    r"(edema is (?:present|absent|noted|not seen|mild|moderate|severe))",
    r"(vasogenic edema)",
    r"(no (?:significant )?(?:surrounding )?edema)",
]

MIDLINE_SHIFT_PATTERNS = [
    r"(midline shift of\s*(?:approximately\s*)?\d+(?:\.\d+)?\s*(?:mm|cm))",
    r"(midline shift[:\-]?\s*(?:present|absent|noted|not seen|none))",
    r"(no (?:evidence of )?midline shift)",
    r"(\d+(?:\.\d+)?\s*(?:mm|cm)\s*(?:of\s*)?midline shift)",
]

CONTRAST_ENHANCEMENT_PATTERNS = [
    r"((?:homogeneous|heterogeneous|ring|peripheral|nodular|avid|mild|no|minimal|patchy)\s*"
    r"(?:contrast\s*)?enhancement)",
    r"(enhances? (?:homogeneously|heterogeneously|avidly|mildly|minimally|following contrast administration))",
    r"(no (?:significant )?(?:contrast )?enhancement)",
]

NECROSIS_PATTERNS = [
    r"(no (?:evidence of )?necrosis)",
    r"((?:central|cystic|extensive)\s*necrosis(?:\s*is\s*(?:present|noted|absent))?)",
    r"(necrotic (?:core|center|component|areas?))",
    r"(necrosis(?:\s*is\s*(?:present|noted|absent))?)",
]

MASS_EFFECT_PATTERNS = [
    r"(no (?:significant )?mass effect)",
    r"((?:mild|moderate|severe|significant)\s*mass effect(?:\s*(?:on|upon)\s*(?:the\s*)?(?:adjacent\s*)?(?:ventric(?:le|ular system)|brainstem|structures))?)",
    r"(mass effect\s*(?:on|upon)\s*(?:the\s*)?(?:adjacent\s*)?(?:ventric(?:le|ular system)|brainstem|structures))",
]

IMPRESSION_PATTERNS = [
    r"(?:impression|conclusion|summary)\s*[:\-]\s*(.+?)(?=\n[A-Z][A-Za-z ]{2,30}:|\Z)",
]


def _search_patterns(text: str, patterns) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if match.groups():
                result = match.group(1) if match.lastindex else match.group(0)
                if match.lastindex and match.lastindex >= 2:
                    result = " ".join(g for g in match.groups() if g)
                return result.strip().rstrip(".,;")
            return match.group(0).strip().rstrip(".,;")
    return None


def _extract_impression(text: str) -> Optional[str]:
    for pattern in IMPRESSION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            impression_text = match.group(1).strip()
            impression_text = re.sub(r"\s+", " ", impression_text)
            if len(impression_text) > 800:
                impression_text = impression_text[:800].rsplit(" ", 1)[0] + "..."
            return impression_text
    return None


def extract_structured_fields(text: str) -> Dict[str, Optional[str]]:
    fields = {
        "Tumor Type": _search_patterns(text, TUMOR_TYPE_PATTERNS),
        "Tumor Size": _search_patterns(text, TUMOR_SIZE_PATTERNS),
        "Tumor Location": _search_patterns(text, TUMOR_LOCATION_PATTERNS),
        "Edema": _search_patterns(text, EDEMA_PATTERNS),
        "Midline Shift": _search_patterns(text, MIDLINE_SHIFT_PATTERNS),
        "Contrast Enhancement": _search_patterns(text, CONTRAST_ENHANCEMENT_PATTERNS),
        "Necrosis": _search_patterns(text, NECROSIS_PATTERNS),
        "Mass Effect": _search_patterns(text, MASS_EFFECT_PATTERNS),
        "Radiologist Impression": _extract_impression(text),
    }

    for key in FIELD_KEYS:
        if fields.get(key) is not None:
            value = re.sub(r"\s+", " ", fields[key]).strip()
            value = re.sub(r"^(the)\s+", "", value, flags=re.IGNORECASE)
            fields[key] = value if value != "" else None

    return fields


def _chunk_text(text: str, tokenizer, max_length: int = 512, stride: int = 50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length - 2:
        return [text]

    chunks = []
    step = max_length - 2 - stride
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + (max_length - 2)]
        if not chunk_tokens:
            break
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if start + (max_length - 2) >= len(tokens):
            break
    return chunks


def get_document_embedding(text: str) -> np.ndarray:
    model_wrapper = get_model()

    chunks = _chunk_text(text, model_wrapper.tokenizer, settings.MAX_LENGTH)

    embeddings = []
    for chunk in chunks:
        emb = model_wrapper.get_cls_embedding(chunk)
        embeddings.append(emb)

    if len(embeddings) == 1:
        return embeddings[0]

    return np.mean(np.stack(embeddings, axis=0), axis=0)


def run_prediction(cleaned_text: str) -> Dict:
    embedding = get_document_embedding(cleaned_text)
    extracted_features = extract_structured_fields(cleaned_text)

    response = {
        "success": True,
        "report_text": cleaned_text,
        "clinicalbert_embedding_dimension": int(embedding.shape[0]),
        "extracted_features": extracted_features,
    }

    return response
