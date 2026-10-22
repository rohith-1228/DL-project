from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
import re
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import easyocr

import config
from registry import normalize_plate, lookup_plate
from database import get_parking_authorization

# Cache singletons for performance across Streamlit reruns
_PLATE_MODEL_INSTANCE: Optional[YOLO] = None
_OCR_READER_INSTANCE: Optional[easyocr.Reader] = None

# Indian state/UT prefixes. This is used to reject arbitrary OCR strings and to
# repair the common ordering artifact where EasyOCR reads the registration body
# first and the two-letter state code afterward.
_INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML",
    "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR",
    "TS", "UK", "UP", "WB",
}
_INDIAN_PLATE_RE = re.compile(r"([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4})")
_GENERIC_PLATE_RE = re.compile(r"(?:[A-Z]{2,4}\d{3,5}|\d{1,4}[A-Z]{1,3}\d{3,4})")
# The project's campus examples also use short alphanumeric inventory tags
# (such as P3RVP). Accept only this constrained shape, not arbitrary OCR text.
_CAMPUS_TAG_RE = re.compile(r"[A-Z]\d[A-Z]{3}")


def _indian_plate_candidate(raw_text: str) -> Tuple[str, bool]:
    """Return a structurally valid Indian plate and whether order was repaired."""
    # EasyOCR commonly returns the blue country badge as a separate small
    # token between the two rows (for example, "AP16 Ind MC8723"). Remove
    # that bounded token before joining the registration characters.
    registration_text = re.sub(
        r"(?i)(?<![A-Z0-9])(?:IND|HND|1ND)(?![A-Z0-9])", " ", str(raw_text or "")
    )
    compact = normalize_plate(registration_text)
    if not compact:
        return "", False

    # IND is the blue country identifier on Indian plates, not part of the
    # registration. OCR may append it before or after the registration.
    # Country marker glyphs are tiny and EasyOCR often confuses IND with HND
    # or 1ND. Ignore only those marker-like strings at the outer edge.
    compact = re.sub(r"^(?:(?:IND|HND|1ND))+|(?:(?:IND|HND|1ND))+$", "", compact)
    # Resolve OCR glyph confusions only inside numeric fields of a known
    # Indian registration pattern (for example AP4O... -> AP40...).
    digit_lookalikes = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "G": "6", "Z": "2"})
    series_lookalikes = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"})
    for state in _INDIAN_STATE_CODES:
        if not compact.startswith(state):
            continue
        tail = compact[2:]
        for district_len in (2, 1):
            for series_len in (1, 2, 3):
                if len(tail) != district_len + series_len + 4:
                    continue
                district = tail[:district_len].translate(digit_lookalikes)
                series = tail[district_len:district_len + series_len].translate(series_lookalikes)
                serial = tail[-4:].translate(digit_lookalikes)
                candidate = state + district + series + serial
                match = _INDIAN_PLATE_RE.fullmatch(candidate)
                if match:
                    return match.group(1), candidate != compact

    direct = _INDIAN_PLATE_RE.search(compact)
    if direct:
        return direct.group(1), False

    # Repair reversed word/box ordering, e.g. "40BS2799APIND" -> AP40BS2799.
    for state in _INDIAN_STATE_CODES:
        if compact.endswith(state):
            candidate = state + compact[:-2]
            match = _INDIAN_PLATE_RE.fullmatch(candidate)
            if match:
                return match.group(1), True

    # Some supplied/test images use non-Indian demonstration registrations
    # (for example PQR 3456, KAB 1234). Keep those readable too, while requiring
    # a plausible mixed letter/digit structure to avoid returning scene text.
    generic = _GENERIC_PLATE_RE.fullmatch(compact)
    if generic and sum(ch.isalpha() for ch in compact) >= 2 and sum(ch.isdigit() for ch in compact) >= 3:
        return generic.group(0), False
    campus_tag = _CAMPUS_TAG_RE.fullmatch(compact)
    if campus_tag:
        return campus_tag.group(0), False
    return "", False

def get_plate_detector(model_path: Optional[Union[str, Path]] = None) -> YOLO:
    """
    Loads and caches the separate YOLO license-plate detector.
    Validates model file existence and raises informative error if missing.
    """
    global _PLATE_MODEL_INSTANCE
    if _PLATE_MODEL_INSTANCE is None:
        target_path = Path(model_path or config.PLATE_MODEL_PATH)
        if not target_path.exists():
            raise FileNotFoundError(
                f"License plate detection model weights not found at: {target_path}\n"
                "Please place a compatible YOLO weights file at models/license_plate_detector.pt "
                "or specify a custom path in config.py / UI settings."
            )
        _PLATE_MODEL_INSTANCE = YOLO(str(target_path))
    return _PLATE_MODEL_INSTANCE

def get_ocr_reader() -> easyocr.Reader:
    """Loads and caches the EasyOCR English recognition reader."""
    global _OCR_READER_INSTANCE
    if _OCR_READER_INSTANCE is None:
        # Initialize EasyOCR reader on CPU with verbose=False
        _OCR_READER_INSTANCE = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER_INSTANCE

def preprocess_plate_image(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Enhances contrast and sharpness of the cropped license plate to assist OCR.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return plate_bgr

    # Resize if very small to improve character clarity
    h, w = plate_bgr.shape[:2]
    if h < 40 or w < 100:
        scale = max(2.0, 100.0 / max(1, w))
        plate_bgr = cv2.resize(plate_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    return plate_bgr

def read_plate_from_vehicle_crop(
    vehicle_crop_rgb: np.ndarray,
    plate_model: Optional[YOLO] = None,
    ocr_reader: Optional[easyocr.Reader] = None,
    conf_threshold: float = config.DEFAULT_PLATE_CONF
) -> Dict[str, Any]:
    """
    Locates license plate inside a vehicle crop using the dedicated plate model,
    crops the plate region, runs EasyOCR, and evaluates normalization & registry match.
    """
    if vehicle_crop_rgb is None or vehicle_crop_rgb.size == 0:
        return {
            "plate_detected": False,
            "plate_crop": None,
            "raw_text": "",
            "normalized_plate": "",
            "ocr_confidence": 0.0,
            "reading_status": "Plate unreadable",
            "requires_manual_review": True,
            "registry_record": None,
            "registry_match_status": "No Plate",
            "eligible_for_notification": False
        }

    # 1. Load Plate Detector
    try:
        detector = plate_model or get_plate_detector()
    except FileNotFoundError as e:
        return {
            "plate_detected": False,
            "plate_crop": None,
            "raw_text": "",
            "normalized_plate": "",
            "ocr_confidence": 0.0,
            "reading_status": "Setup Error: Plate weights missing",
            "status_detail": str(e),
            "requires_manual_review": True,
            "registry_record": None,
            "registry_match_status": "Model Missing",
            "eligible_for_notification": False
        }

    # 2. Run plate detection inference on vehicle crop
    crop_h, crop_w, _ = vehicle_crop_rgb.shape
    results = detector.predict(
        source=vehicle_crop_rgb,
        conf=conf_threshold,
        verbose=False
    )

    best_box = None
    best_conf = 0.0

    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf > best_conf:
                best_conf = conf
                xyxy = box.xyxy[0].cpu().numpy()
                best_box = [int(v) for v in xyxy]

    if best_box is None:
        # Fallback: No plate detected by YOLO
        return {
            "plate_detected": False,
            "plate_crop": None,
            "raw_text": "",
            "normalized_plate": "",
            "ocr_confidence": 0.0,
            "reading_status": "Plate unreadable",
            "status_detail": "No license plate localized within vehicle bounding box.",
            "requires_manual_review": True,
            "registry_record": None,
            "registry_match_status": "Not Applicable",
            "eligible_for_notification": False
        }

    # 3. Crop plate region
    px1, py1, px2, py2 = best_box
    px1 = max(0, min(crop_w - 1, px1))
    px2 = max(0, min(crop_w - 1, px2))
    py1 = max(0, min(crop_h - 1, py1))
    py2 = max(0, min(crop_h - 1, py2))

    if px2 <= px1 or py2 <= py1:
        return {
            "plate_detected": False,
            "plate_crop": None,
            "raw_text": "",
            "normalized_plate": "",
            "ocr_confidence": 0.0,
            "reading_status": "Plate unreadable",
            "status_detail": "Plate bounding box invalid dimensions.",
            "requires_manual_review": True,
            "registry_record": None,
            "registry_match_status": "Not Applicable",
            "eligible_for_notification": False
        }

    plate_crop = vehicle_crop_rgb[py1:py2, px1:px2].copy()
    preprocessed_plate = preprocess_plate_image(plate_crop)

    # 4. OCR character extraction. Upscaling very small crops can make 6/G or
    # 0/O errors worse, so compare native and enlarged crops and keep the
    # highest-confidence structurally valid registration.
    reader = ocr_reader or get_ocr_reader()
    ocr_variants = [plate_crop]
    if preprocessed_plate.shape[:2] != plate_crop.shape[:2]:
        ocr_variants.append(preprocessed_plate)
    best_read = None
    fallback_read = None
    for variant in ocr_variants:
        variant_results = reader.readtext(variant)
        if not variant_results:
            continue
        variant_results = sorted(variant_results, key=lambda r: (r[0][0][1], r[0][0][0]))
        raw_text = " ".join(r[1] for r in variant_results)
        candidate, repaired = _indian_plate_candidate(raw_text)
        confidence = float(np.mean([float(r[2]) for r in variant_results]))
        result = (confidence, raw_text, candidate, repaired)
        if fallback_read is None or confidence > fallback_read[0]:
            fallback_read = result
        if candidate and (best_read is None or confidence > best_read[0]):
            best_read = result

    chosen_read = best_read or fallback_read
    if chosen_read is None:
        return {
            "plate_detected": True,
            "plate_crop": plate_crop,
            "raw_text": "",
            "normalized_plate": "",
            "ocr_confidence": 0.0,
            "reading_status": "Plate unreadable",
            "status_detail": "Plate region localized, but characters were illegible.",
            "requires_manual_review": True,
            "registry_record": None,
            "registry_match_status": "Unreadable",
            "eligible_for_notification": False
        }

    avg_conf, combined_raw, normalized, ordering_repaired = chosen_read

    # 5. Evaluate readability confidence
    if not normalized:
        reading_status = "Plate unreadable"
        requires_manual_review = True
    elif ordering_repaired or avg_conf < config.DEFAULT_OCR_CONF:
        reading_status = "Uncertain (Review required)"
        requires_manual_review = True
    else:
        reading_status = "Recognized"
        requires_manual_review = False

    # 6. Lookup in registry and authorization database
    registry_record = lookup_plate(normalized)
    auth_record = get_parking_authorization(normalized) if normalized else None
    eligible = False

    if registry_record is None:
        registry_match_status = "Not in Registry"
    else:
        if registry_record.get("consent_status", 0) == 1:
            registry_match_status = "Registered & Consented"
            # Eligible if readable and consented
            eligible = (reading_status == "Recognized")
        else:
            registry_match_status = "Consent Disabled"
            eligible = False

    return {
        "plate_detected": True,
        "plate_crop": plate_crop,
        "raw_text": combined_raw,
        "normalized_plate": normalized,
        "ocr_confidence": round(avg_conf, 3),
        "reading_status": reading_status,
        "requires_manual_review": requires_manual_review,
        "registry_record": registry_record,
        "registry_match_status": registry_match_status,
        "authorization": auth_record,
        "eligible_for_notification": eligible
    }
