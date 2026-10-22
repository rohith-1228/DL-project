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

def preprocess_plate_image(plate_rgb: np.ndarray) -> np.ndarray:
    """
    Enhances contrast and sharpness of the cropped license plate to assist OCR.
    """
    if plate_rgb is None or plate_rgb.size == 0:
        return plate_rgb

    # OCR often receives plates below 100 pixels wide after a vehicle crop.
    # Enlarge first, then normalize local contrast and sharpen character edges.
    h, w = plate_rgb.shape[:2]
    scale = max(1.0, min(4.0, 160.0 / max(1, w)))
    if scale > 1.0:
        plate_rgb = cv2.resize(
            plate_rgb, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    lab = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def _ocr_localize_plate(vehicle_crop_rgb: np.ndarray, reader: Any) -> Optional[Dict[str, Any]]:
    """Find a structurally valid plate with OCR when the plate detector misses."""
    height, width = vehicle_crop_rgb.shape[:2]
    y_offset = int(height * 0.20)
    search = vehicle_crop_rgb[y_offset:, :]
    if search.size == 0:
        return None
    scale = min(3.0, max(1.5, 100.0 / max(1, search.shape[0])))
    enlarged = cv2.resize(
        search, (max(1, int(search.shape[1] * scale)), max(1, int(search.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    enhanced = preprocess_plate_image(enlarged)
    try:
        results = reader.readtext(
            enhanced, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
    except TypeError:
        try:
            results = reader.readtext(enhanced, paragraph=False)
        except TypeError:
            results = reader.readtext(enhanced)
    if not results:
        return None

    # Plates are usually low on a vehicle. Keep lower-scene text out of the OCR
    # join and accept only a plausible registration string.
    usable = []
    for item in results:
        if len(item) < 3:
            continue
        points = np.asarray(item[0], dtype=float).reshape(-1, 2)
        center_y = float(points[:, 1].mean()) / max(1.0, scale) + y_offset
        if center_y < height * 0.42:
            continue
        usable.append((float(points[:, 1].mean()), float(points[:, 0].mean()), item, points))
    if not usable:
        return None
    usable.sort(key=lambda row: (row[0], row[1]))
    raw_text = " ".join(str(row[2][1]) for row in usable)
    candidate, repaired = _indian_plate_candidate(raw_text)
    if not candidate:
        return None

    all_points = np.concatenate([row[3] for row in usable], axis=0)
    x1 = max(0, int(np.floor(all_points[:, 0].min() / scale)) - 4)
    x2 = min(width, int(np.ceil(all_points[:, 0].max() / scale)) + 5)
    y1 = max(0, int(np.floor(all_points[:, 1].min() / scale)) + y_offset - 4)
    y2 = min(height, int(np.ceil(all_points[:, 1].max() / scale)) + y_offset + 5)
    if x2 <= x1 or y2 <= y1:
        return None
    confidence = float(np.mean([float(row[2][2]) for row in usable]))
    return {
        "plate_crop": vehicle_crop_rgb[y1:y2, x1:x2].copy(),
        "raw_text": raw_text,
        "normalized_plate": candidate,
        "ocr_confidence": confidence,
        "ordering_repaired": repaired,
    }

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

    # 1. Load the detector. OCR localization remains available when a deployment
    # omitted the optional weight file or the detector misses a small plate.
    model_missing_detail = ""
    try:
        detector = plate_model or get_plate_detector()
    except FileNotFoundError as e:
        detector = None
        model_missing_detail = str(e)

    # 2. Run plate detection inference on vehicle crop
    crop_h, crop_w, _ = vehicle_crop_rgb.shape
    results = []
    if detector is not None:
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

    reader = ocr_reader or get_ocr_reader()
    ocr_localized = None
    if best_box is None:
        ocr_localized = _ocr_localize_plate(vehicle_crop_rgb, reader)
        if ocr_localized is None:
            # Always provide a useful enlarged view for manual verification. It
            # is explicitly marked as estimated and is never treated as OCR.
            fallback_y = int(crop_h * 0.45)
            fallback_crop = vehicle_crop_rgb[fallback_y:, :].copy()
            if fallback_crop.size:
                fallback_crop = cv2.resize(
                    fallback_crop,
                    (max(1, fallback_crop.shape[1] * 2), max(1, fallback_crop.shape[0] * 2)),
                    interpolation=cv2.INTER_CUBIC,
                )
            return {
                "plate_detected": False,
                "plate_crop": fallback_crop if fallback_crop is not None and fallback_crop.size else None,
                "plate_crop_source": "estimated_vehicle_lower_region",
                "raw_text": "",
                "normalized_plate": "",
                "ocr_confidence": 0.0,
                "reading_status": "Plate unreadable",
                "status_detail": model_missing_detail or "No plate box or valid registration text was localized.",
                "requires_manual_review": True,
                "registry_record": None,
                "registry_match_status": "Not Applicable",
                "eligible_for_notification": False
            }
        plate_crop = ocr_localized["plate_crop"]
        preprocessed_plate = preprocess_plate_image(plate_crop)
        chosen_read = (
            ocr_localized["ocr_confidence"],
            ocr_localized["raw_text"],
            ocr_localized["normalized_plate"],
            ocr_localized["ordering_repaired"],
        )
        plate_crop_source = "ocr_localized"
    else:
        plate_crop_source = "plate_detector"

    if best_box is not None:
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
        # Give the OCR a little border so detector boxes do not clip plate edges.
        pad_x = max(2, int((px2 - px1) * 0.06))
        pad_y = max(2, int((py2 - py1) * 0.10))
        px1, px2 = max(0, px1 - pad_x), min(crop_w, px2 + pad_x)
        py1, py2 = max(0, py1 - pad_y), min(crop_h, py2 + pad_y)
        plate_crop = vehicle_crop_rgb[py1:py2, px1:px2].copy()
        preprocessed_plate = preprocess_plate_image(plate_crop)

    # 4. OCR character extraction. Upscaling very small crops can make 6/G or
    # 0/O errors worse, so compare native and enlarged crops and keep the
    # highest-confidence structurally valid registration.
    ocr_candidate_disagreement = False
    if best_box is not None:
        # Keep a color-preserving enlargement as a first-class view. Contrast
        # enhancement can help faint characters but can also erase the tail on
        # Q or turn it into O, so transformed variants receive no automatic
        # preference when their candidate differs.
        color_enlarged = cv2.resize(
            plate_crop,
            (max(1, plate_crop.shape[1] * 3), max(1, plate_crop.shape[0] * 3)),
            interpolation=cv2.INTER_CUBIC,
        )
        ocr_variants = [(color_enlarged, 0.10), (preprocessed_plate, 0.0)]
        if plate_crop.shape[:2] not in (color_enlarged.shape[:2], preprocessed_plate.shape[:2]):
            ocr_variants.append((plate_crop, -0.05))
        best_read = None
        fallback_read = None
        valid_candidates = set()
        for variant, preference in ocr_variants:
            try:
                variant_results = reader.readtext(
                    variant, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
            except TypeError:
                variant_results = reader.readtext(variant)
            if not variant_results:
                continue
            variant_results = sorted(variant_results, key=lambda r: (r[0][0][1], r[0][0][0]))
            raw_text = " ".join(r[1] for r in variant_results)
            candidate, repaired = _indian_plate_candidate(raw_text)
            confidence = float(np.mean([float(r[2]) for r in variant_results]))
            result = (confidence, raw_text, candidate, repaired)
            if candidate:
                valid_candidates.add(candidate)
            if fallback_read is None or confidence > fallback_read[0]:
                fallback_read = result
            if candidate and (
                best_read is None
                or confidence + preference > best_read[0] + best_read[4]
            ):
                best_read = (*result, preference)
        if best_read is not None:
            chosen_read = best_read[:4]
            ocr_candidate_disagreement = len(valid_candidates) > 1
        else:
            chosen_read = fallback_read
    if chosen_read is None:
        return {
            "plate_detected": True,
            "plate_crop": plate_crop,
            "plate_crop_source": plate_crop_source,
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
    elif ordering_repaired or avg_conf < config.DEFAULT_OCR_CONF or ocr_candidate_disagreement:
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
        "plate_crop_source": plate_crop_source,
        "raw_text": combined_raw,
        "normalized_plate": normalized,
        "ocr_confidence": round(avg_conf, 3),
        "ocr_candidate_disagreement": ocr_candidate_disagreement,
        "reading_status": reading_status,
        "requires_manual_review": requires_manual_review,
        "registry_record": registry_record,
        "registry_match_status": registry_match_status,
        "authorization": auth_record,
        "eligible_for_notification": eligible
    }
