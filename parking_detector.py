from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

import config
from campus_violation_engine import (
    detect_footpath_regions,
    detect_entrance_clearance_corridors,
    detect_lawn_green_spaces,
    infer_offsed_surface,
    detect_parking_stall_lines,
    detect_reserved_slot_markings,
    evaluate_all_campus_violations,
)

# Cache for YOLO model instances
_MODEL_CACHE: Dict[str, YOLO] = {}

def get_vehicle_model(model_path: Optional[Union[str, Path]] = None) -> YOLO:
    """
    Loads and caches a YOLO model for vehicle detection.
    Falls back to 'yolov8n.pt' if custom model path is not specified or not found.
    """
    global _MODEL_CACHE
    target_path = Path(model_path or config.VEHICLE_MODEL_PATH)
    if not target_path.exists():
        if Path("yolov8n.pt").exists():
            target_path = Path("yolov8n.pt")
        elif (config.MODELS_DIR / "yolov8n.pt").exists():
            target_path = config.MODELS_DIR / "yolov8n.pt"

    path_key = str(target_path.resolve()) if target_path.exists() else str(target_path)
    if path_key not in _MODEL_CACHE:
        _MODEL_CACHE[path_key] = YOLO(str(target_path))
    return _MODEL_CACHE[path_key]

def get_overhead_vehicle_model(model_path: Optional[Union[str, Path]] = None) -> YOLO:
    """
    Loads and caches the overhead/aerial vehicle detection model (YOLOv8n-VisDrone).
    """
    target = model_path or getattr(config, "OVERHEAD_VEHICLE_MODEL_PATH", config.MODELS_DIR / "yolov8n_visdrone.pt")
    return get_vehicle_model(target)

def get_model_vehicle_classes(model: YOLO) -> Dict[int, str]:
    """
    Dynamically maps model class IDs to canonical vehicle class names.
    Supports COCO (car=2, motorcycle=3, bus=5, truck=7),
    VisDrone (car=3, van=4, truck=5, bus=8, motor=9),
    custom parking models (occupied=1), etc.
    """
    if not hasattr(model, "names") or not model.names:
        return dict(config.VEHICLE_CLASSES)

    names = model.names
    keywords = getattr(config, "VEHICLE_CLASS_KEYWORDS", {
        "car": "car",
        "automobile": "car",
        "vehicle": "car",
        "sedan": "car",
        "suv": "car",
        "motorcycle": "motorcycle",
        "motorbike": "motorcycle",
        "motor": "motorcycle",
        "bike": "motorcycle",
        "bus": "bus",
        "truck": "truck",
        "pickup": "truck",
        "van": "van",
        "ocupied": "car",
        "occupied": "car",
    })

    mapped: Dict[int, str] = {}
    for cls_id, raw_name in names.items():
        name_lower = str(raw_name).lower().strip()
        for kw, cat in keywords.items():
            if kw in name_lower:
                mapped[int(cls_id)] = cat
                break

    if not mapped:
        return dict(config.VEHICLE_CLASSES)

    return mapped

def extract_vehicle_detections_from_model(
    model: YOLO,
    image_rgb: np.ndarray,
    conf_threshold: float,
    imgsz: Optional[int] = None,
    iou_threshold: float = 0.45
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runs YOLO inference on the image, captures all raw detection candidates
    before class filtering, and extracts valid vehicles using dynamic class mapping.
    """
    height, width = image_rgb.shape[:2]
    model_classes = get_model_vehicle_classes(model)

    predict_kwargs = {
        "source": image_rgb,
        "conf": conf_threshold,
        "iou": iou_threshold,
        "verbose": False
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    results = model.predict(**predict_kwargs)

    raw_detections = []
    vehicle_detections = []
    raw_class_counts: Dict[str, int] = {}

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            raw_label = model.names.get(cls_id, f"class_{cls_id}") if hasattr(model, "names") else str(cls_id)
            raw_class_counts[raw_label] = raw_class_counts.get(raw_label, 0) + 1

            raw_detections.append({
                "class_id": cls_id,
                "class_name": raw_label,
                "confidence": round(conf, 3),
                "bbox": (x1, y1, x2, y2)
            })

            if cls_id in model_classes:
                canonical_name = model_classes[cls_id]
                gx, gy = calculate_ground_contact_point((x1, y1, x2, y2))
                vehicle_detections.append({
                    "class_name": canonical_name,
                    "confidence": round(conf, 3),
                    "bbox": (x1, y1, x2, y2),
                    "ground_point": (gx, gy),
                    "crop": image_rgb[y1:y2, x1:x2].copy(),
                    "raw_label": raw_label
                })

    model_name = getattr(model, "ckpt_path", None) or getattr(model, "model_name", "YOLO Model")
    if hasattr(model, "names"):
        class_labels_display = {k: v for k, v in model.names.items() if k in model_classes}
    else:
        class_labels_display = model_classes

    diagnostics = {
        "model_name": str(model_name),
        "class_labels": class_labels_display,
        "conf_threshold": conf_threshold,
        "image_size": (width, height),
        "raw_detection_count": len(raw_detections),
        "vehicle_detection_count": len(vehicle_detections),
        "raw_class_breakdown": raw_class_counts,
        "raw_detections": raw_detections
    }

    return vehicle_detections, diagnostics

def parse_polygon_coordinates(
    zone_dict: Dict[str, Any],
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Converts either normalized or absolute polygon coordinates into an
    integer numpy array suitable for OpenCV contour calculations.
    """
    if "normalized_polygon" in zone_dict:
        pts = [
            [int(p[0] * image_width), int(p[1] * image_height)]
            for p in zone_dict["normalized_polygon"]
        ]
    elif "polygon" in zone_dict:
        pts = [[int(p[0]), int(p[1])] for p in zone_dict["polygon"]]
    else:
        raise ValueError("Zone definition must contain 'normalized_polygon' or 'polygon'.")
    
    return np.array(pts, dtype=np.int32)

def calculate_ground_contact_point(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculates the approximate ground-contact point of a vehicle.
    Standard convention: Midpoint of the bottom edge of the bounding box: ( (x1 + x2) / 2, y2 ).
    """
    x1, y1, x2, y2 = bbox
    gx = int((x1 + x2) / 2)
    gy = int(y2)
    return (gx, gy)

def check_point_in_zone(
    point: Tuple[int, int],
    polygon_contour: np.ndarray
) -> bool:
    """
    Tests if a 2D point (gx, gy) lies inside or on the boundary of a polygon contour.
    Uses OpenCV cv2.pointPolygonTest (>= 0 indicates inside or on edge).
    """
    gx, gy = point
    result = cv2.pointPolygonTest(polygon_contour, (float(gx), float(gy)), False)
    return result >= 0

def check_ground_contact_violation(
    ground_point: Tuple[int, int],
    zones: Optional[List[Dict[str, Any]]],
    image_width: int,
    image_height: int
) -> Tuple[bool, Optional[str]]:
    """
    Checks if a vehicle's ground-contact point falls inside any configured restricted zone.
    Returns: (is_violation, matched_zone_name_or_None)
    If zones is None or empty, returns (False, None).
    """
    if not zones:
        return False, None

    for zone in zones:
        try:
            poly_contour = parse_polygon_coordinates(zone, image_width, image_height)
        except Exception:
            continue
        if poly_contour.shape[0] < 3:
            continue
        if check_point_in_zone(ground_point, poly_contour):
            return True, zone.get("name", "Restricted Area")
    return False, None

def detect_scene_visual_evidence(
    image_rgb: np.ndarray,
    vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    yolo_model: Optional[YOLO] = None,
    ocr_reader: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Analyzes the scene automatically to detect visual evidence of parking restrictions:
    1. Surface Markings (HSV color & morphology):
       - Red curb / emergency fire lane markings
       - Yellow hatched loading bays / yellow curbs
       - Pedestrian walkway striping
       - Accessible blue parking bay demarcations
    2. Street Furniture & Signs:
       - Fire hydrant emergency clearance area
       - No-parking road signs or stop signs (via OCR & pretrained models)
    3. Spatial Boundary Reasoning:
       - If clear surface markings or physical barriers exist -> 'Confirmed Visual Evidence'
       - If a road sign is detected but exact boundaries cannot be reliably delineated
         without road markings -> 'Restricted zone could not be determined reliably'
       - If no signs and no markings -> 'No restricted zone detected'
    """
    height, width, _ = image_rgb.shape
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    evidence_items: List[str] = []
    active_zones: List[Dict[str, Any]] = []
    sign_boxes: List[Dict[str, Any]] = []

    # 1. Surface Markings: Red curb / Fire Lane
    m1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    m2 = cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
    red_mask = m1 | m2

    # Curbs must be detected from the scene, never from vehicle paint or lamps.
    clean_red_mask = red_mask.copy()
    if vehicle_boxes:
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            clean_red_mask[vy1:vy2, vx1:vx2] = 0

    # Search for elongated red curb contours
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 35))
    closed_v = cv2.morphologyEx(clean_red_mask, cv2.MORPH_CLOSE, k_v)
    contours_v, _ = cv2.findContours(closed_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    curb_segments = []
    total_area = width * height
    for c in contours_v:
        area = cv2.contourArea(c)
        if area > 800:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            thickness = min(bw, bh)
            length = max(bw, bh)
            # Exclude red paving blocks, façade details, and distant signs:
            # a painted curb is a thin stripe at ground level, not a broad red
            # patch anywhere in the frame.
            if (
                aspect >= 2.0
                and thickness <= max(18, int(0.025 * height))
                and length >= max(80, int(0.08 * height))
                and area < 0.02 * total_area
                and y + bh >= int(0.48 * height)
            ):
                curb_segments.append((x, y, bw, bh, area))

    if curb_segments:
        evidence_items.append("Confirmed: Painted Red Curb / Fire Lane (Emergency Access)")
        for x, y, bw, bh, area in curb_segments:
            if bh >= bw:
                # Vertical curb: span corridor along curb with clearance into road
                zone_x1 = max(0, x - 60)
                zone_x2 = min(width - 1, x + bw + 40)
                zone_y1 = max(0, y - 40)
                zone_y2 = min(height - 1, y + bh + 100)
                poly = [[zone_x1, zone_y1], [zone_x2, zone_y1], [zone_x2, zone_y2], [zone_x1, zone_y2]]
            else:
                # Horizontal curb: span corridor localized along curb
                zone_x1 = max(0, x - 50)
                zone_x2 = min(width - 1, x + bw + 50)
                zone_y1 = max(0, y - 40)
                zone_y2 = min(height - 1, y + bh + 80)
                poly = [[zone_x1, zone_y1], [zone_x2, zone_y1], [zone_x2, zone_y2], [zone_x1, zone_y2]]
            active_zones.append({
                "id": "auto_zone_red_curb",
                "name": "Emergency Access & Red Curb",
                "polygon": poly,
                "evidence_type": "Confirmed Visual Evidence: Painted Red Curb / Fire Lane",
                "category": "Emergency access area"
            })

    # A broad, continuous red strip can mark a no-parking lane. Keep this
    # separate from the thin-curb detector and inspect its full shape before
    # masking vehicle pixels: a narrow vertical band that spans the scene is
    # distinct from red vehicle paint or a small sign.
    raw_red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in raw_red_contours:
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        if (
            area >= 0.04 * total_area
            and area <= 0.20 * total_area
            and bh >= 0.45 * height
            and 6 <= bw <= 0.12 * width
            and bh / max(1, bw) >= 3.0
            and y + bh >= 0.55 * height
        ):
            poly = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
            active_zones.append({
                "id": "auto_zone_red_restricted_strip",
                "name": "Marked Restricted Parking Strip",
                "polygon": poly,
                "evidence_type": "Confirmed Visual Evidence: Broad continuous red no-parking surface marking",
                "category": "Restricted parking area",
            })
            evidence_items.append("Confirmed: Broad Red No-Parking Surface Marking")

    # Vehicle occlusion can split one long painted strip into top and bottom
    # contours. A vertical pixel projection reconnects those visible segments
    # without treating isolated red car panels as a zone.
    red_columns = np.count_nonzero(red_mask > 0, axis=0)
    active_red_cols = red_columns >= int(0.30 * height)
    starts = np.flatnonzero(active_red_cols & ~np.r_[False, active_red_cols[:-1]])
    ends = np.flatnonzero(active_red_cols & ~np.r_[active_red_cols[1:], False]) + 1
    for x1, x2 in zip(starts, ends):
        if not (6 <= x2 - x1 <= 0.12 * width):
            continue
        visible_y = np.flatnonzero(np.any(red_mask[:, x1:x2] > 0, axis=1))
        if len(visible_y) == 0 or visible_y[-1] - visible_y[0] < 0.45 * height:
            continue
        # Require the red segments to span most of the image vertically and
        # enough total pixels to exclude small signs and red vehicle details.
        if red_columns[x1:x2].sum() < 0.04 * total_area:
            continue
        y1, y2 = int(visible_y[0]), int(visible_y[-1] + 1)
        poly = [[int(x1), y1], [int(x2), y1], [int(x2), y2], [int(x1), y2]]
        if any(zone.get("id") == "auto_zone_red_restricted_strip" for zone in active_zones):
            continue
        active_zones.append({
            "id": "auto_zone_red_restricted_strip",
            "name": "Marked Restricted Parking Strip",
            "polygon": poly,
            "evidence_type": "Confirmed Visual Evidence: Continuous red no-parking strip, partly occluded by a vehicle",
            "category": "Restricted parking area",
        })
        evidence_items.append("Confirmed: Broad Red No-Parking Surface Marking")

    # 2. Pretrained model objects: Fire Hydrant & Stop Sign
    if yolo_model is not None:
        try:
            # Predict with lower confidence for objects of interest (classes 10: fire hydrant, 11: stop sign)
            obj_results = yolo_model.predict(source=image_rgb, conf=0.25, verbose=False)
            if len(obj_results) > 0 and obj_results[0].boxes is not None:
                for b in obj_results[0].boxes:
                    cls_id = int(b.cls[0].cpu().numpy())
                    b_conf = float(b.conf[0].cpu().numpy())
                    xyxy = [int(v) for v in b.xyxy[0].cpu().numpy()]
                    
                    if cls_id == 10:  # Fire Hydrant
                        evidence_items.append(f"Confirmed: Fire Hydrant (Emergency Clearance)")
                        hx1, hy1, hx2, hy2 = xyxy
                        # Generate emergency clearance perimeter at curb level
                        poly = [
                            [max(0, hx1 - 60), min(height - 1, hy2 - 20)],
                            [min(width - 1, hx2 + 60), min(height - 1, hy2 - 20)],
                            [min(width - 1, hx2 + 60), min(height - 1, hy2 + int(0.18 * height))],
                            [max(0, hx1 - 60), min(height - 1, hy2 + int(0.18 * height))]
                        ]
                        active_zones.append({
                            "id": "auto_zone_fire_hydrant",
                            "name": "Emergency Access: Fire Hydrant Clearance",
                            "polygon": poly,
                            "evidence_type": "Confirmed Visual Evidence: Fire Hydrant Emergency Area",
                            "category": "Emergency access area"
                        })
                    elif cls_id == 11:  # Stop Sign
                        evidence_items.append(f"Signage: Stop Sign ({round(b_conf, 2)})")
                        sign_boxes.append({
                            "type": "stop_sign",
                            "label": "STOP SIGN",
                            "bbox": (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                        })
        except Exception:
            pass

    # 3. Sign text detection via OCR for keywords (e.g. 'NO PARKING', 'TOW AWAY', 'FIRE LANE')
    ocr_entrance_corridor = None
    if ocr_reader is not None:
        try:
            # Retain enough pixels to read small entrance plaques and signs.
            scale = 1024.0 / max(height, width) if max(height, width) > 1024 else 1.0
            ocr_img = cv2.resize(image_rgb, (int(width * scale), int(height * scale))) if scale != 1.0 else image_rgb
            ocr_entries = ocr_reader.readtext(ocr_img)
            entrance_sign_bounds = []
            saw_entrance_word = False
            saw_building_word = False
            saw_main_word = False
            
            keywords = getattr(config, "RESTRICTED_SIGN_KEYWORDS", ["NO PARKING", "NO STOPPING", "TOW AWAY", "FIRE LANE", "KEEP CLEAR"])
            for entry in ocr_entries:
                txt = entry[1].upper().strip()
                if "ENTRANCE" in txt:
                    saw_entrance_word = True
                if "BUILDING" in txt:
                    saw_building_word = True
                if "MAIN" in txt:
                    saw_main_word = True
                if "ENTRANCE" in txt or "BUILDING" in txt:
                    pts = np.array(entry[0], dtype=np.float32)
                    if scale != 1.0:
                        pts /= scale
                    entrance_sign_bounds.append((
                        int(np.min(pts[:, 0])), int(np.min(pts[:, 1])),
                        int(np.max(pts[:, 0])), int(np.max(pts[:, 1])),
                    ))
                if any(kw in txt for kw in keywords):
                    evidence_items.append(f"Signage: {txt}")
                    # Map OCR bbox back to image dimensions
                    pts = np.array(entry[0], dtype=np.float32)
                    if scale != 1.0:
                        pts /= scale
                    bx1 = int(np.min(pts[:, 0]))
                    by1 = int(np.min(pts[:, 1]))
                    bx2 = int(np.max(pts[:, 0]))
                    by2 = int(np.max(pts[:, 1]))
                    sign_boxes.append({
                        "type": "road_sign",
                        "label": f"SIGN: {txt}",
                        "bbox": (bx1, by1, bx2, by2)
                    })

            # OCR can read the sign even when the gate bars or clearance
            # markings are fragmented. Project a ground-level approach corridor
            # from the signed entrance toward the side of the frame where the
            # doorway/gate opening lies.
            # Entrance signs may identify the access as MAIN, STAFF, SERVICE,
            # or BUILDING. The word ENTRANCE itself is the general cue.
            if saw_entrance_word and entrance_sign_bounds:
                sx1 = min(box[0] for box in entrance_sign_bounds)
                sy1 = min(box[1] for box in entrance_sign_bounds)
                sx2 = max(box[2] for box in entrance_sign_bounds)
                sy2 = max(box[3] for box in entrance_sign_bounds)
                if (sx1 + sx2) / 2 >= width / 2:
                    corridor_x1, corridor_x2 = max(0, sx1 - int(0.28 * width)), sx2
                else:
                    corridor_x1, corridor_x2 = sx1, min(width - 1, sx2 + int(0.28 * width))
                corridor_bottom = min(height - 1, sy2 + int(0.38 * height))
                corridor = [
                    [corridor_x1, sy1], [corridor_x2, sy1],
                    [corridor_x2, corridor_bottom], [corridor_x1, corridor_bottom],
                ]
                ocr_entrance_corridor = {
                    "id": "ocr_building_entrance_approach",
                    "name": "Building Entrance Access Corridor",
                    "polygon": corridor,
                    "bbox": (corridor_x1, sy1, corridor_x2, corridor_bottom),
                    "evidence_type": "OCR-confirmed entrance sign and projected access corridor",
                }
                evidence_items.append("Confirmed: Building entrance sign and access corridor")
        except Exception:
            pass

    # 4. Campus Feature Detectors: Footpaths, Entrances, Lawns, Stalls, Reserved Slots
    scene_masks: Dict[str, np.ndarray] = {}
    footpaths = detect_footpath_regions(image_rgb, vehicle_boxes=vehicle_boxes, scene_masks=scene_masks)
    entrances = detect_entrance_clearance_corridors(
        image_rgb, vehicle_boxes=vehicle_boxes
    )
    if ocr_entrance_corridor is not None:
        entrances.append(ocr_entrance_corridor)
    surface_prediction = infer_offsed_surface(image_rgb)
    lawns = detect_lawn_green_spaces(
        image_rgb,
        vehicle_boxes=vehicle_boxes,
        surface_prediction=surface_prediction,
    )
    stall_data = detect_parking_stall_lines(image_rgb, vehicle_boxes=vehicle_boxes)
    reserved_slots = detect_reserved_slot_markings(image_rgb, ocr_reader=ocr_reader)

    if footpaths:
        evidence_items.append(f"Confirmed: Pedestrian Footpath / Sidewalk ({len(footpaths)} region(s))")
    if entrances:
        evidence_items.append(f"Confirmed: Gate / Emergency Entrance Clearance ({len(entrances)} corridor(s))")
    if lawns:
        evidence_items.append(f"Confirmed: Campus Lawn / Landscaped Green Space ({len(lawns)} zone(s))")
    if stall_data.get("is_reliable"):
        evidence_items.append(f"Confirmed: Parking Stall Markings ({len(stall_data['divider_x_coords'])} divider lines)")
    if reserved_slots:
        labels_str = ", ".join(s.get("label", "RESERVED") for s in reserved_slots)
        evidence_items.append(f"Confirmed: Reserved Slot Designations ({labels_str})")

    # 5. Synthesize final zone status and reliability
    has_campus_features = bool(active_zones or footpaths or entrances or lawns or reserved_slots)
    if has_campus_features:
        zone_status = "Confirmed Visual Evidence"
        uncertainty_reason = ""
    elif sign_boxes:
        # A road sign or stop sign was detected, but no painted surface markings or physical demarcations exist
        # Requirement: Do not treat the mere presence of a sign as proof every vehicle on the road is violating.
        # If reliable boundary cannot be determined, avoid sending false violation notice.
        zone_status = "Restricted zone could not be determined reliably"
        uncertainty_reason = (
            "A road sign was identified in the scene, but exact spatial restricted boundaries "
            "cannot be determined reliably from this camera view without painted surface markings or 3D calibration."
        )
    else:
        zone_status = "No restricted zone detected"
        uncertainty_reason = ""

    return {
        "zone_status": zone_status,
        "active_zones": active_zones,
        "evidence_items": evidence_items,
        "sign_boxes": sign_boxes,
        "uncertainty_reason": uncertainty_reason,
        "footpaths": footpaths,
        "entrances": entrances,
        "lawns": lawns,
        "_building_mask": scene_masks.get("building"),
        "_cityscapes_prediction": scene_masks.get("cityscapes"),
        "_surface_prediction": surface_prediction,
        "stall_data": stall_data,
        "reserved_slots": reserved_slots
    }

def detect_vehicles_and_violations(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[YOLO] = None,
    conf_threshold: float = config.DEFAULT_VEHICLE_CONF,
    zones: Optional[List[Dict[str, Any]]] = None,
    ocr_reader: Optional[Any] = None,
    plate_model: Optional[YOLO] = None,
    db_path: Optional[Path] = None,
    imgsz: Optional[int] = None,
    iou_threshold: float = 0.45,
    allow_overhead_fallback: bool = True
) -> Dict[str, Any]:
    """
    Automated vehicle detection, scene understanding, and parking violation review pipeline.

    1. Ingests and normalizes image input.
    2. Runs vehicle detection model with dynamic class mapping (COCO, VisDrone, custom).
       Automatically provides adaptive fallback for overhead/aerial parking lot images.
    3. Automatically analyzes the scene for visual evidence across all 6 campus priority categories:
       - Category 1: Unauthorized parking / no-parking zones (red curb, fire hydrant, signage)
       - Category 2: Footpath parking (sidewalks, pedestrian curbs)
       - Category 3: Blocking entrances and emergency access (gates, yellow clearance boxes)
       - Category 4: Parking on green spaces (lawns, landscaped turf)
       - Category 5: Improper parking alignment (divider lines, multi-space overlap)
       - Category 6: Wrong parking slot (faculty/staff/visitor/principal bays with authorization check)
    4. Evaluates vehicle ground-contact point and bounding box against confirmed scene features.
    5. Returns three possible outcomes per category:
       - 'Potential violation detected'
       - 'Not detected'
       - 'Review required'
    6. Produces clear, distinguishable visual annotations and comprehensive diagnostics.
    """
    # 1. Image preprocessing
    if isinstance(image_input, (str, Path)):
        image_pil = Image.open(image_input).convert("RGB")
        image_rgb = np.array(image_pil)
    elif isinstance(image_input, Image.Image):
        image_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    height, width, _ = image_rgb.shape

    # 2. YOLO Model inference with dynamic class mapping and overhead perspective support
    primary_model = model or get_vehicle_model()
    vehicle_detections, diagnostics = extract_vehicle_detections_from_model(
        model=primary_model,
        image_rgb=image_rgb,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        iou_threshold=iou_threshold
    )

    # Adaptive fallback for overhead/aerial images if primary detector found 0 vehicles
    if len(vehicle_detections) == 0 and allow_overhead_fallback:
        overhead_path = getattr(config, "OVERHEAD_VEHICLE_MODEL_PATH", config.MODELS_DIR / "yolov8n_visdrone.pt")
        if overhead_path.exists():
            curr_path_str = str(getattr(primary_model, "ckpt_path", ""))
            if "visdrone" not in curr_path_str.lower():
                overhead_model = get_overhead_vehicle_model(overhead_path)
                oh_vehicles, oh_diag = extract_vehicle_detections_from_model(
                    model=overhead_model,
                    image_rgb=image_rgb,
                    conf_threshold=conf_threshold,
                    imgsz=imgsz,
                    iou_threshold=iou_threshold
                )
                if len(oh_vehicles) > 0:
                    vehicle_detections = oh_vehicles
                    diagnostics = oh_diag
                    diagnostics["adaptive_fallback_used"] = True

    detected_vehicle_boxes = []
    raw_detections = []

    for idx, v_det in enumerate(vehicle_detections):
        v_det["vehicle_id"] = idx + 1
        detected_vehicle_boxes.append(v_det["bbox"])
        raw_detections.append(v_det)

    yolo_model = primary_model

    # 3. Determine restricted zone configuration or run automatic scene analysis
    sign_boxes: List[Dict[str, Any]] = []
    uncertainty_reason = ""
    evidence_items = []

    if zones is not None:
        # Explicit zones provided (e.g., automated test suites or advanced configuration)
        active_zones = [z for z in zones if not z.get("unconfigured", False)]
        is_zone_configured = len(active_zones) > 0
        zone_status = "Configured" if is_zone_configured else "Restricted zone not configured"
        scene_analysis = {
            "zone_status": zone_status,
            "active_zones": active_zones,
            "evidence_items": [],
            "sign_boxes": [],
            "uncertainty_reason": "",
            "footpaths": [],
            "entrances": [],
            "lawns": [],
            "stall_data": {"divider_x_coords": [], "is_reliable": False},
            "reserved_slots": []
        }
    else:
        # Fully automatic scene analysis
        # Lazily get OCR reader for scene sign detection if not supplied
        if ocr_reader is None:
            try:
                from plate_reader import get_ocr_reader
                ocr_reader = get_ocr_reader()
            except Exception:
                ocr_reader = None

        scene_analysis = detect_scene_visual_evidence(
            image_rgb=image_rgb,
            vehicle_boxes=detected_vehicle_boxes,
            yolo_model=yolo_model,
            ocr_reader=ocr_reader
        )
        active_zones = scene_analysis["active_zones"]
        zone_status = scene_analysis["zone_status"]
        sign_boxes = scene_analysis["sign_boxes"]
        uncertainty_reason = scene_analysis["uncertainty_reason"]
        evidence_items = scene_analysis["evidence_items"]
        is_zone_configured = len(active_zones) > 0

    # 4. Evaluate parking violations across all 6 campus priority categories
    detected_vehicles: List[Dict[str, Any]] = []
    total_violations = 0

    for raw in raw_detections:
        gx, gy = raw["ground_point"]
        v_item = {
            "vehicle_id": raw["vehicle_id"],
            "class_name": raw["class_name"],
            "confidence": raw["confidence"],
            "bbox": raw["bbox"],
            "ground_point": (gx, gy),
            "vehicle_crop": raw["crop"],
        }

        # Optional license-plate detection & OCR
        plate_info = None
        if plate_model is not None or ocr_reader is not None:
            try:
                from plate_reader import read_plate_from_vehicle_crop
                plate_info = read_plate_from_vehicle_crop(
                    vehicle_crop_rgb=raw["crop"],
                    plate_model=plate_model,
                    ocr_reader=ocr_reader,
                    conf_threshold=config.DEFAULT_PLATE_CONF
                )
                v_item.update(plate_info)
            except Exception:
                plate_info = None

        # Master Campus Violation Evaluation across 6 categories
        campus_eval = evaluate_all_campus_violations(
            image_rgb=image_rgb,
            vehicle=v_item,
            scene_evidence=scene_analysis,
            plate_data=plate_info,
            db_path=db_path
        )
        v_item["campus_evaluations"] = campus_eval

        # Determine overall violation status
        is_violation = False
        zone_name = None
        violation_status = "No violation detected"

        # Check explicit zones (backwards-compatibility with manual zone tests)
        if is_zone_configured:
            z_viol, z_name = check_ground_contact_violation(
                (gx, gy), active_zones, width, height
            )
            if z_viol:
                is_violation = True
                zone_name = z_name
                violation_status = "Potential violation — review required"

        # Check automated campus violation categories
        if campus_eval["has_potential_violation"]:
            is_violation = True
            primary = campus_eval["primary_violation"]
            if not zone_name:
                zone_name = primary["category_name"]
            violation_status = "Potential violation — review required"
        elif not is_violation:
            if zone_status == "Restricted zone not configured":
                zone_name = None
                violation_status = "Restricted zone not configured"
            elif len(campus_eval["reviews"]) > 0:
                review = campus_eval["reviews"][0]
                zone_name = review.get("category_name")
                violation_status = "Review required"
                v_item["review_reason"] = review.get("explanation", "Evidence needs human review.")
                v_item["review_evidence"] = review.get("evidence", "")
            elif zone_status == "Restricted zone could not be determined reliably":
                zone_name = None
                violation_status = "Restricted zone could not be determined reliably"
                v_item["review_reason"] = scene_analysis.get("uncertainty_reason", "Restricted zone boundaries need human review.")
            else:
                zone_name = None
                violation_status = "No violation detected"

        v_item["is_violation"] = is_violation
        v_item["zone_name"] = zone_name
        v_item["violation_status"] = violation_status

        if is_violation:
            total_violations += 1

        detected_vehicles.append(v_item)

    # Keep full-resolution probability maps internal to evaluation; return only
    # user-facing evidence and avoid putting large NumPy masks in app state.
    scene_analysis.pop("_surface_prediction", None)
    scene_analysis.pop("_building_mask", None)

    # 5. Generate visual annotations
    annotated_rgb = draw_annotations(
        image_rgb,
        active_zones if is_zone_configured else [],
        detected_vehicles,
        sign_boxes=sign_boxes,
        scene_evidence=scene_analysis
    )

    return {
        "annotated_image": annotated_rgb,
        "total_vehicles": len(detected_vehicles),
        "total_violations": total_violations,
        "total_reviews": sum(
            1 for vehicle in detected_vehicles
            if not vehicle["is_violation"] and vehicle.get("violation_status") == "Review required"
        ),
        "vehicles": detected_vehicles,
        "image_size": (width, height),
        "zone_status": zone_status,
        "zones": active_zones if is_zone_configured else [],
        "evidence_items": evidence_items,
        "sign_boxes": sign_boxes,
        "uncertainty_reason": uncertainty_reason,
        "scene_evidence": scene_analysis,
        "ground_contact_explanation": getattr(config, "GROUND_CONTACT_EXPLANATION", ""),
        "diagnostics": diagnostics
    }

def draw_zone_preview(
    base_image_rgb: np.ndarray,
    zones: Optional[List[Dict[str, Any]]]
) -> np.ndarray:
    """Renders the configured restricted zone overlay on the original image."""
    img = base_image_rgb.copy()
    height, width, _ = img.shape

    if not zones:
        return img

    overlay = img.copy()
    for zone in zones:
        try:
            poly = parse_polygon_coordinates(zone, width, height)
        except Exception:
            continue
        if poly.shape[0] < 3:
            continue
        cv2.fillPoly(overlay, [poly], color=(249, 115, 22))
        cv2.polylines(img, [poly], isClosed=True, color=(194, 65, 12), thickness=2)

        top_pt = poly[np.argmin(poly[:, 1])]
        label_text = f"ZONE: {zone.get('name', 'Restricted')}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thick)

        lx = max(10, min(width - tw - 14, int(top_pt[0]) - 8))
        ly = max(th + 6, int(top_pt[1]) - 6)

        cv2.rectangle(img, (lx - 3, ly - th - 4), (lx + tw + 4, ly + 3), (194, 65, 12), -1)
        cv2.putText(img, label_text, (lx, ly), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    img = cv2.addWeighted(overlay, 0.20, img, 0.80, 0)
    return img

def draw_annotations(
    base_image_rgb: np.ndarray,
    zones: Optional[List[Dict[str, Any]]],
    vehicles: List[Dict[str, Any]],
    sign_boxes: Optional[List[Dict[str, Any]]] = None,
    scene_evidence: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Renders clear, professional annotations:
    - Translucent restricted zone overlay with safety-orange boundaries (if confirmed).
    - Campus scene features (footpaths, entrances, lawns, stall lines, reserved slots).
    - Amber indication for road signs where spatial boundaries are uncertain.
    - Red bounding boxes for potential violations requiring review.
    - Green bounding boxes for vehicles with no violation.
    - Amber bounding boxes for vehicles in undetermined zones or requiring review.
    - Clearly marked ground-contact points with crosshairs.
    - Non-overlapping, restrained label badges displaying violation category.
    """
    img = base_image_rgb.copy()
    height, width, _ = img.shape

    # 1. Render restricted zones (if confirmed)
    if zones:
        overlay = img.copy()
        for zone in zones:
            try:
                poly = parse_polygon_coordinates(zone, width, height)
            except Exception:
                continue
            if poly.shape[0] < 3:
                continue
            cv2.fillPoly(overlay, [poly], color=(249, 115, 22))
            cv2.polylines(img, [poly], isClosed=True, color=(194, 65, 12), thickness=2)

            top_pt = poly[np.argmin(poly[:, 1])]
            label_text = f"RESTRICTED ZONE: {zone.get('name', 'Restricted')}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.44
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thick)

            lx = max(10, min(width - tw - 12, int(top_pt[0]) - 5))
            ly = max(th + 6, int(top_pt[1]) - 6)

            cv2.rectangle(img, (lx - 3, ly - th - 4), (lx + tw + 4, ly + 3), (194, 65, 12), -1)
            cv2.putText(img, label_text, (lx, ly), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

    # 2. Render Campus Visual Evidence (Footpaths, Entrances, Lawns, Stall lines, Reserved slots)
    if scene_evidence:
        overlay_campus = img.copy()
        has_campus_overlay = False

        # Category 2: Footpaths
        for fp in scene_evidence.get("footpaths", []):
            raw_p = fp.get("polygon")
            if not raw_p or len(raw_p) < 3:
                continue
            poly = np.array(raw_p, dtype=np.int32)
            fx, fy, fw, fh = cv2.boundingRect(poly)
            if fw > 0.85 * width and fh > 0.85 * height:
                continue
            cv2.fillPoly(overlay_campus, [poly], color=(56, 189, 248))
            cv2.polylines(img, [poly], isClosed=True, color=(14, 165, 233), thickness=2)
            has_campus_overlay = True
            cv2.putText(img, "FOOTPATH / SIDEWALK", (fx + 5, min(height - 10, fy + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (14, 165, 233), 1, cv2.LINE_AA)

        # Category 3: Entrance Clearance Corridors
        for ent in scene_evidence.get("entrances", []):
            raw_p = ent.get("polygon")
            if not raw_p or len(raw_p) < 3:
                continue
            poly = np.array(raw_p, dtype=np.int32)
            ex, ey, ew, eh = cv2.boundingRect(poly)
            if ew > 0.85 * width and eh > 0.85 * height:
                continue
            cv2.fillPoly(overlay_campus, [poly], color=(234, 179, 8))
            cv2.polylines(img, [poly], isClosed=True, color=(202, 138, 4), thickness=2)
            has_campus_overlay = True
            cv2.putText(img, "CLEARANCE CORRIDOR", (ex + 5, min(height - 10, ey + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (161, 98, 7), 1, cv2.LINE_AA)

        # Category 4: Lawns & Green Spaces
        for ln in scene_evidence.get("lawns", []):
            raw_p = ln.get("polygon")
            if not raw_p or len(raw_p) < 3:
                continue
            poly = np.array(raw_p, dtype=np.int32)
            lx, ly, lw, lh = cv2.boundingRect(poly)
            if lw > 0.85 * width and lh > 0.85 * height:
                continue
            cv2.fillPoly(overlay_campus, [poly], color=(34, 197, 94))
            cv2.polylines(img, [poly], isClosed=True, color=(21, 128, 61), thickness=2)
            has_campus_overlay = True
            cv2.putText(img, "CAMPUS LAWN", (lx + 5, min(height - 10, ly + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (21, 128, 61), 1, cv2.LINE_AA)

        if has_campus_overlay:
            img = cv2.addWeighted(overlay_campus, 0.15, img, 0.85, 0)

        # Category 5: Stall Divider Lines
        stall_data = scene_evidence.get("stall_data", {})
        if stall_data.get("is_reliable"):
            divider_lines = stall_data.get("divider_lines", [])
            if divider_lines:
                for divider in divider_lines:
                    x1, y1, x2, y2 = divider.get("points", (0, 0, 0, 0))
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, "STALL LINE", (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                # Preserve overlays for legacy vertical-divider results.
                for dx in stall_data.get("divider_x_coords", []):
                    cv2.line(img, (dx, int(0.2 * height)), (dx, height - 1), (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, "STALL LINE", (dx - 25, int(0.25 * height)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

        # Category 6: Reserved Slot Markings
        for rslot in scene_evidence.get("reserved_slots", []):
            rx1, ry1, rx2, ry2 = rslot["bbox"]
            slot_lbl = rslot.get("label", "RESERVED").upper()
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (147, 51, 234), 2)
            cv2.rectangle(img, (rx1, max(0, ry1 - 18)), (rx1 + 120, ry1), (147, 51, 234), -1)
            cv2.putText(img, f"SLOT: {slot_lbl}", (rx1 + 3, ry1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Render road sign candidates if detected
    if sign_boxes:
        for s in sign_boxes:
            sx1, sy1, sx2, sy2 = s["bbox"]
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (217, 119, 6), 2)
            s_label = s.get("label", "SIGN (UNCERTAIN BOUNDARY)")
            font = cv2.FONT_HERSHEY_SIMPLEX
            (stw, sth), _ = cv2.getTextSize(s_label, font, 0.40, 1)
            cv2.rectangle(img, (sx1, max(0, sy1 - sth - 6)), (sx1 + stw + 6, sy1), (217, 119, 6), -1)
            cv2.putText(img, s_label, (sx1 + 3, sy1 - 4), font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Render vehicles with category badges
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        gx, gy = v["ground_point"]
        is_violation = v["is_violation"]
        v_status = v.get("violation_status", "")
        class_name = v["class_name"].upper()
        v_id = v["vehicle_id"]

        primary_viol = None
        campus_eval = v.get("campus_evaluations")
        if campus_eval and campus_eval.get("primary_violation"):
            primary_viol = campus_eval["primary_violation"]["category_name"]

        if is_violation:
            # Red for potential violation
            box_color = (220, 38, 38)
            cat_display = primary_viol.upper() if primary_viol else (v.get("zone_name", "VIOLATION")).upper()
            badge_text = f"#{v_id} {class_name} | {cat_display}"
        elif v_status in ["Restricted zone could not be determined reliably", "Review required"] or "could not be determined" in v_status.lower() or "review required" in v_status.lower():
            # Amber for uncertain prediction
            box_color = (217, 119, 6)
            review_category = v.get("zone_name", "")
            if "entrance" in review_category.lower() or "emergency access" in review_category.lower():
                review_label = "ENTRANCE | REVIEW"
            elif "green space" in review_category.lower():
                review_label = "GRASS | REVIEW"
            else:
                review_label = "REVIEW REQUIRED"
            badge_text = f"#{v_id} {class_name} | {review_label}"
        else:
            # Green for compliant / no violation detected
            box_color = (22, 163, 74)
            badge_text = f"#{v_id} {class_name} | NO VIOLATION"

        # Draw vehicle bounding box (clean 2px line)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

        # Draw ground contact point (white outer circle, colored center, crosshair)
        cv2.circle(img, (gx, gy), 5, (255, 255, 255), -1)
        cv2.circle(img, (gx, gy), 3, box_color, -1)
        cv2.line(img, (gx - 6, gy), (gx + 6, gy), (255, 255, 255), 1)
        cv2.line(img, (gx, gy - 6), (gx, gy + 6), (255, 255, 255), 1)

        # Draw compact header badge for vehicle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        font_thick = 1
        (text_w, text_h), _ = cv2.getTextSize(badge_text, font, font_scale, font_thick)

        # Prevent label clipping if vehicle is near top edge
        if y1 - text_h - 8 < 0:
            badge_y1 = y1 + 2
            badge_y2 = y1 + text_h + 8
            text_pos = (x1 + 4, y1 + text_h + 4)
        else:
            badge_y1 = y1 - text_h - 8
            badge_y2 = y1
            text_pos = (x1 + 4, y1 - 4)

        badge_x2 = min(width, x1 + text_w + 8)
        cv2.rectangle(img, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
        cv2.putText(
            img, badge_text, text_pos,
            font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
        )

    return img


