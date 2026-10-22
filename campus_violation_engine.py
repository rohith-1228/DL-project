from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import cv2
import re
import warnings

import config
from database import get_parking_authorization, check_slot_authorization

# =====================================================================
# Computer Vision Detectors for Campus Scene Features
# =====================================================================

# Cached SegFormer semantic segmentation model for road/sidewalk understanding
_SEGFORMER_CACHE = None
_OFFSED_SURFACE_CACHE = None


def get_offsed_surface_model():
    """Load the optional OFFSED-adapted surface model from local files only."""
    global _OFFSED_SURFACE_CACHE
    if _OFFSED_SURFACE_CACHE is None:
        model_dir = getattr(config, "OFFSED_SURFACE_MODEL_PATH", config.MODELS_DIR / "offsed_surface_segformer")
        if not model_dir.is_dir():
            _OFFSED_SURFACE_CACHE = False
        else:
            try:
                import torch
                from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
                processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
                model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir), local_files_only=True)
                model.eval()
                _OFFSED_SURFACE_CACHE = (processor, model, torch)
            except Exception as exc:
                warnings.warn(f"Could not load the OFFSED surface model; using color/texture lawn evidence. {exc}", RuntimeWarning, stacklevel=2)
                _OFFSED_SURFACE_CACHE = False
    return _OFFSED_SURFACE_CACHE if _OFFSED_SURFACE_CACHE is not False else None


def infer_offsed_surface(image_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
    """Return grass and pavement masks with per-pixel class confidence, if trained weights exist."""
    tools = get_offsed_surface_model()
    if tools is None:
        return None
    processor, model, torch = tools
    height, width = image_rgb.shape[:2]
    try:
        from PIL import Image as PILImage, ImageOps
        # Match the OFFSED training pipeline's 512px centered letterbox, then
        # remove padding before returning masks in source-image coordinates.
        prepared = ImageOps.pad(
            PILImage.fromarray(image_rgb), (512, 512), method=PILImage.Resampling.BILINEAR,
            color=(0, 0, 0), centering=(0.5, 0.5),
        )
        scale = min(512 / width, 512 / height)
        content_w, content_h = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (512 - content_w) // 2, (512 - content_h) // 2
        inputs = processor(images=prepared, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
            logits = logits[:, :, top:top + content_h, left:left + content_w]
            logits = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            probabilities = logits.softmax(dim=1)[0]
        prediction = probabilities.argmax(dim=0).cpu().numpy()
        return {
            "class_mask": prediction,
            "grass_mask": prediction == 2,
            "grass_confidence": probabilities[2].cpu().numpy(),
            "confidence": probabilities.max(dim=0).values.cpu().numpy(),
            "road_mask": prediction == 0,
            "road_confidence": probabilities[0].cpu().numpy(),
            "pavement_mask": prediction == 1,
            "pavement_confidence": probabilities[1].cpu().numpy(),
        }
    except Exception as exc:
        warnings.warn(f"OFFSED surface inference failed; using fallback lawn evidence. {exc}", RuntimeWarning, stacklevel=2)
        return None

def get_segformer_model():
    """
    Loads the locally fine-tuned BDD100K checkpoint when available, otherwise
    the Cityscapes SegFormer semantic segmentation model.
    Falls back gracefully if transformers is not present.
    """
    global _SEGFORMER_CACHE
    if _SEGFORMER_CACHE is None:
        try:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
            import torch
            local_model = getattr(config, "BDD100K_SIDEWALK_MODEL_PATH", config.MODELS_DIR / "bdd100k_sidewalk_segformer")
            model_source = str(local_model) if (local_model / "config.json").is_file() else "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
            proc = AutoImageProcessor.from_pretrained(model_source, local_files_only=Path(model_source).is_dir())
            mod = SegformerForSemanticSegmentation.from_pretrained(model_source, local_files_only=Path(model_source).is_dir())
            mod.eval()
            _SEGFORMER_CACHE = (proc, mod, model_source)
        except Exception as exc:
            warnings.warn(
                f"Could not load the semantic SegFormer model; footpath analysis will use visual heuristics. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _SEGFORMER_CACHE = False
    return _SEGFORMER_CACHE if _SEGFORMER_CACHE is not False else None

def detect_footpath_regions(
    image_rgb: np.ndarray,
    vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    scene_masks: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """
    Detects pedestrian sidewalks, walkways, and footpaths using a multi-modal approach:
    1. Pretrained semantic segmentation (Cityscapes SegFormer) for road vs sidewalk regions.
    2. Collinear roadside curb demarcation lines (e.g. broken yellow lines in NZ/UK or curb paint).
    3. Concrete pavement texture and corridor morphology.
    Returns localized footpath contours with curb boundaries and semantic masks.
    """
    height, width, _ = image_rgb.shape
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    footpaths: List[Dict[str, Any]] = []

    # 1. Pretrained SegFormer semantic segmentation (Cityscapes Class 1 = 'sidewalk')
    seg_tools = get_segformer_model()
    if seg_tools:
        proc, mod, model_source = seg_tools
        try:
            import torch
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image_rgb)
            inputs = proc(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                out = mod(**inputs)
            upsampled = torch.nn.functional.interpolate(
                out.logits, size=(height, width), mode="bilinear", align_corners=False
            )[0]
            pred_seg = upsampled.argmax(dim=0).cpu().numpy()
            semantic_confidence = upsampled.softmax(dim=0).amax(dim=0).cpu().numpy()
            if scene_masks is not None:
                # Reuse the same Cityscapes pass for entrance obstruction review.
                scene_masks["building"] = pred_seg == 2
                scene_masks["cityscapes"] = {
                    "class_mask": pred_seg,
                    "confidence": semantic_confidence,
                }
            sidewalk_mask_sem = (pred_seg == 1).astype(np.uint8) * 255

            # Mask out vehicle bounding boxes to keep contour boundaries clean for walkway finding
            clean_mask = sidewalk_mask_sem.copy()
            if vehicle_boxes:
                for vx1, vy1, vx2, vy2 in vehicle_boxes:
                    clean_mask[vy1:vy2, vx1:vx2] = 0

            cnts_sem, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area_sem = int(0.035 * width * height)
            for c in cnts_sem:
                area = cv2.contourArea(c)
                if area > min_area_sem:
                    cx, cy, cw, ch = cv2.boundingRect(c)
                    if (cw >= 0.88 * width and ch >= 0.88 * height) or (area > 0.60 * width * height):
                        continue
                    is_vert = (ch >= int(0.20 * height) and int(0.08 * width) <= cw < int(0.85 * width))
                    is_horiz = (cw >= int(0.20 * width) and int(0.08 * height) <= ch < int(0.85 * height))
                    if is_vert or is_horiz:
                        poly = [[cx, cy], [cx + cw, cy], [cx + cw, cy + ch], [cx, cy + ch]]
                        footpaths.append({
                            "id": "footpath_semantic_segformer",
                            "name": "Pedestrian Footpath / Sidewalk",
                            "polygon": poly,
                            "bbox": (cx, cy, cx + cw, cy + ch),
                            "curb_poly": None,
                            "mask": sidewalk_mask_sem,
                            "evidence_type": f"{Path(model_source).name if Path(model_source).is_dir() else 'Cityscapes'} SegFormer semantic sidewalk segmentation"
                        })
        except Exception:
            pass

    # 2. Roadside curb demarcation line (broken yellow or solid edge lines along road edge)
    yellow_mask = cv2.inRange(hsv, (15, 80, 100), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 40, 30), (85, 255, 255))
    yellow_mask[green_mask > 0] = 0  # Exclude yellowish vegetation/turf
    yellow_mask[:int(0.15 * height), :] = 0
    if vehicle_boxes:
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            yellow_mask[vy1:vy2, vx1:vx2] = 0

    cnts_y, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_pts = []
    for c in cnts_y:
        if cv2.contourArea(c) > 30:
            M = cv2.moments(c)
            if M["m00"] > 0:
                valid_pts.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    if len(valid_pts) >= 4:
        xs = np.array([p[0] for p in valid_pts], dtype=np.float32)
        ys = np.array([p[1] for p in valid_pts], dtype=np.float32)
        corr = np.corrcoef(ys, xs)[0, 1]
        if abs(corr) > 0.95:  # Collinear sequence of roadside curb markings
            poly = np.polyfit(ys, xs, 1)  # x = m*y + c
            y_top = int(0.15 * height)
            y_bot = height
            x_top = int(max(0, min(width, poly[0] * y_top + poly[1])))
            x_bot = int(max(0, min(width, poly[0] * y_bot + poly[1])))
            poly_fp = [[0, y_top], [x_top, y_top], [x_bot, y_bot], [0, y_bot]]
            footpaths.append({
                "id": "footpath_curb_demarcation",
                "name": "Pedestrian Footpath / Sidewalk",
                "polygon": poly_fp,
                "bbox": (0, y_top, max(x_top, x_bot), y_bot),
                "curb_poly": (float(poly[0]), float(poly[1])),
                "mask": None,
                "evidence_type": "Curb and roadside demarcation line"
            })

    # Detect the characteristic narrow, elongated yellow tactile strip used
    # to mark pedestrian paths. It remains useful when surface segmentation
    # mistakes red/gray paving tiles for roadway or when the vehicle occludes
    # the sidewalk boundary.
    for contour in cnts_y:
        area = cv2.contourArea(contour)
        if area < 0.004 * width * height:
            continue
        tx, ty, tw, th = cv2.boundingRect(contour)
        elongated = max(tw, th) / max(1, min(tw, th)) >= 1.35
        ground_level = ty + th >= int(0.55 * height)
        near_walkway_edge = tx <= int(0.12 * width) or tx + tw >= int(0.88 * width)
        if not (elongated and ground_level and near_walkway_edge):
            continue
        expand_x = max(20, min(int(0.10 * width), int(0.75 * max(1, min(tw, th)))))
        expand_y = max(12, int(0.025 * height))
        fx1, fy1 = max(0, tx - expand_x), max(int(0.15 * height), ty - expand_y)
        fx2, fy2 = min(width, tx + tw + expand_x), min(height, ty + th + expand_y)
        footpaths.append({
            "id": "footpath_tactile_surface",
            "name": "Pedestrian Footpath / Sidewalk",
            "polygon": [[fx1, fy1], [fx2, fy1], [fx2, fy2], [fx1, fy2]],
            "bbox": (fx1, fy1, fx2, fy2),
            "curb_poly": None,
            "mask": None,
            "evidence_type": "Elongated yellow tactile pedestrian-way surface",
        })

    # 3. Concrete pedestrian pavement corridor segmentation
    sidewalk_mask = cv2.inRange(hsv, (0, 0, 130), (180, 45, 230))
    sidewalk_mask[:int(0.15 * height), :] = 0
    sidewalk_mask[green_mask > 0] = 0  # Subtract green lawn to prevent grass stripes bridging into sidewalk
    if vehicle_boxes:
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            sidewalk_mask[vy1:vy2, vx1:vx2] = 0

    kw = max(5, int(width * 0.025))
    kh = max(9, int(height * 0.055))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_image_area = width * height
    min_area = int(0.035 * total_image_area)
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            if (w >= 0.88 * width and h >= 0.88 * height) or (area > 0.60 * total_image_area):
                continue
            # Sidewalks are at ground level (reject elevated building facades)
            if y + h < int(0.35 * height):
                continue
            is_vert = (h >= int(0.20 * height) and int(0.08 * width) <= w < int(0.85 * width))
            is_horiz = (w >= int(0.20 * width) and int(0.08 * height) <= h < int(0.85 * height))
            if is_vert or is_horiz:
                approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
                poly = approx.reshape(-1, 2).tolist()
                footpaths.append({
                    "id": "footpath_sidewalk",
                    "name": "Pedestrian Footpath / Sidewalk",
                    "polygon": poly,
                    "bbox": (x, y, x + w, y + h),
                    "curb_poly": None,
                    "mask": None,
                    "evidence_type": "Concrete pedestrian sidewalk corridor"
                })

    return footpaths

def detect_entrance_clearance_corridors(
    image_rgb: np.ndarray,
    vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Detects building gates, main entrances, fire exits, and emergency routes:
    - Yellow hatched clearance boxes ('KEEP CLEAR', 'EMERGENCY EXIT', 'MAIN GATE').
    - Excludes ordinary building walls, windows, lawns, and tree canopies.
    - Returns clearance corridor polygons directly in front of the gate/doorway.
    """
    height, width, _ = image_rgb.shape
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    entrances: List[Dict[str, Any]] = []

    # Yellow hatching / emergency clearance line detection on roadway pavement
    yellow_mask = cv2.inRange(hsv, (18, 70, 90), (38, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 40, 30), (85, 255, 255))
    # Exclude vegetative green turf - clearance boxes are painted on pavement, not lawns
    yellow_mask[green_mask > 0] = 0
    # Clearance corridors are on the roadway/ground, not in the sky or upper facade
    yellow_mask[:int(0.20 * height), :] = 0

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = width * height
    for c in contours:
        area = cv2.contourArea(c)
        if area > 2500:
            x, y, w, h = cv2.boundingRect(c)
            # Must be a realistic clearance box, not an entire field/screen
            if w > 0.85 * width or h > 0.85 * height or area > 0.50 * total_area:
                continue
            if w > 75 and h > 75:
                # Check that the region is predominantly paved ground, not lawn or building facade
                roi_hsv = hsv[y:y + h, x:x + w]
                roi_green = cv2.inRange(roi_hsv, (35, 40, 30), (85, 255, 255))
                if np.mean(roi_green > 0) > 0.12:
                    continue  # Lawn/grass false positive
                poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                entrances.append({
                    "id": "entrance_clearance_box",
                    "name": "College Gate / Emergency Entrance Clearance",
                    "polygon": poly,
                    "bbox": (x, y, x + w, y + h),
                    "evidence_type": "Marked yellow emergency gate clearance corridor"
                })

    # Automatic access gates commonly use a thin red-and-white horizontal arm,
    # without painted KEEP CLEAR markings. Detect that visual pattern as a
    # localized entrance corridor instead of letting broad sidewalk geometry
    # become the vehicle's primary label. Mask vehicle boxes first so taillights
    # and red bodywork cannot masquerade as the arm; visible arm fragments on
    # either side of a vehicle are joined by horizontal morphology.
    barrier_red = cv2.inRange(hsv, (0, 75, 55), (12, 255, 255)) | cv2.inRange(
        hsv, (168, 75, 55), (180, 255, 255)
    )
    if vehicle_boxes:
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            barrier_red[max(0, vy1):min(height, vy2), max(0, vx1):min(width, vx2)] = 0
    barrier_red[:int(0.25 * height), :] = 0
    join_width = max(21, int(0.025 * width))
    joined = cv2.morphologyEx(
        barrier_red, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join_width, 3)),
    )
    barrier_contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barrier_candidates = []
    for contour in barrier_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < max(55, int(0.05 * width)) or h > max(24, int(0.045 * height)):
            continue
        if w / max(1, h) < 5.0 or w > 0.55 * width:
            continue
        # Require a thin white/red arm, not a solid red sign or painted curb.
        roi_hsv = hsv[max(0, y - 3):min(height, y + h + 4), x:min(width, x + w)]
        if roi_hsv.size == 0:
            continue
        roi_red = cv2.inRange(roi_hsv, (0, 75, 55), (12, 255, 255)) | cv2.inRange(
            roi_hsv, (168, 75, 55), (180, 255, 255)
        )
        red_share = float(np.mean(roi_red > 0))
        bright_share = float(np.mean(roi_hsv[:, :, 2] >= 165))
        if not (0.08 <= red_share <= 0.72 and bright_share >= 0.12):
            continue
        barrier_candidates.append((x, y, w, h))

    # A vehicle often hides the middle of the arm. Join left/right fragments
    # only when they are horizontally aligned and a vehicle box explains the
    # gap, avoiding the broad zones that caused nearby walkway cars to inherit
    # an entrance label.
    merged_candidates = list(barrier_candidates)
    for i, first in enumerate(barrier_candidates):
        x1, y1, w1, h1 = first
        for second in barrier_candidates[i + 1:]:
            x2, y2, w2, h2 = second
            left, right = (first, second) if x1 <= x2 else (second, first)
            lx, ly, lw, lh = left
            rx, ry, rw, rh = right
            gap = rx - (lx + lw)
            if gap <= 0 or gap > 0.60 * width:
                continue
            if abs((ly + lh / 2) - (ry + rh / 2)) > max(8, int(0.025 * height)):
                continue
            gap_has_vehicle = any(
                bx1 < rx and bx2 > lx + lw
                and by1 <= (ly + ry + lh) / 2 <= by2
                for bx1, by1, bx2, by2 in (vehicle_boxes or [])
            )
            if gap_has_vehicle:
                merged_candidates.append((lx, min(ly, ry), rx + rw - lx, max(lh, rh)))

    # Deduplicate candidates by geometry before projecting approach corridors.
    seen_barriers = set()
    for x, y, w, h in merged_candidates:
        key = (x // 8, y // 8, w // 8, h // 5)
        if key in seen_barriers:
            continue
        seen_barriers.add(key)
        # Access corridor lies immediately beyond/below the arm in image space.
        # Keep it narrow and local; vehicle contact must fall within this box.
        pad_x = max(12, int(0.025 * width))
        x1, x2 = max(0, x - pad_x), min(width - 1, x + w + pad_x)
        y1 = max(0, y - max(8, int(0.015 * height)))
        y2 = min(height - 1, y + max(45, int(0.42 * height)))
        entrances.append({
            "id": "entrance_barrier_arm_corridor",
            "name": "Automatic Gate / Barrier Access Corridor",
            "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "bbox": (x1, y1, x2, y2),
            "evidence_type": "Red-and-white automatic gate barrier arm and projected access corridor",
        })

    return entrances

def detect_lawn_green_spaces(
    image_rgb: np.ndarray,
    vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    surface_prediction: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Detects campus lawns, gardens, and landscaped green spaces:
    - HSV green segmentation combined with Laplacian texture variance.
    - Excludes non-lawn green objects by checking surface texture and ground placement.
    """
    height, width, _ = image_rgb.shape
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lawns: List[Dict[str, Any]] = []

    # Green turf mask: H in [35, 85], S in [40, 255], V in [30, 255]
    surface_prediction = surface_prediction if surface_prediction is not None else infer_offsed_surface(image_rgb)
    if surface_prediction is not None:
        green_mask = surface_prediction["grass_mask"].astype(np.uint8) * 255
    else:
        green_mask = cv2.inRange(hsv, (35, 40, 30), (85, 255, 255))

    # Mask out vehicle interiors to avoid green paint/tint
    if vehicle_boxes:
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            green_mask[vy1:vy2, vx1:vx2] = 0

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = width * height
    for c in contours:
        area = cv2.contourArea(c)
        if area > 8000:  # Substantial lawn patch (> 3% of image)
            x, y, w, h = cv2.boundingRect(c)
            # Lawns are on the ground (reject tree canopies high in the sky: y + h must reach lower half)
            if y + h < int(0.35 * height):
                continue
            # Reject full-screen bounding box
            if area > 0.65 * total_area and w > 0.90 * width:
                continue

            # Texture check: Natural grass has high micro-texture variance
            roi_gray = cv2.cvtColor(image_rgb[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY)
            lap_var = float(np.var(cv2.Laplacian(roi_gray, cv2.CV_64F)))

            if lap_var > 40.0:  # Genuine textured turf, not smooth painted surface
                approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
                poly = approx.reshape(-1, 2).tolist()
                lawns.append({
                    "id": "campus_lawn",
                    "name": "Campus Lawn & Landscaped Green Space",
                    "polygon": poly,
                    "bbox": (x, y, x + w, y + h),
                    "lap_variance": round(lap_var, 1),
                    "evidence_type": "OFFSED-adapted grass segmentation" if surface_prediction is not None else "Color and texture grass heuristic",
                    "mean_grass_confidence": round(float(np.mean(surface_prediction["grass_confidence"][y:y + h, x:x + w])), 3) if surface_prediction is not None else None,
                })

    return lawns

def detect_parking_stall_lines(
    image_rgb: np.ndarray,
    vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None
) -> Dict[str, Any]:
    """
    Detect genuine painted stall dividers, including perspective-slanted lines.
    Require bright, symmetric pavement markings and a consistent multi-stall
    pattern; otherwise leave the alignment result uncertain/not applicable.
    """
    height, width = image_rgb.shape[:2]
    roi_top = int(0.25 * height)
    roi_gray = cv2.cvtColor(image_rgb[roi_top:], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(roi_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=70, maxLineGap=25)

    candidate_lines: List[Dict[str, Any]] = []
    projected_dividers: List[Dict[str, Any]] = []
    # Parking dividers often disappear behind a parked vehicle. Look just
    # beyond the rear/tire contact edge for a short, high-contrast continuation
    # that aligns with the vehicle's interior; that is useful evidence even
    # when the rest of the stall markings are occluded.
    if vehicle_boxes:
        full_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            y_start = min(height - 1, vy2 + 5)
            y_end = min(height, vy2 + max(35, int(0.12 * height)))
            if y_end - y_start < 8 or vx2 - vx1 < 60:
                continue
            x_start = vx1 + int(0.15 * (vx2 - vx1))
            x_end = vx2 - int(0.15 * (vx2 - vx1))
            best_x, best_score = None, 0
            for x in range(max(2, x_start), min(width - 2, x_end)):
                col = full_gray[y_start:y_end, x].astype(np.int16)
                neighbors = (full_gray[y_start:y_end, x - 2].astype(np.int16) +
                             full_gray[y_start:y_end, x + 2].astype(np.int16)) / 2
                contrast = col - neighbors
                score = int(np.count_nonzero((col >= 195) & (contrast >= 24)))
                if score > best_score:
                    best_x, best_score = x, score
            # Require a strong, near-centered marking. Peripheral bright
            # objects (curbs, grass reflections, bumpers) are not stall lines.
            relative_x = (best_x - vx1) / max(1, vx2 - vx1) if best_x is not None else 0.0
            if best_x is not None and best_score >= 7 and 0.30 <= relative_x <= 0.70:
                projected_dividers.append({"bbox": (vx1, vy1, vx2, vy2), "x": int(best_x), "pixels": best_score})
    full_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    reference_y = int(0.82 * height)
    if lines is not None:
        for raw_line in lines:
            coords = np.asarray(raw_line).ravel()
            if len(coords) < 4:
                continue
            x1, y1 = int(coords[0]), int(coords[1] + roi_top)
            x2, y2 = int(coords[2]), int(coords[3] + roi_top)
            if y2 < y1:
                x1, x2, y1, y2 = x2, x1, y2, y1
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            # Stall dividers converge in perspective; accept slanted lines but
            # reject horizontal crosswalk bars and strongly diagonal edges.
            if length < 45 or dy < 30 or abs(dx) / max(1, dy) > 0.75:
                continue
            mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if vehicle_boxes and any(
                bx1 <= mid_x <= bx2 and by1 - 5 <= mid_y <= by2 + 5
                for bx1, by1, bx2, by2 in vehicle_boxes
            ):
                continue

            # A painted stripe should be bright against pavement on both sides
            # of its normal at multiple points, unlike curbs and wall edges.
            normal_x, normal_y = dy / length, -dx / length
            bright_checks = []
            for fraction in (0.25, 0.5, 0.75):
                px = int(round(x1 + fraction * dx))
                py = int(round(y1 + fraction * dy))
                if not (12 <= px < width - 12 and 12 <= py < height - 12):
                    continue
                center = int(full_gray[py, px])
                side_a = int(full_gray[int(round(py + 7 * normal_y)), int(round(px + 7 * normal_x))])
                side_b = int(full_gray[int(round(py - 7 * normal_y)), int(round(px - 7 * normal_x))])
                bright_checks.append(center >= 165 and center - side_a >= 10 and center - side_b >= 10)
            if sum(bright_checks) < 2:
                continue

            slope = dx / max(1, dy)
            projected_x = x1 + slope * (reference_y - y1)
            projection_distance = max(y1 - reference_y, 0, reference_y - y2)
            # A divider can be visible above a parked car and disappear under
            # it, so allow a moderate projection gap when the multi-stall pattern
            # is independently confirmed below.
            if projection_distance > int(0.50 * height) or not (0 <= projected_x < width):
                continue
            candidate_lines.append({
                "points": (int(x1), int(y1), int(x2), int(y2)),
                "slope_dx_dy": float(slope),
                "x_at_reference_y": float(projected_x),
            })

    # Merge Hough fragments from the same stripe before checking stall spacing.
    grouped_lines: List[Dict[str, Any]] = []
    for candidate in sorted(candidate_lines, key=lambda item: item["x_at_reference_y"]):
        match = next((item for item in grouped_lines
                      if abs(item["x_at_reference_y"] - candidate["x_at_reference_y"]) < 28
                      and abs(item["slope_dx_dy"] - candidate["slope_dx_dy"]) < 0.18), None)
        if match is None:
            grouped_lines.append(candidate)
        else:
            old, new = match["points"], candidate["points"]
            if np.hypot(new[2] - new[0], new[3] - new[1]) > np.hypot(old[2] - old[0], old[3] - old[1]):
                match.update(candidate)

    grouped_lines.sort(key=lambda item: item["x_at_reference_y"])
    unique_x = [int(round(item["x_at_reference_y"])) for item in grouped_lines]

    # Require at least two adjacent, similarly sized stalls, not just arbitrary
    # parallel scene edges. Widths are measured at the same perspective row.
    is_reliable = False
    if len(unique_x) >= 3:
        widths = [right - left for left, right in zip(unique_x, unique_x[1:])]
        for first, second in zip(widths, widths[1:]):
            if 55 <= first <= 360 and 55 <= second <= 360 and abs(first - second) <= max(40, 0.25 * max(first, second)):
                is_reliable = True
                break

    return {
        "divider_x_coords": unique_x if is_reliable else [],
        "divider_lines": grouped_lines if is_reliable else [],
        "observed_divider_lines": grouped_lines,
        "vehicle_boxes": [tuple(map(int, box)) for box in (vehicle_boxes or [])],
        "reference_y": reference_y,
        "image_height": height,
        "is_reliable": is_reliable,
        "vehicle_divider_continuations": projected_dividers,
    }

def detect_reserved_slot_markings(
    image_rgb: np.ndarray,
    ocr_reader: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Scans pavement markings and headboard signs for reserved slot types:
    - 'FACULTY', 'STAFF', 'PRINCIPAL', 'VISITOR', 'RESERVED', 'ACCESSIBLE'.
    - Returns localized slot type and bounding region.
    """
    reserved_slots: List[Dict[str, Any]] = []

    if ocr_reader is not None:
        try:
            h, w = image_rgb.shape[:2]
            scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
            small = cv2.resize(image_rgb, (int(w * scale), int(h * scale))) if scale != 1.0 else image_rgb
            ocr_res = ocr_reader.readtext(small)

            slot_keywords = {
                "FACULTY": "faculty",
                "STAFF": "staff",
                "PRINCIPAL": "principal",
                "VISITOR": "visitor",
                "RESERVED": "reserved",
                "ACCESSIBLE": "accessible",
                "DIRECTOR": "principal"
            }

            for entry in ocr_res:
                txt = entry[1].upper().strip()
                for kw, slot_type in slot_keywords.items():
                    if kw in txt:
                        pts = np.array(entry[0], dtype=np.float32)
                        if scale != 1.0:
                            pts /= scale
                        bx1, by1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                        bx2, by2 = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))
                        reserved_slots.append({
                            "slot_type": slot_type,
                            "label": txt,
                            "bbox": (bx1, by1, bx2, by2)
                        })
        except Exception:
            pass

    return reserved_slots

# =====================================================================
# Individual Category Classifiers
# =====================================================================

def evaluate_unauthorized_no_parking_zone(
    vehicle: Dict[str, Any],
    scene_evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Category 1: Unauthorized Parking / No-Parking Zones
    Three outcomes:
    - 'Potential violation detected'
    - 'Not detected'
    - 'Review required'
    """
    gx, gy = vehicle["ground_point"]
    active_zones = scene_evidence.get("active_zones", [])
    zone_status = scene_evidence.get("zone_status", "")

    # Check confirmed red curbs, fire lanes, fire hydrants
    for z in active_zones:
        poly = np.array(z["polygon"], dtype=np.int32)
        if cv2.pointPolygonTest(poly, (float(gx), float(gy)), False) >= 0:
            return {
                "category_key": "cat1_unauthorized_zone",
                "category_name": "Unauthorized Parking / No-Parking Zone",
                "status": "Potential violation detected",
                "explanation": f"Vehicle ground-contact point occupies {z['name']}.",
                "evidence": z.get("evidence_type", "Painted red curb / emergency zone"),
                "is_violation": True
            }

    # If no-parking sign detected without clear road boundaries -> Review required
    if zone_status == "Restricted zone could not be determined reliably":
        return {
            "category_key": "cat1_unauthorized_zone",
            "category_name": "Unauthorized Parking / No-Parking Zone",
            "status": "Review required",
            "explanation": "No-parking sign identified, but exact restricted road boundaries cannot be determined reliably from this view without surface markings or 3D calibration.",
            "evidence": "Signage detected; spatial boundary review required",
            "is_violation": False
        }

    return {
        "category_key": "cat1_unauthorized_zone",
        "category_name": "Unauthorized Parking / No-Parking Zone",
        "status": "Not detected",
        "explanation": "Vehicle is outside all designated restricted parking zones.",
        "evidence": "Compliant road position",
        "is_violation": False
    }

def _vehicle_wheel_contact_slices(
    image_shape: Tuple[int, ...], bbox: Tuple[int, int, int, int],
    class_name: str = "",
) -> List[Tuple[slice, slice]]:
    """Return contact patches around likely wheel locations at the vehicle base."""
    height, width = image_shape[:2]
    x1, _y1, x2, y2 = [int(value) for value in bbox]
    box_w, box_h = max(1, x2 - x1), max(1, y2 - _y1)
    # Motorcycle detections are loose silhouettes: the box includes handlebars,
    # mirrors, and a narrow pair of tires. Broader patches better cover both tire
    # contacts while retaining two spatially separate observations.
    is_two_wheeler = str(class_name).lower() in {"motorcycle", "motorbike", "bike", "motor"}
    patch_w = max(4, int(box_w * (0.22 if is_two_wheeler else 0.16)))
    patch_h = max(3, int(box_h * (0.16 if is_two_wheeler else 0.10)))
    contact_y2 = min(height, max(0, y2))
    contact_y1 = max(0, contact_y2 - patch_h)
    patches = []
    for fraction in ((0.16, 0.84) if is_two_wheeler else (0.20, 0.80)):
        center_x = int(x1 + fraction * box_w)
        left = max(0, center_x - patch_w // 2)
        right = min(width, center_x + (patch_w + 1) // 2)
        if right > left and contact_y2 > contact_y1:
            patches.append((slice(contact_y1, contact_y2), slice(left, right)))
    return patches


def _contact_class_evidence(
    prediction: Optional[Dict[str, np.ndarray]],
    patches: List[Tuple[slice, slice]],
    class_ids: Tuple[int, ...],
    confidence_threshold: float = 0.60,
) -> Tuple[List[float], List[float]]:
    """Measure class occupancy and confidence only at estimated tire contacts."""
    if prediction is None or prediction.get("class_mask") is None:
        return [], []
    class_mask = prediction["class_mask"]
    confidence = prediction.get("confidence")
    ratios, certainties = [], []
    for ys, xs in patches:
        patch = class_mask[ys, xs]
        if patch.size == 0:
            continue
        confident = np.ones(patch.shape, dtype=bool) if confidence is None else confidence[ys, xs] >= confidence_threshold
        ratios.append(float(np.mean(np.isin(patch, class_ids) & confident)))
        certainties.append(float(np.mean(confidence[ys, xs])) if confidence is not None else 1.0)
    return ratios, certainties


def evaluate_footpath_parking(
    image_rgb: np.ndarray,
    vehicle: Dict[str, Any],
    footpaths: List[Dict[str, Any]],
    cityscapes_prediction: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Category 2: Footpath Parking
    Detects cars and motorcycles occupying pedestrian sidewalks or footpaths:
    - Evaluates vehicle bounding box and tire ground-contact footprint.
    - Evaluates curb demarcation line (straddling curb vs roadway).
    - Evaluates semantic segmentation mask from SegFormer.
    - Evaluates elevated concrete pavement corridor geometry.
    - Returns 'Review required' when sidewalk presence is ambiguous or boundary is uncalibrated.
    Three outcomes:
    - 'Potential violation detected'
    - 'Not detected'
    - 'Review required'
    """
    gx, gy = vehicle["ground_point"]
    vx1, vy1, vx2, vy2 = vehicle["bbox"]
    vw = max(1, vx2 - vx1)
    vh = max(1, vy2 - vy1)
    image_h, image_w = image_rgb.shape[:2]
    edge_truncated = vx1 <= 1 or vy1 <= 1 or vx2 >= image_w - 1 or vy2 >= image_h - 1
    cname = vehicle["class_name"].title()

    # Wheel ground contact estimates
    left_wheel_x = vx1 + int(0.20 * vw)
    right_wheel_x = vx1 + int(0.82 * vw)
    contact_patches = _vehicle_wheel_contact_slices(
        image_rgb.shape, vehicle["bbox"], vehicle.get("class_name", "")
    )
    sidewalk_ratios, sidewalk_confidences = _contact_class_evidence(
        cityscapes_prediction, contact_patches, (1,)
    )
    road_ratios, road_confidences = _contact_class_evidence(
        cityscapes_prediction, contact_patches, (0,)
    )
    sidewalk_contact = max(sidewalk_ratios, default=0.0)
    road_contact = min(road_ratios) if len(road_ratios) == 2 else 0.0
    mean_sidewalk_confidence = float(np.mean(sidewalk_confidences)) if sidewalk_confidences else 0.0
    mean_road_confidence = float(np.mean(road_confidences)) if road_confidences else 0.0

    # Use direct contact-patch evidence, not broad overlap with the car box.
    # A clearly predicted sidewalk under even one wheel/contact region is
    # sufficient evidence. Averaging confidence over both wheels used to hide
    # a real sidewalk contact when only one side was visible or unobstructed.
    strongest_sidewalk_confidence = max(sidewalk_confidences, default=0.0)
    if sidewalk_contact >= 0.20 and strongest_sidewalk_confidence >= 0.50:
        return {
            "category_key": "cat2_footpath_parking",
            "category_name": "Footpath Parking",
            "status": "Review required" if edge_truncated else "Potential violation detected",
            "explanation": (
                f"{cname} may occupy a segmented pedestrian sidewalk, but the vehicle is clipped by the image edge."
                if edge_truncated else f"{cname} has a wheel contact area on the segmented pedestrian sidewalk / footpath."
            ),
            "evidence": f"Cityscapes sidewalk occupies {sidewalk_contact:.0%} of a tire-contact patch; confidence {mean_sidewalk_confidence:.0%}" + ("; vehicle box is image-edge truncated" if edge_truncated else ""),
            "is_violation": not edge_truncated
        }

    if not footpaths:
        return {
            "category_key": "cat2_footpath_parking",
            "category_name": "Footpath Parking",
            "status": "Not detected",
            "explanation": "No pedestrian footpath or sidewalk occupied by the vehicle.",
            "evidence": "Vehicle on roadway or parking bay",
            "is_violation": False
        }

    for fp in footpaths:
        # Check if sidewalk boundary cannot be determined reliably
        if fp.get("is_reliable") is False or fp.get("status") == "Review required" or fp.get("review_required") is True:
            return {
                "category_key": "cat2_footpath_parking",
                "category_name": "Footpath Parking",
                "status": "Review required",
                "explanation": "Pedestrian sidewalk / footpath boundary cannot be identified reliably from this camera perspective; manual inspection required.",
                "evidence": "Ambiguous / uncalibrated sidewalk boundary",
                "is_violation": False
            }

        # 1. Roadside curb demarcation line evaluation
        curb_poly = fp.get("curb_poly")
        if curb_poly is not None:
            m, c = curb_poly
            curb_x_at_base = m * vy2 + c
            # Check wheel positions relative to the curb boundary
            left_on_fp = left_wheel_x < curb_x_at_base
            right_on_fp = right_wheel_x < curb_x_at_base

            if left_on_fp and right_on_fp:
                return {
                    "category_key": "cat2_footpath_parking",
                    "category_name": "Footpath Parking",
                    "status": "Potential violation detected",
                    "explanation": f"{cname} fully parked on pedestrian sidewalk / footpath.",
                    "evidence": f"Both vehicle wheels located on sidewalk across curb (curb_x={int(curb_x_at_base)})",
                    "is_violation": True
                }
            elif left_on_fp or right_on_fp:
                overlap_pct = int(((curb_x_at_base - vx1) / vw if left_on_fp else (vx2 - curb_x_at_base) / vw) * 100)
                return {
                    "category_key": "cat2_footpath_parking",
                    "category_name": "Footpath Parking",
                    "status": "Potential violation detected",
                    "explanation": f"{cname} mounted on pedestrian sidewalk / footpath (straddling roadside curb; {overlap_pct}% on footpath).",
                    "evidence": f"Vehicle straddles curb boundary line (curb_x={int(curb_x_at_base)})",
                    "is_violation": True
                }
            else:
                continue

        # 2. Semantic segmentation mask evaluation (SegFormer)
        mask = fp.get("mask")
        if mask is not None:
            tire_ratios = []
            for ys, xs in contact_patches:
                tire_patch = mask[ys, xs]
                if tire_patch.size > 0:
                    tire_ratios.append(float(np.mean(tire_patch > 0)))
            if tire_ratios:
                occupied_ratio = max(tire_ratios)
                if occupied_ratio >= 0.25:
                    return {
                        "category_key": "cat2_footpath_parking",
                        "category_name": "Footpath Parking",
                        "status": "Review required" if edge_truncated else "Potential violation detected",
                        "explanation": f"{cname} may occupy the pedestrian sidewalk / footpath." + (" The vehicle is clipped by the image edge." if edge_truncated else ""),
                        "evidence": f"Vehicle tire footprint occupies semantic sidewalk mask ({int(occupied_ratio*100)}% contact)" + ("; vehicle box is image-edge truncated" if edge_truncated else ""),
                        "is_violation": not edge_truncated
                    }

        # 3. Geometric corridor polygon ground-contact evaluation
        # Note: Do NOT use 2D vehicle bounding-box overlap alone to establish footpath occupation,
        # as vehicles on adjacent lawns or travel lanes may overlap the sidewalk in perspective projection.
        raw_poly = fp.get("polygon")
        if raw_poly and len(raw_poly) >= 3:
            poly = np.array(raw_poly, dtype=np.int32)
            in_center = cv2.pointPolygonTest(poly, (float(gx), float(gy)), False) >= 0
            in_left = cv2.pointPolygonTest(poly, (float(left_wheel_x), float(vy2)), False) >= 0
            in_right = cv2.pointPolygonTest(poly, (float(right_wheel_x), float(vy2)), False) >= 0

            # For two-wheelers the box bottom often lands between the tires. Use
            # a lower-body footprint as corroboration when a detected walkway
            # corridor touches the bottom of the vehicle box.
            lower_y1 = max(0, vy2 - int(0.22 * vh))
            lower_y2 = min(image_h, vy2 + 1)
            lower_x1 = max(0, vx1 + int(0.08 * vw))
            lower_x2 = min(image_w, vx2 - int(0.08 * vw))
            lower_region = np.zeros((image_h, image_w), dtype=np.uint8)
            cv2.fillPoly(lower_region, [np.array([
                [lower_x1, lower_y1], [lower_x2, lower_y1],
                [lower_x2, lower_y2], [lower_x1, lower_y2],
            ], dtype=np.int32)], 1)
            polygon_region = np.zeros_like(lower_region)
            cv2.fillPoly(polygon_region, [poly], 1)
            lower_area = int(np.count_nonzero(lower_region))
            lower_path_overlap = (
                float(np.count_nonzero((lower_region > 0) & (polygon_region > 0))) / lower_area
                if lower_area else 0.0
            )
            # Contour polygons can be ragged or fragmented by the vehicle
            # occlusion. For motorcycles, corroborate the lower footprint with
            # the contour's ground-level bounding corridor. This catches a
            # two-wheeler whose kickstand/wheels are on a tiled path even when
            # Cityscapes calls those tiles "road".
            lower_box_overlap = 0.0
            fp_bbox = fp.get("bbox")
            if fp_bbox and lower_area:
                fx1, fy1, fx2, fy2 = [int(value) for value in fp_bbox]
                ix1, iy1 = max(lower_x1, fx1), max(lower_y1, fy1)
                ix2, iy2 = min(lower_x2, fx2), min(lower_y2, fy2)
                lower_box_overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1) / lower_area
            is_two_wheeler = str(vehicle.get("class_name", "")).lower() in {
                "motorcycle", "motorbike", "bike", "motor"
            }
            ground_in_path_corridor = bool(
                fp_bbox and fp_bbox[0] <= gx <= fp_bbox[2] and fp_bbox[1] <= gy <= fp_bbox[3]
            )
            if fp.get("id") == "footpath_tactile_surface" and ground_in_path_corridor:
                return {
                    "category_key": "cat2_footpath_parking",
                    "category_name": "Footpath Parking",
                    "status": "Potential violation detected",
                    "explanation": f"{cname} is parked on a pedestrian walkway marked by tactile paving.",
                    "evidence": "Vehicle ground-contact point lies within the adjacent tactile pedestrian path corridor.",
                    "is_violation": True
                }
            if (
                is_two_wheeler
                and fp.get("id") == "footpath_sidewalk"
                and lower_box_overlap >= 0.40
                and ground_in_path_corridor
            ):
                return {
                    "category_key": "cat2_footpath_parking",
                    "category_name": "Footpath Parking",
                    "status": "Potential violation detected",
                    "explanation": f"{cname} is parked across a detected pedestrian pavement corridor.",
                    "evidence": f"Ground contact lies inside the footpath corridor; {lower_box_overlap:.0%} of the lower vehicle footprint overlaps it. Tiled footpaths may be classified as road by the surface model.",
                    "is_violation": True
                }

            if in_center or in_left or in_right or lower_path_overlap >= 0.20:
                if edge_truncated:
                    return {
                        "category_key": "cat2_footpath_parking",
                        "category_name": "Footpath Parking",
                        "status": "Review required",
                        "explanation": f"{cname} overlaps a possible footpath region, but its image-edge-truncated box makes wheel placement uncertain.",
                        "evidence": "Coarse footpath corridor overlap; vehicle is clipped by the image edge",
                        "is_violation": False
                    }
                # A confidently road-classified contact on both sides is strong
                # evidence against a coarse geometric sidewalk polygon.
                if road_contact >= 0.50 and mean_road_confidence >= 0.65:
                    # The geometric footpath and semantic road labels disagree.
                    # Surface class models often confuse tiled sidewalks with
                    # roads; preserve the conflict for human review instead of
                    # silently declaring the vehicle compliant.
                    return {
                        "category_key": "cat2_footpath_parking",
                        "category_name": "Footpath Parking",
                        "status": "Review required",
                        "explanation": f"{cname} overlaps a detected footpath, but the surface classifier disagrees.",
                        "evidence": f"Footpath geometry overlaps {lower_path_overlap:.0%} of the vehicle's lower footprint; semantic road confidence {mean_road_confidence:.0%}",
                        "is_violation": False
                    }
                # Check whether ground surface is vegetative lawn rather than pedestrian pavement
                h_img, w_img = image_rgb.shape[:2]
                patch = image_rgb[max(0, gy - 20):min(h_img, gy + 15), max(0, gx - 40):min(w_img, gx + 40)]
                if patch.size > 0:
                    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
                    patch_green = cv2.inRange(patch_hsv, (35, 40, 30), (85, 255, 255))
                    if np.mean(patch_green > 0) > 0.35:
                        continue  # Ground contact is grass/turf, not concrete sidewalk

                return {
                    "category_key": "cat2_footpath_parking",
                    "category_name": "Footpath Parking",
                    "status": "Potential violation detected",
                    "explanation": f"{cname} mounted on pedestrian sidewalk / footpath.",
                    "evidence": "Vehicle wheel/footprint ground contact occupies pedestrian sidewalk corridor",
                    "is_violation": True
                }

    return {
        "category_key": "cat2_footpath_parking",
        "category_name": "Footpath Parking",
        "status": "Not detected",
        "explanation": "Vehicle is on the roadway and not occupying the pedestrian footpath.",
        "evidence": "Footpath remains clear of vehicles",
        "is_violation": False
    }

def evaluate_blocking_entrance(
    vehicle: Dict[str, Any],
    entrances: List[Dict[str, Any]],
    building_mask: Optional[np.ndarray] = None,
    image_rgb: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Category 3: Blocking Entrances and Emergency Access
    Detects vehicles obstructing college gates, building entrances, fire exits, and emergency routes.
    Three outcomes:
    - 'Potential violation detected'
    - 'Not detected'
    - 'Review required'
    """
    vx1, vy1, vx2, vy2 = vehicle["bbox"]
    gx, gy = vehicle["ground_point"]
    vw = max(1, vx2 - vx1)
    left_wheel_x = vx1 + int(0.20 * vw)
    right_wheel_x = vx1 + int(0.80 * vw)

    # When no painted clearance zone was found, use façade overlap as a review
    # cue and a clear, framed doorway behind the vehicle as stronger evidence.
    building_overlap = 0.0
    if building_mask is not None:
        mask_h, mask_w = building_mask.shape[:2]
        bx1, bx2 = max(0, vx1), min(mask_w, vx2)
        by1, by2 = max(0, vy1), min(mask_h, vy2)
        if bx2 > bx1 and by2 > by1:
            building_patch = building_mask[by1:by2, bx1:bx2]
            if building_patch.size:
                building_overlap = float(np.mean(building_patch))

    def visible_doorway_behind_vehicle() -> bool:
        """Detect a large, dark entrance opening immediately behind the vehicle."""
        if image_rgb is None:
            return False
        image_h, image_w = image_rgb.shape[:2]
        # The visible strip above the vehicle roof often contains the entrance
        # interior even when the vehicle hides the doorway threshold itself.
        roi_x1 = max(0, vx1 + int(0.10 * vw))
        roi_x2 = min(image_w, vx2 - int(0.10 * vw))
        roi_y1 = max(0, vy1 - int(0.12 * image_h))
        roi_y2 = min(image_h, vy1)
        if roi_x2 - roi_x1 < max(60, int(0.55 * vw)) or roi_y2 - roi_y1 < 30:
            return False
        gray = cv2.cvtColor(image_rgb[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_RGB2GRAY)
        dark_ratio = float(np.mean(gray < 85))
        mean_opening = float(np.mean(gray))

        # Require a large, very dark, structured opening directly above the
        # vehicle. Façade-side contrast alone misses glass doors because the
        # pillars and shaded entry often share the same exposure.
        edges = cv2.Canny(gray, 40, 100)
        segments = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=18,
            minLineLength=max(18, int(0.12 * (roi_y2 - roi_y1))), maxLineGap=6,
        )
        vertical_edges = []
        if segments is not None:
            for line in segments.reshape(-1, 4):
                lx1, ly1, lx2, ly2 = map(int, line)
                if abs(lx2 - lx1) <= 3 and abs(ly2 - ly1) >= max(18, int(0.12 * gray.shape[0])):
                    vertical_edges.append((lx1 + lx2) // 2)
        structured_opening = len(vertical_edges) >= 6
        return dark_ratio >= 0.70 and mean_opening <= 70 and structured_opening

    dark_doorway = visible_doorway_behind_vehicle()

    def probable_unmarked_entrance_result() -> Optional[Dict[str, Any]]:
        if building_overlap < 0.10 and not dark_doorway:
            return None
        if dark_doorway:
            return {
                "category_key": "cat3_blocking_entrance",
                "category_name": "Blocking Entrances and Emergency Access",
                "status": "Potential violation detected",
                "explanation": "The vehicle is positioned directly in front of a visible building entrance and obstructs its access path.",
                "evidence": "A broad dark doorway is visible immediately above the vehicle, framed by a contrasting building façade; the vehicle occupies the entrance approach.",
                "is_violation": True
            }
        return {
            "category_key": "cat3_blocking_entrance",
            "category_name": "Blocking Entrances and Emergency Access",
            "status": "Review required",
            "explanation": "The vehicle overlaps a building façade where an unmarked garage or entrance may be obstructed; verify the doorway and access path from the scene.",
            "evidence": f"Building-segmentation overlap within vehicle box ({building_overlap:.0%}); entrance boundary is unmarked",
            "is_violation": False
        }

    if not entrances:
        review = probable_unmarked_entrance_result()
        if review is not None:
            return review
        return {
            "category_key": "cat3_blocking_entrance",
            "category_name": "Blocking Entrances and Emergency Access",
            "status": "Not detected",
            "explanation": "No building entrance, gate, or fire exit clearance corridor obstructed.",
            "evidence": "Access corridors clear",
            "is_violation": False
        }

    for ent in entrances:
        poly_pts = ent.get("polygon")
        if poly_pts and len(poly_pts) >= 3:
            poly = np.array(poly_pts, dtype=np.int32)
            in_center = cv2.pointPolygonTest(poly, (float(gx), float(gy)), False) >= 0
            in_left = cv2.pointPolygonTest(poly, (float(left_wheel_x), float(vy2)), False) >= 0
            in_right = cv2.pointPolygonTest(poly, (float(right_wheel_x), float(vy2)), False) >= 0

            # Compute bounding box intersection area with clearance corridor
            ex1, ey1, ex2, ey2 = ent["bbox"]
            inter_x1 = max(vx1, ex1)
            inter_y1 = max(vy1, ey1)
            inter_x2 = min(vx2, ex2)
            inter_y2 = min(vy2, ey2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            vehicle_area = max(1, (vx2 - vx1) * (vy2 - vy1))

            if in_center or in_left or in_right or (inter_area / vehicle_area > 0.20 and (ex1 <= gx <= ex2 and ey1 <= gy <= ey2)):
                return {
                    "category_key": "cat3_blocking_entrance",
                    "category_name": "Blocking Entrances and Emergency Access",
                    "status": "Potential violation detected",
                    "explanation": "Vehicle directly obstructs designated college gate / building entrance clearance corridor.",
                    "evidence": "Direct spatial obstruction of entrance clearance box",
                    "is_violation": True
                }

    review = probable_unmarked_entrance_result()
    if review is not None:
        return review

    return {
        "category_key": "cat3_blocking_entrance",
        "category_name": "Blocking Entrances and Emergency Access",
        "status": "Not detected",
        "explanation": "Entrance and emergency access routes are clear of obstruction.",
        "evidence": "Vehicle clear of entrance corridor",
        "is_violation": False
    }

def evaluate_lawn_parking(
    image_rgb: np.ndarray,
    vehicle: Dict[str, Any],
    lawns: List[Dict[str, Any]],
    surface_prediction: Optional[Dict[str, Any]] = None,
    cityscapes_prediction: Optional[Dict[str, Any]] = None,
    entrance_obstruction: bool = False,
) -> Dict[str, Any]:
    """
    Category 4: Parking on Green Spaces
    Detects vehicles occupying lawns, gardens, and landscaped turf.
    Three outcomes:
    - 'Potential violation detected'
    - 'Not detected'
    - 'Review required'
    """
    contact_patches = _vehicle_wheel_contact_slices(image_rgb.shape, vehicle["bbox"])
    offsed_grass_ratios, offsed_confidences = _contact_class_evidence(
        surface_prediction, contact_patches, (2,)
    )
    city_paved_ratios, city_paved_confidences = _contact_class_evidence(
        cityscapes_prediction, contact_patches, (0, 1), confidence_threshold=0.65
    )
    city_green_ratios, city_green_confidences = _contact_class_evidence(
        cityscapes_prediction, contact_patches, (8, 9), confidence_threshold=0.60
    )
    grass_ratio = max(offsed_grass_ratios, default=0.0)
    grass_confidence = float(np.mean(offsed_confidences)) if offsed_confidences else 0.0
    paved_contact = min(city_paved_ratios) if len(city_paved_ratios) == 2 else 0.0
    paved_confidence = float(np.mean(city_paved_confidences)) if city_paved_confidences else 0.0
    green_contact = max(city_green_ratios, default=0.0)

    # Strong road/sidewalk evidence at both tire contacts should override nearby
    # green pixels, tree foliage, or low-confidence OFFSED predictions.
    if (
        paved_contact >= 0.50
        and paved_confidence >= 0.65
        and green_contact < 0.20
        and grass_ratio < 0.70
    ):
        return {
            "category_key": "cat4_lawn_parking",
            "category_name": "Parking on Green Spaces",
            "status": "Not detected",
            "explanation": "Both estimated tire-contact areas are classified as road or sidewalk, with no meaningful grass overlap.",
            "evidence": f"Cityscapes paved-surface contact {paved_contact:.0%}; OFFSED grass contact {grass_ratio:.0%}",
            "is_violation": False
        }

    # Strong, agreeing ground-contact evidence should be confirmed instead of
    # left in the generic review state. In particular, full OFFSED grass
    # coverage at high confidence plus Cityscapes vegetation/terrain support
    # identifies a vehicle parked on lawn rather than merely beside a lawn.
    if (
        len(offsed_grass_ratios) == 2
        and min(offsed_grass_ratios) >= 0.70
        and grass_confidence >= 0.65
    ):
        return {
            "category_key": "cat4_lawn_parking",
            "category_name": "Parking on Green Spaces",
            "status": "Potential violation detected",
            "explanation": "Both estimated tire-contact areas are confidently classified as lawn grass / landscaped ground.",
            "evidence": f"OFFSED grass occupies {offsed_grass_ratios[0]:.0%} and {offsed_grass_ratios[1]:.0%} of the tire-contact patches at {grass_confidence:.0%} mean confidence",
            "is_violation": True
        }

    # A low-confidence grass mask often bleeds onto the paved apron in front
    # of a doorway. Strong two-contact grass evidence above wins; suppress this
    # weak signal only when a separate entrance obstruction is already found.
    if entrance_obstruction and grass_confidence < 0.75 and green_contact < 0.20:
        return {
            "category_key": "cat4_lawn_parking",
            "category_name": "Parking on Green Spaces",
            "status": "Not detected",
            "explanation": "The vehicle obstructs an entrance, but the weak grass signal is unsupported by vegetation at its tire contacts.",
            "evidence": f"OFFSED grass confidence {grass_confidence:.0%}; Cityscapes vegetation/terrain contact {green_contact:.0%}",
            "is_violation": False
        }

    # A low-confidence grass mask on only one side of a vehicle commonly
    # captures the lawn beside a paved stall. Without vegetation support or
    # agreement from the other tire patch, treat that as pavement-adjacent
    # greenery rather than a grass-parking violation.
    if (
        len(offsed_grass_ratios) == 2
        and max(offsed_grass_ratios) >= 0.50
        and min(offsed_grass_ratios) < 0.15
        and grass_confidence < 0.65
        and green_contact < 0.20
    ):
        return {
            "category_key": "cat4_lawn_parking",
            "category_name": "Parking on Green Spaces",
            "status": "Not detected",
            "explanation": "Grass-like pixels occur beside only one estimated tire contact and are low confidence; vegetation is not confirmed under the vehicle.",
            "evidence": f"OFFSED contact ratios {offsed_grass_ratios[0]:.0%}/{offsed_grass_ratios[1]:.0%} at {grass_confidence:.0%} mean confidence; no semantic vegetation support",
            "is_violation": False
        }

    if grass_ratio >= 0.75 and grass_confidence >= 0.80 and green_contact >= 0.30:
        return {
            "category_key": "cat4_lawn_parking",
            "category_name": "Parking on Green Spaces",
            "status": "Potential violation detected",
            "explanation": "The vehicle's tire-contact area is confidently classified as grass / landscaped ground.",
            "evidence": f"OFFSED grass contact {grass_ratio:.0%} at {grass_confidence:.0%} confidence, corroborated by Cityscapes vegetation/terrain contact {green_contact:.0%}",
            "is_violation": True
        }

    if grass_ratio >= 0.20 or green_contact >= 0.30:
        strength = "strong" if grass_ratio >= 0.45 and grass_confidence >= 0.65 else "ambiguous"
        return {
            "category_key": "cat4_lawn_parking",
            "category_name": "Parking on Green Spaces",
            "status": "Review required",
            "explanation": "One or both tire-contact areas may overlap grass or unpaved ground; verify the surface before deciding.",
            "evidence": f"{strength.title()} contact evidence: OFFSED grass {grass_ratio:.0%} (confidence {grass_confidence:.0%}); Cityscapes vegetation/terrain {green_contact:.0%}; paved confidence {paved_confidence:.0%}",
            "is_violation": False
        }

    # A local color fallback is used only when semantic evidence is unavailable.
    if surface_prediction is None and cityscapes_prediction is None and contact_patches:
        patch_pixels = [image_rgb[ys, xs] for ys, xs in contact_patches if image_rgb[ys, xs].size]
        if patch_pixels:
            tire_patch = np.concatenate([patch.reshape(-1, 1, 3) for patch in patch_pixels], axis=0)
            patch_hsv = cv2.cvtColor(tire_patch, cv2.COLOR_RGB2HSV)
            green_mask = cv2.inRange(patch_hsv, (35, 40, 30), (85, 255, 255))
            green_ratio = float(np.mean(green_mask > 0))
            if green_ratio > 0.15:
                return {
                    "category_key": "cat4_lawn_parking",
                    "category_name": "Parking on Green Spaces",
                    "status": "Review required",
                    "explanation": "Color near a tire contact suggests possible grass; verify the surface before deciding.",
                    "evidence": f"Tire-contact patches contain {green_ratio:.0%} green pixels (heuristic only)",
                    "is_violation": False
                }

    return {
        "category_key": "cat4_lawn_parking",
        "category_name": "Parking on Green Spaces",
        "status": "Not detected",
        "explanation": "Vehicle is on paved roadway or parking stall and not occupying green spaces.",
        "evidence": "Paved surface contact verified",
        "is_violation": False
    }

def evaluate_parking_alignment(
    vehicle: Dict[str, Any],
    stall_data: Dict[str, Any],
    competing_surface_or_access_violation: bool = False,
) -> Dict[str, Any]:
    """
    Category 5: Improper Parking Alignment
    Identifies vehicles extending outside marked spaces or occupying multiple spaces.
    Three outcomes:
    - 'Potential violation detected'
    - 'Not detected'
    - 'Review required'
    """
    vx1, vy1, vx2, vy2 = vehicle["bbox"]
    divider_x = stall_data.get("divider_x_coords", [])
    is_reliable = stall_data.get("is_reliable", False)

    # A strong stripe continuing under a car is evidence of possible line
    # straddling, but can also be a lane marking. Confirm it only when the scene
    # establishes a stall grid; otherwise show it for human review instead of
    # silently reporting no alignment concern.
    continuation = next((item for item in stall_data.get("vehicle_divider_continuations", [])
                         if tuple(item.get("bbox", ())) == tuple(vehicle["bbox"])
                         and int(item.get("pixels", 0)) >= 6), None)
    if continuation is not None and not competing_surface_or_access_violation:
        x = int(continuation["x"])
        pixels = int(continuation.get("pixels", 0))
        image_height = max(1, int(stall_data.get("image_height", max(vy2, 1))))
        row_neighbors = sum(
            1 for other in stall_data.get("vehicle_boxes", [])
            if tuple(other) != tuple(vehicle["bbox"])
            and abs(int(other[3]) - vy2) <= max(25, int(0.07 * image_height))
            and abs((int(other[0]) + int(other[2])) // 2 - (vx1 + vx2) // 2)
                >= max(80, int(0.45 * max(vx2 - vx1, int(other[2]) - int(other[0]))))
        )
        # A strong stripe hidden beneath the car, together with neighboring
        # vehicles at the same depth, is useful marked-row context even when
        # Hough cannot reconstruct three full dividers through occlusion.
        # Keep this as a potential finding, never a confirmed enforcement action.
        if pixels >= 24 and row_neighbors >= 1 and len(stall_data.get("vehicle_boxes", [])) >= 3:
            return {
                "category_key": "cat5_improper_alignment",
                "category_name": "Improper Parking Alignment",
                "status": "Potential violation detected",
                "explanation": "A strong painted-line continuation crosses the vehicle base in a row of adjacent parked vehicles, suggesting it may straddle a stall boundary.",
                "evidence": f"Bright line continues beneath vehicle at X={x} ({pixels} pixels); {row_neighbors} neighboring vehicle(s) share the parking row",
                "is_violation": True
            }
        if is_reliable and pixels >= 12:
            return {
                "category_key": "cat5_improper_alignment",
                "category_name": "Improper Parking Alignment",
                "status": "Potential violation detected",
                "explanation": "A painted parking divider continues from beneath the vehicle, indicating it may be straddling a stall boundary.",
                "evidence": f"High-contrast divider continuation at X={x} aligns with the vehicle interior and a confirmed multi-stall pattern",
                "is_violation": True
            }
        return {
            "category_key": "cat5_improper_alignment",
            "category_name": "Improper Parking Alignment",
            "status": "Review required",
            "explanation": "A painted-line segment continues beneath the vehicle, but the image does not provide enough clear stall context to confirm an alignment violation.",
            "evidence": f"Possible line continuation at X={x} beneath the vehicle ({pixels} pixels); confirm stall context manually",
            "is_violation": False
        }

    if not is_reliable and len(divider_x) > 0:
        return {
            "category_key": "cat5_improper_alignment",
            "category_name": "Improper Parking Alignment",
            "status": "Review required",
            "explanation": "Stall divider markings cannot be identified reliably or are partially degraded; manual inspection required.",
            "evidence": "Ambiguous / unverified painted lines",
            "is_violation": False
        }

    if not is_reliable or len(divider_x) == 0:
        return {
            "category_key": "cat5_improper_alignment",
            "category_name": "Improper Parking Alignment",
            "status": "Not detected",
            "explanation": "No marked parking stall boundaries identified in the scene; vehicle on general roadway (Alignment check not applicable / unable to determine).",
            "evidence": "General roadway / unmarked parking surface",
            "is_violation": False
        }

    # Project perspective-slanted dividers to the vehicle's ground-contact row.
    # Comparing their X positions at one fixed image row causes missed/incorrect
    # results in angled camera views where parking lines converge.
    vw = max(1, vx2 - vx1)
    margin = max(25, int(0.15 * vw))
    divider_lines = stall_data.get("divider_lines", [])
    if divider_lines:
        ground_y = int(vehicle.get("ground_point", ((vx1 + vx2) // 2, vy2))[1])
        for divider in divider_lines:
            x1, y1, x2, y2 = divider.get("points", (0, 0, 0, 0))
            dy = y2 - y1
            if dy == 0:
                continue
            slope = (x2 - x1) / dy
            distance = max(y1 - ground_y, 0, ground_y - y2)
            # Avoid a long, unsupported extrapolation beyond the visible stripe.
            if distance > max(45, int(0.45 * stall_data.get("image_height", max(vy2, y2, 1)))):
                continue
            projected_x = int(round(x1 + slope * (ground_y - y1)))
            if vx1 + margin < projected_x < vx2 - margin:
                return {
                    "category_key": "cat5_improper_alignment",
                    "category_name": "Improper Parking Alignment",
                    "status": "Potential violation detected",
                    "explanation": "A confirmed painted stall divider projects through the vehicle's ground-contact area, indicating it may straddle a marked space.",
                    "evidence": f"Perspective-adjusted stall divider at X={projected_x}, vehicle ground row Y={ground_y}",
                    "is_violation": True
                }

    # Backward-compatible path for callers supplying the older vertical-divider schema.
    for x in divider_x:
        if vx1 + margin < x < vx2 - margin:
            return {
                "category_key": "cat5_improper_alignment",
                "category_name": "Improper Parking Alignment",
                "status": "Potential violation detected",
                "explanation": "Vehicle straddles marked parking stall dividing lines, occupying multiple spaces.",
                "evidence": f"Stall dividing line at X={x} passes through vehicle body",
                "is_violation": True
            }

    return {
        "category_key": "cat5_improper_alignment",
        "category_name": "Improper Parking Alignment",
        "status": "Not detected",
        "explanation": "Vehicle is properly aligned within marked parking stall divider lines.",
        "evidence": "Compliant single-space stall alignment",
        "is_violation": False
    }

def evaluate_wrong_parking_slot(
    vehicle: Dict[str, Any],
    reserved_slots: List[Dict[str, Any]],
    plate_data: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Category 6: Wrong Parking Slot
    Identifies vehicles occupying designated staff, faculty, visitor, or principal spaces without authorization.
    Three outcomes:
    - 'Potential violation detected'
    - 'Not detected'
    - 'Review required'
    """
    vx1, vy1, vx2, vy2 = vehicle["bbox"]
    gx, gy = vehicle["ground_point"]

    if not reserved_slots:
        return {
            "category_key": "cat6_wrong_parking_slot",
            "category_name": "Wrong Parking Slot",
            "status": "Not detected",
            "explanation": "Vehicle in standard / general open parking space.",
            "evidence": "General parking bay",
            "is_violation": False
        }

    # Find which reserved slot the vehicle occupies
    matched_slot = None
    for slot in reserved_slots:
        sx1, sy1, sx2, sy2 = slot["bbox"]
        # Check if vehicle is in or adjacent to the slot marking
        if (sx1 - 120 <= gx <= sx2 + 120 and (sy1 - 80 <= gy <= sy2 + 380 or abs(vy1 - sy2) < 220)):
            matched_slot = slot
            break

    if not matched_slot:
        return {
            "category_key": "cat6_wrong_parking_slot",
            "category_name": "Wrong Parking Slot",
            "status": "Not detected",
            "explanation": "Vehicle is not occupying a designated reserved parking slot.",
            "evidence": "General parking space",
            "is_violation": False
        }

    slot_type = matched_slot["slot_type"]
    slot_label = matched_slot["label"]

    norm_plate = (plate_data or {}).get("normalized_plate", "").strip()
    ocr_status = (plate_data or {}).get("reading_status", "")

    if not norm_plate or ocr_status not in ["Recognized"]:
        return {
            "category_key": "cat6_wrong_parking_slot",
            "category_name": "Wrong Parking Slot",
            "status": "Review required",
            "explanation": f"Vehicle is parked in a '{slot_label}' reserved slot, but number plate could not be verified to check authorization.",
            "evidence": "Unverified plate in reserved slot",
            "is_violation": False
        }

    # Query authorization database
    status, explanation, record = check_slot_authorization(norm_plate, slot_type, db_path=db_path)
    is_viol = (status == "Potential violation detected")

    return {
        "category_key": "cat6_wrong_parking_slot",
        "category_name": "Wrong Parking Slot",
        "status": status,
        "explanation": explanation,
        "evidence": f"Slot: {slot_label} • Permit Status: {record['role'].title() if record else 'Unregistered'}",
        "is_violation": is_viol
    }

# =====================================================================
# Master Campus Violation Evaluation Pipeline
# =====================================================================

def evaluate_all_campus_violations(
    image_rgb: np.ndarray,
    vehicle: Dict[str, Any],
    scene_evidence: Dict[str, Any],
    plate_data: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates all six priority college campus violation categories for a single vehicle.
    Returns:
    - has_potential_violation: bool
    - primary_violation: Dict[str, Any] or None
    - categories: Dict[str, Dict[str, Any]] (all 6 categories evaluated)
    """
    footpaths = scene_evidence.get("footpaths", [])
    entrances = scene_evidence.get("entrances", [])
    lawns = scene_evidence.get("lawns", [])
    stall_data = scene_evidence.get("stall_data", {"divider_x_coords": [], "is_reliable": False})
    reserved_slots = scene_evidence.get("reserved_slots", [])

    cat1 = evaluate_unauthorized_no_parking_zone(vehicle, scene_evidence)
    cat2 = evaluate_footpath_parking(
        image_rgb, vehicle, footpaths,
        cityscapes_prediction=scene_evidence.get("_cityscapes_prediction"),
    )
    cat3 = evaluate_blocking_entrance(
        vehicle, entrances, scene_evidence.get("_building_mask"), image_rgb=image_rgb
    )
    cat4 = evaluate_lawn_parking(
        image_rgb, vehicle, lawns,
        surface_prediction=scene_evidence.get("_surface_prediction"),
        cityscapes_prediction=scene_evidence.get("_cityscapes_prediction"),
        entrance_obstruction=cat3.get("status") == "Potential violation detected",
    )
    surface_or_access_conflict = any(
        category.get("status") == "Potential violation detected"
        for category in (cat2, cat3, cat4)
    )
    cat5 = evaluate_parking_alignment(
        vehicle, stall_data,
        competing_surface_or_access_violation=surface_or_access_conflict,
    )
    cat6 = evaluate_wrong_parking_slot(vehicle, reserved_slots, plate_data=plate_data, db_path=db_path)

    all_cats = {
        "cat1_unauthorized_zone": cat1,
        "cat2_footpath_parking": cat2,
        "cat3_blocking_entrance": cat3,
        "cat4_lawn_parking": cat4,
        "cat5_improper_alignment": cat5,
        "cat6_wrong_parking_slot": cat6,
    }

    # Priority ordering when determining primary violation:
    # 1. Blocking a marked entrance or gate (direct access obstruction)
    # 2. Footpath parking (pedestrian safety & accessibility)
    # 3. Unauthorized parking / no-parking zone (red curbs, fire lanes)
    # 4. Green spaces
    # 5. Wrong parking slot
    # 6. Improper parking alignment
    priority_order = [
        "cat3_blocking_entrance",
        "cat2_footpath_parking",
        "cat1_unauthorized_zone",
        "cat4_lawn_parking",
        "cat6_wrong_parking_slot",
        "cat5_improper_alignment",
    ]

    violations = [c for c in all_cats.values() if c["status"] == "Potential violation detected"]
    reviews = [c for c in all_cats.values() if c["status"] == "Review required"]

    violations_sorted = sorted(
        violations,
        key=lambda c: priority_order.index(c["category_key"]) if c["category_key"] in priority_order else 99
    )
    reviews_sorted = sorted(
        reviews,
        key=lambda c: priority_order.index(c["category_key"]) if c["category_key"] in priority_order else 99
    )

    has_violation = len(violations_sorted) > 0
    primary = violations_sorted[0] if violations_sorted else (reviews_sorted[0] if reviews_sorted else None)

    return {
        "has_potential_violation": has_violation,
        "primary_violation": primary,
        "violations": violations_sorted,
        "reviews": reviews_sorted,
        "categories": all_cats
    }
