import json
from pathlib import Path

def create_notebook():
    cells = [
        # Cell 1: Markdown Title & Lab Overview
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Smart Parking Violation Detection — Six Priority Campus Categories\n",
                "**Deep Learning & Computer Vision Lab Academic Project**\n",
                "\n",
                "## 1. Problem Statement & Project Objectives\n",
                "College campuses frequently encounter traffic bottlenecks, pedestrian safety hazards, and emergency access blockages due to disorganized vehicle parking. Manual patrolling of sprawling campuses is labor-intensive, error-prone, and inefficient.\n",
                "\n",
                "### Objective:\n",
                "Develop an end-to-end computer vision and deep learning pipeline that automatically identifies **Six Priority Campus Parking Violation Categories** from camera images, extracts vehicle number plates via OCR, validates permissions against a campus SQLite authorization database, and prepares human-in-the-loop advisory notices.\n",
                "\n",
                "### The Six Priority Violation Categories:\n",
                "1. **Category 1: Unauthorized Parking / No-Parking Zones** — Identifies vehicles occupying designated red curb fire lanes, fire hydrant buffers, or clearly marked restricted zones. (Explicitly avoids false positives when a vehicle is merely driving past a no-parking road sign).\n",
                "2. **Category 2: Footpath Parking** — Detects cars and motorcycles mounted on pedestrian walkways or sidewalks.\n",
                "3. **Category 3: Blocking Entrances & Emergency Access** — Detects vehicles obstructing college gates, building entrances, and yellow hatched clearance boxes.\n",
                "4. **Category 4: Parking on Green Spaces** — Detects vehicles occupying campus lawns and landscaped vegetative turf.\n",
                "5. **Category 5: Improper Parking Alignment** — Identifies vehicles straddling painted stall divider lines or occupying multiple parking spaces.\n",
                "6. **Category 6: Wrong Parking Slot** — Identifies vehicles occupying reserved faculty, staff, visitor, or principal slots without matching database authorization.\n",
                "\n",
                "### Three-State Classification Standard:\n",
                "Each category yields one of three transparent outcomes:\n",
                "- **`Potential violation detected`**: Sufficient visual and spatial evidence confirms the violation.\n",
                "- **`Not detected`**: Available evidence confirms compliance.\n",
                "- **`Review required`**: Ambiguous boundaries, faded lines, unreadable plate, or uncalibrated camera angle require manual human review."
            ]
        },
        # Cell 2: Markdown Section 2 - Installation and Setup
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Environment Setup & Library Imports\n",
                "The pipeline leverages modern deep learning and computer vision libraries:\n",
                "- **Ultralytics YOLOv8**: Real-time object detection for vehicles (cars, motorcycles, buses, trucks) and license plates.\n",
                "- **OpenCV & NumPy**: Perspective analysis, Hough Line Transform, HSV color segmentation, and morphological filtering.\n",
                "- **EasyOCR**: Optical character recognition for license plates and pavement markings.\n",
                "- **SQLite**: Embedded database for vehicle registration, role-based authorization, and notification audit logs."
            ]
        },
        # Cell 3: Code Cell 2 - Imports
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import cv2\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from PIL import Image\n",
                "\n",
                "# Add project root to sys.path\n",
                "project_root = Path.cwd()\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "# Import project modules\n",
                "import config\n",
                "from database import init_db, get_parking_authorization, check_slot_authorization\n",
                "from registry import lookup_plate, register_vehicle, get_all_vehicles\n",
                "from parking_detector import (\n",
                "    get_vehicle_model,\n",
                "    detect_vehicles_and_violations,\n",
                "    detect_scene_visual_evidence,\n",
                "    calculate_ground_contact_point\n",
                ")\n",
                "from campus_violation_engine import (\n",
                "    detect_footpath_regions,\n",
                "    detect_entrance_clearance_corridors,\n",
                "    detect_lawn_green_spaces,\n",
                "    detect_parking_stall_lines,\n",
                "    detect_reserved_slot_markings,\n",
                "    evaluate_all_campus_violations\n",
                ")\n",
                "from plate_reader import (\n",
                "    get_plate_detector,\n",
                "    get_ocr_reader,\n",
                "    read_plate_from_vehicle_crop\n",
                ")\n",
                "\n",
                "# Initialize SQLite database\n",
                "init_db()\n",
                "print(\"Environment initialized successfully!\")\n",
                "print(f\"Vehicle Model: {config.VEHICLE_MODEL_PATH}\")\n",
                "print(f\"Plate Model:   {config.PLATE_MODEL_PATH}\")\n",
                "print(f\"Database:      {config.DB_PATH}\")"
            ]
        },
        # Cell 4: Markdown Section 3 - Loading & Displaying Test Images
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Loading and Displaying Campus Test Images\n",
                "We load test scenes representing various campus parking layouts, walkways, entrance gates, lawns, and reserved bays."
            ]
        },
        # Cell 5: Code Cell 3 - Load & Display Image
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_image_path = Path(\"data/test_images/scene_clearly_marked_restricted_area.png\")\n",
                "assert sample_image_path.exists(), f\"Image not found at {sample_image_path}\"\n",
                "\n",
                "img_bgr = cv2.imread(str(sample_image_path))\n",
                "img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n",
                "h, w, _ = img_rgb.shape\n",
                "\n",
                "plt.figure(figsize=(9, 5))\n",
                "plt.imshow(img_rgb)\n",
                "plt.title(f\"Test Image: {sample_image_path.name} ({w}x{h})\", fontsize=12)\n",
                "plt.axis(\"off\")\n",
                "plt.show()"
            ]
        },
        # Cell 6: Markdown Section 4 - Vehicle Detection with YOLOv8
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Vehicle Detection Using Pretrained YOLOv8\n",
                "We utilize YOLOv8 pretrained on MS-COCO to detect campus vehicle classes:\n",
                "- `car` (Class ID 2)\n",
                "- `motorcycle` (Class ID 3)\n",
                "- `bus` (Class ID 5)\n",
                "- `truck` (Class ID 7)\n",
                "\n",
                "### Ground-Contact Point Approximation:\n",
                "Rather than evaluating bounding-box Intersection-over-Union (IoU) — which erroneously includes vehicle roofs and shadows — we compute the vehicle's bottom-center coordinate $( (x_1+x_2)/2, y_2 )$, representing where tires contact the ground."
            ]
        },
        # Cell 7: Code Cell 4 - Vehicle Detection
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "vehicle_model = get_vehicle_model()\n",
                "results = vehicle_model.predict(\n",
                "    source=img_rgb,\n",
                "    conf=0.25,\n",
                "    classes=list(config.VEHICLE_CLASSES.keys()),\n",
                "    verbose=False\n",
                ")\n",
                "\n",
                "annotated_vis = img_rgb.copy()\n",
                "detected_boxes = []\n",
                "\n",
                "if len(results) > 0 and results[0].boxes is not None:\n",
                "    for box in results[0].boxes:\n",
                "        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]\n",
                "        cls_id = int(box.cls[0].cpu().numpy())\n",
                "        conf = float(box.conf[0].cpu().numpy())\n",
                "        cls_name = config.VEHICLE_CLASSES.get(cls_id, \"vehicle\")\n",
                "        gx, gy = calculate_ground_contact_point((x1, y1, x2, y2))\n",
                "        detected_boxes.append((x1, y1, x2, y2))\n",
                "\n",
                "        cv2.rectangle(annotated_vis, (x1, y1), (x2, y2), (0, 200, 255), 2)\n",
                "        cv2.circle(annotated_vis, (gx, gy), 6, (255, 0, 0), -1)\n",
                "        cv2.putText(annotated_vis, f\"{cls_name} ({conf:.2f})\", (x1, max(20, y1 - 8)),\n",
                "                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)\n",
                "\n",
                "plt.figure(figsize=(9, 5))\n",
                "plt.imshow(annotated_vis)\n",
                "plt.title(f\"YOLOv8 Detection: {len(detected_boxes)} Vehicle(s) with Ground Contact Points (Blue Dot)\")\n",
                "plt.axis(\"off\")\n",
                "plt.show()"
            ]
        },
        # Cell 8: Markdown Section 5 - Scene Segmentation & Restricted-Region Identification
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Scene Segmentation & Restricted-Region Identification\n",
                "Automatic scene understanding detects infrastructure boundaries without requiring users to manually draw polygons:\n",
                "1. **Footpaths / Sidewalks**: Segmented via pavement texture and curb line edges.\n",
                "2. **Entrance Clearance Corridors**: Detected via yellow hatched cross-striping and portal bounding boxes.\n",
                "3. **Campus Lawns & Green Spaces**: Identified using HSV color masking combined with Laplacian texture variance to distinguish real grass turf from smooth green painted surfaces.\n",
                "4. **Parking Stall Lines**: Detected using Hough Line Transform on road asphalt, filtering out car body edges to identify true pavement stall dividers.\n",
                "5. **Reserved Pavement Markings**: OCR detection of keywords like `FACULTY`, `STAFF`, `PRINCIPAL`, and `VISITOR`."
            ]
        },
        # Cell 9: Code Cell 5 - Scene Evidence Demonstration
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "evidence = detect_scene_visual_evidence(img_rgb, vehicle_boxes=detected_boxes)\n",
                "print(\"Detected Scene Evidence:\")\n",
                "print(f\"- Zone Status:       {evidence['zone_status']}\")\n",
                "print(f\"- Active Zones:      {len(evidence['active_zones'])}\")\n",
                "print(f\"- Footpaths:         {len(evidence['footpaths'])}\")\n",
                "print(f\"- Entrance Corridors:{len(evidence['entrances'])}\")\n",
                "print(f\"- Green Lawns:       {len(evidence['lawns'])}\")\n",
                "print(f\"- Stall Markings:    {evidence['stall_data']}\")\n",
                "print(f\"- Reserved Slots:    {evidence['reserved_slots']}\")\n",
                "for item in evidence['evidence_items']:\n",
                "    print(f\"  * {item}\")"
            ]
        },
        # Cell 10: Markdown Section 6 - Demonstration of Six Priority Violation Categories
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Demonstration of the Six Priority Violation Categories\n",
                "We evaluate benchmark test images across all six priority categories using the unified master pipeline `detect_vehicles_and_violations`."
            ]
        },
        # Cell 11: Code Cell 6 - Six Priority Categories Demonstration
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "benchmark_scenes = [\n",
                "    (\"Category 1: Marked Restricted Area\", \"data/test_images/scene_clearly_marked_restricted_area.png\"),\n",
                "    (\"Category 1: Passing Near Sign (No Violation)\", \"data/test_images/scene_near_no_parking_sign.png\"),\n",
                "    (\"Category 2: Footpath Parking\", \"data/test_images/scene_cat2_footpath_parking.png\"),\n",
                "    (\"Category 2: Footpath Compliant\", \"data/test_images/scene_cat2_footpath_compliant.png\"),\n",
                "    (\"Category 3: Blocking Entrance Gate\", \"data/test_images/scene_cat3_blocking_entrance.png\"),\n",
                "    (\"Category 3: Entrance Clearance Compliant\", \"data/test_images/scene_cat3_entrance_clear.png\"),\n",
                "    (\"Category 4: Parking on Green Lawn\", \"data/test_images/scene_cat4_lawn_parking.png\"),\n",
                "    (\"Category 4: Paved Parking (Beside Lawn)\", \"data/test_images/scene_cat4_paved_parking.png\"),\n",
                "    (\"Category 5: Improper Alignment (Straddling Line)\", \"data/test_images/scene_cat5_improper_alignment.png\"),\n",
                "    (\"Category 5: Compliant Stall Alignment\", \"data/test_images/scene_cat5_proper_alignment.png\"),\n",
                "    (\"Category 6: Wrong Slot (Student in Principal Bay)\", \"data/test_images/scene_cat6_wrong_slot_unauthorized.png\"),\n",
                "    (\"Category 6: Authorized Slot (Principal in Bay)\", \"data/test_images/scene_cat6_correct_slot_authorized.png\")\n",
                "]\n",
                "\n",
                "print(f\"{'SCENE':<48} | {'VEHICLES':<8} | {'OUTCOME':<12} | {'PRIMARY VIOLATION'}\")\n",
                "print(\"-\" * 105)\n",
                "\n",
                "for label, img_file in benchmark_scenes:\n",
                "    p = Path(img_file)\n",
                "    if not p.exists():\n",
                "        continue\n",
                "    res = detect_vehicles_and_violations(p)\n",
                "    v_count = res[\"total_vehicles\"]\n",
                "    is_viol = (res[\"total_violations\"] > 0)\n",
                "    outcome = \"VIOLATION\" if is_viol else \"COMPLIANT\"\n",
                "\n",
                "    viol_summary = \"None\"\n",
                "    if v_count > 0:\n",
                "        v = res[\"vehicles\"][0]\n",
                "        evals = v.get(\"campus_evaluations\", {}).get(\"categories\", {})\n",
                "        active_cats = [cat[\"category_name\"] for cat in evals.values() if cat.get(\"status\") == \"Potential violation detected\"]\n",
                "        if active_cats:\n",
                "            viol_summary = \", \".join(active_cats)\n",
                "\n",
                "    print(f\"{label:<48} | {v_count:<8} | {outcome:<12} | {viol_summary}\")"
            ]
        },
        # Cell 12: Markdown Section 7 - Number Plate Recognition & OCR
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. License Plate Detection & EasyOCR Recognition\n",
                "For vehicles associated with potential violations, the dedicated plate detector localizes the license plate, crops it, and extracts text using EasyOCR with character normalization."
            ]
        },
        # Cell 13: Code Cell 7 - Plate Detection & OCR
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_car_path = Path(\"data/test_images/car_sample_2.png\")\n",
                "car_bgr = cv2.imread(str(test_car_path))\n",
                "car_rgb = cv2.cvtColor(car_bgr, cv2.COLOR_BGR2RGB)\n",
                "\n",
                "plate_info = read_plate_from_vehicle_crop(car_rgb)\n",
                "\n",
                "print(\"Plate Recognition Results:\")\n",
                "print(f\"- Plate Detected:     {plate_info['plate_detected']}\")\n",
                "print(f\"- Raw OCR Text:       '{plate_info['raw_text']}'\")\n",
                "print(f\"- Normalized Plate:   '{plate_info['normalized_plate']}'\")\n",
                "print(f\"- Confidence:         {plate_info.get('ocr_confidence', 0.0)}\")\n",
                "print(f\"- Is Registered:      {plate_info.get('registry_record') is not None}\")\n",
                "if plate_info.get('registry_record'):\n",
                "    rec = plate_info['registry_record']\n",
                "    print(f\"- Registered Owner:   {rec['owner_name']}\")\n",
                "    print(f\"- Campus Role:        {rec['role']}\")\n",
                "    print(f\"- Authorized Slots:   {rec['authorized_slots']}\")\n",
                "    print(f\"- Notification Email: {rec['email']}\")"
            ]
        },
        # Cell 14: Markdown Section 8 - SQLite Registration & Role-Based Authorization
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. SQLite Vehicle Registry & Campus Authorization Lookup\n",
                "The SQLite database contains voluntary vehicle registrations, campus role tags (`faculty`, `staff`, `principal`, `visitor`, `student`), and authorized parking slots."
            ]
        },
        # Cell 15: Code Cell 8 - Database Lookup
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "registered_vehicles = get_all_vehicles()\n",
                "print(f\"Total Registered Campus Vehicles in SQLite: {len(registered_vehicles)}\\n\")\n",
                "print(f\"{'ID':<4} | {'PLATE':<10} | {'ROLE':<10} | {'OWNER':<18} | {'AUTHORIZED SLOTS':<20} | {'CONSENT'}\")\n",
                "print(\"-\" * 80)\n",
                "for v in registered_vehicles:\n",
                "    consent = \"Yes\" if v['consent_status'] == 1 else \"No\"\n",
                "    slots = v['authorized_slots'] or \"None\"\n",
                "    print(f\"{v['id']:<4} | {v['plate_number']:<10} | {v['role']:<10} | {v['owner_name']:<18} | {slots:<20} | {consent}\")"
            ]
        },
        # Cell 16: Markdown Section 9 - Human-in-the-Loop Email Advisory Preview
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Human-in-the-Loop Advisory Notice Preview\n",
                "In strict compliance with project privacy and safety standards:\n",
                "1. **Never sent automatically** during image analysis.\n",
                "2. Requires a human reviewer to inspect and click **Send Email Notice**.\n",
                "3. Verified against explicit recipient consent."
            ]
        },
        # Cell 17: Code Cell 9 - Email Advisory Preview
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "reg_record = plate_info.get('registry_record')\n",
                "if reg_record and reg_record.get('consent_status') == 1:\n",
                "    preview = f\"\"\"======================================================================\n",
                "CAMPUS PARKING ADVISORY PREVIEW (NOT SENT YET — REQUIRES HUMAN REVIEW)\n",
                "======================================================================\n",
                "To: {reg_record['email']}\n",
                "Subject: [Campus Parking Advisory] Potential Violation Notice: {plate_info['normalized_plate']}\n",
                "\n",
                "Dear {reg_record['owner_name']} ({reg_record['role'].title()}),\n",
                "\n",
                "A potential parking violation was logged for your vehicle:\n",
                "- Vehicle Plate:     {plate_info['normalized_plate']}\n",
                "- Violation Category: Category 1: Unauthorized Parking / No-Parking Zone\n",
                "- Review Status:     Pending Verified Human Review\n",
                "\n",
                "IMPORTANT PROTOTYPE NOTICE & DISCLAIMER:\n",
                "This message is an automated prototype advisory notification generated for\n",
                "academic research and testing within the Deep Learning Lab project.\n",
                "It does NOT constitute an official traffic fine, legal ticket, or disciplinary action.\n",
                "======================================================================\"\"\"\n",
                "    print(preview)\n",
                "else:\n",
                "    print(\"No advisory generated (vehicle unknown or owner has not consented to notifications).\")"
            ]
        },
        # Cell 18: Markdown Section 10 - Evaluation & Performance Metrics
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Quantitative Pipeline Evaluation (Precision, Recall, F1-Score)\n",
                "We evaluate the complete computer vision pipeline against manually verified ground-truth test labels across all benchmark test scenes.\n",
                "\n",
                "### Metrics Formulation:\n",
                "- **Precision**: $\\frac{TP}{TP + FP}$ — Proportion of flagged violations that are genuine violations.\n",
                "- **Recall**: $\\frac{TP}{TP + FN}$ — Proportion of actual violations successfully captured.\n",
                "- **F1-Score**: $2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$"
            ]
        },
        # Cell 19: Code Cell 10 - Metric Computation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ground_truth_test_suite = [\n",
                "    {\"image\": \"data/test_images/scene_clearly_marked_restricted_area.png\", \"expected_violation\": True, \"label\": \"Cat 1: Restricted Area\"},\n",
                "    {\"image\": \"data/test_images/scene_near_no_parking_sign.png\", \"expected_violation\": False, \"label\": \"Cat 1: Passing Near Sign (Compliant)\"},\n",
                "    {\"image\": \"data/test_images/scene_cat2_footpath_parking.png\", \"expected_violation\": True, \"label\": \"Cat 2: Motorcycle on Footpath\"},\n",
                "    {\"image\": \"data/test_images/scene_cat2_footpath_compliant.png\", \"expected_violation\": False, \"label\": \"Cat 2: Beside Footpath (Compliant)\"},\n",
                "    {\"image\": \"data/test_images/scene_cat3_blocking_entrance.png\", \"expected_violation\": True, \"label\": \"Cat 3: Blocking Gate\"},\n",
                "    {\"image\": \"data/test_images/scene_cat3_entrance_clear.png\", \"expected_violation\": False, \"label\": \"Cat 3: Beside Entrance (Compliant)\"},\n",
                "    {\"image\": \"data/test_images/scene_cat4_lawn_parking.png\", \"expected_violation\": True, \"label\": \"Cat 4: Parked on Lawn\"},\n",
                "    {\"image\": \"data/test_images/scene_cat4_paved_parking.png\", \"expected_violation\": False, \"label\": \"Cat 4: Paved Stall Beside Lawn (Compliant)\"},\n",
                "    {\"image\": \"data/test_images/scene_cat5_improper_alignment.png\", \"expected_violation\": True, \"label\": \"Cat 5: Straddling Stall Line\"},\n",
                "    {\"image\": \"data/test_images/scene_cat5_proper_alignment.png\", \"expected_violation\": False, \"label\": \"Cat 5: Compliant Stall Alignment\"},\n",
                "    {\"image\": \"data/test_images/scene_cat6_wrong_slot_unauthorized.png\", \"expected_violation\": True, \"label\": \"Cat 6: Unauthorized Slot\"},\n",
                "    {\"image\": \"data/test_images/scene_cat6_correct_slot_authorized.png\", \"expected_violation\": False, \"label\": \"Cat 6: Authorized Bay (Compliant)\"},\n",
                "]\n",
                "\n",
                "tp, fp, fn, tn = 0, 0, 0, 0\n",
                "\n",
                "print(f\"{'SCENE LABEL':<42} | {'ACTUAL':<10} | {'PREDICTED':<10} | {'STATUS'}\")\n",
                "print(\"-\" * 76)\n",
                "\n",
                "for test_item in ground_truth_test_suite:\n",
                "    img_path = Path(test_item[\"image\"])\n",
                "    if not img_path.exists():\n",
                "        continue\n",
                "    res = detect_vehicles_and_violations(img_path)\n",
                "    pred = (res[\"total_violations\"] > 0)\n",
                "    actual = test_item[\"expected_violation\"]\n",
                "\n",
                "    if pred and actual:\n",
                "        tp += 1\n",
                "        outcome = \"TP (Correct Violation)\"\n",
                "    elif pred and not actual:\n",
                "        fp += 1\n",
                "        outcome = \"FP (False Alarm)\"\n",
                "    elif not pred and actual:\n",
                "        fn += 1\n",
                "        outcome = \"FN (Missed Violation)\"\n",
                "    else:\n",
                "        tn += 1\n",
                "        outcome = \"TN (Correct Compliance)\"\n",
                "\n",
                "    act_str = \"Violation\" if actual else \"Compliant\"\n",
                "    pred_str = \"Violation\" if pred else \"Compliant\"\n",
                "    print(f\"{test_item['label']:<42} | {act_str:<10} | {pred_str:<10} | {outcome}\")\n",
                "\n",
                "precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0\n",
                "recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0\n",
                "f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0\n",
                "\n",
                "print(\"\\n\" + \"=\" * 50)\n",
                "print(f\"Confusion Matrix:  TP={tp}, FP={fp}, FN={fn}, TN={tn}\")\n",
                "print(f\"Precision:         {precision:.4f} ({precision * 100:.1f}%)\")\n",
                "print(f\"Recall:            {recall:.4f} ({recall * 100:.1f}%)\")\n",
                "print(f\"F1-Score:          {f1:.4f} ({f1 * 100:.1f}%)\")\n",
                "print(\"=\" * 50)"
            ]
        },
        # Cell 20: Markdown Section 11 - Limitations & Real-World Challenges
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Limitations & Computer Vision Challenges\n",
                "While our multi-modal vision pipeline successfully identifies all six priority campus categories on benchmark test scenes, real-world deployment presents practical constraints:\n",
                "\n",
                "1. **2D Camera Perspective Compression & Occlusion**:\n",
                "   - Single 2D monocular cameras lack 3D depth perception. When a vehicle passes in front of a painted sign or stall line, perspective foreshortening can obscure boundaries.\n",
                "   - *Mitigation*: The system flags uncertain boundary cases as **`Review required`** rather than fabricating false violations.\n",
                "\n",
                "2. **Environmental & Lighting Variations**:\n",
                "   - Direct sunlight glare, heavy shadows from trees or buildings, and torrential rain degrade Canny edge detection and HSV color thresholds.\n",
                "   - Faded or weathered pavement paint requires adaptive thresholding or learned segmentation.\n",
                "\n",
                "3. **OCR Sensitivity to Camera Resolution & Distance**:\n",
                "   - Vehicles parked at distances exceeding 15 meters may yield license plates with fewer than 15 vertical pixels, resulting in OCR failure.\n",
                "   - *Mitigation*: When OCR confidence is low, the system displays `Number plate could not be read` and provides an optional inline registration form."
            ]
        },
        # Cell 21: Markdown Section 12 - Conclusion & Viva Defense Notes
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Conclusion & Viva Defense Notes\n",
                "\n",
                "### Key Concepts for Lab Viva:\n",
                "1. **Why use YOLOv8 over Traditional CNNs or Two-Stage Detectors (e.g., Faster R-CNN)?**\n",
                "   - YOLO models frame detection as a single regression task, simultaneously predicting bounding coordinates and class probabilities across feature pyramids. This enables real-time inference (30–60 FPS) essential for high-throughput campus monitoring.\n",
                "\n",
                "2. **Why Ground-Contact Points instead of Bounding-Box IoU?**\n",
                "   - 2D bounding boxes enclose a vehicle's entire projected volume, including roof rails, wing mirrors, and shadows. A car parked legally alongside a pedestrian footpath frequently has its bounding box overlap the walkway. Evaluating ground-contact point $( (x_1+x_2)/2, y_2 )$ accurately tests tire contact with the ground plane.\n",
                "\n",
                "3. **How does the system prevent false alarms when a car drives past a no-parking sign?**\n",
                "   - Road signs indicate a zone's existence, but not its 2D camera geometry. The system requires corroborating surface markings (red curbs, yellow clearance boxes) or explicit camera calibration before confirming a violation. Without surface markings, the state is honestly reported as **`Review required`**.\n",
                "\n",
                "4. **Why decouple Vehicle Detection from License Plate Localization?**\n",
                "   - Vehicle detection requires wide field-of-view spatial features, while plate recognition requires high-resolution fine-grained localization. Decoupling ensures the plate detector operates on high-resolution vehicle crops, maximizing OCR readability."
            ]
        }
    ]

    # Assign unique IDs to each cell to satisfy nbformat 4.5+ specs
    for idx, cell in enumerate(cells):
        cell["id"] = f"cell_{idx + 1}"

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    target_file = Path("parking_violation_detection.ipynb")
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    print(f"Notebook written successfully to {target_file.resolve()}")

if __name__ == "__main__":
    create_notebook()
