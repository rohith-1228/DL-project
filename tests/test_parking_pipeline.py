import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import smtplib

from config import DEFAULT_RESTRICTED_ZONES, DEFAULT_VEHICLE_CONF
from database import init_db, get_connection
from registry import normalize_plate, lookup_plate, add_vehicle
from parking_detector import (
    calculate_ground_contact_point,
    check_point_in_zone,
    check_ground_contact_violation,
    detect_vehicles_and_violations
)
from plate_reader import read_plate_from_vehicle_crop
from notifications import send_violation_notice, is_duplicate_event

@pytest.fixture
def temp_db():
    """Provides a fresh isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = Path(tf.name)
    init_db(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()

# Scenario 1 & 2: Image handling and Vehicle Detection
def test_vehicle_detection_on_clear_image():
    sample_path = Path("data/test_images/car_sample_2.png")
    assert sample_path.exists(), "Sample test image must exist for testing."
    
    result = detect_vehicles_and_violations(sample_path)
    assert "annotated_image" in result
    assert "vehicles" in result
    assert result["total_vehicles"] > 0
    
    v = result["vehicles"][0]
    assert v["class_name"] in ["car", "motorcycle", "bus", "truck"]
    assert len(v["bbox"]) == 4

# Scenario 3: Vehicle detected inside restricted polygon
def test_vehicle_inside_restricted_polygon():
    # Polygon covering center of a 400x250 image
    test_zone = {
        "id": "test_zone",
        "name": "Test Walkway",
        "polygon": [[100, 100], [300, 100], [300, 250], [100, 250]]
    }
    # Point at (200, 200) is strictly inside
    ground_point = (200, 200)
    is_viol, zone_name = check_ground_contact_violation(ground_point, [test_zone], 400, 250)
    assert is_viol is True
    assert zone_name == "Test Walkway"

# Scenario 4: Vehicle outside restricted polygon
def test_vehicle_outside_restricted_polygon():
    test_zone = {
        "id": "test_zone",
        "name": "Test Walkway",
        "polygon": [[100, 100], [300, 100], [300, 250], [100, 250]]
    }
    # Point at (50, 50) is outside
    ground_point = (50, 50)
    is_viol, zone_name = check_ground_contact_violation(ground_point, [test_zone], 400, 250)
    assert is_viol is False
    assert zone_name is None

# Scenario 4b: Unconfigured zone does not flag violations
def test_unconfigured_zone_does_not_flag_violation():
    ground_point = (200, 200)
    # When zones list is empty (unconfigured)
    is_viol, zone_name = check_ground_contact_violation(ground_point, [], 400, 250)
    assert is_viol is False
    assert zone_name is None

    # When tested with detect_vehicles_and_violations on image
    sample_path = Path("data/test_images/car_sample_2.png")
    det_res = detect_vehicles_and_violations(sample_path, zones=[])
    assert det_res["total_violations"] == 0
    assert det_res["zone_status"] == "Restricted zone not configured"
    for v in det_res["vehicles"]:
        assert v["is_violation"] is False
        assert v["violation_status"] == "Restricted zone not configured"

# Scenario 4c: Inaccurate polygon causes false positives; calibrated polygon resolves them
def test_inaccurate_polygon_vs_calibrated_polygon():
    # Vehicle is on the road at (120, 190)
    vehicle_ground_point = (120, 190)

    # Inaccurate polygon: overly broad, covering the entire road and sidewalk
    inaccurate_polygon = {
        "id": "inaccurate_blanket_zone",
        "name": "Inaccurate Blanket Zone",
        "polygon": [[50, 50], [350, 50], [350, 240], [50, 240]]
    }
    is_viol_bad, _ = check_ground_contact_violation(vehicle_ground_point, [inaccurate_polygon], 400, 250)
    # The inaccurate polygon falsely flags the vehicle on the road
    assert is_viol_bad is True

    # Calibrated, scene-specific polygon: precisely restricted to the red curb on the right
    calibrated_polygon = {
        "id": "calibrated_curb_zone",
        "name": "Calibrated Red Curb Zone",
        "polygon": [[240, 150], [360, 150], [360, 240], [240, 240]]
    }
    is_viol_good, zone_name_good = check_ground_contact_violation(vehicle_ground_point, [calibrated_polygon], 400, 250)
    # The calibrated polygon correctly evaluates the vehicle as outside the restricted zone
    assert is_viol_good is False
    assert zone_name_good is None

# Scenario 4d: Vehicle near a no-parking sign (Must NOT flag every vehicle on the road)
def test_vehicle_near_no_parking_sign_does_not_flag_violation():
    sample_path = Path("data/test_images/scene_near_no_parking_sign.png")
    assert sample_path.exists(), "Scene test image must exist."
    det_res = detect_vehicles_and_violations(sample_path)
    assert det_res["total_vehicles"] > 0
    # Mere presence of a no-parking sign must not flag vehicles traveling or outside restricted zone
    assert det_res["total_violations"] == 0
    for v in det_res["vehicles"]:
        assert v["is_violation"] is False

# Scenario 4e: Vehicle in a clearly marked restricted area (Red curb emergency access)
def test_vehicle_in_clearly_marked_restricted_area():
    sample_path = Path("data/test_images/scene_clearly_marked_restricted_area.png")
    assert sample_path.exists(), "Scene test image must exist."
    det_res = detect_vehicles_and_violations(sample_path)
    assert det_res["total_vehicles"] > 0
    assert det_res["total_violations"] > 0
    assert det_res["vehicles"][0]["is_violation"] is True
    assert det_res["vehicles"][0]["violation_status"] == "Potential violation — review required"

# Scenario 4f: Vehicle in an ordinary parking area (Compliant stall)
def test_vehicle_in_ordinary_parking_area():
    sample_path = Path("data/test_images/scene_ordinary_parking_area.png")
    assert sample_path.exists(), "Scene test image must exist."
    det_res = detect_vehicles_and_violations(sample_path)
    assert det_res["total_vehicles"] > 0
    assert det_res["total_violations"] == 0
    assert det_res["vehicles"][0]["is_violation"] is False

# Scenario 4g: Image with no vehicles
def test_image_with_no_vehicles():
    sample_path = Path("data/test_images/scene_no_vehicles.png")
    assert sample_path.exists(), "Scene test image must exist."
    det_res = detect_vehicles_and_violations(sample_path)
    assert det_res["total_vehicles"] == 0
    assert det_res["total_violations"] == 0
    assert len(det_res["vehicles"]) == 0

# Scenario 4h: Blurry or unreadable number plate
def test_blurry_or_unreadable_number_plate():
    sample_path = Path("data/test_images/scene_unreadable_blurry_plate.png")
    assert sample_path.exists(), "Scene test image must exist."
    det_res = detect_vehicles_and_violations(sample_path)
    assert det_res["total_vehicles"] > 0
    v_crop = det_res["vehicles"][0]["vehicle_crop"]
    plate_res = read_plate_from_vehicle_crop(v_crop)
    # Must show unreadable and not fabricate a plate number
    assert plate_res["reading_status"] in ["Plate unreadable", "Number plate could not be read"]
    assert plate_res["normalized_plate"] == ""
    assert plate_res["eligible_for_notification"] is False


# Scenario 5: Number plate recognized & normalized
def test_plate_normalization():
    assert normalize_plate("ka-01 ab 1234") == "KA01AB1234"
    assert normalize_plate("  p3rv  p ") == "P3RVP"
    assert normalize_plate("MH_12-DE.1432") == "MH12DE1432"
    assert normalize_plate("") == ""
    assert normalize_plate(None) == ""

def test_number_plate_recognized_from_real_crop():
    sample_path = Path("data/test_images/car_sample_2.png")
    det_res = detect_vehicles_and_violations(sample_path)
    assert det_res["total_vehicles"] > 0
    vehicle_crop = det_res["vehicles"][0]["vehicle_crop"]
    plate_res = read_plate_from_vehicle_crop(vehicle_crop)
    assert plate_res["plate_detected"] is True
    assert plate_res["normalized_plate"] == "P3RVP"
    assert plate_res["reading_status"] == "Recognized"


# Scenario 6: Unreadable number plate handling
def test_unreadable_number_plate():
    # Blank black crop with no characters
    blank_crop = np.zeros((100, 200, 3), dtype=np.uint8)
    result = read_plate_from_vehicle_crop(blank_crop)
    assert result["reading_status"] in ["Plate unreadable", "Setup Error: Plate weights missing"]
    assert result["requires_manual_review"] is True
    assert result["eligible_for_notification"] is False

# Scenario 7: Plate not present in registry
def test_plate_not_in_registry(temp_db):
    record = lookup_plate("NONEXISTENT99", db_path=temp_db)
    assert record is None

# Scenario 8: Registry match without consent
def test_registry_match_without_consent(temp_db):
    record = lookup_plate("CORSI6K", db_path=temp_db)
    assert record is not None
    assert record["consent_status"] == 0
    # Should not be eligible for automatic notification

# Scenario 9: Verified, consenting registry match
def test_verified_consenting_registry_match(temp_db):
    record = lookup_plate("P3RVP", db_path=temp_db)
    assert record is not None
    assert record["consent_status"] == 1
    assert "prof.patel" in record["email"]

# Scenario 10: Missing SMTP credentials handling
def test_missing_smtp_credentials_handling(temp_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "SMTP_USERNAME", "")
    monkeypatch.setattr(config, "SMTP_PASSWORD", "")
    monkeypatch.setattr(config, "SMTP_SENDER_EMAIL", "")

    result = send_violation_notice(
        event_id="test_event_missing_smtp",
        plate_number="P3RVP",
        recipient_email="prof.patel.test@campus-demo.edu",
        zone_name="Zone A",
        db_path=temp_db
    )
    assert result["success"] is False
    assert result["status"] == "FAILED_CONFIG"

# Scenario 11: SMTP failure handling
def test_smtp_failure_handling(temp_db):
    mock_server = MagicMock()
    mock_server.send_message.side_effect = smtplib.SMTPException("Simulated connection timeout")

    result = send_violation_notice(
        event_id="test_event_smtp_fail",
        plate_number="P3RVP",
        recipient_email="prof.patel.test@campus-demo.edu",
        zone_name="Zone A",
        server_override=mock_server,
        db_path=temp_db
    )
    assert result["success"] is False
    assert result["status"] == "SMTP_ERROR"

# Scenario 12: Duplicate notification prevention (Idempotency)
def test_duplicate_notification_prevention(temp_db):
    mock_server = MagicMock()
    mock_server.send_message.return_value = {}

    event_id = "test_event_dedup_001"
    # First delivery succeeds
    res1 = send_violation_notice(
        event_id=event_id,
        plate_number="P3RVP",
        recipient_email="prof.patel.test@campus-demo.edu",
        zone_name="Zone A",
        server_override=mock_server,
        db_path=temp_db
    )
    assert res1["success"] is True
    assert res1["status"] == "SENT"

    # Second delivery with same event_id must be blocked
    res2 = send_violation_notice(
        event_id=event_id,
        plate_number="P3RVP",
        recipient_email="prof.patel.test@campus-demo.edu",
        zone_name="Zone A",
        server_override=mock_server,
        db_path=temp_db
    )
    assert res2["success"] is False
    assert res2["status"] == "PREVENTED_DUPLICATE"

# -----------------------------------------------------------------
# Overhead Parking Image Vehicle Detection Regression Tests
# -----------------------------------------------------------------
def test_overhead_parking_multiple_cars_localized():
    """
    Regression Test: Overhead parking lot image with ~15 clearly visible cars
    must localize multiple vehicles (>= 10 cars) and draw bounding boxes.
    """
    sample_path = Path("data/test_images/sample_parking_1.jpg")
    if not sample_path.exists():
        sample_path = Path("data/test_images/overhead_parking_sample.jpg")
    assert sample_path.exists(), "Overhead parking test image must exist."

    det_res = detect_vehicles_and_violations(sample_path)
    total_v = det_res["total_vehicles"]
    assert total_v >= 10, f"Expected at least 10 cars localized, got {total_v}"
    assert len(det_res["vehicles"]) == total_v

    img_w, img_h = det_res["image_size"]
    for v in det_res["vehicles"]:
        x1, y1, x2, y2 = v["bbox"]
        gx, gy = v["ground_point"]
        assert 0 <= x1 < x2 <= img_w, f"Invalid x coordinates: ({x1}, {x2}) in width {img_w}"
        assert 0 <= y1 < y2 <= img_h, f"Invalid y coordinates: ({y1}, {y2}) in height {img_h}"
        assert x1 <= gx <= x2, "Ground point gx must lie within horizontal bbox"
        assert gy == y2, "Ground point gy must be the bottom of bbox"
        assert v["class_name"] in ["car", "van", "truck", "bus", "motorcycle"]

    assert det_res["annotated_image"] is not None
    assert det_res["annotated_image"].shape[:2] == (img_h, img_w)

def test_overhead_parking_zone_analysis_receives_coordinates():
    """
    Verifies that the parking-violation zone logic receives the actual coordinates
    of localized vehicles in the overhead parking image.
    """
    sample_path = Path("data/test_images/sample_parking_1.jpg")
    if not sample_path.exists():
        sample_path = Path("data/test_images/overhead_parking_sample.jpg")
    assert sample_path.exists()

    # Define a restricted zone covering the lower parking bay rows (y in 210 to 370)
    test_zone = {
        "id": "zone_lower_bay",
        "name": "Restricted Lower Stalls",
        "polygon": [[100, 210], [550, 210], [550, 370], [100, 370]]
    }

    det_res = detect_vehicles_and_violations(sample_path, zones=[test_zone])
    assert det_res["total_vehicles"] >= 10

    # At least some vehicles fall inside the defined lower bay zone
    inside_count = sum(1 for v in det_res["vehicles"] if v["is_violation"])
    outside_count = sum(1 for v in det_res["vehicles"] if not v["is_violation"])

    assert inside_count > 0, "Vehicles inside the lower bay zone must be flagged as violations"
    assert outside_count > 0, "Vehicles outside the lower bay zone must remain non-violations"

def test_scene_with_no_vehicles_regression():
    """
    Verifies that an empty scene with no vehicles is correctly handled
    without false positives.
    """
    sample_path = Path("data/test_images/scene_no_vehicles.png")
    assert sample_path.exists()

    det_res = detect_vehicles_and_violations(sample_path)
    assert det_res["total_vehicles"] == 0
    assert len(det_res["vehicles"]) == 0
    assert det_res["total_violations"] == 0

def test_changing_confidence_threshold_affects_inference():
    """
    Verifies that changing the confidence threshold affects inference detections as expected.
    """
    sample_path = Path("data/test_images/sample_parking_1.jpg")
    if not sample_path.exists():
        sample_path = Path("data/test_images/overhead_parking_sample.jpg")
    assert sample_path.exists()

    res_low = detect_vehicles_and_violations(sample_path, conf_threshold=0.15)
    res_high = detect_vehicles_and_violations(sample_path, conf_threshold=0.85)

    assert res_low["total_vehicles"] >= res_high["total_vehicles"]
    assert res_high["total_vehicles"] > 0
    assert res_low["total_vehicles"] >= 10

def test_detection_diagnostics_metadata():
    """
    Verifies that the inference pipeline returns complete diagnostics
    for debugging and inspection.
    """
    sample_path = Path("data/test_images/sample_parking_1.jpg")
    if not sample_path.exists():
        sample_path = Path("data/test_images/overhead_parking_sample.jpg")
    assert sample_path.exists()

    det_res = detect_vehicles_and_violations(sample_path, conf_threshold=0.25)
    assert "diagnostics" in det_res
    diag = det_res["diagnostics"]

    assert "model_name" in diag
    assert "class_labels" in diag
    assert "conf_threshold" in diag
    assert "image_size" in diag
    assert "raw_detection_count" in diag
    assert "vehicle_detection_count" in diag
    assert diag["raw_detection_count"] >= diag["vehicle_detection_count"]
    assert diag["vehicle_detection_count"] == det_res["total_vehicles"]
    assert diag["image_size"] == det_res["image_size"]
