import pytest
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import smtplib

from database import init_db, get_connection, check_slot_authorization, get_parking_authorization
from registry import register_vehicle, lookup_plate, normalize_plate
from parking_detector import (
    detect_vehicles_and_violations,
    detect_scene_visual_evidence
)
from campus_violation_engine import (
    detect_footpath_regions,
    detect_entrance_clearance_corridors,
    detect_lawn_green_spaces,
    detect_parking_stall_lines,
    detect_reserved_slot_markings,
    evaluate_all_campus_violations,
    evaluate_unauthorized_no_parking_zone,
    evaluate_footpath_parking,
    evaluate_blocking_entrance,
    evaluate_lawn_parking,
    evaluate_parking_alignment,
    evaluate_wrong_parking_slot
)
from plate_reader import read_plate_from_vehicle_crop
from notifications import send_violation_notice, get_recent_audit_logs

TEST_DATA_DIR = Path("data/test_images")

@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Initializes isolated SQLite test database with campus permissions schema."""
    test_db = tmp_path / "test_campus_violations.db"
    monkeypatch.setattr("config.DB_PATH", test_db)
    init_db(test_db)
    yield test_db

# -----------------------------------------------------------------
# 1. Category 1: Unauthorized Parking / No-Parking Zones
# -----------------------------------------------------------------
def test_category1_vehicle_in_clearly_marked_no_parking_zone():
    img_path = TEST_DATA_DIR / "scene_clearly_marked_restricted_area.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat1 = evals.get("categories", {}).get("cat1_unauthorized_zone", {})
    assert cat1.get("status") == "Potential violation detected"
    assert v["is_violation"] is True

def test_category1_vehicle_legally_passing_near_no_parking_sign():
    img_path = TEST_DATA_DIR / "scene_near_no_parking_sign.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat1 = evals.get("categories", {}).get("cat1_unauthorized_zone", {})
    # Mere proximity to sign must NOT trigger false violation notice
    assert cat1.get("status") in ["Not detected", "Review required"]
    assert cat1.get("status") != "Potential violation detected"
    assert v["is_violation"] is False

# -----------------------------------------------------------------
# 2. Category 2: Footpath Parking
# -----------------------------------------------------------------
def test_category2_motorcycle_occupying_footpath():
    img_path = TEST_DATA_DIR / "scene_cat2_footpath_parking.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat2 = evals.get("categories", {}).get("cat2_footpath_parking", {})
    assert cat2.get("status") == "Potential violation detected"
    assert "footpath" in cat2.get("explanation", "").lower() or "sidewalk" in cat2.get("explanation", "").lower()

def test_category2_vehicle_beside_footpath_compliant():
    img_path = TEST_DATA_DIR / "scene_cat2_footpath_compliant.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat2 = evals.get("categories", {}).get("cat2_footpath_parking", {})
    assert cat2.get("status") == "Not detected"

def test_category2_car_half_mounted_on_footpath():
    img_path = TEST_DATA_DIR / "car_parked_footpath_real.jpg"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    footpath_viol = None
    for v in result["vehicles"]:
        cat2 = v.get("campus_evaluations", {}).get("categories", {}).get("cat2_footpath_parking", {})
        if cat2.get("status") == "Potential violation detected":
            footpath_viol = v
            break
    assert footpath_viol is not None, "Vehicle partially mounted on footpath must trigger Category 2 violation"
    cat2_stat = footpath_viol["campus_evaluations"]["categories"]["cat2_footpath_parking"]
    assert "mounted on pedestrian sidewalk" in cat2_stat.get("explanation", "").lower()

def test_category2_red_pickup_mounted_on_curb_footpath():
    """
    Image regression test:
    Red pickup truck parked partly on raised roadside footpath with broken yellow curb line.
    Must correctly identify Footpath Parking violation, NOT Improper Parking Alignment.
    """
    img_path = TEST_DATA_DIR / "red_pickup_footpath.jpg"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1

    truck = None
    for v in result["vehicles"]:
        if v["class_name"].lower() == "truck" or (v["bbox"][2] - v["bbox"][0]) > 200:
            truck = v
            break

    assert truck is not None, "Red pickup truck must be detected"
    evals = truck.get("campus_evaluations", {})
    cats = evals.get("categories", {})
    cat2 = cats.get("cat2_footpath_parking", {})
    cat5 = cats.get("cat5_improper_alignment", {})

    # Category 2 must detect Footpath Parking violation
    assert cat2.get("status") == "Potential violation detected"
    assert truck["is_violation"] is True
    assert "footpath" in cat2.get("explanation", "").lower() or "sidewalk" in cat2.get("explanation", "").lower()

    # Category 5 must NOT be falsely triggered by the curb or stone wall
    assert cat5.get("status") in ["Not detected", "Not applicable"]
    assert cat5.get("is_violation") is False

def test_category2_footpath_partial_occupancy_rule():
    """
    Rule test: Evaluates vehicle whose left wheel is across curb on footpath.
    """
    dummy_vehicle = {
        "vehicle_id": 1,
        "class_name": "truck",
        "bbox": (145, 264, 406, 537),
        "ground_point": (275, 537)
    }
    dummy_footpaths = [{
        "id": "footpath_curb_demarcation",
        "name": "Pedestrian Footpath / Sidewalk",
        "bbox": (0, 100, 310, 600),
        "curb_poly": (0.825, -136.5),
        "mask": None
    }]
    dummy_img = np.zeros((600, 500, 3), dtype=np.uint8)
    cat2 = evaluate_footpath_parking(dummy_img, dummy_vehicle, dummy_footpaths)
    assert cat2["status"] == "Potential violation detected"
    assert cat2["is_violation"] is True
    assert "footpath" in cat2["explanation"].lower() or "sidewalk" in cat2["explanation"].lower()

def test_category2_vehicle_entirely_on_roadway_rule():
    """
    Rule test: Vehicle situated on roadway clear of curb and footpath.
    """
    dummy_vehicle = {
        "vehicle_id": 2,
        "class_name": "car",
        "bbox": (330, 264, 480, 537),
        "ground_point": (405, 537)
    }
    dummy_footpaths = [{
        "id": "footpath_curb_demarcation",
        "name": "Pedestrian Footpath / Sidewalk",
        "bbox": (0, 100, 310, 600),
        "curb_poly": (0.825, -136.5),
        "mask": None
    }]
    dummy_img = np.zeros((600, 500, 3), dtype=np.uint8)
    cat2 = evaluate_footpath_parking(dummy_img, dummy_vehicle, dummy_footpaths)
    assert cat2["status"] == "Not detected"
    assert cat2["is_violation"] is False

def test_category2_sidewalk_boundary_uncertain_returns_review_required():
    """
    Rule test: When sidewalk boundary cannot be determined reliably,
    must return 'Review required' instead of guessing.
    """
    dummy_vehicle = {
        "vehicle_id": 3,
        "class_name": "car",
        "bbox": (100, 100, 200, 200),
        "ground_point": (150, 200)
    }
    unclear_footpaths = [{
        "id": "footpath_ambiguous",
        "name": "Pedestrian Footpath",
        "bbox": (50, 50, 250, 250),
        "curb_poly": None,
        "mask": None,
        "is_reliable": False
    }]
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cat2 = evaluate_footpath_parking(dummy_img, dummy_vehicle, unclear_footpaths)
    assert cat2["status"] == "Review required"
    assert cat2["is_violation"] is False
    assert "cannot be identified reliably" in cat2["explanation"] or "review required" in cat2["explanation"].lower()

# -----------------------------------------------------------------
# 3. Category 3: Blocking Entrances and Emergency Access
# -----------------------------------------------------------------
def test_category3_vehicle_blocking_entrance_gate():
    img_path = TEST_DATA_DIR / "scene_cat3_blocking_entrance.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat3 = evals.get("categories", {}).get("cat3_blocking_entrance", {})
    assert cat3.get("status") == "Potential violation detected"
    assert "entrance" in cat3.get("explanation", "").lower() or "clearance" in cat3.get("explanation", "").lower()

def test_category3_vehicle_beside_entrance_compliant():
    img_path = TEST_DATA_DIR / "scene_cat3_entrance_clear.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat3 = evals.get("categories", {}).get("cat3_blocking_entrance", {})
    assert cat3.get("status") == "Not detected"

# -----------------------------------------------------------------
# 4. Category 4: Parking on Green Spaces
# -----------------------------------------------------------------
def test_category4_vehicle_occupying_lawn():
    img_path = TEST_DATA_DIR / "scene_cat4_lawn_parking.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat4 = evals.get("categories", {}).get("cat4_lawn_parking", {})
    assert cat4.get("status") == "Potential violation detected"
    assert "lawn" in cat4.get("explanation", "").lower() or "green" in cat4.get("explanation", "").lower()

def test_category4_vehicle_on_paved_parking_beside_lawn():
    img_path = TEST_DATA_DIR / "scene_cat4_paved_parking.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat4 = evals.get("categories", {}).get("cat4_lawn_parking", {})
    assert cat4.get("status") == "Not detected"

# -----------------------------------------------------------------
# 5. Category 5: Improper Parking Alignment
# -----------------------------------------------------------------
def test_category5_vehicle_straddling_stall_divider():
    img_path = TEST_DATA_DIR / "scene_cat5_improper_alignment.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat5 = evals.get("categories", {}).get("cat5_improper_alignment", {})
    assert cat5.get("status") == "Potential violation detected"
    assert "straddle" in cat5.get("explanation", "").lower() or "stall" in cat5.get("explanation", "").lower()

def test_category5_vehicle_properly_aligned_within_stall():
    img_path = TEST_DATA_DIR / "scene_cat5_proper_alignment.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    evals = v.get("campus_evaluations", {})
    cat5 = evals.get("categories", {}).get("cat5_improper_alignment", {})
    assert cat5.get("status") == "Not detected"

def test_category5_detects_vehicle_straddling_perspective_slanted_divider():
    """Angled camera views have converging stall lines, not fixed vertical Xs."""
    image = np.full((400, 640, 3), 70, dtype=np.uint8)
    for start, end in [((180, 80), (150, 390)), ((320, 80), (280, 390)), ((460, 80), (410, 390))]:
        cv2.line(image, start, end, (245, 245, 245), 6)

    stall_data = detect_parking_stall_lines(image)
    assert stall_data["is_reliable"] is True
    assert len(stall_data["divider_lines"]) >= 3

    vehicle = {"bbox": (250, 220, 390, 350), "ground_point": (320, 350)}
    result = evaluate_parking_alignment(vehicle, stall_data)
    assert result["status"] == "Potential violation detected"
    assert "ground row" in result["evidence"]

def test_category5_hidden_stripe_without_confirmed_stall_grid_requests_review():
    """A short divider visible beyond a vehicle should not silently become compliant."""
    vehicle = {"bbox": (736, 522, 1139, 840), "ground_point": (937, 840)}
    stall_data = {
        "divider_x_coords": [],
        "divider_lines": [],
        "image_height": 1024,
        "is_reliable": False,
        "vehicle_divider_continuations": [
            {"bbox": vehicle["bbox"], "x": 908, "pixels": 8}
        ],
    }

    result = evaluate_parking_alignment(vehicle, stall_data)
    assert result["status"] == "Review required"
    assert result["is_violation"] is False
    assert "confirm stall context" in result["evidence"]

def test_category5_curb_or_wall_not_mistaken_for_parking_stalls():
    """
    Regression test: Curbs, stone walls, and roadside demarcation lines
    must NOT be classified as parking stall divider lines.
    If no genuine parking bay can be identified, result should be Not detected / Not applicable,
    never a confirmed alignment violation.
    """
    img_path = TEST_DATA_DIR / "red_pickup_footpath.jpg"
    assert img_path.exists()
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    stall_data = detect_parking_stall_lines(img_rgb)
    assert stall_data["is_reliable"] is False
    assert len(stall_data["divider_x_coords"]) == 0

    dummy_vehicle = {
        "vehicle_id": 1,
        "class_name": "truck",
        "bbox": (145, 264, 406, 537),
        "ground_point": (275, 537)
    }
    cat5 = evaluate_parking_alignment(dummy_vehicle, stall_data)
    assert cat5["status"] in ["Not detected", "Not applicable"]
    assert cat5["is_violation"] is False
    assert "not applicable" in cat5["explanation"].lower() or "unable to determine" in cat5["explanation"].lower()

# -----------------------------------------------------------------
# 6. Category 6: Wrong Parking Slot
# -----------------------------------------------------------------
def test_category6_unauthorized_student_in_principal_slot(setup_test_db):
    img_path = TEST_DATA_DIR / "scene_cat6_wrong_slot_unauthorized.png"
    assert img_path.exists()

    # Register unauthorized student vehicle in DB
    register_vehicle(
        plate_number="STUDENT01",
        email="student@campus.edu",
        role="student",
        authorized_slots=["student"],
        consent_status=1,
        db_path=setup_test_db
    )

    result = detect_vehicles_and_violations(img_path, db_path=setup_test_db)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]

    # Explicit evaluation check with student plate
    plate_data = {"normalized_plate": "STUDENT01", "reading_status": "Recognized"}
    evals = evaluate_all_campus_violations(
        image_rgb=np.array(cv2.imread(str(img_path))),
        vehicle=v,
        scene_evidence=result.get("scene_evidence", {}),
        plate_data=plate_data,
        db_path=setup_test_db
    )
    cat6 = evals["categories"]["cat6_wrong_parking_slot"]
    assert cat6["status"] == "Potential violation detected"
    assert "not authorized" in cat6["explanation"].lower() or "faculty" in cat6["explanation"].lower() or "principal" in cat6["explanation"].lower()

def test_category6_authorized_principal_in_principal_slot(setup_test_db):
    img_path = TEST_DATA_DIR / "scene_cat6_correct_slot_authorized.png"
    assert img_path.exists()

    # Register principal with principal slot permit
    register_vehicle(
        plate_number="PRINCIPAL1",
        email="principal@campus.edu",
        role="principal",
        authorized_slots=["principal", "faculty", "staff"],
        consent_status=1,
        db_path=setup_test_db
    )

    result = detect_vehicles_and_violations(img_path, db_path=setup_test_db)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]

    plate_data = {"normalized_plate": "PRINCIPAL1", "reading_status": "Recognized"}
    evals = evaluate_all_campus_violations(
        image_rgb=np.array(cv2.imread(str(img_path))),
        vehicle=v,
        scene_evidence=result.get("scene_evidence", {}),
        plate_data=plate_data,
        db_path=setup_test_db
    )
    cat6 = evals["categories"]["cat6_wrong_parking_slot"]
    assert cat6["status"] == "Not detected"
    assert "authorized" in cat6["explanation"].lower()

# -----------------------------------------------------------------
# 7. Unclear Scene Boundaries -> Review Required (No False Violation)
# -----------------------------------------------------------------
def test_unclear_scene_boundaries_returns_review_required():
    # Empty stall line data simulates worn/unclear painted lines
    dummy_vehicle = {
        "vehicle_id": 1,
        "class_name": "car",
        "bbox": (100, 100, 200, 200),
        "ground_point": (150, 200)
    }
    unclear_stalls = {"divider_x_coords": [120, 240], "is_reliable": False}
    cat5 = evaluate_parking_alignment(dummy_vehicle, unclear_stalls)

    # Must NOT invent boundaries or claim a confirmed violation
    assert cat5["status"] == "Review required"
    assert cat5["is_violation"] is False
    assert "cannot be identified reliably" in cat5["explanation"]

# -----------------------------------------------------------------
# 8. Unreadable Number Plate Handling
# -----------------------------------------------------------------
def test_unreadable_number_plate_handling():
    img_path = TEST_DATA_DIR / "scene_unreadable_blurry_plate.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]

    # Test OCR extraction on blurry crop
    plate_res = read_plate_from_vehicle_crop(v["vehicle_crop"])
    # Plate must either not be detected or reading_status is unreadable
    assert plate_res["reading_status"] in ["Plate unreadable", "Uncertain (Review required)", "Setup Error: Plate weights missing"]
    assert plate_res["eligible_for_notification"] is False

# -----------------------------------------------------------------
# 9. Unknown / Unregistered Vehicle Workflow
# -----------------------------------------------------------------
def test_unknown_vehicle_registration_and_lookup(setup_test_db):
    unknown_plate = "UNKNOWN99"
    rec = lookup_plate(unknown_plate, db_path=setup_test_db)
    assert rec is None

    # Register unknown vehicle
    success, msg = register_vehicle(
        plate_number=unknown_plate,
        email="newuser@example.com",
        role="visitor",
        consent_status=1,
        db_path=setup_test_db
    )
    assert success is True
    rec_after = lookup_plate(unknown_plate, db_path=setup_test_db)
    assert rec_after is not None
    assert rec_after["email"] == "newuser@example.com"
    assert rec_after["role"] == "visitor"

# -----------------------------------------------------------------
# 10. Registered Vehicle Without Notification Consent
# -----------------------------------------------------------------
def test_registered_vehicle_without_notification_consent(setup_test_db):
    plate = "NOCONSENT1"
    register_vehicle(
        plate_number=plate,
        email="noconsent@example.com",
        role="student",
        consent_status=0,  # Consent disabled
        db_path=setup_test_db
    )

    rec = lookup_plate(plate, db_path=setup_test_db)
    assert rec["consent_status"] == 0

    # Attempt to dispatch notice
    with patch("smtplib.SMTP") as mock_smtp:
        res = send_violation_notice(
            event_id="EVT-TEST-NOCONSENT",
            plate_number=plate,
            recipient_email="noconsent@example.com",
            zone_name="Pedestrian Footpath / Sidewalk",
            db_path=setup_test_db
        )
        assert res["success"] is False
        assert "consent" in res["message"].lower()
        mock_smtp.assert_not_called()

# -----------------------------------------------------------------
# 11. Mocked SMTP Email Delivery Failure
# -----------------------------------------------------------------
def test_email_delivery_failure_mocked(setup_test_db):
    plate = "SMTPFAIL1"
    register_vehicle(
        plate_number=plate,
        email="fail@example.com",
        role="faculty",
        consent_status=1,
        db_path=setup_test_db
    )

    # Mock SMTP to simulate network / connection failure
    with patch("smtplib.SMTP", side_effect=smtplib.SMTPConnectError(421, b"Connection refused")):
        res = send_violation_notice(
            event_id="EVT-TEST-FAIL",
            plate_number=plate,
            recipient_email="fail@example.com",
            zone_name="Campus Lawn / Green Space",
            db_path=setup_test_db
        )
        assert res["success"] is False
        assert "failed" in res["message"].lower()

        # Check that failure was recorded in audit log
        logs = get_recent_audit_logs(limit=5, db_path=setup_test_db)
        assert len(logs) >= 1
        assert logs[0]["status"].lower() == "failed"

# -----------------------------------------------------------------
# 12. Explicit Section 6 Regression Tests
# -----------------------------------------------------------------
def test_regression_vehicle_parked_on_grass_away_from_footpath():
    """
    Regression Test 1: A vehicle parked on grass, away from the footpath.
    Image: data/test_images/white_car_grass_parking.jpg
    The white car is parked on the lawn beside a diagonal walkway and building.
    - Category 4 (Parking on Green Spaces): Potential violation detected.
    - Category 2 (Footpath Parking): Not detected (no 2D perspective false positive).
    - Category 3 (Blocking Entrances): Not detected (building wall/lawn is not an entrance).
    - Category 1 (Unauthorized Zone): Not detected (no red curb on building).
    - Category 5 (Alignment): Not detected (lawn is not a parking bay).
    """
    img_path = TEST_DATA_DIR / "white_car_grass_parking.jpg"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1

    # Locate the primary foreground white car (largest bounding box in the scene)
    white_car = max(result["vehicles"], key=lambda v: (v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]))
    evals = white_car["campus_evaluations"]["categories"]

    assert evals["cat4_lawn_parking"]["status"] == "Potential violation detected", "Green space parking must be detected"
    assert evals["cat2_footpath_parking"]["status"] == "Not detected", "Must NOT falsely flag Footpath Parking"
    assert evals["cat3_blocking_entrance"]["status"] == "Not detected", "Must NOT falsely flag Blocking Entrances"
    assert evals["cat1_unauthorized_zone"]["status"] == "Not detected", "Must NOT falsely flag Red Curb / No-Parking"
    assert evals["cat5_improper_alignment"]["status"] == "Not detected", "Must NOT falsely flag Improper Alignment"
    assert white_car["campus_evaluations"]["primary_violation"]["category_key"] == "cat4_lawn_parking"

def test_regression_vehicle_parked_on_footpath():
    """
    Regression Test 2: A vehicle parked on a footpath.
    Image: data/test_images/red_pickup_footpath.jpg
    Must correctly detect Footpath Parking violation.
    """
    img_path = TEST_DATA_DIR / "red_pickup_footpath.jpg"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    truck = next(v for v in result["vehicles"] if v["class_name"].lower() == "truck" or (v["bbox"][2] - v["bbox"][0]) > 200)
    cat2 = truck["campus_evaluations"]["categories"]["cat2_footpath_parking"]
    assert cat2["status"] == "Potential violation detected"

def test_regression_vehicle_beside_building_without_blocking_entrance():
    """
    Regression Test 3: A vehicle beside a building without blocking an entrance.
    Image: data/test_images/scene_cat3_entrance_clear.png
    Must NOT flag Blocking Entrances and Emergency Access.
    """
    img_path = TEST_DATA_DIR / "scene_cat3_entrance_clear.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    cat3 = v["campus_evaluations"]["categories"]["cat3_blocking_entrance"]
    assert cat3["status"] == "Not detected"

def test_regression_vehicle_actually_obstructing_designated_entrance():
    """
    Regression Test 4: A vehicle actually obstructing a designated entrance.
    Image: data/test_images/scene_cat3_blocking_entrance.png
    Vehicle parked in marked yellow hatched emergency gate clearance corridor.
    Must detect Blocking Entrances and Emergency Access violation.
    """
    img_path = TEST_DATA_DIR / "scene_cat3_blocking_entrance.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    cat3 = v["campus_evaluations"]["categories"]["cat3_blocking_entrance"]
    assert cat3["status"] == "Potential violation detected"

def test_regression_vehicle_parked_on_marked_parking_bay():
    """
    Regression Test 5: A vehicle parked on a marked parking bay.
    Image: data/test_images/scene_cat5_proper_alignment.png
    Vehicle centered neatly inside genuine painted parking stall lines.
    Must NOT detect any alignment violation.
    """
    img_path = TEST_DATA_DIR / "scene_cat5_proper_alignment.png"
    assert img_path.exists()

    result = detect_vehicles_and_violations(img_path)
    assert result["total_vehicles"] >= 1
    v = result["vehicles"][0]
    cat5 = v["campus_evaluations"]["categories"]["cat5_improper_alignment"]
    assert cat5["status"] == "Not detected"

def test_regression_vehicle_outside_identifiable_parking_space_markings():
    """
    Regression Test 6: A vehicle outside identifiable parking-space markings.
    When parking space markings are absent or cannot be established reliably,
    must NOT invent a false confirmed alignment violation.
    """
    dummy_vehicle = {
        "vehicle_id": 99,
        "class_name": "car",
        "bbox": (100, 100, 250, 250),
        "ground_point": (175, 250)
    }
    # Case A: No stall lines exist in the scene (e.g. open road or unmarked lot)
    no_stalls = {"divider_x_coords": [], "is_reliable": False}
    cat5_a = evaluate_parking_alignment(dummy_vehicle, no_stalls)
    assert cat5_a["status"] == "Not detected"
    assert cat5_a["is_violation"] is False

    # Case B: Unclear / degraded line evidence where boundaries cannot be determined
    unclear_stalls = {"divider_x_coords": [150], "is_reliable": False}
    cat5_b = evaluate_parking_alignment(dummy_vehicle, unclear_stalls)
    assert cat5_b["status"] == "Review required"
    assert cat5_b["is_violation"] is False
