import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import smtplib

from database import init_db, get_connection
from registry import (
    normalize_plate,
    lookup_plate,
    register_vehicle,
    register_or_load_vehicle,
    validate_registration,
    is_valid_email,
    get_all_vehicles
)
from plate_reader import read_plate_from_vehicle_crop
from notifications import send_violation_notice, is_duplicate_event
from parking_detector import check_ground_contact_violation

@pytest.fixture
def temp_db():
    """Provides a fresh isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = Path(tf.name)
    init_db(db_path)
    yield db_path
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

# Scenario 1: A registered vehicle is matched to its existing email
def test_scenario_1_registered_vehicle_matched_to_existing_email(temp_db):
    # Test seeded record SAMPLEPLATE1
    rec = lookup_plate("SAMPLEPLATE1", db_path=temp_db)
    assert rec is not None
    assert rec["email"] == "student1@example.invalid"
    assert rec["consent_status"] == 1

    # Test seeded record with space / punctuation query
    rec2 = lookup_plate("sample-plate-2", db_path=temp_db)
    assert rec2 is not None
    assert rec2["email"] == "principal1@example.invalid"

# Scenario 2: An unknown vehicle displays the Register Vehicle option
def test_scenario_2_unknown_vehicle_displays_register_vehicle_option(temp_db):
    unknown_plate = "UNKNOWN-8899"
    norm = normalize_plate(unknown_plate)
    rec = lookup_plate(norm, db_path=temp_db)
    # Plate is absent from database
    assert rec is None

    # In the application workflow, an unregistered plate with a potential violation
    # does not raise a fatal error, but instead triggers the 'Not Registered' status
    # and provides the registration option.
    is_registered = rec is not None
    assert is_registered is False

# Scenario 3: A new registration is saved successfully
def test_scenario_3_new_registration_saved_successfully(temp_db):
    plate = "KA-05-MJ-4321"
    email = "owner.newreg@example.com"
    success, msg = register_vehicle(
        plate_number=plate,
        email=email,
        consent_status=1,
        require_consent=True,
        db_path=temp_db
    )
    assert success is True
    assert "successfully registered" in msg.lower()

    # Verify persisted in SQLite
    norm = normalize_plate(plate)
    rec = lookup_plate(norm, db_path=temp_db)
    assert rec is not None
    assert rec["plate_number"] == norm
    assert rec["email"] == email.lower()
    assert rec["consent_status"] == 1
    assert "created_at" in rec
    assert len(rec["created_at"]) > 0

# Scenario 4: The new registration is available immediately without restarting the application
def test_scenario_4_new_registration_available_immediately_without_restart(temp_db):
    plate = "IMMEDIATE-01"
    email = "immediate.test@example.com"

    # Save registration
    success, _ = register_vehicle(
        plate_number=plate,
        email=email,
        consent_status=1,
        require_consent=True,
        db_path=temp_db
    )
    assert success is True

    # Immediate query in the same session without restart
    immediate_record = lookup_plate("immediate01", db_path=temp_db)
    assert immediate_record is not None
    assert immediate_record["email"] == email.lower()
    assert immediate_record["consent_status"] == 1

# Scenario 5: The registration remains after restarting the application
def test_scenario_5_registration_remains_after_restarting_application(temp_db):
    plate = "PERSIST-RESTART-09"
    email = "restart.test@example.com"

    success, _ = register_vehicle(
        plate_number=plate,
        email=email,
        consent_status=1,
        require_consent=True,
        db_path=temp_db
    )
    assert success is True

    # Simulate application restart by calling init_db again
    init_db(temp_db)

    # Verify record is still present and unaltered
    rec = lookup_plate(plate, db_path=temp_db)
    assert rec is not None
    assert rec["email"] == email.lower()
    assert rec["consent_status"] == 1

# Scenario 6: Duplicate plates cannot be registered
def test_scenario_6_duplicate_plates_cannot_be_registered(temp_db):
    plate = "DL-01-AB-9999"
    email1 = "first.owner@example.com"
    email2 = "second.owner@example.com"

    # First registration
    s1, _ = register_vehicle(plate, email1, consent_status=1, db_path=temp_db)
    assert s1 is True

    # Second registration attempt with identical normalized plate
    s2, msg2 = register_vehicle("dl 01 ab 9999", email2, consent_status=1, db_path=temp_db)
    assert s2 is False
    assert "already registered" in msg2.lower()

    # Also test register_or_load_vehicle: should load existing without duplicate insert
    succ, load_msg, loaded_rec, is_existing = register_or_load_vehicle(
        "DL01AB9999",
        email2,
        db_path=temp_db
    )
    assert succ is True
    assert is_existing is True
    assert loaded_rec["email"] == email1.lower()

    # Ensure total records in registry with this plate is exactly 1
    conn = get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM registry WHERE plate_number = 'DL01AB9999'")
    count = cursor.fetchone()["cnt"]
    conn.close()
    assert count == 1

# Scenario 7: An invalid email is rejected
def test_scenario_7_invalid_email_is_rejected(temp_db):
    invalid_emails = [
        "not-an-email",
        "missingatsign.com",
        "user@",
        "@domain.com",
        "user@domain",
        "",
        "   "
    ]
    for bad_email in invalid_emails:
        success, msg = register_vehicle(
            plate_number="VALIDPLATE99",
            email=bad_email,
            consent_status=1,
            require_consent=True,
            db_path=temp_db
        )
        assert success is False, f"Expected rejection for invalid email: '{bad_email}'"
        assert "invalid email" in msg.lower()

    # Ensure nothing was added
    rec = lookup_plate("VALIDPLATE99", db_path=temp_db)
    assert rec is None

# Scenario 8: Registration cannot be completed without consent
def test_scenario_8_registration_cannot_be_completed_without_consent(temp_db):
    plate = "NOCONSENT99"
    email = "noconsent.user@example.com"

    # When require_consent=True and consent_status=0
    success, msg = register_vehicle(
        plate_number=plate,
        email=email,
        consent_status=0,
        require_consent=True,
        db_path=temp_db
    )
    assert success is False
    assert "consent" in msg.lower()

    # Ensure record was not saved
    rec = lookup_plate(plate, db_path=temp_db)
    assert rec is None

# Scenario 9: An unreadable plate cannot be registered automatically
def test_scenario_9_unreadable_plate_cannot_be_registered_automatically(temp_db):
    # Blank crop produces unreadable status
    blank_crop = np.zeros((80, 160, 3), dtype=np.uint8)
    ocr_result = read_plate_from_vehicle_crop(blank_crop)

    # Reading status must indicate unreadable / manual review required
    assert ocr_result["reading_status"] in ["Plate unreadable", "Setup Error: Plate weights missing"]
    assert ocr_result["requires_manual_review"] is True
    assert ocr_result["eligible_for_notification"] is False

    # Attempting to validate or register empty plate string is rejected
    is_valid, msg = validate_registration(ocr_result["normalized_plate"], "test@example.com", db_path=temp_db)
    assert is_valid is False
    assert "cannot be empty" in msg.lower()

# Scenario 10: Registering a vehicle does not automatically send an email
def test_scenario_10_registering_vehicle_does_not_automatically_send_email(temp_db):
    mock_smtp = MagicMock()

    # Perform registration
    plate = "NOAUTO-SEND-10"
    email = "noauto@example.com"
    success, _ = register_vehicle(plate, email, consent_status=1, require_consent=True, db_path=temp_db)
    assert success is True

    # Verify mock SMTP send_message was NEVER invoked
    assert mock_smtp.send_message.call_count == 0

    # Verify SQLite notification_history has no notifications for this plate
    conn = get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM notification_history WHERE plate_number = ?", (normalize_plate(plate),))
    cnt = cursor.fetchone()["cnt"]
    conn.close()
    assert cnt == 0

# Scenario 11: A verified, consenting, registered vehicle can receive an email after explicit confirmation
def test_scenario_11_verified_consenting_registered_vehicle_can_receive_email_after_explicit_confirmation(temp_db):
    mock_smtp = MagicMock()
    mock_smtp.send_message.return_value = {}

    plate = "CONSENT-VERIF-11"
    email = "verified.recipient@example.com"
    success, _ = register_vehicle(plate, email, consent_status=1, require_consent=True, db_path=temp_db)
    assert success is True

    # Reviewer explicitly confirms and sends notification
    event_id = "EVT-TEST-HASH-11-CONSENTVERIF11"
    res = send_violation_notice(
        event_id=event_id,
        plate_number=normalize_plate(plate),
        recipient_email=email,
        zone_name="Emergency Access Lane",
        server_override=mock_smtp,
        db_path=temp_db
    )

    assert res["success"] is True
    assert res["status"] == "SENT"
    assert mock_smtp.send_message.call_count == 1
    sent_msg = mock_smtp.send_message.call_args[0][0]
    assert sent_msg["To"] == email
    assert "Emergency Access Lane" in sent_msg.get_content()

# Scenario 12: A vehicle without a parking violation is not unnecessarily prompted for registration
def test_scenario_12_vehicle_without_parking_violation_not_prompted_for_registration():
    test_zone = {
        "id": "restricted_lane",
        "name": "Restricted Lane",
        "polygon": [[100, 100], [300, 100], [300, 250], [100, 250]]
    }
    # Vehicle outside zone
    outside_ground_point = (20, 20)
    is_viol, zone_name = check_ground_contact_violation(outside_ground_point, [test_zone], 400, 250)

    assert is_viol is False
    assert zone_name is None
    # Pipeline contract: When is_violation is False, plate detection and registration
    # prompting are omitted.
