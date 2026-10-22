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
    add_vehicle,
    register_vehicle,
    update_vehicle,
    remove_vehicle,
    update_consent,
    is_valid_email
)
from plate_reader import read_plate_from_vehicle_crop
from notifications import send_violation_notice, is_duplicate_event, get_recent_audit_logs

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

# 1. A registered plate matches the correct email
def test_registered_plate_matches_correct_email(temp_db):
    # Test the seeded sample records
    rec1 = lookup_plate("SAMPLEPLATE1", db_path=temp_db)
    assert rec1 is not None
    assert rec1["email"] == "student1@example.invalid"

    rec2 = lookup_plate("SAMPLEPLATE2", db_path=temp_db)
    assert rec2 is not None
    assert rec2["email"] == "principal1@example.invalid"

    rec3 = lookup_plate("SAMPLEPLATE3", db_path=temp_db)
    assert rec3 is not None
    assert rec3["email"] == "visitor1@example.invalid"

# 2. A plate with spaces or punctuation matches its normalized record
def test_plate_with_spaces_or_punctuation_matches_normalized_record(temp_db):
    success, msg = register_vehicle("DL-01 AB.1234", "owner1@example.com", db_path=temp_db)
    assert success is True

    # Look up with diverse spaces, casing, and punctuation
    queries = ["dl 01 ab 1234", "DL01AB-1234", "  dl-01.ab.1234  ", "DL01AB1234"]
    for q in queries:
        record = lookup_plate(q, db_path=temp_db)
        assert record is not None, f"Query '{q}' failed to match normalized plate."
        assert record["plate_number"] == "DL01AB1234"
        assert record["email"] == "owner1@example.com"

# 3. An unknown plate cannot trigger an email
def test_unknown_plate_cannot_trigger_email(temp_db):
    record = lookup_plate("UNKNOWN999XYZ", db_path=temp_db)
    assert record is None

    # Pipeline logic check: unknown plate must not be eligible
    reading_status = "Recognized"
    eligible = (reading_status == "Recognized" and record is not None and record.get("consent_status") == 1)
    assert eligible is False

# 4. An unreadable plate cannot trigger an email
def test_unreadable_plate_cannot_trigger_email():
    blank_crop = np.zeros((100, 200, 3), dtype=np.uint8)
    res = read_plate_from_vehicle_crop(blank_crop)

    assert res["reading_status"] in ["Plate unreadable", "Setup Error: Plate weights missing"]
    assert res["requires_manual_review"] is True
    assert res["eligible_for_notification"] is False

# 5. A registration without consent cannot trigger an email
def test_registration_without_consent_cannot_trigger_email(temp_db):
    success, _ = register_vehicle("NOCONSENT01", "noconsent@example.com", consent_status=0, db_path=temp_db)
    assert success is True

    record = lookup_plate("NOCONSENT01", db_path=temp_db)
    assert record is not None
    assert record["consent_status"] == 0

    reading_status = "Recognized"
    eligible = (reading_status == "Recognized" and record is not None and record.get("consent_status") == 1)
    assert eligible is False

# 6. Clicking Send Email Notice sends the email to the correct registered recipient
def test_send_email_notice_sends_to_correct_registered_recipient(temp_db):
    mock_smtp = MagicMock()
    mock_smtp.send_message.return_value = {}

    event_id = "test_event_send_001"
    target_email = "student1@example.invalid"
    plate_number = "SAMPLEPLATE1"
    zone_name = "Zone A: Emergency Access"

    result = send_violation_notice(
        event_id=event_id,
        plate_number=plate_number,
        recipient_email=target_email,
        zone_name=zone_name,
        server_override=mock_smtp,
        db_path=temp_db
    )

    assert result["success"] is True
    assert result["status"] == "SENT"
    assert mock_smtp.send_message.called is True

    # Inspect the delivered message
    sent_msg = mock_smtp.send_message.call_args[0][0]
    assert sent_msg["To"] == target_email
    assert sent_msg["Subject"] == "Parking Notice — Vehicle Detected in a Restricted Zone"
    body_content = sent_msg.get_content()
    assert plate_number in body_content
    assert zone_name in body_content
    assert "Smart Parking Violation Detection" in body_content

# 7. Missing SMTP configuration produces a clear error
def test_missing_smtp_configuration_produces_clear_error(temp_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "SMTP_EMAIL", "")
    monkeypatch.setattr(config, "SMTP_APP_PASSWORD", "")
    monkeypatch.setattr(config, "SMTP_USERNAME", "")
    monkeypatch.setattr(config, "SMTP_PASSWORD", "")
    monkeypatch.setattr(config, "SMTP_SENDER_EMAIL", "")

    result = send_violation_notice(
        event_id="test_event_missing_cfg",
        plate_number="SAMPLEPLATE1",
        recipient_email="student1@example.invalid",
        zone_name="Zone A",
        server_override=None,
        db_path=temp_db
    )

    assert result["success"] is False
    assert result["status"] == "FAILED_CONFIG"
    assert "SMTP configuration is missing" in result["message"]

# 8. SMTP failure is recorded correctly
def test_smtp_failure_recorded_correctly(temp_db):
    mock_smtp = MagicMock()
    mock_smtp.send_message.side_effect = smtplib.SMTPException("Connection reset by remote host")

    event_id = "test_event_smtp_error_log"
    result = send_violation_notice(
        event_id=event_id,
        plate_number="SAMPLEPLATE1",
        recipient_email="student1@example.invalid",
        zone_name="Zone A",
        server_override=mock_smtp,
        db_path=temp_db
    )

    assert result["success"] is False
    assert result["status"] == "SMTP_ERROR"

    # Verify notification_history audit table in SQLite recorded failure
    conn = get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notification_history WHERE event_id = ?", (event_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["status"] == "FAILED"
    assert "Connection reset" in row["error_summary"]

# 9. Repeated clicks do not send duplicate notices for the same event (Idempotency)
def test_repeated_clicks_do_not_send_duplicate_notices(temp_db):
    mock_smtp = MagicMock()
    mock_smtp.send_message.return_value = {}

    event_id = "test_event_idempotency_check"
    # First delivery
    res1 = send_violation_notice(
        event_id=event_id,
        plate_number="SAMPLEPLATE2",
        recipient_email="principal1@example.invalid",
        zone_name="Zone B",
        server_override=mock_smtp,
        db_path=temp_db
    )
    assert res1["success"] is True
    assert res1["status"] == "SENT"
    assert mock_smtp.send_message.call_count == 1

    # Second delivery attempt with identical event_id
    res2 = send_violation_notice(
        event_id=event_id,
        plate_number="SAMPLEPLATE2",
        recipient_email="principal1@example.invalid",
        zone_name="Zone B",
        server_override=mock_smtp,
        db_path=temp_db
    )
    assert res2["success"] is False
    assert res2["status"] == "PREVENTED_DUPLICATE"
    # Mock SMTP must not have been invoked a second time
    assert mock_smtp.send_message.call_count == 1

# 10. Registrations remain available after restarting the application
def test_registrations_remain_available_after_restart(temp_db):
    # Add a custom vehicle
    success, _ = register_vehicle(
        plate_number="PERSIST-100",
        email="persist.test@example.com",
        consent_status=1,
        db_path=temp_db
    )
    assert success is True

    # Simulate restart by re-invoking init_db
    init_db(temp_db)

    # Verify persisted record is still present
    rec = lookup_plate("PERSIST100", db_path=temp_db)
    assert rec is not None
    assert rec["email"] == "persist.test@example.com"
    assert rec["consent_status"] == 1

    # Verify restart does not create duplicate sample records
    conn = get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM registry WHERE LOWER(email) = 'student1@example.invalid'")
    count = cursor.fetchone()["cnt"]
    conn.close()
    assert count == 1, "Restart created duplicate sample records!"
