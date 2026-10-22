import smtplib
from email.message import EmailMessage
import datetime
import hmac
from pathlib import Path
from typing import Optional, Dict, Any, List

import config
from database import get_connection, init_db

def is_duplicate_event(event_id: str, db_path: Optional[Path] = None) -> bool:
    """
    Checks if an event with this ID has already been successfully notified.
    Uses persistent SQLite notification_history table for idempotency.
    """
    if not event_id:
        return False
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT status, timestamp FROM notification_history WHERE event_id = ?", (event_id,)).fetchone()
        if not row:
            return False
        return row["status"] == "SENT"
    finally:
        conn.close()

def log_notification_event(
    event_id: str,
    plate_number: str,
    recipient_email: str,
    zone_name: str,
    status: str,
    error_summary: str = "",
    db_path: Optional[Path] = None
) -> None:
    """
    Records the notification attempt in both notification_history and audit_log tables.
    Never logs credentials or private passwords.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()
    try:
        # Table 1: notification_history
        cursor.execute("""
            INSERT INTO notification_history (event_id, plate_number, zone_name, recipient_email, timestamp, status, error_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                plate_number=excluded.plate_number,
                zone_name=excluded.zone_name,
                recipient_email=excluded.recipient_email,
                timestamp=excluded.timestamp,
                status=excluded.status,
                error_summary=excluded.error_summary
        """, (event_id, plate_number, zone_name, recipient_email, now, status, error_summary))

        cursor.execute("""
            INSERT INTO notification_attempts (event_id, plate_number, zone_name, recipient_email, timestamp, status, error_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event_id, plate_number, zone_name, recipient_email, now, status, error_summary))

        # Table 2: audit_log (backward compatibility)
        cursor.execute("""
            INSERT INTO audit_log (event_id, plate_number, zone_name, recipient_email, status, details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                plate_number=excluded.plate_number,
                zone_name=excluded.zone_name,
                recipient_email=excluded.recipient_email,
                status=excluded.status,
                details=excluded.details,
                timestamp=excluded.timestamp
        """, (event_id, plate_number, zone_name, recipient_email, status, error_summary, now))

        conn.commit()
    except Exception as e:
        print(f"Audit log write notice: {e}")
    finally:
        conn.close()

def send_violation_notice(
    event_id: str,
    plate_number: str,
    recipient_email: str,
    zone_name: str,
    location: Optional[str] = None,
    detection_time: Optional[str] = None,
    server_override: Optional[smtplib.SMTP] = None,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Sends an advisory email notice after manual reviewer verification.
    
    Adheres to strict safety and idempotency rules:
    - Never sends automatically without manual trigger.
    - Idempotency: checks persistent notification_history table to prevent duplicate delivery.
    - Uses exact specified professional notice template.
    - Supports dependency injection of a mock SMTP server for automated testing.
    - Logs delivery status or error summary to SQLite.
    """
    if not event_id:
        return {
            "success": False,
            "message": "Missing event identifier.",
            "status": "INVALID_EVENT"
        }
    # Authorization is enforced here as well as in the UI: this function is the send boundary.
    from registry import lookup_plate, normalize_plate, is_valid_email
    normalized_plate = normalize_plate(plate_number)
    rec = lookup_plate(normalized_plate, db_path=db_path)
    if not rec:
        return {"success": False, "message": "No registered vehicle matches this plate.", "status": "FAILED_REGISTRATION"}
    if int(rec.get("consent_status", 0)) != 1:
        return {"success": False, "message": "The registered owner has not consented to parking notifications.", "status": "FAILED_CONSENT"}
    if not is_valid_email(recipient_email) or not hmac.compare_digest(
        str(rec.get("email", "")).strip().casefold(), str(recipient_email).strip().casefold()
    ):
        return {"success": False, "message": "Recipient must match the registered owner email.", "status": "FAILED_RECIPIENT"}

    # Claim the event atomically, so simultaneous requests cannot dispatch twice.
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status, timestamp FROM notification_history WHERE event_id = ?", (event_id,)).fetchone()
        if row and row["status"] == "SENT":
            conn.rollback()
            return {"success": False, "message": "Notice has already been sent for this event.", "status": "PREVENTED_DUPLICATE"}
        if row and row["status"] == "SENDING":
            try:
                started = datetime.datetime.fromisoformat(row["timestamp"])
                if datetime.datetime.now() - started < datetime.timedelta(minutes=15):
                    conn.rollback()
                    return {"success": False, "message": "Notice is already being dispatched for this event.", "status": "PREVENTED_DUPLICATE"}
            except ValueError:
                conn.rollback()
                return {"success": False, "message": "Notice dispatch state needs administrator review.", "status": "PREVENTED_DUPLICATE"}
        now = datetime.datetime.now().isoformat()
        conn.execute("""
            INSERT INTO notification_history (event_id, plate_number, zone_name, recipient_email, timestamp, status, error_summary)
            VALUES (?, ?, ?, ?, ?, 'SENDING', '')
            ON CONFLICT(event_id) DO UPDATE SET
                plate_number=excluded.plate_number, zone_name=excluded.zone_name,
                recipient_email=excluded.recipient_email, timestamp=excluded.timestamp,
                status='SENDING', error_summary=''
        """, (event_id, normalized_plate, zone_name, recipient_email, now))
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        return {"success": False, "message": "Could not reserve this notification event.", "status": "DB_ERROR"}
    finally:
        if conn:
            conn.close()

    # 2. Check SMTP configuration
    smtp_server = getattr(config, "SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(getattr(config, "SMTP_PORT", 587))
    sender_email = (
        getattr(config, "SMTP_EMAIL", "")
        or getattr(config, "SMTP_USERNAME", "")
        or getattr(config, "SMTP_SENDER_EMAIL", "")
    )
    smtp_pass = getattr(config, "SMTP_APP_PASSWORD", "") or getattr(config, "SMTP_PASSWORD", "")

    if not server_override and (not sender_email or not smtp_pass):
        error_msg = "SMTP configuration is missing. Please set SMTP_EMAIL and SMTP_APP_PASSWORD in your .env file."
        log_notification_event(
            event_id=event_id,
            plate_number=plate_number,
            recipient_email=recipient_email,
            zone_name=zone_name,
            status="FAILED_CONFIG",
            error_summary=error_msg,
            db_path=db_path
        )
        return {
            "success": False,
            "message": error_msg,
            "status": "FAILED_CONFIG"
        }

    # 3. Construct Email using required professional template
    loc_str = location or config.CONFIGURED_LOCATION
    time_str = detection_time or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = EmailMessage()
    msg["Subject"] = "Parking Notice — Vehicle Detected in a Restricted Zone"
    msg["From"] = sender_email if sender_email else "smart-parking@system.local"
    msg["To"] = recipient_email

    body_text = f"""Hello,

Our Smart Parking Violation Detection system detected your registered vehicle in a configured restricted parking zone.

Vehicle number: {plate_number}
Location: {loc_str}
Restricted zone: {zone_name}
Detection time: {time_str}

Please review this information and verify the situation.

This notification was generated by an academic prototype and may contain detection or recognition errors. It is not an official fine or legal determination.

If you believe this notification is incorrect, please contact the project administrator.

Regards,
Smart Parking Violation Detection System
"""
    msg.set_content(body_text)

    # 4. Dispatch via SMTP
    try:
        if server_override is not None:
            # Used for testing / mock verification
            server_override.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                if config.SMTP_USE_TLS:
                    server.starttls()
                server.login(sender_email, smtp_pass)
                server.send_message(msg)

        # Log success
        log_notification_event(
            event_id=event_id,
            plate_number=plate_number,
            recipient_email=recipient_email,
            zone_name=zone_name,
            status="SENT",
            error_summary="",
            db_path=db_path
        )
        return {
            "success": True,
            "message": f"Notice sent successfully to {recipient_email}.",
            "status": "SENT"
        }

    except smtplib.SMTPAuthenticationError:
        err = "SMTP authentication failed. Verify your SMTP_EMAIL and Gmail App Password."
        log_notification_event(
            event_id=event_id,
            plate_number=plate_number,
            recipient_email=recipient_email,
            zone_name=zone_name,
            status="FAILED",
            error_summary=err,
            db_path=db_path
        )
        return {"success": False, "message": err, "status": "AUTH_ERROR"}

    except Exception as e:
        err = f"Failed to send email: {type(e).__name__} ({e})"
        log_notification_event(
            event_id=event_id,
            plate_number=plate_number,
            recipient_email=recipient_email,
            zone_name=zone_name,
            status="FAILED",
            error_summary=str(e),
            db_path=db_path
        )
        return {"success": False, "message": err, "status": "SMTP_ERROR"}

def get_recent_audit_logs(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves recent notification history logs from SQLite."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, event_id, plate_number, zone_name, recipient_email, timestamp, status, error_summary
        FROM notification_attempts
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
