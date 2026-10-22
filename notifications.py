import smtplib
from email.message import EmailMessage
import datetime
import hmac
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List

import config
from database import get_connection, init_db


def _deployment_secret(name: str, default: str = "") -> str:
    """Read an environment variable or a Streamlit Community Cloud secret."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        value = str(st.secrets.get(name, default)).strip()
    except Exception:
        value = default
    return value


def _send_via_resend(api_key: str, from_email: str, recipient: str, subject: str, body: str) -> None:
    """Send email through Resend's HTTPS API (works on hosts blocking SMTP)."""
    payload = json.dumps({
        "from": from_email,
        "to": [recipient],
        "subject": subject,
        "text": body,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Resend returned HTTP {response.status}")

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

    # HTTPS email APIs are supported for hosted deployments where SMTP ports
    # are commonly blocked. A configured Resend API key takes precedence.
    resend_api_key = _deployment_secret("RESEND_API_KEY")
    resend_from_email = _deployment_secret("RESEND_FROM_EMAIL")

    # 2. Check SMTP configuration
    smtp_server = getattr(config, "SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(getattr(config, "SMTP_PORT", 587))
    sender_email = (
        _deployment_secret("SMTP_EMAIL", getattr(config, "SMTP_EMAIL", ""))
        or _deployment_secret("SMTP_USERNAME", getattr(config, "SMTP_USERNAME", ""))
        or _deployment_secret("SMTP_SENDER_EMAIL", getattr(config, "SMTP_SENDER_EMAIL", ""))
    )
    smtp_pass = (
        _deployment_secret("SMTP_APP_PASSWORD", getattr(config, "SMTP_APP_PASSWORD", ""))
        or _deployment_secret("SMTP_PASSWORD", getattr(config, "SMTP_PASSWORD", ""))
    )

    if not server_override and not resend_api_key and (not sender_email or not smtp_pass):
        error_msg = (
            "Email is not configured. For Streamlit Cloud, add RESEND_API_KEY and "
            "RESEND_FROM_EMAIL in App settings → Secrets (the sender must be verified "
            "with Resend). For local SMTP, set SMTP_EMAIL and SMTP_APP_PASSWORD."
        )
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
    msg["From"] = (resend_from_email or sender_email) if (resend_api_key or sender_email) else "smart-parking@system.local"
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
        elif resend_api_key:
            if not resend_from_email:
                raise ValueError("RESEND_FROM_EMAIL is required when using Resend and must be a verified sender.")
            _send_via_resend(
                resend_api_key,
                resend_from_email,
                recipient_email,
                str(msg["Subject"]),
                body_text,
            )
        else:
            if smtp_port == 465:
                smtp_client = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=20)
            else:
                smtp_client = smtplib.SMTP(smtp_server, smtp_port, timeout=20)
            with smtp_client as server:
                if smtp_port != 465 and config.SMTP_USE_TLS:
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

    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            detail = ""
        err = f"Email provider rejected the request (HTTP {e.code}). Check the API key and verified sender. {detail}".strip()
        log_notification_event(
            event_id=event_id,
            plate_number=plate_number,
            recipient_email=recipient_email,
            zone_name=zone_name,
            status="FAILED",
            error_summary=err,
            db_path=db_path
        )
        return {"success": False, "message": err, "status": "EMAIL_API_ERROR"}

    except Exception as e:
        if isinstance(e, smtplib.SMTPServerDisconnected):
            err = (
                "SMTP server closed the connection. Streamlit Cloud may block outbound SMTP; "
                "configure RESEND_API_KEY and RESEND_FROM_EMAIL in Streamlit Secrets, or "
                "try SMTP port 465 with implicit SSL."
            )
        else:
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
