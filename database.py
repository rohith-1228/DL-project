import sqlite3
import datetime
import json
import re
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from config import DB_PATH, DATA_DIR, SEED_DATA_FILE

DEFAULT_SAMPLE_RECORDS = [
    {
        "plate_number": "SAMPLEPLATE1",
        "email": "student1@example.invalid",
        "consent_status": 1,
        "owner_name": "Sample Vehicle 1 (Editable Placeholder)",
        "vehicle_type": "car",
        "notes": "Testing Sample 1 - Placeholder plate"
    },
    {
        "plate_number": "SAMPLEPLATE2",
        "email": "principal1@example.invalid",
        "consent_status": 1,
        "owner_name": "Sample Vehicle 2 (Editable Placeholder)",
        "vehicle_type": "car",
        "notes": "Testing Sample 2 - Placeholder plate"
    },
    {
        "plate_number": "SAMPLEPLATE3",
        "email": "visitor1@example.invalid",
        "consent_status": 1,
        "owner_name": "Sample Vehicle 3 (Editable Placeholder)",
        "vehicle_type": "car",
        "notes": "Testing Sample 3 - Placeholder plate"
    }
]

def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory enabled."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # A bounded wait avoids transient SQLITE_BUSY errors when Streamlit sessions
    # submit feedback or notification events at the same time.
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def load_seed_vehicles() -> List[Dict[str, Any]]:
    """Loads seed records from JSON file if available, or returns default sample records."""
    if SEED_DATA_FILE.exists():
        try:
            with open(SEED_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            print(f"Notice: Could not load seed file ({e}), using default sample records.")
    return DEFAULT_SAMPLE_RECORDS

def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initializes the SQLite database with:
    1. Vehicle registry table (plate, email, consent status, date added).
    2. Notification history table (event_id, plate, zone, recipient, timestamp, status, error_summary).
    3. Audit log table (maintained for backward compatibility).
    4. Seeds required test sample records without creating duplicates.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")

    # Table 1: Vehicle Registry (with campus role and authorized slots)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            vehicle_type TEXT DEFAULT 'car',
            owner_name TEXT NOT NULL,
            email TEXT NOT NULL,
            consent_status INTEGER NOT NULL DEFAULT 1,
            role TEXT DEFAULT 'general',
            authorized_slots TEXT DEFAULT 'general',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)

    # Ensure role and authorized_slots columns exist if database was previously created
    cursor.execute("PRAGMA table_info(registry)")
    existing_cols = [row["name"] for row in cursor.fetchall()]
    if "role" not in existing_cols:
        cursor.execute("ALTER TABLE registry ADD COLUMN role TEXT DEFAULT 'general'")
    if "authorized_slots" not in existing_cols:
        cursor.execute("ALTER TABLE registry ADD COLUMN authorized_slots TEXT DEFAULT 'general'")

    # Table 2: Parking Permissions & Authorization Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_permissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL DEFAULT 'general',
            authorized_slots TEXT NOT NULL DEFAULT 'general',
            permit_id TEXT DEFAULT '',
            valid_until TEXT DEFAULT ''
        )
    """)

    # Table 3: Notification History (Persistent Audit & Idempotency Log)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notification_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            plate_number TEXT NOT NULL,
            zone_name TEXT NOT NULL,
            recipient_email TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            error_summary TEXT DEFAULT ''
        )
    """)

    # Append-only record of each dispatch outcome, including retries.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notification_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            plate_number TEXT NOT NULL,
            zone_name TEXT NOT NULL,
            recipient_email TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            error_summary TEXT DEFAULT ''
        )
    """)

    # Table 4: Audit log table (for backward compatibility)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            plate_number TEXT NOT NULL,
            zone_name TEXT NOT NULL,
            recipient_email TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT DEFAULT '',
            timestamp TEXT NOT NULL
        )
    """)

    # Human-reviewed model feedback. Image files are stored locally only when a
    # reviewer submits a vote, and are referenced here for later curation.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            image_path TEXT NOT NULL,
            vehicle_id INTEGER NOT NULL,
            category_key TEXT NOT NULL,
            predicted_status TEXT NOT NULL,
            feedback_vote TEXT NOT NULL CHECK (feedback_vote IN ('correct', 'wrong')),
            corrected_label TEXT NOT NULL,
            bbox_json TEXT NOT NULL DEFAULT '[]',
            context_json TEXT NOT NULL DEFAULT '{}',
            model_name TEXT NOT NULL DEFAULT 'unknown',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (analysis_id, vehicle_id, category_key)
        )
    """)

    now = datetime.datetime.now().isoformat()

    # Replace legacy real-looking addresses attached to built-in placeholder plates.
    demo_email_migrations = {
        "SAMPLEPLATE1": ("vivekpastor30@gmail.com", "student1@example.invalid"),
        "SAMPLEPLATE2": ("parvathi.veeravilli@gmail.com", "principal1@example.invalid"),
        "SAMPLEPLATE3": ("radhasunith.veeravilli@gmail.com", "visitor1@example.invalid"),
    }
    for plate, (old_email, demo_email) in demo_email_migrations.items():
        cursor.execute(
            "UPDATE registry SET email = ? WHERE plate_number = ? AND LOWER(email) = LOWER(?)",
            (demo_email, plate, old_email),
        )

    # Seed the sample records without creating duplicates
    seed_records = load_seed_vehicles()
    for rec in seed_records:
        raw_plate = rec.get("plate_number", "")
        clean_plate = re.sub(r"[^A-Za-z0-9]", "", str(raw_plate).strip().upper())
        rec_email = str(rec.get("email", "")).strip().lower()
        if not clean_plate or not rec_email:
            continue

        # Check if email already exists in registry
        cursor.execute("SELECT id FROM registry WHERE LOWER(email) = ?", (rec_email,))
        existing_email = cursor.fetchone()

        if not existing_email:
            # Check if plate already exists
            cursor.execute("SELECT id FROM registry WHERE plate_number = ?", (clean_plate,))
            existing_plate = cursor.fetchone()
            if not existing_plate:
                cursor.execute("""
                    INSERT INTO registry (plate_number, vehicle_type, owner_name, email, consent_status, role, authorized_slots, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    clean_plate,
                    rec.get("vehicle_type", "car"),
                    rec.get("owner_name", "Registered Owner"),
                    rec_email,
                    int(rec.get("consent_status", 1)),
                    rec.get("role", "student"),
                    rec.get("authorized_slots", "general,student"),
                    rec.get("notes", "Sample record with editable placeholder plate"),
                    now
                ))

    # Pre-populate campus test scene records with explicit authorization permissions
    test_scene_records = [
        ("P3RVP", "car", "Prof. Patel (Faculty)", "prof.patel.test@campus-demo.edu", 1, "faculty", "general,faculty,staff", "Authorized for faculty & staff parking bays", now),
        ("PZRVP", "car", "Prof. Patel (Faculty)", "prof.patel.ocr@campus-demo.edu", 1, "faculty", "general,faculty,staff", "OCR variant for faculty vehicle", now),
        ("CRAIG", "car", "Campus Security (Staff)", "security.patrol@campus-demo.edu", 1, "staff", "general,staff,emergency", "Authorized for staff & emergency bays", now),
        ("CORSI6K", "truck", "Delivery Van (Vendor)", "vendor.delivery@campus-demo.edu", 0, "vendor", "loading,delivery", "Authorized for loading dock only", now),
        ("SAMPLEPLATE1", "car", "Student Vehicle (Sample 1)", "student1@example.invalid", 1, "student", "general,student", "Authorized for student/general lots only", now),
        ("SAMPLEPLATE2", "car", "Principal Vehicle (Sample 2)", "principal1@example.invalid", 1, "principal", "general,faculty,staff,principal,reserved", "Authorized for principal & reserved slots", now),
        ("SAMPLEPLATE3", "car", "Visitor Vehicle (Sample 3)", "visitor1@example.invalid", 1, "visitor", "general,visitor", "Authorized for visitor parking only", now),
    ]
    for plate, vtype, owner, email, consent, role, auth_slots, notes, created in test_scene_records:
        cursor.execute("SELECT id FROM registry WHERE plate_number = ?", (plate,))
        row = cursor.fetchone()
        if not row:
            cursor.execute("""
                INSERT INTO registry (plate_number, vehicle_type, owner_name, email, consent_status, role, authorized_slots, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (plate, vtype, owner, email, consent, role, auth_slots, notes, created))
        else:
            # Update role and authorized_slots if not already populated
            cursor.execute("""
                UPDATE registry SET role = ?, authorized_slots = ? WHERE plate_number = ?
            """, (role, auth_slots, plate))

        # Synchronize parking_permissions table
        cursor.execute("SELECT id FROM parking_permissions WHERE plate_number = ?", (plate,))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO parking_permissions (plate_number, role, authorized_slots, permit_id, valid_until)
                VALUES (?, ?, ?, ?, ?)
            """, (plate, role, auth_slots, f"PERMIT-{plate}", "2027-12-31"))

    conn.commit()
    conn.close()


def save_prediction_feedback(
    *,
    analysis_id: str,
    image_rgb,
    vehicle_id: int,
    category_key: str,
    predicted_status: str,
    feedback_vote: str,
    corrected_label: str,
    bbox: List[int],
    context: Optional[Dict[str, Any]] = None,
    model_name: str = "unknown",
    db_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Persist one human vote plus its source image for later curated training."""
    if feedback_vote not in {"correct", "wrong"}:
        raise ValueError("feedback_vote must be 'correct' or 'wrong'")
    if not analysis_id or not category_key or not corrected_label:
        raise ValueError("analysis_id, category_key, and corrected_label are required")
    if image_rgb is None:
        raise ValueError("The source image is unavailable; feedback was not saved")

    import numpy as np
    from PIL import Image

    image_array = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("The feedback image must be an RGB image")
    image_hash = hashlib.sha256(image_array.tobytes()).hexdigest()
    feedback_dir = DATA_DIR / "feedback_images"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    image_path = feedback_dir / f"{image_hash}.jpg"
    if not image_path.exists():
        temp_path = feedback_dir / f"{image_hash}.jpg.tmp"
        Image.fromarray(image_array, mode="RGB").save(temp_path, format="JPEG", quality=92, optimize=True)
        temp_path.replace(image_path)

    now = datetime.datetime.now().isoformat(timespec="seconds")
    relative_image_path = image_path.relative_to(DATA_DIR.parent).as_posix()
    conn = get_connection(db_path)
    try:
        conn.execute("""
            INSERT INTO prediction_feedback (
                analysis_id, image_hash, image_path, vehicle_id, category_key,
                predicted_status, feedback_vote, corrected_label, bbox_json,
                context_json, model_name, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (analysis_id, vehicle_id, category_key) DO UPDATE SET
                image_hash=excluded.image_hash,
                image_path=excluded.image_path,
                predicted_status=excluded.predicted_status,
                feedback_vote=excluded.feedback_vote,
                corrected_label=excluded.corrected_label,
                bbox_json=excluded.bbox_json,
                context_json=excluded.context_json,
                model_name=excluded.model_name,
                updated_at=excluded.updated_at
        """, (
            analysis_id, image_hash, relative_image_path, int(vehicle_id), category_key,
            predicted_status, feedback_vote, corrected_label,
            json.dumps(bbox), json.dumps(context or {}, ensure_ascii=False),
            model_name, now, now,
        ))
        conn.commit()
    finally:
        conn.close()
    return {"image_hash": image_hash, "image_path": relative_image_path, "updated_at": now}


def get_prediction_feedback(
    analysis_id: str,
    db_path: Optional[Path] = None,
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Return saved votes for one analysis, keyed by vehicle and category."""
    if not analysis_id:
        return {}
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM prediction_feedback WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchall()
        return {(int(row["vehicle_id"]), row["category_key"]): dict(row) for row in rows}
    finally:
        conn.close()

def get_parking_authorization(plate_number: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves the campus parking authorization and role for a normalized plate.
    Returns dictionary with role, authorized_slots, and permit details, or None.
    """
    clean_plate = re.sub(r"[^A-Za-z0-9]", "", str(plate_number).strip().upper())
    if not clean_plate:
        return None

    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.plate_number, r.owner_name, r.email, r.role, r.authorized_slots,
               r.consent_status, p.valid_until
        FROM registry AS r
        LEFT JOIN parking_permissions AS p ON p.plate_number = r.plate_number
        WHERE r.plate_number = ?
    """, (clean_plate,))
    row = cursor.fetchone()
    conn.close()

    if row:
        rec = dict(row)
        rec["authorized_list"] = [s.strip().lower() for s in rec.get("authorized_slots", "general").split(",")]
        return rec
    return None

def check_slot_authorization(
    plate_number: str,
    slot_type: str,
    db_path: Optional[Path] = None
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Checks if a detected plate is authorized to park in a designated slot type
    (e.g. 'faculty', 'staff', 'principal', 'visitor', 'emergency').
    
    Returns:
    (status, explanation, record)
    status: 'Potential violation detected' | 'Not detected' | 'Review required'
    """
    clean_plate = re.sub(r"[^A-Za-z0-9]", "", str(plate_number).strip().upper())
    if not clean_plate or len(clean_plate) < 2:
        return "Review required", "Number plate could not be read or verified for slot authorization.", None

    auth = get_parking_authorization(clean_plate, db_path=db_path)
    if auth is None:
        return "Potential violation detected", f"Vehicle '{clean_plate}' is not registered in the campus authorization database.", None

    clean_slot = slot_type.strip().lower()
    authorized_list = auth.get("authorized_list", ["general"])

    # If slot is general or open, anyone is authorized
    if clean_slot in ["general", "open", "standard"]:
        return "Not detected", f"Vehicle '{clean_plate}' ({auth['role'].title()}) is authorized in general parking.", auth

    valid_until = (auth.get("valid_until") or "").strip()
    if not valid_until:
        return "Review required", f"No validity date is recorded for vehicle '{clean_plate}' in this reserved slot.", auth
    try:
        expiry_date = datetime.date.fromisoformat(valid_until[:10])
    except ValueError:
        return "Review required", f"Permit validity for vehicle '{clean_plate}' could not be verified.", auth
    if expiry_date < datetime.date.today():
        return "Potential violation detected", f"The parking permit for vehicle '{clean_plate}' expired on {expiry_date.isoformat()}.", auth

    # Check if vehicle has permission for this reserved slot type
    if clean_slot in authorized_list or "all" in authorized_list or auth.get("role") == "principal":
        return "Not detected", f"Vehicle '{clean_plate}' is authorized for {slot_type.title()} parking (Role: {auth['role'].title()}).", auth
    else:
        return (
            "Potential violation detected",
            f"Vehicle '{clean_plate}' has campus role '{auth['role'].title()}' and is not authorized for {slot_type.title()} reserved parking.",
            auth
        )

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)

