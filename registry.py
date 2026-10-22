import re
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

from database import get_connection, init_db

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

def normalize_plate(raw_text: Optional[str]) -> str:
    """
    Normalizes a license plate string:
    - Uppercase
    - Removes whitespace, dashes, dots, and non-alphanumeric punctuation.
    e.g. 'ka-01 ab 1234' -> 'KA01AB1234'
    """
    if not raw_text:
        return ""
    # Strip any characters that are not ASCII letters or digits
    return re.sub(r"[^A-Za-z0-9]", "", str(raw_text).strip().upper())

def is_valid_email(email: Optional[str]) -> bool:
    """Validates email format using standard RFC-compatible regex."""
    if not email:
        return False
    return bool(EMAIL_REGEX.match(email.strip()))

def lookup_plate(plate_number: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Looks up a vehicle in the registry by exact normalized plate match.
    Returns record dictionary or None if not found.
    """
    normalized = normalize_plate(plate_number)
    if not normalized:
        return None

    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, plate_number, vehicle_type, owner_name, email, consent_status, role, authorized_slots, notes, created_at
        FROM registry
        WHERE plate_number = ?
    """, (normalized,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None

def validate_registration(
    plate_number: str,
    email: str,
    consent_status: int = 1,
    require_consent: bool = False,
    exclude_id: Optional[int] = None,
    db_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Validates email format, consent requirement, and ensures normalized plate uniqueness in the database.
    """
    normalized = normalize_plate(plate_number)
    if not normalized:
        return False, "Vehicle plate number cannot be empty."
    if len(normalized) < 2:
        return False, f"Plate number '{normalized}' is too short."

    if not is_valid_email(email):
        return False, f"Invalid email address '{email}'. Please enter a valid email (e.g., user@example.com)."

    if require_consent and not consent_status:
        return False, "Registration cannot be completed without consent confirmation."

    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if exclude_id is not None:
        cursor.execute("SELECT id FROM registry WHERE plate_number = ? AND id != ?", (normalized, exclude_id))
    else:
        cursor.execute("SELECT id FROM registry WHERE plate_number = ?", (normalized,))

    existing = cursor.fetchone()
    conn.close()

    if existing:
        return False, f"Normalized plate number '{normalized}' is already registered to another record."

    return True, "Validation successful."

def add_vehicle(
    plate_number: str,
    email: str,
    owner_name: str = "Registered Owner",
    vehicle_type: str = "car",
    consent_status: int = 1,
    notes: str = "",
    db_path: Optional[Path] = None
) -> bool:
    """
    Adds a vehicle to the registry with validation.
    Returns True if successfully added, False otherwise.
    """
    success, _ = register_vehicle(
        plate_number=plate_number,
        email=email,
        owner_name=owner_name,
        vehicle_type=vehicle_type,
        consent_status=consent_status,
        notes=notes,
        db_path=db_path
    )
    return success

def register_vehicle(
    plate_number: str,
    email: str,
    owner_name: str = "Registered Owner",
    vehicle_type: str = "car",
    consent_status: int = 1,
    role: str = "general",
    authorized_slots: str = "general",
    notes: str = "",
    require_consent: bool = False,
    db_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Registers a new vehicle record after validation.
    Returns (success, message).
    """
    is_valid, msg = validate_registration(
        plate_number,
        email,
        consent_status=consent_status,
        require_consent=require_consent,
        exclude_id=None,
        db_path=db_path
    )
    if not is_valid:
        return False, msg

    normalized = normalize_plate(plate_number)
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()

    if isinstance(authorized_slots, (list, tuple, set)):
        slots_str = ",".join(str(s).strip() for s in authorized_slots)
    else:
        slots_str = str(authorized_slots).strip()

    try:
        cursor.execute("""
            INSERT INTO registry (plate_number, vehicle_type, owner_name, email, consent_status, role, authorized_slots, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (normalized, vehicle_type, owner_name, email.strip().lower(), int(consent_status), role, slots_str, notes, now))
        
        # Also add to parking_permissions table
        cursor.execute("""
            INSERT OR REPLACE INTO parking_permissions (plate_number, role, authorized_slots, permit_id, valid_until)
            VALUES (?, ?, ?, ?, ?)
        """, (normalized, role, slots_str, f"PERMIT-{normalized}", "2027-12-31"))

        conn.commit()
        return True, f"Vehicle '{normalized}' successfully registered with {email.strip().lower()}."
    except Exception as e:
        return False, f"Database error during registration: {e}"
    finally:
        conn.close()


def register_or_load_vehicle(
    plate_number: str,
    email: str,
    owner_name: str = "Registered Owner",
    vehicle_type: str = "car",
    consent_status: int = 1,
    notes: str = "",
    require_consent: bool = False,
    db_path: Optional[Path] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]], bool]:
    """
    Attempts to register a vehicle after validation.
    If the normalized plate is already registered in the database,
    it avoids inserting a duplicate and instead loads and returns the existing record.

    Returns:
        (success, message, record_dict_or_None, is_existing)
    """
    normalized = normalize_plate(plate_number)
    if not normalized:
        return False, "Vehicle plate number cannot be empty.", None, False
    if len(normalized) < 2:
        return False, f"Plate number '{normalized}' is too short.", None, False
    if not is_valid_email(email):
        return False, f"Invalid email address '{email}'. Please enter a valid email.", None, False
    if require_consent and not consent_status:
        return False, "Registration cannot be completed without consent confirmation.", None, False

    # Check if plate already registered in DB
    existing = lookup_plate(normalized, db_path=db_path)
    if existing:
        return True, f"Vehicle '{normalized}' is already registered with {existing['email']}. Loaded existing record.", existing, True

    # Register new vehicle
    success, msg = register_vehicle(
        plate_number=normalized,
        email=email,
        owner_name=owner_name,
        vehicle_type=vehicle_type,
        consent_status=consent_status,
        notes=notes,
        require_consent=require_consent,
        db_path=db_path
    )
    if success:
        new_rec = lookup_plate(normalized, db_path=db_path)
        return True, msg, new_rec, False
    return False, msg, None, False

def update_vehicle(
    vehicle_id: int,
    plate_number: str,
    email: str,
    consent_status: int = 1,
    owner_name: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Updates an existing vehicle registration (plate, email, consent status).
    Returns (success, message).
    """
    is_valid, msg = validate_registration(plate_number, email, exclude_id=vehicle_id, db_path=db_path)
    if not is_valid:
        return False, msg

    normalized = normalize_plate(plate_number)
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id, owner_name, vehicle_type, notes FROM registry WHERE id = ?", (vehicle_id,))
        current = cursor.fetchone()
        if not current:
            return False, f"Vehicle with ID {vehicle_id} not found."

        new_owner = owner_name if owner_name is not None else current["owner_name"]
        new_type = vehicle_type if vehicle_type is not None else current["vehicle_type"]
        new_notes = notes if notes is not None else current["notes"]

        cursor.execute("""
            UPDATE registry
            SET plate_number = ?,
                email = ?,
                consent_status = ?,
                owner_name = ?,
                vehicle_type = ?,
                notes = ?
            WHERE id = ?
        """, (normalized, email.strip().lower(), int(consent_status), new_owner, new_type, new_notes, vehicle_id))
        conn.commit()
        return True, f"Registration updated: '{normalized}' -> {email.strip().lower()} (Consent: {'Enabled' if consent_status else 'Disabled'})."
    except Exception as e:
        return False, f"Database error updating record: {e}"
    finally:
        conn.close()

def remove_vehicle(identifier: Union[int, str], db_path: Optional[Path] = None) -> bool:
    """Removes a vehicle from the registry by ID or normalized plate number."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        cursor.execute("DELETE FROM registry WHERE id = ?", (int(identifier),))
    else:
        normalized = normalize_plate(str(identifier))
        cursor.execute("DELETE FROM registry WHERE plate_number = ?", (normalized,))

    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def update_consent(identifier: Union[int, str], consent_status: int, db_path: Optional[Path] = None) -> bool:
    """Updates the consent status of a vehicle by ID or normalized plate."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        cursor.execute("UPDATE registry SET consent_status = ? WHERE id = ?", (int(consent_status), int(identifier)))
    else:
        normalized = normalize_plate(str(identifier))
        cursor.execute("UPDATE registry SET consent_status = ? WHERE plate_number = ?", (int(consent_status), normalized))

    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def get_all_vehicles(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns all registered vehicles ordered by ID ascending."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, plate_number, vehicle_type, owner_name, email, consent_status, role, authorized_slots, notes, created_at
        FROM registry
        ORDER BY id ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_vehicle_by_id(vehicle_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Returns a single vehicle record by ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, plate_number, vehicle_type, owner_name, email, consent_status, role, authorized_slots, notes, created_at
        FROM registry
        WHERE id = ?
    """, (vehicle_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None
