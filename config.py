import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")


def _configured_path(variable: str, default: Path) -> Path:
    """Resolve a path override relative to the project unless it is absolute."""
    value = os.getenv(variable, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path).resolve()

# Database configuration
DB_PATH = _configured_path("DB_PATH", DATA_DIR / "parking_system.db")

# Model configurations
VEHICLE_MODEL_PATH = _configured_path("VEHICLE_MODEL_PATH", MODELS_DIR / "yolov8n.pt")
OVERHEAD_VEHICLE_MODEL_PATH = _configured_path("OVERHEAD_VEHICLE_MODEL_PATH", MODELS_DIR / "yolov8n_visdrone.pt")
PLATE_MODEL_PATH = _configured_path("PLATE_MODEL_PATH", MODELS_DIR / "license_plate_detector.pt")
OFFSED_SURFACE_MODEL_PATH = _configured_path("OFFSED_SURFACE_MODEL_PATH", MODELS_DIR / "offsed_surface_segformer")
BDD100K_SIDEWALK_MODEL_PATH = _configured_path("BDD100K_SIDEWALK_MODEL_PATH", MODELS_DIR / "bdd100k_sidewalk_segformer")

# Supported vehicle classes in COCO (YOLOv8)
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Overhead / Aerial vehicle classes (VisDrone: 3: car, 4: van, 5: truck, 8: bus, 9: motor)
OVERHEAD_VEHICLE_CLASSES = {
    3: "car",
    4: "van",
    5: "truck",
    8: "bus",
    9: "motorcycle"
}

# Generic vehicle keyword mapping for dynamic class inspection
VEHICLE_CLASS_KEYWORDS = {
    "car": "car",
    "automobile": "car",
    "vehicle": "car",
    "sedan": "car",
    "suv": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "motor": "motorcycle",
    "bike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "pickup": "truck",
    "van": "van",
    "ocupied": "car",
    "occupied": "car",
}

# Detection confidence thresholds
DEFAULT_VEHICLE_CONF = 0.25
DEFAULT_PLATE_CONF = 0.25
DEFAULT_OCR_CONF = 0.30

# Predefined restricted zones (Normalized coordinates 0.0 - 1.0 for image-agnostic scaling)
# Calibrated specific zones (avoiding overly broad blanket coverage of the road)
DEFAULT_RESTRICTED_ZONES = [
    {
        "id": "zone_emergency_lane",
        "name": "Zone A: Emergency Access & Red Curb",
        "normalized_polygon": [
            [0.20, 0.55],
            [0.80, 0.55],
            [0.80, 0.98],
            [0.20, 0.98]
        ],
        "description": "Designated emergency clearance lane and red curb. Strictly no parking."
    },
    {
        "id": "zone_loading_dock",
        "name": "Zone B: Campus Delivery Bay",
        "normalized_polygon": [
            [0.65, 0.05],
            [0.98, 0.05],
            [0.98, 0.35],
            [0.65, 0.35]
        ],
        "description": "Upper bay reserved for campus maintenance and delivery vans."
    }
]

# Keywords indicating potential parking restrictions on road signage
RESTRICTED_SIGN_KEYWORDS = [
    "NO PARKING", "NO STOPPING", "TOW AWAY", "FIRE LANE",
    "RESTRICTED", "EMERGENCY", "BUS ONLY", "KEEP CLEAR", "HANDICAP",
    "FACULTY", "STAFF", "PRINCIPAL", "VISITOR", "RESERVED"
]

# Six Priority College Campus Parking Violation Categories
VIOLATION_CATEGORIES = {
    "cat1_unauthorized_zone": {
        "id": 1,
        "name": "Unauthorized Parking / No-Parking Zone",
        "short_name": "No-Parking Zone",
        "description": "Vehicle occupying red curb, fire lane, fire hydrant clearance, or designated no-parking zone."
    },
    "cat2_footpath_parking": {
        "id": 2,
        "name": "Footpath Parking",
        "short_name": "Footpath Parking",
        "description": "Car or motorcycle mounted on pedestrian sidewalk, walkway, or footpath."
    },
    "cat3_blocking_entrance": {
        "id": 3,
        "name": "Blocking Entrances and Emergency Access",
        "short_name": "Blocking Entrance",
        "description": "Vehicle obstructing college gates, building entrances, fire exits, or emergency access routes."
    },
    "cat4_lawn_parking": {
        "id": 4,
        "name": "Parking on Green Spaces",
        "short_name": "Lawn Parking",
        "description": "Vehicle occupying campus lawns, gardens, landscaped areas, or flowerbeds."
    },
    "cat5_improper_alignment": {
        "id": 5,
        "name": "Improper Parking Alignment",
        "short_name": "Improper Alignment",
        "description": "Vehicle straddling marked parking stall divider lines or occupying multiple spaces."
    },
    "cat6_wrong_parking_slot": {
        "id": 6,
        "name": "Wrong Parking Slot",
        "short_name": "Wrong Slot",
        "description": "Vehicle occupying designated faculty, staff, visitor, or principal reserved spaces without authorization."
    }
}



# Ground-contact estimation notice text
GROUND_CONTACT_EXPLANATION = (
    "The vehicle ground-contact location is estimated using the midpoint of the "
    "bottom edge of its bounding box ((x1 + x2)/2, y2). This 2D geometric estimate "
    "serves as an approximation of the tire-road interface and requires visual human review."
)

# Seed data configuration
SEED_DATA_FILE = DATA_DIR / "seed_vehicles.json"

# Location configuration
CONFIGURED_LOCATION = os.getenv("CONFIGURED_LOCATION", "Main Campus Parking Area")

# SMTP Configuration (supports SMTP_EMAIL and SMTP_APP_PASSWORD for Gmail, plus SMTP_USERNAME/SMTP_PASSWORD)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD", "")

# Fallback aliases for backward compatibility
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

# The app fails closed unless an access password is configured.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
AUTH_IDLE_TIMEOUT_MINUTES = int(os.getenv("AUTH_IDLE_TIMEOUT_MINUTES", "30"))

# Streamlit upload byte limit should be kept aligned with this setting.
MAX_IMAGE_UPLOAD_MB = int(os.getenv("MAX_IMAGE_UPLOAD_MB", "15"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))

if SMTP_PORT < 1 or SMTP_PORT > 65535:
    raise ValueError("SMTP_PORT must be between 1 and 65535")
if MAX_IMAGE_UPLOAD_MB < 1 or MAX_IMAGE_PIXELS < 1:
    raise ValueError("Upload size and pixel limits must be positive integers")
if AUTH_IDLE_TIMEOUT_MINUTES < 1:
    raise ValueError("AUTH_IDLE_TIMEOUT_MINUTES must be a positive integer")

# Notification cooldown / deduplication window in minutes
NOTIFICATION_COOLDOWN_MINUTES = 60
