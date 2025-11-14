import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def save_image(base64_string: str) -> str:
    """Decode a base64 PNG string, write it into /generated, and return the relative path."""
    if not base64_string:
        raise ValueError("Missing base64 image data")

    # Strip optional data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(base64_string)
    with BytesIO(image_bytes) as buffer:
        image = Image.open(buffer)
        image = image.convert("RGBA")

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        filename = GENERATED_DIR / f"scene_{timestamp}.png"
        image.save(filename, format="PNG")

    return str(filename.relative_to(BASE_DIR))