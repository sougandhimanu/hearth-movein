"""Draw numbered outline boxes on a photo (used for the report and for the outline check)."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

PALETTE = [(220, 38, 38), (37, 99, 235), (217, 119, 6), (5, 150, 105), (147, 51, 234), (219, 39, 119)]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def _drawable(text: str) -> bool:
    """The bundled font only covers Latin letters, and Pillow here cannot shape scripts such as Telugu,
    so other labels are left off the image (the report lists them next to the box number)."""
    return all(ord(c) < 0x250 for c in text)


def draw_boxes(jpeg: bytes, boxes: list[list[int]], labels: list[str] | None = None) -> bytes:
    """boxes are [x1, y1, x2, y2] in the pixels of `jpeg`."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    d = ImageDraw.Draw(img)
    width = max(3, img.width // 250)
    size = max(14, img.width // 45)
    font = _font(size)
    for n, box in enumerate(boxes):
        color = PALETTE[n % len(PALETTE)]
        x1, y1, x2, y2 = box
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)
        tag = f"{n + 1} {labels[n]}" if labels and _drawable(labels[n]) else f"{n + 1}"
        tw = d.textlength(tag, font=font)
        top = max(0, y1 - size - 8)
        d.rectangle([x1, top, x1 + tw + 10, top + size + 8], fill=color)
        d.text((x1 + 5, top + 3), tag, fill=(255, 255, 255), font=font)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue()
