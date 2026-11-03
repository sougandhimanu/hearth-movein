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
    for n, box in enumerate(boxes):  # all outlines first, so no outline is drawn across a label
        d.rectangle(box, outline=PALETTE[n % len(PALETTE)], width=width)
    placed: list[tuple[float, float, float, float]] = []
    tag_h = size + 8
    for n, (x1, y1, x2, y2) in enumerate(boxes):
        tag = f"{n + 1} {labels[n]}" if labels and _drawable(labels[n]) else f"{n + 1}"
        tw = d.textlength(tag, font=font) + 10
        left = max(0, min(x1, img.width - tw))  # keep the label inside the photo
        top = y1 - tag_h if y1 >= tag_h else y1  # above the box, or just inside it at the top edge
        # move down past any label already drawn there, so two labels never cover each other
        while any(left < r[2] and left + tw > r[0] and top < r[3] and top + tag_h > r[1] for r in placed):
            top += tag_h
        placed.append((left, top, left + tw, top + tag_h))
        d.rectangle([left, top, left + tw, top + tag_h], fill=PALETTE[n % len(PALETTE)])
        d.text((left + 5, top + 3), tag, fill=(255, 255, 255), font=font)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue()
