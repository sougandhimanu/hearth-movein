"""Generate synthetic sample files (no real people, places or documents).

    python scripts/make_samples.py

Creates samples/room_living.jpg, samples/room_bedroom.jpg and, if reportlab is installed, samples/sample_lease.pdf.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(exist_ok=True)


def room(name: str, wall, floor, stain_at, crack_from):
    w, h = 1200, 800
    img = Image.new("RGB", (w, h), wall)
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.72), w, h], fill=floor)  # floor
    d.rectangle([0, int(h * 0.70), w, int(h * 0.72)], fill=(235, 235, 230))  # baseboard
    sx, sy = stain_at
    blob = Image.new("L", (w, h), 0)
    ImageDraw.Draw(blob).ellipse([sx, sy, sx + 220, sy + 140], fill=255)
    blob = blob.filter(ImageFilter.GaussianBlur(14))
    img.paste(Image.new("RGB", (w, h), (150, 110, 60)), mask=blob.point(lambda v: int(v * 0.65)))  # water stain
    x, y = crack_from
    pts = [(x, y)]
    for i in range(1, 9):
        x += (-1) ** i * 12
        y += 38
        pts.append((x, y))
    d.line(pts, fill=(60, 55, 50), width=3)  # crack
    img.save(OUT / f"room_{name}.jpg", quality=90)


room("living", (232, 226, 214), (160, 120, 80), (140, 60), (820, 150))
room("bedroom", (214, 224, 232), (140, 130, 120), (760, 90), (300, 200))
print("wrote room_living.jpg, room_bedroom.jpg")

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    text = (OUT / "sample_lease.txt").read_text().splitlines()
    c = canvas.Canvas(str(OUT / "sample_lease.pdf"), pagesize=letter)
    y = 750
    for line in text:
        for chunk in [line[i : i + 95] for i in range(0, max(len(line), 1), 95)]:
            if y < 60:
                c.showPage()
                y = 750
            c.drawString(50, y, chunk)
            y -= 15
    c.save()
    print("wrote sample_lease.pdf")
except ImportError:
    print("reportlab not installed; skipped sample_lease.pdf")
