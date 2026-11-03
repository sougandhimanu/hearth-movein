"""Generate samples/sample_lease.pdf from samples/sample_lease.txt (an invented lease, no real people or places).

    python scripts/make_samples.py

The sample room photos are real photos taken by the author, not generated.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples"

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
