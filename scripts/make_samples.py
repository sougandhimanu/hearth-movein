"""Generate the sample PDFs from their invented text versions (no real people or places).

    python scripts/make_samples.py

Writes samples/sample_lease.pdf and samples/sample_deductions.pdf.
The sample room photos are real photos taken by the author, not generated.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples"

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    raise SystemExit("reportlab not installed; run: python -m pip install reportlab")


def to_pdf(name: str) -> None:
    text = (OUT / f"{name}.txt").read_text().splitlines()
    c = canvas.Canvas(str(OUT / f"{name}.pdf"), pagesize=letter)
    y = 750
    for line in text:
        for chunk in [line[i : i + 95] for i in range(0, max(len(line), 1), 95)]:
            if y < 60:
                c.showPage()
                y = 750
            c.drawString(50, y, chunk)
            y -= 15
    c.save()
    print(f"wrote {name}.pdf")


for name in ("sample_lease", "sample_deductions"):
    to_pdf(name)
