"""Combine native Prism exports and audited radar panels on the original AI artboards."""

from io import BytesIO
from pathlib import Path
import math
import numpy as np
import pandas as pd
from PIL import ImageDraw
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.utils import ImageReader

root = Path.cwd()
assets = root / "publication_artwork"
out = root / "figures"
values = pd.read_csv(root / "corrected_results/all_associations.csv")
order = [
    "Fish and seafood",
    "Starchy Roots",
    "Fruit",
    "Other meat",
    "Red meat",
    "Alcoholic Beverages",
    "Animal fats group",
    "Milk",
    "Eggs",
    "Sugar and Sweeteners",
    "Vegetable Oils",
    "Nuts",
    "Poultry meat",
    "Vegetables",
    "Pulses",
    "Cereals",
]
labels = {
    "Animal fats group": "Animal fats",
    "Alcoholic Beverages": "Alcoholic beverages",
    "Sugar and Sweeteners": "Sugar / sweeteners",
    "Starchy Roots": "Starchy roots",
    "Vegetable Oils": "Vegetable oils",
}
# Rectangles in original Illustrator artboard coordinates, measured from the upper left.
tiles = {
    "GBR": (150, 616.4, 297, 689.3),
    "SAU": (446, 690, 591, 762.1),
    "TUR": (296.9, 762.2, 444.8, 837.2),
}


def radar(c, cx, cy, rows, colour):
    radius = 65
    label_radius = 78
    vals = rows.loc[order, "r"]
    lo = min(-0.4, math.floor(vals.min() * 5) / 5)
    angles = np.arange(16) * 2 * np.pi / 16

    def point(angle, rr):
        return cx + rr * np.cos(angle), cy + rr * np.sin(angle)

    def radial(r):
        return radius * (r - lo) / (1 - lo)

    c.setStrokeColor(HexColor("#cbcbcb"))
    c.setLineWidth(0.27)
    for angle in angles:
        c.line(cx, cy, *point(angle, radius))
    ticks = np.arange(round(lo * 5), 6) / 5
    for tick in ticks:
        if radial(tick) > 0:
            c.circle(cx, cy, radial(tick), stroke=1, fill=0)
    path = c.beginPath()
    for j, (a, r) in enumerate(zip(angles, vals)):
        (path.moveTo if j == 0 else path.lineTo)(*point(a, radial(r)))
    path.close()
    c.setStrokeColor(HexColor(colour))
    col = HexColor(colour)
    c.setFillColor(Color(*[1 - 0.27 * (1 - v) for v in (col.red, col.green, col.blue)]))
    c.setLineWidth(0.45)
    c.drawPath(path, stroke=1, fill=1)
    c.setFillColor(HexColor("#686868"))
    c.setFont("Helvetica", 3.3)
    for tick in ticks[1:]:
        x, y = point(np.pi, radial(tick))
        c.drawCentredString(x, y + 0.65, f"{tick:.1f}" if abs(tick) > 1e-6 else "0")
    for a, name in zip(angles, order):
        x, y = point(a, label_radius)
        q = rows.loc[name, "q_BH"]
        c.setFillColor(HexColor("#222222"))
        c.setFont("Helvetica", 4.35)
        c.drawCentredString(x, y + 0.2, labels.get(name, name))
        c.setFont("Helvetica-Bold" if q < 0.05 else "Helvetica", 4.05)
        c.drawCentredString(x, y - 4.45, "q<0.001" if q < 0.001 else f"q={q:.3f}")


def place(page, source, rect):
    x1, y1, x2, y2 = rect
    h = float(page.mediabox.height)
    obj = PdfReader(source).pages[0]
    ow = float(obj.mediabox.width)
    oh = float(obj.mediabox.height)
    scale = min((x2 - x1) / ow, (y2 - y1) / oh)
    dx = x1 + ((x2 - x1) - ow * scale) / 2
    dy = h - y2 + ((y2 - y1) - oh * scale) / 2
    page.merge_transformed_page(obj, Transformation().scale(scale).translate(dx, dy))


for i, (ai, outcome, palette) in enumerate(
    [
        ("Non-HDL组图.ai", "Non-HDL", ("#a56ff6", "#4caf50")),
        ("Diabetes组图.ai", "Diabetes1992-2014", ("#e6aa08", "#d22d9c")),
        ("Hypertension.ai", "Hypertension", ("#e7564d", "#3187c2")),
    ],
    2,
):
    original = pdfium.PdfDocument(assets / ai)
    pg = original[0]
    width, height = pg.get_size()
    scale = 5
    split = 347
    im = pg.render(scale=scale).to_pil().convert("RGB")
    im = im.crop((0, split * scale, im.width, im.height))
    if i == 4:
        draw = ImageDraw.Draw(im)
        for x1, y1, x2, y2 in tiles.values():
            draw.rectangle(
                (
                    round(x1 * scale),
                    round((y1 - split) * scale),
                    round(x2 * scale),
                    round((y2 - split) * scale),
                ),
                fill="white",
            )
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=(width, height), invariant=1)
    c.drawImage(ImageReader(im), 0, 0, width, height - split)
    for y, label, spec, color in [
        (86, "a", "Pearson_158", palette[0]),
        (259, "c", "GDP_partial_143", palette[1]),
    ]:
        rows = values[
            (values.specification == spec) & (values.outcome == outcome)
        ].set_index("exposure")
        assert len(rows) == 16
        radar(c, 106, height - y, rows, color)
        c.setFillColor(HexColor("#000000"))
        c.setFont("Helvetica-Bold", 15)
        c.drawString(6, height - y + 73, label)
    c.drawString(192, height - 13, "b")
    c.showPage()
    c.save()
    stream.seek(0)
    writer = PdfWriter()
    page = writer.add_page(PdfReader(stream).pages[0])
    place(page, assets / f"Prism_{i}b.pdf", (211, 2, 592, 344))
    if i == 4:
        for code, box in tiles.items():
            place(page, assets / f"Prism_4d_{code}.pdf", box)
    dest = out / f"Figure_{i}.pdf"
    with dest.open("wb") as f:
        writer.write(f)
    image = pdfium.PdfDocument(dest)[0].render(scale=4.25).to_pil().convert("RGB")
    image.save(out / f"Figure_{i}.png", dpi=(306, 306))
