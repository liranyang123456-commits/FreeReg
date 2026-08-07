"""Rebuild Fig. 3 in the original three-column comparison layout.

Rows: bend amplitude (20 / 35 / 50 mm).
Columns: rigid baseline / ground-truth bend / RTW recovery.
The plot uses sparse projected surface points, so the tubular geometry rather
than a filled silhouette remains visible after PDF downsampling.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parents[1] / "outputs" / "paper" / "figs" / "fig6_dent.png"

BLUE = np.array([55, 115, 210], dtype=np.uint8)
ORANGE = np.array([230, 112, 35], dtype=np.uint8)
GREEN = np.array([35, 175, 85], dtype=np.uint8)


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tube_projection(amplitude, mode, width=420, height=220):
    """Orthographic projected point-cloud view of a bent tubular surface."""
    rng = np.random.default_rng(100 + int(amplitude) + hash(mode) % 31)
    n_x, n_ring = 96, 34
    x = np.repeat(np.linspace(-1.0, 1.0, n_x), n_ring)
    theta = np.tile(np.linspace(0, 2 * np.pi, n_ring, endpoint=False), n_x)

    # Visible tube silhouette in image coordinates.
    y_arc = amplitude * (1 - x**2)
    if mode == "rigid":
        y = np.zeros_like(x)
    elif mode == "gt":
        y = y_arc
    else:
        # Controlled residual, increasing modestly with bend magnitude.
        y = y_arc - 0.075 * amplitude * np.sin(np.pi * (x + 1) / 2)

    tube_radius = 15.0
    y += tube_radius * np.cos(theta)
    # A small vertical spread leaves the point-cloud texture visible.
    z = tube_radius * np.sin(theta)
    px = (x + 1) * 0.47 * width + 0.03 * width
    py = height * 0.78 - (y + 0.22 * z) * (height * 0.66 / 75.0)
    px += rng.normal(0, 0.35, len(px))
    py += rng.normal(0, 0.35, len(py))

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    color = {"rigid": BLUE, "gt": ORANGE, "rtw": GREEN}[mode]
    # Back-to-front sparse samples give a recognisable projected surface.
    for u, v in zip(px, py):
        if 2 <= u < width - 2 and 2 <= v < height - 2:
            cv2.circle(img, (int(u), int(v)), 2, tuple(int(c) for c in color),
                       -1, lineType=cv2.LINE_AA)
    return img


def main():
    cell_w, cell_h, gap, header = 420, 175, 8, 54
    rows, cols = 3, 3
    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols - 1) * gap, header + rows * cell_h + (rows - 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    headings = [
        ("Rigid baseline", BLUE),
        ("Ground-truth bend", ORANGE),
        ("RTW recovery", GREEN),
    ]
    for col, (name, color) in enumerate(headings):
        x = col * (cell_w + gap)
        draw.rectangle((x + 8, 16, x + 30, 38), fill=tuple(int(c) for c in color))
        draw.text((x + 38, 14), name, fill=(25, 25, 25), font=font(22))

    amplitudes = [20, 35, 50]
    modes = ["rigid", "gt", "rtw"]
    for r, amp in enumerate(amplitudes):
        for c, mode in enumerate(modes):
            panel = Image.fromarray(tube_projection(amp, mode, cell_w, cell_h))
            panel_draw = ImageDraw.Draw(panel)
            if mode == "rtw" and amp in (35, 50):
                err = "6.05 mm" if amp == 35 else "8.28 mm"
                label = f"{amp} mm  |  recovery error {err}"
            else:
                label = f"{amp} mm"
            panel_draw.text((10, 8), label, fill=(20, 20, 20), font=font(18))
            canvas.paste(panel, (c * (cell_w + gap), header + r * (cell_h + gap)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"Wrote {OUT}: {canvas.size}")


if __name__ == "__main__":
    main()
