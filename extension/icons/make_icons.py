"""Vẽ icon ChordTrack (bàn phím 3 phím trắng, 2 phím đen trên nền tím xanh) ra SVG và PNG 16/32/48/128.

Chạy: <venv python> extension/icons/make_icons.py
Hình học dùng chung cho cả SVG (icon.svg) và PNG (vẽ bằng Pillow ở 512 px rồi thu nhỏ), vì venv không có
cairosvg. Ở 16 px bỏ chấm đỏ (quá nhỏ, chỉ thành nhiễu).
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
VIEW = 128
BG_TOP, BG_BOTTOM = (79, 70, 229), (124, 58, 237)       # indigo-600 -> violet-600
WHITE, GAP, BLACK, DOT = (255, 255, 255), (199, 201, 240), (30, 27, 75), (244, 63, 94)
RADIUS = 28
KEYS = (22, 42, 106, 102)                                 # x0, y0, x1, y1 của cả bàn phím
KEY_W = (KEYS[2] - KEYS[0]) / 3
BLACK_W, BLACK_H = 16, 36
DOT_C, DOT_R = (100, 25), 9


def _hex(c):
    return "#%02x%02x%02x" % c


def svg() -> str:
    x0, y0, x1, y1 = KEYS
    gaps = "".join(
        f'<rect x="{x0 + KEY_W * i - 1.5:.1f}" y="{y0}" width="3" height="{y1 - y0}" fill="{_hex(GAP)}"/>'
        for i in (1, 2))
    blacks = "".join(
        f'<rect x="{x0 + KEY_W * i - BLACK_W / 2:.1f}" y="{y0}" width="{BLACK_W}" height="{BLACK_H}" rx="3" '
        f'fill="{_hex(BLACK)}"/>' for i in (1, 2))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW} {VIEW}" width="{VIEW}" height="{VIEW}">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{_hex(BG_TOP)}"/><stop offset="1" stop-color="{_hex(BG_BOTTOM)}"/>'
        f'</linearGradient></defs>'
        f'<rect width="{VIEW}" height="{VIEW}" rx="{RADIUS}" fill="url(#g)"/>'
        f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="8" fill="{_hex(WHITE)}"/>'
        f'{gaps}{blacks}'
        f'<circle cx="{DOT_C[0]}" cy="{DOT_C[1]}" r="{DOT_R}" fill="{_hex(DOT)}"/>'
        f'</svg>\n')


def render(size: int, dot: bool) -> Image.Image:
    S = 512
    k = S / VIEW
    sc = lambda *v: [round(a * k) for a in v]
    # nền gradient chéo
    grad = Image.new("RGB", (S, S))
    px = grad.load()
    for y in range(S):
        for x in range(S):
            t = (x + y) / (2 * (S - 1))
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=round(RADIUS * k), fill=255)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = KEYS
    d.rounded_rectangle(sc(x0, y0, x1, y1), radius=round(8 * k), fill=WHITE)
    for i in (1, 2):
        cx = x0 + KEY_W * i
        d.rectangle(sc(cx - 1.5, y0, cx + 1.5, y1), fill=GAP)
    for i in (1, 2):
        cx = x0 + KEY_W * i
        d.rounded_rectangle(sc(cx - BLACK_W / 2, y0, cx + BLACK_W / 2, y0 + BLACK_H), radius=round(3 * k), fill=BLACK)
    if dot:
        cx, cy = DOT_C
        d.ellipse(sc(cx - DOT_R, cy - DOT_R, cx + DOT_R, cy + DOT_R), fill=DOT)
    return img.resize((size, size), Image.LANCZOS)


def main():
    (HERE / "icon.svg").write_text(svg(), encoding="utf-8")
    for size in (16, 32, 48, 128):
        render(size, dot=size >= 32).save(HERE / f"icon{size}.png", optimize=True)
    print("wrote icon.svg, icon{16,32,48,128}.png ->", HERE)


if __name__ == "__main__":
    main()
