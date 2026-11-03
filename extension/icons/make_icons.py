"""Vẽ icon ChordTrack: nền vuông bo góc đen bóng, ký hiệu hồng đỏ chuyển sắc nổi khối.

Ký hiệu là logo trong trang (src/ui/icons.js): hợp âm ba ghi trên khuông (ba đầu nốt chung một đuôi)
cạnh tam giác play, nghĩa là "hợp âm chạy theo video".

Chạy: <venv python> extension/icons/make_icons.py   -> icon.svg, icon16/32/48/128.png
PNG vẽ bằng Pillow ở 1024 px rồi thu nhỏ (venv không có cairosvg); SVG dùng cùng hình học.
"""
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
VIEW = 128
RADIUS = 29                                  # bo góc nền (đơn vị VIEW)
BG_TOP, BG_BOTTOM = (52, 52, 54), (18, 18, 19)
GLYPH_TOP, GLYPH_BOTTOM = (255, 64, 129), (232, 10, 44)

# Hình học ký hiệu trong lưới 24×24 (giống src/ui/icons.js, đuôi dày hơn chút cho icon lớn).
HEADS = [(7.4, 5.5), (7.4, 11.65), (7.4, 17.8)]
HEAD_RX, HEAD_RY, HEAD_ROT = 3.1, 2.15, -22
STEM = (9.6, 1.3, 1.85, 16.5)                # x, y, w, h
TRI = [(14.7, 7.8), (22.6, 12.3), (14.7, 16.8)]
TRI_ROUND = 1.0
# Đặt ký hiệu (bbox ~ x 4.4..22.6, y 1.5..19.6) vào giữa nền, chiếm ~60% chiều cao.
K = VIEW * 0.6 / 18.1
TX = VIEW / 2 - 13.5 * K
TY = VIEW / 2 - 10.75 * K + 1


def _hex(c):
    return "#%02x%02x%02x" % c


def P(x, y):
    return TX + x * K, TY + y * K


def ellipse_pts(cx, cy, n=72):
    a = math.radians(HEAD_ROT)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = HEAD_RX * math.cos(t), HEAD_RY * math.sin(t)
        pts.append(P(cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)))
    return pts


def rounded_poly(pts, r, n=10):
    """Đa giác lồi bo góc bán kính r (đơn vị lưới 24)."""
    out = []
    m = len(pts)
    for i in range(m):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % m]
        v1 = (p0[0] - p1[0], p0[1] - p1[1]); v2 = (p2[0] - p1[0], p2[1] - p1[1])
        l1, l2 = math.hypot(*v1), math.hypot(*v2)
        u1, u2 = (v1[0] / l1, v1[1] / l1), (v2[0] / l2, v2[1] / l2)
        ang = math.acos(max(-1, min(1, u1[0] * u2[0] + u1[1] * u2[1])))
        d = r / math.tan(ang / 2)
        a, b = (p1[0] + u1[0] * d, p1[1] + u1[1] * d), (p1[0] + u2[0] * d, p1[1] + u2[1] * d)
        # cung bậc hai từ a qua p1 tới b (gần đúng cung tròn, đủ mịn ở cỡ icon)
        for j in range(n + 1):
            t = j / n
            out.append(((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * p1[0] + t * t * b[0],
                        (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * p1[1] + t * t * b[1]))
    return out


def glyph_parts():
    """Các hình của ký hiệu, tọa độ VIEW: list (kind, data)."""
    parts = [("poly", ellipse_pts(cx, cy)) for cx, cy in HEADS]
    x, y, w, h = STEM
    parts.append(("rrect", (*P(x, y), *P(x + w, y + h), w / 2 * K)))
    parts.append(("poly", [P(*p) for p in rounded_poly(TRI, TRI_ROUND)]))
    return parts


def svg() -> str:
    shapes = []
    for kind, d in glyph_parts():
        if kind == "poly":
            shapes.append('<path d="M' + " L".join(f"{x:.2f} {y:.2f}" for x, y in d) + 'Z"/>')
        else:
            x0, y0, x1, y1, r = d
            shapes.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" rx="{r:.2f}"/>')
    g = "".join(shapes)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW} {VIEW}" width="{VIEW}" height="{VIEW}">'
        '<defs>'
        f'<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{_hex(BG_TOP)}"/>'
        f'<stop offset="1" stop-color="{_hex(BG_BOTTOM)}"/></linearGradient>'
        '<linearGradient id="rim" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".28"/>'
        '<stop offset=".5" stop-color="#fff" stop-opacity="0"/></linearGradient>'
        f'<linearGradient id="gl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{_hex(GLYPH_TOP)}"/>'
        f'<stop offset="1" stop-color="{_hex(GLYPH_BOTTOM)}"/></linearGradient>'
        '<filter id="sh" x="-20%" y="-20%" width="140%" height="150%">'
        '<feDropShadow dx="0" dy="2.6" stdDeviation="3" flood-color="#000" flood-opacity=".6"/></filter>'
        '</defs>'
        f'<rect width="{VIEW}" height="{VIEW}" rx="{RADIUS}" fill="url(#bg)"/>'
        f'<rect x=".5" y=".5" width="{VIEW - 1}" height="{VIEW - 1}" rx="{RADIUS - .5}" fill="none" stroke="url(#rim)"/>'
        f'<g fill="url(#gl)" filter="url(#sh)">{g}</g>'
        '</svg>\n')


def _vgrad(S, top, bottom):
    col = Image.new("RGB", (1, S))
    for y in range(S):
        t = y / (S - 1)
        col.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    return col.resize((S, S))


def render(size: int) -> Image.Image:
    S = 1024
    k = S / VIEW
    sc = lambda pts: [(x * k, y * k) for x, y in pts]

    # nền
    bg_mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(bg_mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=round(RADIUS * k), fill=255)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(_vgrad(S, BG_TOP, BG_BOTTOM), (0, 0), bg_mask)
    # viền sáng mảnh ở nửa trên (như cạnh kính)
    inner = bg_mask.filter(ImageFilter.MinFilter(2 * round(0.9 * k) + 1))
    rim = ImageChops.subtract(bg_mask, inner)
    fade = _vgrad(S, (60,) * 3, (0,) * 3).convert("L")
    rim = ImageChops.multiply(rim, fade)
    img.paste((255, 255, 255), (0, 0), rim)

    # mặt nạ ký hiệu
    gm = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(gm)
    for kind, dat in glyph_parts():
        if kind == "poly":
            d.polygon(sc(dat), fill=255)
        else:
            x0, y0, x1, y1, r = dat
            d.rounded_rectangle((x0 * k, y0 * k, x1 * k, y1 * k), radius=r * k, fill=255)

    # bóng đổ
    sh = gm.filter(ImageFilter.GaussianBlur(3 * k)).point(lambda v: v * 0.62)
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow.putalpha(ImageChops.offset(sh, 0, round(2.6 * k)))
    img = Image.alpha_composite(img, shadow)

    # thân ký hiệu: gradient hồng -> đỏ
    body = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    body.paste(_vgrad(S, GLYPH_TOP, GLYPH_BOTTOM), (0, 0), gm)
    # cạnh trên sáng, cạnh dưới tối (nổi khối)
    off = round(1.1 * k)
    hl = ImageChops.subtract(gm, ImageChops.offset(gm, 0, off)).filter(ImageFilter.GaussianBlur(0.5 * k))
    hl = ImageChops.multiply(hl, gm).point(lambda v: v * 0.45)
    lo = ImageChops.subtract(gm, ImageChops.offset(gm, 0, -off)).filter(ImageFilter.GaussianBlur(0.6 * k))
    lo = ImageChops.multiply(lo, gm).point(lambda v: v * 0.35)
    body.paste((255, 200, 220), (0, 0), hl)
    body.paste((120, 0, 20), (0, 0), lo)
    body.putalpha(gm)
    img = Image.alpha_composite(img, body)
    return img.resize((size, size), Image.LANCZOS)


def main():
    (HERE / "icon.svg").write_text(svg(), encoding="utf-8")
    for size in (16, 32, 48, 128):
        render(size).save(HERE / f"icon{size}.png", optimize=True)
    print("wrote icon.svg, icon{16,32,48,128}.png ->", HERE)


if __name__ == "__main__":
    main()
