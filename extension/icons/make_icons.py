"""Vẽ icon ChordTrack: nền vuông bo góc đen bóng, bàn phím piano cách điệu màu hồng đỏ nổi khối.

Bốn phím trắng tô hồng -> đỏ, cả bàn phím nghiêng như đang lướt; hai "phím đen" đổi thành trắng sáng, tách khỏi
phím trắng bằng khe hở. Cùng hình học với CT.icons.avatar / CT.icons.logo trong src/ui/icons.js.

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
BLACK_TOP, BLACK_BOTTOM = (255, 255, 255), (236, 226, 232)   # "phím đen" đổi thành trắng sáng

# Ký hiệu: bàn phím piano cách điệu trong lưới 24×24 (giống CT.icons.logo của src/ui/icons.js).
# Bốn phím trắng tô hồng đỏ, cả bàn phím nghiêng (skew) như đang lướt; phím đen đổi thành trắng sáng,
# có khe hở quanh để tách khỏi phím trắng.
WHITE_X = [3.0, 7.7, 12.4, 17.1]
WHITE_W, WHITE_Y0, WHITE_Y1, WHITE_R = 4.1, 3.5, 20.5, 1.5
BLACK_CX = [7.4, 12.1]
BLACK_W, BLACK_Y1, BLACK_R, BLACK_GAP = 3.0, 12.6, 1.1, 0.8
SKEW = math.radians(-11)                     # nghiêng sang phải phía trên
# Đặt ký hiệu vào giữa nền, chiếm ~62% chiều cao.
K = VIEW * 0.62 / 17
TX = VIEW / 2 - 12.1 * K
TY = VIEW / 2 - 12 * K


def _hex(c):
    return "#%02x%02x%02x" % c


def P(x, y):
    x = x - (y - 12) * math.tan(SKEW)
    return TX + x * K, TY + y * K


def rrect_pts(x0, y0, x1, y1, r, n=12):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for j in range(n + 1):
            a = math.radians(a0 + 90 * j / n)
            pts.append(P(cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def glyph_parts():
    """(keys, cuts, blacks): đa giác tọa độ VIEW. keys trừ cuts = phần hồng đỏ; blacks = phần trắng sáng."""
    keys = [rrect_pts(x, WHITE_Y0, x + WHITE_W, WHITE_Y1, WHITE_R) for x in WHITE_X]
    g = BLACK_GAP
    cuts = [rrect_pts(c - BLACK_W / 2 - g, WHITE_Y0 - 2, c + BLACK_W / 2 + g, BLACK_Y1 + g, BLACK_R + g) for c in BLACK_CX]
    blacks = [rrect_pts(c - BLACK_W / 2, WHITE_Y0, c + BLACK_W / 2, BLACK_Y1, BLACK_R) for c in BLACK_CX]
    return keys, cuts, blacks


def _path(pts):
    return "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + "Z"


def svg() -> str:
    keys, cuts, blacks = glyph_parts()
    cut = "".join(f'<path d="{_path(c)}" fill="#000"/>' for c in cuts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW} {VIEW}" width="{VIEW}" height="{VIEW}">'
        '<defs>'
        f'<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{_hex(BG_TOP)}"/>'
        f'<stop offset="1" stop-color="{_hex(BG_BOTTOM)}"/></linearGradient>'
        '<linearGradient id="rim" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".2"/>'
        '<stop offset=".5" stop-color="#fff" stop-opacity="0"/></linearGradient>'
        f'<linearGradient id="gl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{_hex(GLYPH_TOP)}"/>'
        f'<stop offset="1" stop-color="{_hex(GLYPH_BOTTOM)}"/></linearGradient>'
        f'<linearGradient id="bk" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{_hex(BLACK_TOP)}"/>'
        f'<stop offset="1" stop-color="{_hex(BLACK_BOTTOM)}"/></linearGradient>'
        f'<mask id="cut" maskUnits="userSpaceOnUse" x="0" y="0" width="{VIEW}" height="{VIEW}">'
        f'<rect width="{VIEW}" height="{VIEW}" fill="#fff"/>{cut}</mask>'
        '<filter id="sh" x="-20%" y="-20%" width="140%" height="150%">'
        '<feDropShadow dx="0" dy="2.6" stdDeviation="3" flood-color="#000" flood-opacity=".6"/></filter>'
        '</defs>'
        f'<rect width="{VIEW}" height="{VIEW}" rx="{RADIUS}" fill="url(#bg)"/>'
        f'<rect x=".5" y=".5" width="{VIEW - 1}" height="{VIEW - 1}" rx="{RADIUS - .5}" fill="none" stroke="url(#rim)"/>'
        '<g filter="url(#sh)">'
        '<g fill="url(#gl)" mask="url(#cut)">' + "".join(f'<path d="{_path(k)}"/>' for k in keys) + '</g>'
        '<g fill="url(#bk)">' + "".join(f'<path d="{_path(k)}"/>' for k in blacks) + '</g>'
        '</g></svg>\n')


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

    # mặt nạ: phím trắng (trừ khe quanh phím đen) và phím đen
    def poly_mask(polys):
        m = Image.new("L", (S, S), 0)
        dr = ImageDraw.Draw(m)
        for pts in polys:
            dr.polygon(sc(pts), fill=255)
        return m
    keys, cuts, blacks = glyph_parts()
    km = ImageChops.subtract(poly_mask(keys), poly_mask(cuts))
    bm = poly_mask(blacks)
    allm = ImageChops.lighter(km, bm)

    # bóng đổ chung
    sh = allm.filter(ImageFilter.GaussianBlur(3 * k)).point(lambda v: v * 0.62)
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow.putalpha(ImageChops.offset(sh, 0, round(2.6 * k)))
    img = Image.alpha_composite(img, shadow)

    def emboss(mask, top, bottom, hi, lo):
        """Tô gradient dọc, cạnh trên sáng, cạnh dưới tối (nổi khối)."""
        layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        layer.paste(_vgrad(S, top, bottom), (0, 0), mask)
        off = round(1.1 * k)
        h = ImageChops.subtract(mask, ImageChops.offset(mask, 0, off)).filter(ImageFilter.GaussianBlur(0.5 * k))
        h = ImageChops.multiply(h, mask).point(lambda v: v * 0.45)
        l = ImageChops.subtract(mask, ImageChops.offset(mask, 0, -off)).filter(ImageFilter.GaussianBlur(0.6 * k))
        l = ImageChops.multiply(l, mask).point(lambda v: v * 0.35)
        layer.paste(hi, (0, 0), h)
        layer.paste(lo, (0, 0), l)
        layer.putalpha(mask)
        return layer

    img = Image.alpha_composite(img, emboss(km, GLYPH_TOP, GLYPH_BOTTOM, (255, 200, 220), (120, 0, 20)))
    img = Image.alpha_composite(img, emboss(bm, BLACK_TOP, BLACK_BOTTOM, (255, 255, 255), (150, 130, 140)))
    return img.resize((size, size), Image.LANCZOS)


def main():
    (HERE / "icon.svg").write_text(svg(), encoding="utf-8")
    for size in (16, 32, 48, 128):
        render(size).save(HERE / f"icon{size}.png", optimize=True)
    print("wrote icon.svg, icon{16,32,48,128}.png ->", HERE)


if __name__ == "__main__":
    main()
