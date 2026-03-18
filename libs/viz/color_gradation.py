from typing import Tuple


SLAHMR_COLORS: tuple[tuple[int, int, int], ...] = (
    (50, 210, 250),
    (255, 93, 81),
    (253, 208, 242),
    (207, 250, 231),
    (241, 91, 181),
    (0, 187, 249),
    (0, 245, 212),
    (222, 158, 54),
    (254, 228, 64),
    (93, 169, 233),
    (155, 93, 229),
    (24, 242, 178),
    (252, 116, 186),
    (138, 225, 252),
    (178, 171, 242),
    (24, 231, 98),
    (232, 93, 117),
    (109, 157, 197),
    (166, 189, 219),
    (253, 146, 207),
    (167, 201, 87),
    (117, 68, 177),
    (255, 229, 50),
    (251, 202, 239),
    (58, 134, 255),
    (255, 0, 110),
    (251, 86, 7),
    (188, 51, 209),
    (122, 229, 130),
    (0, 48, 73),
    (214, 40, 40),
    (229, 179, 179),
    (0, 187, 249),
    (255, 190, 11),
    (204, 213, 174),
    (0, 245, 212),
    (255, 153, 200),
    (144, 251, 146),
    (189, 211, 147),
    (230, 0, 86),
    (0, 95, 57),
    (0, 174, 126),
    (255, 116, 163),
    (189, 198, 255),
    (90, 219, 255),
    (158, 0, 142),
    (255, 147, 126),
    (164, 36, 0),
    (0, 21, 68),
    (145, 208, 203),
    (95, 173, 78),
    (107, 104, 130),
    (0, 125, 181),
    (106, 130, 108),
    (252, 246, 189),
    (208, 244, 222),
    (169, 222, 249),
    (228, 193, 249),
    (122, 204, 174),
    (194, 140, 159),
    (0, 143, 156),
    (235, 0, 0),
    (255, 2, 157),
    (104, 61, 59),
    (150, 138, 232),
    (152, 255, 82),
    (167, 87, 64),
    (1, 255, 254),
    (255, 238, 232),
    (254, 137, 0),
    (1, 208, 255),
    (187, 136, 0),
    (165, 255, 210),
    (255, 166, 254),
    (119, 77, 0),
    (122, 71, 130),
    (38, 52, 0),
    (0, 71, 84),
    (67, 0, 44),
    (181, 0, 255),
    (255, 177, 103),
    (255, 219, 102),
    (126, 45, 210),
    (229, 111, 254),
    (222, 255, 116),
    (0, 255, 120),
    (0, 155, 255),
    (0, 100, 1),
    (0, 118, 255),
    (133, 169, 0),
    (0, 185, 23),
    (120, 130, 49),
    (0, 255, 198),
    (255, 110, 65),
    (232, 94, 190),
    (1, 0, 103),
    (149, 0, 58),
    (98, 14, 0),
    (0, 0, 0),
)

BERKELEY_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 75, 174),
    (231, 17, 95),
    (255, 194, 27),
    (129, 54, 199),
    (1, 137, 67),
    (159, 209, 255),
    (255, 207, 229),
    (255, 232, 141),
    (217, 206, 255),
    (179, 229, 154),
)

# ---------- sRGB <-> linear sRGB ----------
def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _clamp255(x: float) -> int:
    return int(round(max(0, min(255, x))))

# ---------- sRGB(0-255) <-> OKLab ----------
def rgb_to_oklab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    # 0-255 -> 0-1 sRGB
    sr = r / 255.0
    sg = g / 255.0
    sb = b / 255.0
    # gamma decode
    lr = _srgb_to_linear(sr)
    lg = _srgb_to_linear(sg)
    lb = _srgb_to_linear(sb)
    # linear RGB -> LMS
    l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb
    m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb
    s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb
    l_ = l ** (1/3)
    m_ = m ** (1/3)
    s_ = s ** (1/3)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return (L, a, b)

def oklab_to_rgb(L: float, a: float, b: float) -> Tuple[int, int, int]:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3
    lr = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    lg = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    lb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    # clamp in linear, then encode
    sr = _linear_to_srgb(_clamp01(lr))
    sg = _linear_to_srgb(_clamp01(lg))
    sb = _linear_to_srgb(_clamp01(lb))
    return (_clamp255(sr * 255), _clamp255(sg * 255), _clamp255(sb * 255))

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def gradient_oklab(start_rgb: Tuple[int,int,int],
                   end_rgb: Tuple[int,int,int],
                #    n: int) -> List[Tuple[int,int,int]]:
                   i: int, n: int) -> Tuple[int,int,int]:
    if n <= 1:
        t = 0.0
    else:
        t = i / (n - 1)
    L0, a0, b0 = rgb_to_oklab(*start_rgb)
    L1, a1, b1 = rgb_to_oklab(*end_rgb)
    L = _lerp(L0, L1, t)
    a = _lerp(a0, a1, t)
    b = _lerp(b0, b1, t)
    return oklab_to_rgb(L, a, b)

def make_oklab_shade_fn(base_rgb, limit=0.7, curve=1.0):
    L0, a0, b0 = rgb_to_oklab(*base_rgb)

    def shade(t):
        # clamp
        t = max(-1.0, min(1.0, float(t)))
        alpha = (abs(t) * limit) ** curve

        if t >= 0:  # lighten (to white: (1,0,0))
            L = (1 - alpha) * L0 + alpha * 1.0
            a = (1 - alpha) * a0  # keep hue, but naturally reduce chroma
            b = (1 - alpha) * b0
        else:       # darken (to black: (0,0,0))
            L = (1 - alpha) * L0
            a = (1 - alpha) * a0
            b = (1 - alpha) * b0

        return oklab_to_rgb(L, a, b)

    return shade