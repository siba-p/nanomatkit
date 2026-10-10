"""Powder X-ray diffraction of cubic crystals, with Scherrer and Williamson-Hall broadening.

I(hkl) = |F|^2 * multiplicity * LP(theta) * exp(-2 B s^2), with s = sin(theta)/lambda and
LP = (1 + cos^2 2theta) / (sin^2 theta cos theta). Atomic form factors use the Cromer-Mann
coefficients (International Tables for Crystallography, Vol. C, Table 6.1.1.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

# a1 b1 a2 b2 a3 b3 a4 b4 c
CROMER_MANN = {
    "Au": (16.8819, 0.4611, 18.5913, 8.6216, 25.5582, 1.4826, 5.86, 36.3956, 12.0658),
    "Ag": (19.2808, 0.6446, 16.6885, 7.4726, 4.8045, 24.6605, 1.0463, 99.8156, 5.179),
    "Cu": (13.338, 3.5828, 7.1676, 0.247, 5.6158, 11.3966, 1.6735, 64.8126, 1.191),
    "Pt": (27.0059, 1.51293, 17.7639, 8.81174, 15.7131, 0.424593, 5.7837, 38.6103, 11.6883),
    "Pd": (19.3319, 0.698655, 15.5017, 7.98929, 5.29537, 25.2052, 0.605844, 76.8986, 5.26593),
    "Ni": (12.8376, 3.8785, 7.292, 0.2565, 4.4438, 12.1763, 2.38, 66.3421, 1.0341),
    "Al": (6.4202, 3.0387, 1.9002, 0.7426, 1.5936, 31.5472, 1.9646, 85.0886, 1.1151),
    "Fe": (11.7695, 4.7611, 7.3573, 0.3072, 3.5222, 15.3535, 2.3045, 76.8805, 1.0369),
    "W": (29.0818, 1.72029, 15.43, 9.2259, 14.4327, 0.321703, 5.11982, 57.056, 9.8875),
    "Si": (6.2915, 2.4386, 3.0353, 32.3337, 1.9891, 0.6785, 1.541, 81.6937, 1.1407),
    "Na": (4.7626, 3.285, 3.1736, 8.8422, 1.2674, 0.3136, 1.1128, 129.424, 0.676),
    "Cl": (11.4604, 0.0104, 7.1962, 1.1662, 6.2556, 18.5194, 1.6455, 47.7784, -9.5574),
    "Mg": (5.4204, 2.8275, 2.1735, 79.2611, 1.2269, 0.3808, 2.3073, 7.1937, 0.8584),
    "O": (3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 0.867, 32.9089, 0.2508),
    "Cs": (20.3892, 3.569, 19.1062, 0.3107, 10.662, 24.3879, 1.4953, 213.904, 3.3352),
}
FCC = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
STRUCTURES = {
    "fcc": lambda A, B: [(A, p) for p in FCC],
    "bcc": lambda A, B: [(A, (0, 0, 0)), (A, (.5, .5, .5))],
    "sc": lambda A, B: [(A, (0, 0, 0))],
    "diamond": lambda A, B: [(A, p) for p in FCC] + [(A, tuple(c + .25 for c in p)) for p in FCC],
    "rocksalt": lambda A, B: [(A, p) for p in FCC] + [(B, ((p[0] + .5) % 1, p[1], p[2])) for p in FCC],
    "cscl": lambda A, B: [(A, (0, 0, 0)), (B, (.5, .5, .5))],
}
MATERIALS = {
    "Au": ("fcc", "Au", None, 4.0782), "Ag": ("fcc", "Ag", None, 4.0862), "Cu": ("fcc", "Cu", None, 3.6149),
    "Pt": ("fcc", "Pt", None, 3.9242), "Pd": ("fcc", "Pd", None, 3.8907), "Ni": ("fcc", "Ni", None, 3.5240),
    "Al": ("fcc", "Al", None, 4.0495), "Fe": ("bcc", "Fe", None, 2.8665), "W": ("bcc", "W", None, 3.1652),
    "Si": ("diamond", "Si", None, 5.4309), "NaCl": ("rocksalt", "Na", "Cl", 5.6402),
    "MgO": ("rocksalt", "Mg", "O", 4.2112), "CsCl": ("cscl", "Cs", "Cl", 4.123),
}
WAVELENGTHS = {"Cu Kα": 1.54056, "Co Kα": 1.78897, "Mo Kα": 0.70930, "Cr Kα": 2.28970, "Ag Kα": 0.55941}


def form_factor(el: str, s: np.ndarray | float) -> np.ndarray:
    a1, b1, a2, b2, a3, b3, a4, b4, c = CROMER_MANN[el]
    s2 = np.asarray(s) ** 2
    return a1 * np.exp(-b1 * s2) + a2 * np.exp(-b2 * s2) + a3 * np.exp(-b3 * s2) + a4 * np.exp(-b4 * s2) + c


def multiplicity(h: int, k: int, l: int) -> int:
    return len({tuple(sg * v for sg, v in zip(signs, perm)) for perm in permutations((h, k, l)) for signs in product((1, -1), repeat=3)})


@dataclass
class Peak:
    hkl: tuple
    d: float
    two_theta: float
    multiplicity: int
    F2: float
    intensity: float


def peaks(material: str = "Au", wavelength: float = 1.54056, a: float | None = None, tt_min: float = 20,
          tt_max: float = 100, B: float = 0.5) -> list[Peak]:
    struct, A, Bel, a0 = MATERIALS[material]
    a = a or a0
    basis = STRUCTURES[struct](A, Bel)
    out = []
    hmax = int(np.ceil(2 * a / wavelength)) + 1
    for h in range(hmax + 1):
        for k in range(h + 1):
            for l in range(k + 1):
                if h == k == l == 0:
                    continue
                d = a / np.sqrt(h * h + k * k + l * l)
                x = wavelength / (2 * d)
                if x >= 1:
                    continue
                th = np.arcsin(x)
                tt = np.degrees(2 * th)
                if not tt_min <= tt <= tt_max:
                    continue
                s = x / wavelength
                F = sum(form_factor(el, s) * np.exp(2j * np.pi * (h * p[0] + k * p[1] + l * p[2])) for el, p in basis)
                F2 = float(abs(F) ** 2)
                if F2 < 1e-6:
                    continue
                m = multiplicity(h, k, l)
                lp = (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))
                I = F2 * m * lp * np.exp(-2 * B * s * s)
                out.append(Peak((h, k, l), d, tt, m, F2, I))
    if out:
        imax = max(p.intensity for p in out)
        for p in out:
            p.intensity = 100 * p.intensity / imax
    return sorted(out, key=lambda p: p.two_theta)


def fwhm(two_theta: float, wavelength: float, size_nm: float | None, strain: float = 0.0, instrumental: float = 0.05,
         K: float = 0.9) -> float:
    """Peak FWHM in degrees 2theta: Scherrer size term + Williamson-Hall strain term + instrument, in quadrature."""
    th = np.radians(two_theta / 2)
    b_size = K * wavelength / (size_nm * 10 * np.cos(th)) if size_nm else 0.0
    b_strain = 4 * strain * np.tan(th)
    return float(np.degrees(np.hypot(b_size + b_strain, np.radians(instrumental))))


def scherrer_size(two_theta: float, fwhm_deg: float, wavelength: float, instrumental: float = 0.0, K: float = 0.9) -> float:
    """Crystallite size (nm) from an observed FWHM (degrees 2theta)."""
    b = np.sqrt(max(np.radians(fwhm_deg) ** 2 - np.radians(instrumental) ** 2, 1e-30))
    return float(K * wavelength / (b * np.cos(np.radians(two_theta / 2))) / 10)


def pattern(pk: list[Peak], wavelength: float, size_nm: float | None = 20, strain: float = 0.0,
            instrumental: float = 0.05, tt_min: float = 20, tt_max: float = 100, n: int = 4000, eta: float = 0.5):
    """Pseudo-Voigt powder profile (normalised to max 100)."""
    x = np.linspace(tt_min, tt_max, n)
    y = np.zeros_like(x)
    for p in pk:
        w = fwhm(p.two_theta, wavelength, size_nm, strain, instrumental)
        g = np.exp(-4 * np.log(2) * ((x - p.two_theta) / w) ** 2)
        lor = 1 / (1 + 4 * ((x - p.two_theta) / w) ** 2)
        y += p.intensity * (eta * lor + (1 - eta) * g)
    return x, 100 * y / y.max() if y.max() > 0 else y
