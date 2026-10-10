"""FCC metal nanoparticles: construction, surface-site analysis and ligand coverage.

Lattice sites are stored in units of a/2 (integer vectors with an even component sum).
Shapes are intersections of {111} planes (|x|+|y|+|z| <= S) and {100} planes (max|x_i| <= M);
the Wulff construction sets M/S = (gamma_100/gamma_111)/sqrt(3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# lattice constant (Angstrom) and molar mass (g/mol) at room temperature
METALS = {
    "Au": (4.0782, 196.967), "Ag": (4.0862, 107.868), "Cu": (3.6149, 63.546), "Pt": (3.9242, 195.084),
    "Pd": (3.8907, 106.42), "Ni": (3.5240, 58.693), "Al": (4.0495, 26.982), "Ir": (3.8390, 192.217),
    "Rh": (3.8034, 102.906),
}
NA = 6.02214076e23
SHAPES = ("cuboctahedron", "truncated_octahedron", "octahedron", "cube", "sphere", "wulff")


def _sites(extent: int, hole: bool) -> np.ndarray:
    r = np.arange(-extent, extent + 1)
    P = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    P = P[(P.sum(1) % 2) == 0].astype(float)
    if hole:  # centre on an octahedral interstitial
        P = P - np.array([1.0, 0.0, 0.0])
    return P


def build(shape: str = "truncated_octahedron", k: int = 2, gamma_ratio: float = 2 / np.sqrt(3),
          radius: float | None = None) -> np.ndarray:
    """Return site vectors (units of a/2) of a nanoparticle.

    cuboctahedron(k): S = 2k, M = k (13, 55, 147, 309 ... atoms)
    truncated_octahedron(k): S = 3k, M = 2k (38, 201, 586, 1289 ...)
    octahedron(k): S = k (6, 19, 44, 85, 146 ...)
    cube(k): M = k (14, 63, 172 ...)
    sphere: |r| <= radius (units of a/2)
    wulff(k): S = 2k, M = gamma_ratio * S / sqrt(3)
    """
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    S = M = np.inf
    hole = False
    if shape == "cuboctahedron":
        S, M = 2 * k, k
    elif shape == "truncated_octahedron":
        S, M, hole = 3 * k, 2 * k, k % 2 == 1
    elif shape == "octahedron":
        S, hole = k, k % 2 == 1
    elif shape == "cube":
        M, hole = k, k % 2 == 1
    elif shape == "wulff":
        S = 2 * k
        M = gamma_ratio * S / np.sqrt(3)
    ext = int(np.ceil(min(S, 3 * M) if np.isfinite(min(S, 3 * M)) else 0)) + 3
    if shape == "sphere":
        R = float(radius if radius is not None else 2 * k)
        ext = int(np.ceil(R)) + 3
    P = _sites(ext, hole)
    if shape == "sphere":
        m = np.sqrt((P ** 2).sum(1)) <= R + 1e-9
    else:
        m = (np.abs(P).sum(1) <= S + 1e-9) & (np.abs(P).max(1) <= M + 1e-9)
    return P[m]


def neighbours(P: np.ndarray) -> list[np.ndarray]:
    """Nearest neighbours (|d|^2 = 2 in units of a/2)."""
    key = {tuple(np.round(p * 2).astype(int)): i for i, p in enumerate(P)}
    offs = np.array([(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                     (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)], float)
    out = []
    for p in P:
        idx = [key.get(tuple(np.round((p + o) * 2).astype(int))) for o in offs]
        out.append(np.array([j for j in idx if j is not None], int))
    return out


SITE_LABELS = {12: "bulk", 9: "{111} facet", 8: "{100} facet", 7: "edge"}


def site_class(cn: int) -> str:
    return SITE_LABELS.get(cn, "bulk" if cn > 9 else "vertex" if cn <= 6 else "edge")


@dataclass
class Analysis:
    metal: str
    shape: str
    positions: np.ndarray          # Angstrom
    cn: np.ndarray                 # coordination numbers
    gcn: np.ndarray                # generalized coordination numbers (Calle-Vallejo)
    classes: list[str]
    a: float
    n_atoms: int
    n_surface: int
    dispersion: float
    d_equivalent: float            # nm, sphere of equal volume
    d_max: float                   # nm, largest atom-centre distance + one atomic diameter
    mass: float                    # g/mol
    counts: dict = field(default_factory=dict)
    surface_area: float = 0.0      # nm^2, sum of per-site areas
    facet_fraction_111: float = 0.0


def analyse(P: np.ndarray, metal: str = "Au", shape: str = "") -> Analysis:
    a, molar = METALS[metal]
    nb = neighbours(P)
    cn = np.array([len(n) for n in nb])
    gcn = np.array([cn[n].sum() / 12.0 for n in nb])
    classes = [site_class(int(c)) for c in cn]
    pos = P * a / 2
    n = len(P)
    surf = cn < 12
    counts = {c: int(sum(1 for x in classes if x == c)) for c in ("{111} facet", "{100} facet", "edge", "vertex", "bulk")}
    a_nm = a / 10
    area = counts["{111} facet"] * np.sqrt(3) / 4 * a_nm ** 2 + counts["{100} facet"] * a_nm ** 2 / 2 + \
        (counts["edge"] + counts["vertex"]) * np.sqrt(3) / 4 * a_nm ** 2
    d_eq = (6 / np.pi * n * a_nm ** 3 / 4) ** (1 / 3)
    span = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)).max() if n < 4000 else 2 * np.sqrt((pos ** 2).sum(1)).max()
    d_max = (span + a / np.sqrt(2)) / 10
    f111 = counts["{111} facet"] / max(1, counts["{111} facet"] + counts["{100} facet"])
    return Analysis(metal, shape, pos, cn, gcn, classes, a, n, int(surf.sum()), float(surf.mean()), float(d_eq),
                    float(d_max), n * molar, counts, float(area), float(f111))


def ligands(an: Analysis, grafting_density: float) -> dict:
    """Ligand count for a grafting density (nm^-2) on the atomistic surface area."""
    nl = grafting_density * an.surface_area
    return {"ligands": nl, "per_surface_atom": nl / max(1, an.n_surface),
            "sphere_area_nm2": np.pi * an.d_equivalent ** 2, "atomistic_area_nm2": an.surface_area}


def to_xyz(an: Analysis, comment: str = "") -> str:
    lines = [str(an.n_atoms), comment or f"{an.metal} {an.shape} N={an.n_atoms} (nanomatkit)"]
    lines += [f"{an.metal} {x:12.5f} {y:12.5f} {z:12.5f}" for x, y, z in an.positions]
    return "\n".join(lines) + "\n"
