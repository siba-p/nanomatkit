# nanomatkit

**Materials-science tools for nanoparticle and simulation work: FCC nanoparticle builder with surface-site analysis, powder XRD of cubic crystals, and simulation-box composition.**

[![tests](https://github.com/siba-p/nanomatkit/actions/workflows/ci.yml/badge.svg)](https://github.com/siba-p/nanomatkit/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Each module also runs as an interactive web app, with nothing to install:

| module | web app | what it does |
|---|---|---|
| `nanomatkit.nanoparticle` | [FacetLab](https://siba-p.github.io/facetlab/) | Builds FCC metal nanoparticles (Au, Ag, Cu, Pt, Pd, Ni, Al, Ir, Rh) as cuboctahedra, truncated octahedra, octahedra, cubes, spheres or Wulff shapes. Classifies every surface atom ({111}, {100}, edge, vertex), computes generalized coordination numbers, estimates ligand coverage and writes XYZ. |
| `nanomatkit.xrd` | [Bragg](https://siba-p.github.io/bragg/) | Powder diffraction of cubic crystals (fcc, bcc, diamond, rock salt, CsCl) from structure factors with Cromer–Mann form factors, Lorentz–polarization and Debye–Waller factors. Adds Scherrer size and Williamson–Hall strain broadening. |
| `nanomatkit.box` | [BoxMaker](https://siba-p.github.io/boxmaker/) | Converts molarity, molality, mass fraction or mole fraction into molecule counts for a simulation box, with a consistent mass balance and the concentration actually reached after rounding. |

## Install

```bash
pip install "git+https://github.com/siba-p/nanomatkit.git"
```

## Command line

```bash
nanomatkit np --metal Au --shape truncated_octahedron -k 2 --grafting 4.6 -o Au201.xyz
nanomatkit xrd Au --size 8 --strain 0.1 -o au_pattern.csv
nanomatkit box --edge 4 --solute glycine:75.07:2.0:m --solute NaCl:58.44:0.15:M
```

```
Au truncated_octahedron (k=2): 201 atoms, 122 on the surface (dispersion 0.607)
  equivalent-sphere diameter 1.867 nm, largest dimension 2.112 nm, mass 39,590 g/mol
  {111} facet      56   mean GCN 7.00
  {100} facet       6   mean GCN 6.33
  edge             36   mean GCN 5.11
  vertex           24   mean GCN 4.25
```

## Python

```python
from nanomatkit import nanoparticle as npk, xrd, box

an = npk.analyse(npk.build("cuboctahedron", 3), "Pt")      # Pt147
print(an.n_surface, an.counts, an.gcn.mean())

for p in xrd.peaks("Au", wavelength=1.54056):
    print(p.hkl, round(p.two_theta, 2), round(p.intensity, 1))

r = box.counts(64.0, [box.Solute("NaCl", 58.44, 0.15, "M")])
print(r["counts"], r["achieved"])
```

## Validation (`pytest`)

| check | result |
|---|---|
| cuboctahedral magic numbers | 13, 55, 147, 309 |
| truncated-octahedral magic numbers | 38, 201, 586, 1289 (Au₂₀₁: 122 surface atoms, 79 core) |
| octahedra / cubes | 6, 19, 44, 85, 146 / 14, 63, 172 |
| generalized coordination on large particles | 7.5 on {111} and 6.67 on {100} terraces (infinite-surface values) |
| gold, Cu Kα | (111) 38.19°, (200) 44.39°, (220) 64.58°, (311) 77.58°; intensities 100 : 49 : 31 : 35 : 10 against 100 : 52 : 32 : 36 : 12 (reference pattern) |
| extinction rules | fcc mixed-parity and diamond (200)/(222) reflections absent |
| Scherrer | size → FWHM → size round trip exact |
| water box | 3 nm cube at 0.997 g/cm³ holds 900 molecules; molality and mole-fraction targets exact before rounding |

## Methods and references

- Shapes: FCC sites inside {111} planes |x|+|y|+|z| ≤ S and {100} planes max|xᵢ| ≤ M; Wulff ratio M/(S/√3) = γ₁₀₀/γ₁₁₁.
- Generalized coordination number: F. Calle-Vallejo *et al.*, *Science* **350**, 185 (2015).
- XRD: B. D. Cullity & S. R. Stock, *Elements of X-Ray Diffraction* (2001); form factors from *International Tables for Crystallography*, Vol. C, Table 6.1.1.4; G. K. Williamson & W. H. Hall, *Acta Metall.* **1**, 22 (1953).

## Citation

> S. Panigrahy, *nanomatkit: nanoparticle structure, powder XRD and simulation-box composition*, 2026. https://github.com/siba-p/nanomatkit

## License

MIT © Sibasankar Panigrahy
