"""Command-line interface: ``nanomatkit np|xrd|box``."""
from __future__ import annotations

import argparse

from . import __version__
from . import box as bx
from . import nanoparticle as npk
from . import xrd


def cmd_np(a):
    P = npk.build(a.shape, a.k, a.gamma_ratio, a.radius)
    an = npk.analyse(P, a.metal, a.shape)
    print(f"{a.metal} {a.shape} (k={a.k}): {an.n_atoms} atoms, {an.n_surface} on the surface (dispersion {an.dispersion:.3f})")
    print(f"  equivalent-sphere diameter {an.d_equivalent:.3f} nm, largest dimension {an.d_max:.3f} nm, mass {an.mass:,.0f} g/mol")
    for c, n in an.counts.items():
        sel = [x == c for x in an.classes]
        g = an.gcn[sel].mean() if n else float("nan")
        print(f"  {c:12s} {n:6d}   mean GCN {g:.2f}")
    print(f"  atomistic surface area {an.surface_area:.2f} nm^2, {{111}} share of facet atoms {an.facet_fraction_111:.2f}")
    if a.grafting:
        L = npk.ligands(an, a.grafting)
        print(f"  ligands at {a.grafting} nm^-2: {L['ligands']:.1f} ({L['per_surface_atom']:.2f} per surface atom)")
    if a.output:
        with open(a.output, "w") as fh:
            fh.write(npk.to_xyz(an))
        print(f"wrote {a.output}")


def cmd_xrd(a):
    lam = xrd.WAVELENGTHS.get(a.source, None) or float(a.source)
    pk = xrd.peaks(a.material, lam, a.a, a.tt_min, a.tt_max, a.B)
    print(f"{a.material}, lambda = {lam} A")
    print(f"{'hkl':>8s}{'d (A)':>10s}{'2theta':>10s}{'m':>5s}{'I/I0':>8s}{'FWHM':>8s}")
    for p in pk:
        w = xrd.fwhm(p.two_theta, lam, a.size, a.strain / 100, a.instrumental)
        print(f"{''.join(map(str, p.hkl)):>8s}{p.d:10.4f}{p.two_theta:10.3f}{p.multiplicity:5d}{p.intensity:8.1f}{w:8.3f}")
    if a.output:
        x, y = xrd.pattern(pk, lam, a.size, a.strain / 100, a.instrumental, a.tt_min, a.tt_max)
        with open(a.output, "w") as fh:
            fh.write("two_theta,intensity\n" + "\n".join(f"{u:.4f},{v:.4f}" for u, v in zip(x, y)))
        print(f"wrote {a.output}")


def cmd_box(a):
    sol = []
    for spec in a.solute or []:
        name, M, val, unit, *q = spec.split(":")
        sol.append(bx.Solute(name, float(M), float(val), unit, int(q[0]) if q else 0))
    V = a.edge ** 3 if a.edge else a.volume
    r = bx.counts(V, sol, a.solvent, a.density)
    print(f"box {r['box_edge_nm']:.3f} nm (V = {V:.3f} nm^3), solvent {a.solvent}")
    for k, v in r["counts"].items():
        extra = ""
        if k in r["achieved"]:
            ac = r["achieved"][k]
            extra = f"   -> {ac['M']:.4f} M, {ac['m']:.4f} mol/kg, x = {ac['x']:.4f}"
        print(f"  {k:14s} {v:8d}{extra}")
    if r["net_charge"]:
        print(f"  net charge {r['net_charge']:+d}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nanomatkit", description="Nanoparticles, powder XRD and box composition.")
    ap.add_argument("--version", action="version", version=f"nanomatkit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("np", help="build and analyse an FCC nanoparticle")
    p.add_argument("--metal", default="Au", choices=list(npk.METALS))
    p.add_argument("--shape", default="truncated_octahedron", choices=npk.SHAPES)
    p.add_argument("-k", type=int, default=2, help="size index (see docs)")
    p.add_argument("--gamma-ratio", type=float, default=2 / 3 ** 0.5, help="gamma_100/gamma_111 for --shape wulff")
    p.add_argument("--radius", type=float, help="sphere radius in units of a/2")
    p.add_argument("--grafting", type=float, help="ligand grafting density, nm^-2")
    p.add_argument("-o", "--output", help="write XYZ file")
    p.set_defaults(func=cmd_np)
    p = sub.add_parser("xrd", help="powder XRD of a cubic crystal")
    p.add_argument("material", choices=list(xrd.MATERIALS))
    p.add_argument("--source", default="Cu Kα", help="tube (Cu Kα, Co Kα, Mo Kα, Cr Kα, Ag Kα) or wavelength in A")
    p.add_argument("-a", type=float, help="lattice parameter (A)")
    p.add_argument("--size", type=float, default=20, help="crystallite size (nm)")
    p.add_argument("--strain", type=float, default=0.0, help="microstrain (%%)")
    p.add_argument("--instrumental", type=float, default=0.05, help="instrumental FWHM (deg)")
    p.add_argument("-B", type=float, default=0.5, help="Debye-Waller B (A^2)")
    p.add_argument("--tt-min", type=float, default=20)
    p.add_argument("--tt-max", type=float, default=100)
    p.add_argument("-o", "--output", help="write the pattern as CSV")
    p.set_defaults(func=cmd_xrd)
    p = sub.add_parser("box", help="molecule counts for a simulation box")
    p.add_argument("--edge", type=float, help="cubic box edge (nm)")
    p.add_argument("--volume", type=float, help="box volume (nm^3)")
    p.add_argument("--solvent", default="water", choices=list(bx.SOLVENTS))
    p.add_argument("--density", type=float, help="solution density (g/cm^3); default: pure solvent")
    p.add_argument("--solute", action="append", help="name:molar_mass:value:unit[:charge], unit = M | m | wt%% | x | count")
    p.set_defaults(func=cmd_box)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":  # pragma: no cover
    main()
