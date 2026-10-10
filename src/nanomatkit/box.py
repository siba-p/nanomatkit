"""Molecule counts for a simulation box at a target composition.

Solvent count follows from the solution density: rho_sol * V = N_w M_w / N_A + sum_s N_s M_s / N_A.
Concentrations may be given as molarity (mol/L), molality (mol/kg solvent), mass fraction, mole fraction
or an explicit count.
"""
from __future__ import annotations

from dataclasses import dataclass

NA = 6.02214076e23
SOLVENTS = {  # density g/cm^3 at 298 K, molar mass g/mol
    "water": (0.997, 18.015), "methanol": (0.7866, 32.042), "ethanol": (0.7849, 46.069),
    "DMSO": (1.0955, 78.13), "acetonitrile": (0.7766, 41.053), "chloroform": (1.4788, 119.38),
}


@dataclass
class Solute:
    name: str
    molar_mass: float           # g/mol
    value: float
    unit: str = "M"             # "M", "m", "wt%", "x", "count"
    charge: int = 0


def counts(volume_nm3: float, solutes: list[Solute], solvent: str = "water", density: float | None = None,
           solvent_molar_mass: float | None = None, round_counts: bool = True) -> dict:
    """Return integer (or exact) counts of solvent and each solute, and the achieved concentrations."""
    rho, Mw = SOLVENTS.get(solvent, (density or 1.0, solvent_molar_mass or 18.015))
    rho = density or rho
    Mw = solvent_molar_mass or Mw
    V_L = volume_nm3 * 1e-24           # litres
    mass_g = rho * volume_nm3 * 1e-21  # g in the box
    n = {}
    # fixed-count and molarity solutes are independent of the solvent amount
    for s in solutes:
        if s.unit == "count":
            n[s.name] = s.value
        elif s.unit == "M":
            n[s.name] = s.value * NA * V_L
    # molality / mass fraction / mole fraction depend on the solvent amount: solve for N_w
    fixed_mass = sum(n[s.name] * s.molar_mass for s in solutes if s.name in n) / NA
    molal = sum(s.value * s.molar_mass / 1000 for s in solutes if s.unit == "m")  # g solute per g solvent
    wt = sum(s.value / 100 for s in solutes if s.unit == "wt%")
    xs = [s for s in solutes if s.unit == "x"]
    # mass balance: mass = N_w Mw/NA (1 + molal) + fixed_mass, with wt% solutes taking a fraction of total mass
    solvent_mass = (mass_g * (1 - wt) - fixed_mass) / (1 + molal)
    if xs:
        # mole fractions relative to all species: N_s = x_s/(1 - sum x) * (N_w + N_other)
        xsum = sum(s.value for s in xs)
        Mx = sum(s.value * s.molar_mass for s in xs) / (1 - xsum)
        n_other = sum(n.values())
        # N_w Mw + N_x-weighted mass = available solvent mass (in molecules)
        Nw = (solvent_mass * NA - Mx * n_other) / (Mw + Mx)
        Nw = max(Nw, 0)
        for s in xs:
            n[s.name] = s.value / (1 - xsum) * (Nw + n_other)
    else:
        Nw = max(solvent_mass * NA / Mw, 0)
    for s in solutes:
        if s.unit == "m":
            n[s.name] = s.value * Nw * Mw / NA / 1000 * NA
        elif s.unit == "wt%":
            n[s.name] = s.value / 100 * mass_g * NA / s.molar_mass
    exact = {"solvent": Nw, **n}
    out = {k: (int(round(v)) if round_counts else v) for k, v in exact.items()}
    Nw_r = out["solvent"]
    achieved = {}
    for s in solutes:
        Ns = out[s.name]
        achieved[s.name] = {"M": Ns / NA / V_L, "m": Ns * 1000 / (Nw_r * Mw) if Nw_r else float("nan"),
                            "x": Ns / max(1, sum(out.values()))}
    charge = sum(s.charge * out[s.name] for s in solutes)
    return {"counts": out, "exact": exact, "achieved": achieved, "net_charge": charge,
            "box_edge_nm": volume_nm3 ** (1 / 3)}


def box_for_count(n_solute: float, molarity: float) -> float:
    """Cubic box edge (nm) that holds n_solute molecules at the given molarity."""
    V_L = n_solute / (molarity * NA)
    return (V_L * 1e24) ** (1 / 3)
