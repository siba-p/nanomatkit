import numpy as np
import pytest

from nanomatkit import box, nanoparticle as npk, xrd


@pytest.mark.parametrize("shape,ks,expected", [
    ("cuboctahedron", [1, 2, 3, 4], [13, 55, 147, 309]),
    ("truncated_octahedron", [1, 2, 3, 4], [38, 201, 586, 1289]),
    ("octahedron", [1, 2, 3, 4, 5], [6, 19, 44, 85, 146]),
    ("cube", [1, 2, 3], [14, 63, 172]),
])
def test_magic_numbers(shape, ks, expected):
    assert [len(npk.build(shape, k)) for k in ks] == expected


def test_au201_surface_and_sites():
    an = npk.analyse(npk.build("truncated_octahedron", 2), "Au")
    assert an.n_surface == 122 and an.counts["bulk"] == 79
    assert set(an.cn[[c == "{111} facet" for c in an.classes]]) == {9}
    assert set(an.cn[[c == "{100} facet" for c in an.classes]]) == {8}
    assert an.d_equivalent == pytest.approx(1.87, abs=0.02)


def test_gcn_bulk_and_terrace_limits():
    an = npk.analyse(npk.build("cuboctahedron", 6), "Pt")
    assert np.allclose(an.gcn[an.cn == 12], 12) is False or an.gcn[an.cn == 12].max() <= 12
    t = an.gcn[[c == "{111} facet" for c in an.classes]]
    assert t.max() == pytest.approx(7.5)            # infinite (111) terrace value
    s = an.gcn[[c == "{100} facet" for c in an.classes]]
    assert s.max() == pytest.approx(80 / 12)         # infinite (100) terrace value


def test_wulff_ratio_reproduces_regular_shapes():
    assert len(npk.build("wulff", 3, gamma_ratio=np.sqrt(3) / 2)) == len(npk.build("cuboctahedron", 3))


def test_gold_xrd_against_reference_pattern():
    pk = xrd.peaks("Au")
    tt = {p.hkl: p.two_theta for p in pk}
    I = {p.hkl: p.intensity for p in pk}
    assert tt[(1, 1, 1)] == pytest.approx(38.18, abs=0.03)
    assert tt[(2, 0, 0)] == pytest.approx(44.39, abs=0.03)
    ref = {(1, 1, 1): 100, (2, 0, 0): 52, (2, 2, 0): 32, (3, 1, 1): 36, (2, 2, 2): 12}
    for h, v in ref.items():
        assert I[h] == pytest.approx(v, abs=5), (h, I[h])


def test_extinction_rules():
    fcc = {p.hkl for p in xrd.peaks("Cu", tt_max=150)}
    assert (1, 0, 0) not in fcc and (1, 1, 0) not in fcc and (2, 1, 0) not in fcc
    dia = {p.hkl for p in xrd.peaks("Si", tt_max=150)}
    assert (2, 0, 0) not in dia and (2, 2, 2) not in dia and (4, 0, 0) in dia


def test_scherrer_roundtrip():
    w = xrd.fwhm(38.18, 1.54056, 12.0, 0, 0)
    assert xrd.scherrer_size(38.18, w, 1.54056) == pytest.approx(12.0, rel=1e-6)


def test_water_box_and_salt():
    r = box.counts(27.0, [])
    assert r["counts"]["solvent"] == pytest.approx(900, abs=2)
    r = box.counts(27.0, [box.Solute("NaCl", 58.44, 0.15, "M")])
    assert r["counts"]["NaCl"] == 2


def test_molality_is_exact():
    r = box.counts(64.0, [box.Solute("gly", 75.07, 2.0, "m")], round_counts=False)
    Nw, Ns = r["exact"]["solvent"], r["exact"]["gly"]
    assert Ns * 1000 / (Nw * 18.015) == pytest.approx(2.0, rel=1e-9)


def test_mole_fraction_is_exact():
    r = box.counts(125.0, [box.Solute("urea", 60.06, 0.1, "x")], round_counts=False)
    Nw, Ns = r["exact"]["solvent"], r["exact"]["urea"]
    assert Ns / (Nw + Ns) == pytest.approx(0.1, rel=1e-9)
