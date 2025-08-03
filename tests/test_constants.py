from dispersionrelations.constants import *

def test_energy_unit_conversion():
    assert GeV / MeV == 1e3
    assert MeV / eV == 1e6
    assert GeV / keV == 1e6
    assert TeV / GeV == 1e6


def test_rounding():
    a1, b1 = rounding_PDG(0.827, 0.367)
    assert a1 == 0.8 and b1 == 0.4
