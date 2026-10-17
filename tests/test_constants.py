import numpy as np
from dispersionrelations.constants import *

def test_energy_unit_conversion():
    assert GeV / MeV == 1e3
    assert MeV / eV == 1e6
    assert GeV / keV == 1e6
    assert TeV / GeV == 1e6
    assert 90 * degrees == np.pi / 2
