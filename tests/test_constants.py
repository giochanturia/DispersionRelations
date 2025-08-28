import numpy as np
from dispersionrelations.constants import *


def test_energy_unit_conversion():
    assert GeV / MeV == 1e3
    assert MeV / eV == 1e6
    assert GeV / keV == 1e6
    assert TeV / GeV == 1e6
    assert 90 * degrees == np.pi / 2


def test_scientific_notation():
    assert scientific_notation(np.pi) == "3.14"
    assert scientific_notation(np.pi, 10) == "3.1415926536"
    assert scientific_notation(13.4) == r"1.34 \times 10^{1}"
    assert scientific_notation(13.4, 10) == r"1.34 \times 10^{1}"
    assert scientific_notation(0.134) == r"1.34 \times 10^{-1}"
    assert scientific_notation(0.000134) == r"1.34 \times 10^{-4}"
    assert scientific_notation(13495) == r"1.35 \times 10^{4}"
    assert scientific_notation(13495, 20) == r"1.3495 \times 10^{4}"


def test_rounding():
    a1, b1 = rounding_PDG(0.827, 0.367)
    assert a1 == 0.8 and b1 == 0.4
    a2, b2 = rounding_PDG(0.827, 0.119)
    assert a2 == 0.83 and b2 == 0.12


def test_sR():
    M1 = 500
    G1 = 100
    s1 = sR(M1, G1)
    assert np.real(np.sqrt(s1)) == M1
    assert np.imag(np.sqrt(s1)) == -G1 / 2
    assert np.real(s1) == M1**2 - G1**2 / 4
    assert np.imag(s1) == -M1 * G1
