from dispersionrelations.dynamics import *

def test_thresholds():
    M1 = 0.5
    M2 = 0.2
    STHR = (M1 + M2)**2
    assert vertex_VPP(STHR, M1, M2) == 0.0
    assert vertex_VVP(STHR, M1, M2) == 0.0


def test_sR():
    M1 = 500
    G1 = 100
    s1 = sR(M1, G1)
    assert np.real(np.sqrt(s1)) == M1
    assert np.imag(np.sqrt(s1)) == - G1 / 2
    assert np.real(s1) == M1**2 - G1**2 / 4
    assert np.imag(s1) == - M1 * G1