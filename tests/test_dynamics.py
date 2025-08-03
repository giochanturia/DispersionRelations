from dispersionrelations.dynamics import *

def test_thresholds():
    M1 = 0.5
    M2 = 0.2
    STHR = (M1 + M2)**2
    assert vertex_VPP(STHR, M1, M2) == 0.0
    assert vertex_VVP(STHR, M1, M2) == 0.0