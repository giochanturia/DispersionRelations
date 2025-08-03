from dispersionrelations.kinematics import *

def test_kallen_points():
    assert Kallen(1, 1, 1) == -3
    assert Kallen(2, 2, 2) == -12