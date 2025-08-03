import numpy as np
from dispersionrelations.utils import *

def test_sqrt_along_rhc():
    sample_points = np.linspace(0, 1e4, 10)
    np.testing.assert_allclose(np.sqrt(sample_points), sqrtRHC(sample_points))
    np.testing.assert_allclose(np.sqrt(sample_points), np.imag(sqrtLHC(sample_points)))