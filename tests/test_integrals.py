import numpy as np
from dispersionrelations.integrals import *

def test_gl_quadrature():
    np.testing.assert_almost_equal(integrate_gl(lambda x: x**2, 1, 2), 7/3)