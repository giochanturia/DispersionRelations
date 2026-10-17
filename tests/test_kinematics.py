from dispersionrelations.kinematics import *

def test_kallen_points():
    assert kallen(1, 1, 1) == -3
    assert kallen(2, 2, 2) == -12

def test_kallen_symmetry():
    ma2 = 1
    mb2 = 2
    mc2 = 3
    assert kallen(ma2, mb2, mc2) == kallen(ma2, mc2, mb2)
    assert kallen(ma2, mb2, mc2) == kallen(mb2, ma2, mc2)
    assert kallen(ma2, mb2, mc2) == kallen(mc2, mb2, ma2)
    assert kallen(ma2, mb2, mc2) == kallen(mb2, mc2, ma2)
    assert kallen(ma2, mb2, mc2) == kallen(mc2, ma2, mb2)

def test_kallen_kin_thresholds():
    ma2 = 1
    mb2 = 2
    s = (np.sqrt(ma2) + np.sqrt(mb2))**2
    np.testing.assert_allclose(kallen(s, ma2, mb2), 0, atol = 1e-10)
    s = (np.sqrt(ma2) - np.sqrt(mb2))**2
    np.testing.assert_allclose(kallen(s, ma2, mb2), 0, atol = 1e-10)


def test_kallen_shapes():
    svals = 10.0 * np.random.random_sample((3, 6, 4)) - 5.0
    m1vals = 2.0 * np.random.random_sample((3, 6, 4))
    m2vals = 3.0 * np.random.random_sample((3, 6, 4))

    np.testing.assert_allclose(np.shape(kallen(svals, m1vals, m2vals)), (3,6,4))

    m1vals = 2.0
    m2vals = 3.0

    np.testing.assert_allclose(np.shape(kallen(svals, m1vals, m2vals)), (3, 6, 4))


def test_alternative_form():
    def alt(s, m1, m2):
        return (s - (m1 + m2)**2) * (s - (m1 - m2)**2)

    svals = 10.0 * np.random.random_sample((3, 6, 4)) - 5.0
    m1vals = 2.0 * np.random.random_sample((3, 6, 4))
    m2vals = 3.0 * np.random.random_sample((3, 6, 4))

    np.testing.assert_allclose(kallen(svals, m1vals**2, m2vals**2), alt(svals, m1vals, m2vals))

    m1vals = 2.0
    m2vals = 3.0

    np.testing.assert_allclose(
        kallen(svals, m1vals**2, m2vals**2), alt(svals, m1vals, m2vals)
    )


def test_cms_kin_thresholds():
    ma2 = 2.0
    mb2 = 3.0

    s = (np.sqrt(ma2) + np.sqrt(mb2)) ** 2
    np.testing.assert_allclose(momentum_cms(s, ma2, mb2), 0, atol=1e-7)
    s = (np.sqrt(ma2) - np.sqrt(mb2)) ** 2
    np.testing.assert_allclose(momentum_cms(s, ma2, mb2), 0, atol=1e-7)


def test_cms_values():
    def alt(s, m1, m2):
        return np.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / s + 0j) / 2.

    svals = 10.0 * np.random.random_sample((3, 6, 4)) - 5.0
    m1vals = 2.0 * np.random.random_sample((3, 6, 4))
    m2vals = 3.0 * np.random.random_sample((3, 6, 4))

    np.testing.assert_allclose(
        momentum_cms(svals, m1vals**2, m2vals**2), alt(svals, m1vals, m2vals)
    )

    m1vals = 2.0
    m2vals = 3.0

    np.testing.assert_allclose(
        momentum_cms(svals, m1vals**2, m2vals**2), alt(svals, m1vals, m2vals)
    )


def test_phase_space_twobody_kin_thresholds():
    ma2 = 2.0
    mb2 = 3.0

    s = (np.sqrt(ma2) + np.sqrt(mb2)) ** 2
    np.testing.assert_allclose(phase_space_twobody(s, ma2, mb2), 0, atol=1e-7)
    s = (np.sqrt(ma2) - np.sqrt(mb2)) ** 2
    np.testing.assert_allclose(phase_space_twobody(s, ma2, mb2), 0, atol=1e-7)


def test_phase_space_twobody_values():
    def alt(s, m1, m2):
        return np.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / s**2 + 0j) / (16. * np.pi)

    svals = 10.0 * np.random.random_sample((3, 6, 4)) - 5.0
    m1vals = 2.0 * np.random.random_sample((3, 6, 4))
    m2vals = 3.0 * np.random.random_sample((3, 6, 4))

    np.testing.assert_allclose(
        phase_space_twobody(svals, m1vals**2, m2vals**2), alt(svals, m1vals, m2vals)
    )

    m1vals = 2.0
    m2vals = 3.0

    np.testing.assert_allclose(
        phase_space_twobody(svals, m1vals**2, m2vals**2), alt(svals, m1vals, m2vals)
    )
