import pytest
# from scipy.optimize.elementwise import find_root
from scipy.optimize import fsolve
from dispersionrelations.dynamics import *
from dispersionrelations.integrals import SimplePoleMonomial


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_s_pole():
    M1 = 500
    G1 = 100
    s1 = s_pole(M1, G1)
    assert np.real(np.sqrt(s1)) == M1
    assert np.imag(np.sqrt(s1)) == -G1 / 2
    assert np.real(s1) == M1**2 - G1**2 / 4
    assert np.imag(s1) == -M1 * G1


def test_thresholds():
    M1 = 0.5
    M2 = 0.2

    # STHR:

    STHR = (M1 + M2)**2

    # squared:

    np.testing.assert_allclose(
        vertex_squared_vpp(STHR, M1**2.0, M2**2.0), 0.0, atol=1e-8
    )
    np.testing.assert_allclose(
        vertex_squared_vvp(STHR, M1**2.0, M2**2.0), 0.0, atol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_avp(STHR, M1**2.0, M2**2.0), STHR * M1**2., rtol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_vap(STHR, M1**2.0, M2**2.0), STHR * M1**2.0, rtol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_vap(STHR, M1**2.0, M2**2.0), STHR * M1**2.0, rtol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_vap_t1(STHR, M1**2.0, M2**2.0), STHR / 3.0, rtol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_vap_t2(STHR, M1**2.0, M2**2.0), 1. / 6., rtol=1e-8
    )


    # not squared:

    np.testing.assert_allclose(vertex_vpp(STHR, M1**2., M2**2.), 0.0, atol = 1e-8)
    np.testing.assert_allclose(vertex_vvp(STHR, M1**2.0, M2**2.0), 0.0, atol = 1e-8)
    np.testing.assert_allclose(
        vertex_avp(STHR, M1**2.0, M2**2.0), np.sqrt(STHR * M1**2.0), rtol=1e-8
    )

    np.testing.assert_allclose(
        vertex_vap(STHR, M1**2.0, M2**2.0), np.sqrt(STHR * M1**2.0), rtol=1e-8
    )


    # PTHR:

    PTHR = (M1 - M2) ** 2

    # squared:

    np.testing.assert_allclose(
        vertex_squared_vpp(PTHR, M1**2.0, M2**2.0), 0.0, atol=1e-8
    )
    np.testing.assert_allclose(
        vertex_squared_vvp(PTHR, M1**2.0, M2**2.0), 0.0, atol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_avp(PTHR, M1**2.0, M2**2.0), PTHR * M1**2.0, rtol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_vap(PTHR, M1**2.0, M2**2.0), PTHR * M1**2.0, rtol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_vap_t1(PTHR, M1**2.0, M2**2.0), PTHR / 3.0, rtol=1e-8
    )

    np.testing.assert_allclose(
        vertex_squared_vap_t2(PTHR, M1**2.0, M2**2.0), 1. / 6., rtol=1e-8
    )


    # not squared:

    np.testing.assert_allclose(vertex_vpp(PTHR, M1**2.0, M2**2.0), 0.0, atol=1e-8)
    np.testing.assert_allclose(vertex_vvp(PTHR, M1**2.0, M2**2.0), 0.0, atol=1e-8)
    np.testing.assert_allclose(
        vertex_avp(PTHR, M1**2.0, M2**2.0), np.sqrt(PTHR * M1**2.0), rtol=1e-8
    )
    np.testing.assert_allclose(
        vertex_vap(PTHR, M1**2.0, M2**2.0), np.sqrt(PTHR * M1**2.0), rtol=1e-8
    )


def test_BlWe_error():

    sthr = 0.5
    sb = 2.0
    norm_at_sb = False

    with pytest.raises(NotImplementedError):
        taming_blatt_weisskopf(3.0, sthr, sb, -1, norm_at_sb)

    with pytest.raises(NotImplementedError):
        taming_blatt_weisskopf(3.0, sthr, sb, 5, norm_at_sb)


def test_BlWe_0(rng):

    sthr = 0.5
    sb = 2.
    lang = 0
    norm_at_sb = False

    assert taming_blatt_weisskopf(3., sthr, sb, lang, norm_at_sb) == 1

    svals = 3. * rng.random((2,4,3))

    np.testing.assert_allclose(np.shape(taming_blatt_weisskopf(svals, sthr, sb, lang, norm_at_sb)), (2,4,3))

    # print(np.shape(taming_blatt_weisskopf(svals, sthr, sb, lang, norm_at_sb)))


def test_BlWe_not_norm():

    sthr = 0.5
    sb = 2.0
    langarr = [1, 2]
    norm_at_sb = False

    svals = np.linspace(50., 200., 200)

    result_sthr = [1, 1./9., -1./45.]
    result_sb = [0.5, 1 / 13.0]

    for i, lang in enumerate(langarr):
        np.testing.assert_allclose(taming_blatt_weisskopf(sthr, sthr, sb, lang, norm_at_sb), result_sthr[i])
        np.testing.assert_allclose(
            taming_blatt_weisskopf(sb, sthr, sb, lang, norm_at_sb), result_sb[i]
        )

        assert(np.all(np.diff(taming_blatt_weisskopf(svals, sthr, sb, lang, norm_at_sb)) < 0.))


def test_BlWe_not_norm():

    sthr = 0.5
    sb = 2.0
    langarr = [1, 2]
    norm_at_sb = True

    svals = np.linspace(1000.0, 5000.0, 200)

    result_sthr = [0, 0]
    result_sb = [1, 1]
    result_high_en = [2., 13.]

    for i, lang in enumerate(langarr):
        np.testing.assert_allclose(
            taming_blatt_weisskopf(sthr, sthr, sb, lang, norm_at_sb), result_sthr[i]
        )
        np.testing.assert_allclose(
            taming_blatt_weisskopf(sb, sthr, sb, lang, norm_at_sb), result_sb[i]
        )

        np.testing.assert_allclose(
            taming_blatt_weisskopf(5000, sthr, sb, lang, norm_at_sb), result_high_en[i], rtol = 1e-3
        )

        assert np.all(
            np.abs(np.diff(taming_blatt_weisskopf(svals, sthr, sb, lang, norm_at_sb)) / np.diff(svals)) < 1e-3
        )


def test_space_pole_polepos():
    sb = 4.
    narr = np.asarray([1, 3, 5])
    
    for n in narr:
        np.testing.assert_allclose(fsolve(lambda s: 1./taming_spacelike_pole(s, sb, n), (-1.)), [-sb])


def test_space_pole_normalisation():
    sb = 3.
    narr = np.linspace(1, 5, 5)
    svals = np.linspace(0.5, 5., 100)

    for n in narr:
        np.testing.assert_allclose(taming_spacelike_pole(0, sb, n), 1)
        np.testing.assert_allclose(taming_spacelike_pole(sb, sb, n), 0.5**n)
        assert np.all(np.diff(taming_spacelike_pole(svals, sb, n)) < 0)


def test_breit_wigner_normalisation():
    mass = 2.
    width = 0.5

    assert breit_wigner(0., mass, width) == 1./(-mass**2 + 1j * mass * width)


def test_breit_wigner_pole_position():
    mass = 2.
    width = 0.5

    def func(z, mass, width):
        x, y = z

        val = np.abs(1./breit_wigner(x + 1j * y, mass, width))

        return [np.real(val), np.imag(val)]

    np.testing.assert_allclose(fsolve(lambda z: func(z, mass, width), (1.,1.)), [mass**2., -mass*width])


def test_breit_wigner_high_en():
    mass = 2.
    width = 0.5

    svals = np.linspace(4.5, 10., 100)

    assert(np.all(np.diff(np.abs(breit_wigner(svals, mass, width))) < 0))

    # assert np.all(np.diff(np.abs(breit_wigner(-svals, mass, width))) < 0)


def test_breit_wigner_ed_const_polePos():
    mass = 2.0
    width = 0.5

    def Gs(s, m, gamma):
        return gamma

    def func(z, mass, width):
        x, y = z

        val = np.abs(1.0 / breit_wigner_energy_dependent(x + 1j * y, mass, Gs, width))

        return [np.real(val), np.imag(val)]

    np.testing.assert_allclose(
        fsolve(lambda z: func(z, mass, width), (1.0, 1.0)), [mass**2.0, -mass * width]
    )


def test_breit_wigner_ed_const():
    mass = 2.0
    width = 0.5

    def Gs(s, m, gamma):
        return gamma


    svals = np.linspace(0.1, 10., 200)

    np.testing.assert_allclose(breit_wigner_energy_dependent(svals, mass, Gs, width), breit_wigner(svals, mass, width))


def test_breit_wigner_ed_narrow_width_approximation():
    pass


def test_VP_normalisation_squared():
    rw = [0.8, 0.7, 0.0, 12.]
    mV = [1.2, 0.77, 1.2, 0.8]
    mP = [0.139, 0.445, 0.6, 0.135]

    sols = [1585.5931372986518, 17070.145143907295, 0.0, 84061.54223986717]

    for i in range(len(rw)):
        np.testing.assert_allclose(vector_pseudoscalar_normalisation_squared(rw[i], mV[i], mP[i]), sols[i])


def test_VP_normalisation_squared_error():
    rw = [0.8, 0.7, 0.0, 12.0]
    mV = [1.2, 0.77, 1.2, 0.8]
    mP = [0.139, 0.445, 0.6, 0.135]

    Drw = [0.1, 0.075, 0.05, 0.01]
    DmV = [0.2, 0.05, 0.3, 0.25]
    DmP = [0.01, 0.02, 0.1, 0.012]

    sols = [201.0830908650855, 1852.3790957791805, 225.5736625514402, 3477.2868546288755]

    for i in range(len(rw)):
        np.testing.assert_allclose(
            vector_pseudoscalar_normalisation_squared_error(rw[i], mV[i], mP[i], Drw[i], DmV[i], DmP[i]), sols[i]
        )


def prepare_Omnes_tests():

    threshold = 1.0
    cutoff = 2.0
    asymptotic = np.pi
    infinity = 1e6

    def integrand(s):
        s = np.asarray(s)
        results = np.zeros_like(s)

        mask0 = s <= threshold
        mask1 = (s > threshold) * (s < cutoff)
        mask2 = s >= cutoff

        results[mask1] = asymptotic * (
            3.0 * (s[mask1] - threshold) ** 2.0 / (cutoff - threshold) ** 2.0
            - 2.0 * (s[mask1] - threshold) ** 3.0 / (cutoff - threshold) ** 3.0
        )
        results[mask2] = asymptotic

        return results

    DI = OmnesFunction(
        integrand,
        threshold,
        infinity=infinity,
        integration_split_points=[
            1.0,
            2.0,
            2.5,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            25.0,
            50.0,
            75.0,
            100.0,
        ],
        integration_order=600,
    )

    # prepare solutions:
    SPMm1 = SimplePoleMonomial(-1)
    SPM0 = SimplePoleMonomial(0)
    SPM1 = SimplePoleMonomial(1)
    SPM2 = SimplePoleMonomial(2)

    def solution(s):
        s = s + 0j
        return np.exp(
            s
            * asymptotic
            / np.pi
            * (
                3.0
                / (cutoff - threshold) ** 2.0
                * (
                    SPM1(s, 0.0, threshold, cutoff)
                    - 2.0 * threshold * SPM0(s, 0.0, threshold, cutoff)
                    + threshold**2.0 * SPMm1(s, 0.0, threshold, cutoff)
                )
                - 2.0
                / (cutoff - threshold) ** 3.0
                * (
                    SPM2(s, 0.0, threshold, cutoff)
                    - 3.0 * threshold * SPM1(s, 0.0, threshold, cutoff)
                    + 3.0 * threshold**2.0 * SPM0(s, 0.0, threshold, cutoff)
                    - threshold**3.0 * SPMm1(s, 0.0, threshold, cutoff)
                )
                + SPMm1(s, 0, cutoff, np.inf)
            )
        )

    return DI, solution, integrand


def test_Omnes_real(rng):

    DI, solution, ig = prepare_Omnes_tests()

    vals = [
        3.0,
        1.5,
        np.asarray([-1.0, 0.0, 0.9, 1.1, 1.5, 2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
        12.0 * rng.random((4, 5, 6)) - 1.0,
    ]

    for i, val in enumerate(vals):
        np.testing.assert_allclose(DI(val), solution(val), rtol=2e-6)


def test_DispersionIntegralOmnes_complex(rng):
    DI, solution, ig = prepare_Omnes_tests()

    vals_real = [
        3.0,
        1.5,
        np.asarray([-1.0, 0.0, 0.9, 1.1, 1.5, 2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
    ]

    vals_imag = [
        1.5,
        -3.0,
        np.asarray([-10.0, 1e-3, 0.9, -1.1, 1.5, -2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
    ]

    vals = []

    for i in range(len(vals_real)):
        vals.append(vals_real[i] + 1j * vals_imag[i])

    for i, val in enumerate(vals):
        np.testing.assert_allclose(DI(val), solution(val), rtol=1e-4)

    for i, val in enumerate(vals_real):
        np.testing.assert_allclose(DI(np.conj(val + 1e-3j)), np.conj(DI(val + 1e-3j)))


def test_DispersionIntegralOmnes_fail_subtraction():

    threshold = 1.0
    cutoff = 2.0
    asymptotic = np.pi

    def integrand(s):
        s = np.asarray(s)
        results = np.zeros_like(s)

        mask0 = s <= threshold
        mask1 = (s > threshold) * (s < cutoff)
        mask2 = s >= cutoff

        results[mask1] = asymptotic * (
            3.0 * (s[mask1] - threshold) ** 2.0 / (cutoff - threshold) ** 2.0
            - 2.0 * (s[mask1] - threshold) ** 3.0 / (cutoff - threshold) ** 3.0
        )
        results[mask2] = asymptotic

        return results

    with pytest.raises(ValueError):
        DI = DispersionIntegralRHC(integrand, threshold, subtraction_point=1.5)


def prepare_Omnes_tests_2():

    threshold = 1.0
    infinity = 1e6

    mass = 1.5
    width = 0.2

    def integrand(s):
        s = np.asarray(s)
        s0 = np.angle(breit_wigner(1.0, mass, width))
        mask = s > threshold
        return np.where(s > threshold, np.angle(breit_wigner(s, mass, width)) - s0, 0.0)

    DI = OmnesFunction(
            integrand,
            threshold,
            infinity=infinity,
            integration_split_points=[
                1.0,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                25.0,
                50.0,
                75.0,
                100.0,
            ],
            integration_order=500,
        )

    return DI, integrand


def test_Omnes_schwarz(rng):
    Om, integrand = prepare_Omnes_tests_2()

    revals = 10. * rng.random((44, 2, 3)) + 0.5
    imvals = 5. * rng.random((44, 2, 3)) + 0.01

    svals = revals + 1j * imvals

    np.testing.assert_allclose(Om(np.conj(svals)), np.conj(Om(svals)))


def test_Omnes_phase(rng):

    Om, integrand = prepare_Omnes_tests_2()

    svals = 10. * rng.random((3,2,5)) + 0.5

    np.testing.assert_allclose(np.angle(Om(svals)), integrand(svals))


def test_Omnes_high_energy():

    Om, integrand = prepare_Omnes_tests_2()

    svals = np.linspace(100., 500., 500)

    omvals = Om(svals)

    assert(np.all(np.diff(np.abs(omvals)) < 0))
