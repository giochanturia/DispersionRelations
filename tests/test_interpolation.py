import pytest
import numpy as np
from dispersionrelations.interpolation import *
from dispersionrelations.domains import Domainify


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_RG_gen_level_tuple_fail():
    RG = RecursiveGrid(scale=2.0, dimension=3)

    with pytest.raises(ValueError):
        RG.generate_level(n_gridpoints=(42, 42))


def test_RG_gen_level():
    RG = RecursiveGrid(scale=2.0, dimension=3)

    lel = RG.generate_level(level=0, n_gridpoints=5)
    solel = np.meshgrid(*(3 * [[-2.0, -1.0, 0.0, 1.0, 2.0]]), indexing="ij")

    np.testing.assert_allclose(lel, solel)

    lel = RG.generate_level(level=2, n_gridpoints=5)
    solel = np.meshgrid(*(3 * [[-18, -9, 0, 9, 18]]), indexing = "ij")

    np.testing.assert_allclose(lel, solel)


def test_RG_gen_level_tuple():
    RG = RecursiveGrid(scale=2.0, dimension=3)

    lel = RG.generate_level(level=0, n_gridpoints=(3, 5, 3))
    solel = np.meshgrid(
        *[[-2.0, 0, 2.0], [-2.0, -1.0, 0.0, 1.0, 2.0], [-2.0, 0, 2.0]], indexing="ij"
    )

    np.testing.assert_allclose(lel, solel)

    lel = RG.generate_level(level=2, n_gridpoints=(3, 5, 3))
    solel = np.meshgrid(
        *[[-18.0, 0.0, 18.0], [-18, -9, 0, 9, 18], [-18.0, 0.0, 18.0]],
        indexing="ij",
    )
    np.testing.assert_allclose(lel, solel)


def test_determine_level():
    RG = RecursiveGrid(scale=2.0, dimension=3)

    point_arrs = [
        np.asarray([1, 1, 6]),
        np.asarray([1, 2, 3]),
        np.asarray([0.0, 1.2, -1.8]),
        np.asarray([0, 1, 42]),
        np.asarray([[1, 2, 3], [0.0, 1.2, -1.8]]),
        np.asarray([[[1, 2, 3], [0.0, 1.2, -1.8]], [[1, 2, 8], [0.0, 42, -1.8]]]),
    ]

    solutions = [1, 1, 0, 3, np.asarray([1, 0]), np.asarray([[1, 0], [2, 3]])]

    for i, points in enumerate(point_arrs):
        np.testing.assert_equal(RG.determine_level(points), solutions[i])


def test_level_dictionary():
    RG = RecursiveGrid(scale=2.0, dimension=3)

    point_arrs = [
        np.asarray([[1, 2, 3], [0.0, 1.2, -1.8]]),
        np.asarray(
            [
                [[1, 2, 3], [0.0, 1.2, -1.8], [4, 5, 6]],
                [[1, 2, 8], [0.0, 42, -1.8], [7, 8, 9]],
            ]
        ),
    ]

    soldics = [
        {0: [1], 1: [0]},
        {
            0: [[0], [1]],
            1: [[0, 0], [0, 2]],
            2: [[1, 1], [0, 2]],
            3: [[1], [1]],
        },
    ]

    for i, points in enumerate(point_arrs):
        np.testing.assert_equal(RG.level_index_dictionary(points), soldics[i])


# def test_CP_dim_error():
#     RG = RecursiveGrid(scale=2.0, dimension=3)

#     def f(x, y):
#         return np.sin(x * np.pi) * np.cos(y * np.pi) + 1j * np.sin(x * y * np.pi)

#     func = Domainify(f)

#     with pytest.raises(ValueError):
#         CP = ComplexPrecomputatinator(func, RG, "full")


def inst_Complex_Precomputatinator(domain="full"):
    RG = RecursiveGrid(scale=2.0, dimension=2)

    def f(z):
        x = np.real(z)
        y = np.imag(z)
        return np.sin(x * np.pi) * np.cos(y * np.pi) + 1j * np.sin(x * y * np.pi)

    func = Domainify(f)

    CP = ComplexPrecomputatinator(func, RG, domain)

    return CP


def test_CP_evaluate_level():
    CP = inst_Complex_Precomputatinator()

    RG = RecursiveGrid(scale=2.0, dimension=2)

    lev = 2
    shap = (15, 13)
    mk = {"indexing": "ij"}
    X, Y = RG.generate_level(lev, shap)
    fvals = CP.f(X + 1j * Y)
    soldic = {
        "level": lev,
        "shape": shap,
        "meshgrid": mk,
        "data": {
            "z_RE": X,
            "z_IM": Y,
            "f_RE": np.real(fvals),
            "f_IM": np.imag(fvals),
        },
    }

    el_dic = CP.evaluate_level(lev, shap, mk)

    np.testing.assert_equal(el_dic, soldic)


def test_CP_evaluate_level_upper():
    CP = inst_Complex_Precomputatinator("upper")

    RG = RecursiveGrid(scale=2.0, dimension=2)

    lev = 2
    shap = (15, 13)
    mk = {"indexing": "ij"}
    X, Y = RG.generate_level(lev, shap, mk)
    mask = Y >= 0
    newshap = (shap[0], int(np.floor(shap[1] / 2) + 1))
    print(newshap)
    X = X[mask].reshape(newshap)
    Y = Y[mask].reshape(newshap)
    fvals = CP.f(X + 1j * Y)
    soldic = {
        "level": lev,
        "shape": newshap,
        "meshgrid": mk,
        "data": {
            "z_RE": X,
            "z_IM": Y,
            "f_RE": np.real(fvals),
            "f_IM": np.imag(fvals),
        },
    }

    el_dic = CP.evaluate_level(lev, shap, mk)

    np.testing.assert_equal(el_dic, soldic)


def test_CP_evaluate_level_lower():
    CP = inst_Complex_Precomputatinator("lower")

    RG = RecursiveGrid(scale=2.0, dimension=2)

    lev = 2
    shap = (15, 13)
    mk = {"indexing": "ij"}
    X, Y = RG.generate_level(lev, shap, mk)
    mask = Y <= 0
    newshap = (shap[0], int(np.floor(shap[1] / 2) + 1))
    print(newshap)
    X = X[mask].reshape(newshap)
    Y = Y[mask].reshape(newshap)
    fvals = CP.f(X + 1j * Y)
    soldic = {
        "level": lev,
        "shape": newshap,
        "meshgrid": mk,
        "data": {
            "z_RE": X,
            "z_IM": Y,
            "f_RE": np.real(fvals),
            "f_IM": np.imag(fvals),
        },
    }

    el_dic = CP.evaluate_level(lev, shap, mk)

    np.testing.assert_equal(el_dic, soldic)


def make_CI(domain="full"):

    RG = RecursiveGrid(scale=2.0, dimension=2)

    CI = ComplexInterpolator(RG, domain)

    return CI


def test_zero_real_domain(rng):
    CI = make_CI()

    zs = [
        np.asarray([1]),
        np.asarray([1, 2]),
        rng.random(size=(2, 4)),
        rng.random(size=(3, 6, 9)),
        rng.random(size=(3, 2, 4, 5, 1, 6, 7)),
    ]

    for i, z in enumerate(zs):
        np.testing.assert_equal(CI.__in_real_line_domain__(z), np.zeros_like(z))


def exfunc(z):
    # return np.sin(z * np.pi) + 1j * np.sin(z * np.pi)
    return np.sin(z * np.pi) + 5.0 * np.sqrt(2.0 - z + 0j)


def test_add_real_line_interpolator_fail(rng):
    CI = make_CI()

    zarr = rng.random(size=(1))
    z2arr = rng.random(size=(2))

    fvals = exfunc(z2arr)

    with pytest.raises(ValueError):
        CI.add_real_line_interpolator(zarr, np.real(fvals), np.imag(fvals))


def test_add_real_line_interpolator(rng):
    CI = make_CI()

    zarr = np.sort(2.0 * rng.random(size=(5000)) - 1.0)
    zarr[0] = -1.0
    zarr[-1] = 1.0
    z2arr = np.sort(2.0 * rng.random(size=(5000)) - 1.0)

    fvals = exfunc(zarr)

    CI.add_real_line_interpolator(zarr, np.real(fvals), np.imag(fvals))

    np.testing.assert_allclose(CI.interpolators[-1](z2arr), exfunc(z2arr), 1e-4)


def test_real_domain(rng):
    CI = make_CI()

    zarr = np.sort(2.0 * rng.random(size=(5000)) - 0.9)
    zarr[0] = -0.9
    zarr[-1] = 1.1

    fvals = exfunc(zarr)

    CI.add_real_line_interpolator(zarr, np.real(fvals), np.imag(fvals))

    z2arr = 4.0 * rng.random(size=(100)) - 1.1

    boolvals = np.ones_like(z2arr)
    boolvals[z2arr < CI.real_line_bounds[0]] = 0
    boolvals[z2arr > CI.real_line_bounds[1]] = 0

    np.testing.assert_equal(CI.__in_real_line_domain__(z2arr), boolvals)


def test_zero_complex_domain(rng):
    CI = make_CI()

    zs = [
        np.asarray([1 + 1j]),
        np.asarray([1 + 3j, 2 + 4j]),
        rng.random(size=(2, 4))
        + 1j * rng.random(size=(2, 4)),
        rng.random(size=(3, 6, 9))
        + 1j * rng.random(size=(3, 6, 9)),
        rng.random(size=(3, 2, 4, 5, 1, 6, 7))
        + 1j * rng.random(size=(3, 2, 4, 5, 1, 6, 7)),
    ]

    for i, z in enumerate(zs):
        np.testing.assert_equal(CI.__in_complex_grid_domain__(z), np.zeros_like(z))


def test_add_level_interpolator_fail(rng):
    CI = make_CI()

    zarr_re = np.sort(3.0 * rng.random(size=(5000)) - 1.0)
    zarr_re[0] = -1.5
    zarr_re[-1] = 1.5
    zarr_re = np.sort(zarr_re)

    zarr_im = np.sort(2.0 * rng.random(size=(5000)) - 1.0)
    zarr_im[0] = -1.0
    zarr_im[-1] = 1.0
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    # Re axis is wrong:

    z2arr_re = np.sort(3.0 * rng.random(size=(3)) - 1.0)
    z2arr_im = np.sort(2.0 * rng.random(size=(5000)) - 1.0)

    Z2arr_re, Z2arr_im = np.meshgrid(z2arr_re, z2arr_im, indexing="ij")

    Z2arr = Z2arr_re + 1j * Z2arr_im

    fvals = exfunc(Z2arr)

    level = np.max(CI.grid.determine_level(Zarr))

    with pytest.raises(ValueError):

        CI.add_level_interpolator(
            level,
            Zarr_re,
            Zarr_im,
            np.real(fvals),
            np.imag(fvals),
            impose_schwarz_reflection=False,
        )

    # Im axis is wrong:

    z2arr_re = np.sort(3.0 * rng.random(size=(5000)) - 1.0)
    z2arr_im = np.sort(2.0 * rng.random(size=(3)) - 1.0)

    Z2arr_re, Z2arr_im = np.meshgrid(z2arr_re, z2arr_im, indexing="ij")

    Z2arr = Z2arr_re + 1j * Z2arr_im

    fvals = exfunc(Z2arr)

    level = np.max(CI.grid.determine_level(Zarr))

    with pytest.raises(ValueError):

        CI.add_level_interpolator(
            level,
            Zarr_re,
            Zarr_im,
            np.real(fvals),
            np.imag(fvals),
            impose_schwarz_reflection=False,
        )


def test_add_level_interpolator_without_schwarz(rng):
    CI = make_CI()

    zarr_re = np.sort(3.0 * rng.random(size=(5000)) - 1.0)
    zarr_re[0] = -1.5
    zarr_re[-1] = 1.5
    zarr_re = np.sort(zarr_re)

    zarr_im = np.sort(2.0 * rng.random(size=(5000)) - 1.0)
    zarr_im[0] = -1.0
    zarr_im[-1] = 1.0
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=False,
    )

    z2arr_re = np.sort(3.0 * rng.random(size=(3)) - 1.0)
    z2arr_im = np.sort(2.0 * rng.random(size=(3)) - 1.0)

    Z2arr_re, Z2arr_im = np.meshgrid(z2arr_re, z2arr_im, indexing="ij")

    Z2arr = Z2arr_re + 1j * Z2arr_im

    np.testing.assert_allclose(CI.interpolators[1](Z2arr), exfunc(Z2arr), 5e-5)


def test_add_level_interpolator_with_schwarz(rng):
    CI = make_CI()

    zarr_re = np.sort(3.0 * rng.random(size=(5000)) - 1.0)
    zarr_re[0] = -1.5
    zarr_re[-1] = 1.5
    zarr_re = np.sort(zarr_re)

    zarr_im = np.sort(2.0 * rng.random(size=(5000)) - 1.0)
    zarr_im[0] = -1.0
    zarr_im[-1] = 1.0
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    z2arr_re = np.sort(3.0 * rng.random(size=(3)) - 1.0)
    z2arr_im = np.sort(2.0 * rng.random(size=(3)) - 1.0)

    Z2arr_re, Z2arr_im = np.meshgrid(z2arr_re, z2arr_im, indexing="ij")

    Z2arr = Z2arr_re + 1j * Z2arr_im

    np.testing.assert_allclose(CI.interpolators[1](Z2arr), exfunc(Z2arr), 1e-4)


# test add_level_interpolator:
def test_add_level_interpolator_with_schwarz_upper(rng):
    CI = make_CI("upper")

    zarr_re = np.sort(3.0 * rng.random(size=(1000)) - 1.0)
    zarr_re[0] = -1.5
    zarr_re[-1] = 1.5
    zarr_re = np.sort(zarr_re)

    zarr_im = np.sort(2.0 * rng.random(size=(1000)) - 1.0)
    zarr_im[0] = -1.0
    zarr_im[-1] = 1.0
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    z2arr_re = np.sort(3.0 * rng.random(size=(3)) - 1.0)
    z2arr_im = np.sort(2.0 * rng.random(size=(3)) - 1.0)

    Z2arr_re, Z2arr_im = np.meshgrid(z2arr_re, z2arr_im, indexing="ij")

    Z2arr = Z2arr_re + 1j * Z2arr_im

    np.testing.assert_allclose(CI.interpolators[1](Z2arr), exfunc(Z2arr), 5e-4)


# test add_level_interpolator:
def test_add_level_interpolator_with_schwarz_lower(rng):
    CI = make_CI("lower")

    zarr_re = np.sort(3.0 * rng.random(size=(1000)) - 1.0)
    zarr_re[0] = -1.5
    zarr_re[-1] = 1.5
    zarr_re = np.sort(zarr_re)

    zarr_im = np.sort(2.0 * rng.random(size=(1000)) - 1.0)
    zarr_im[0] = -1.0
    zarr_im[-1] = 1.0
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    z2arr_re = np.sort(3.0 * rng.random(size=(3)) - 1.0)
    z2arr_im = np.sort(2.0 * rng.random(size=(3)) - 1.0)

    Z2arr_re, Z2arr_im = np.meshgrid(z2arr_re, z2arr_im, indexing="ij")

    Z2arr = Z2arr_re + 1j * Z2arr_im

    np.testing.assert_allclose(CI.interpolators[1](Z2arr), exfunc(Z2arr), 6e-4)


# test add_level_interpolator:
def test_add_level_interpolator_with_schwarz_upper_lownovals(rng):
    CI = make_CI("upper")

    zarr_re = np.sort(3.0 * rng.random(size=(5000)) - 1.0)
    zarr_re[0] = -1.0
    zarr_re[-1] = 2.0
    zarr_re = np.sort(zarr_re)

    zarr_im = np.sort(2.0 * rng.random(size=(5000)) + 1.0)
    zarr_im[0] = 1.0
    zarr_im[-1] = 3.0
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    z2arr_re = np.sort(3.0 * rng.random(size=(3)) - 1.0)
    z2arr_im = np.sort(2.0 * rng.random(size=(3)) + 1.0)

    Z2arr_re, Z2arr_im = np.meshgrid(z2arr_re, z2arr_im, indexing="ij")

    Z2arr = Z2arr_re + 1j * Z2arr_im

    np.testing.assert_allclose(CI.interpolators[1](Z2arr), exfunc(Z2arr), 5e-5)


# test add_level_interpolator:
def test_add_level_interpolator_with_schwarz_lower_upnovals(rng):
    CI = make_CI("lower")

    zarr_re = np.sort(3.0 * rng.random(size=(5000)) - 1.0)
    zarr_re[0] = -1.0
    zarr_re[-1] = 2.0
    zarr_re = np.sort(zarr_re)

    zarr_im = np.sort(2.0 * rng.random(size=(5000)) - 3.0)
    zarr_im[0] = -3.0
    zarr_im[-1] = -1.0
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    z2arr_re = np.sort(3.0 * rng.random(size=(3)) - 1.0)
    z2arr_im = np.sort(2.0 * rng.random(size=(3)) - 3.0)

    Z2arr_re, Z2arr_im = np.meshgrid(z2arr_re, z2arr_im, indexing="ij")

    Z2arr = Z2arr_re + 1j * Z2arr_im

    np.testing.assert_allclose(CI.interpolators[1](Z2arr), exfunc(Z2arr), 5e-5)

    print(level)


# test add_level_interpolator:
def test_add_level_interpolator_highest_level(rng):
    CI = make_CI()

    np.testing.assert_equal(CI.highest_level, -1)

    xdiam = 5.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)
    zarr_re[0] = xshift
    zarr_re[-1] = xshift * xdiam
    zarr_re = np.sort(zarr_re)

    ydiam = 8.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)
    zarr_im[0] = yshift
    zarr_im[-1] = yshift + ydiam
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    np.testing.assert_equal(CI.highest_level, 2)

    xdiam = 3.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)
    zarr_re[0] = xshift  # test add_level_interpolator:


# test add_level_interpolator:
def test_add_level_interpolator_highest_level(rng):
    CI = make_CI()

    np.testing.assert_equal(CI.highest_level, -1)

    xdiam = 5.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)
    zarr_re[0] = xshift
    zarr_re[-1] = xshift * xdiam
    zarr_re = np.sort(zarr_re)

    ydiam = 8.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)
    zarr_im[0] = yshift
    zarr_im[-1] = yshift + ydiam
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    np.testing.assert_equal(CI.highest_level, 2)

    xdiam = 3.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)
    zarr_re[0] = xshift
    zarr_re[-1] = xshift * xdiam
    zarr_re = np.sort(zarr_re)

    ydiam = 2.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)
    zarr_im[0] = yshift
    zarr_im[-1] = yshift + ydiam
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    np.testing.assert_equal(CI.highest_level, 2)


def test_in_complex_grid_domain(rng):
    CI = make_CI()

    xdiam = 5.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)
    zarr_re[0] = xshift
    zarr_re[-1] = xshift * xdiam
    zarr_re = np.sort(zarr_re)

    ydiam = 8.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)
    zarr_im[0] = yshift
    zarr_im[-1] = yshift + ydiam
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    xdiam = 5.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)

    ydiam = 8.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    np.testing.assert_equal(CI.__in_complex_grid_domain__(Zarr), np.ones_like(Zarr))

    zs = [
        np.asarray([1 + 1j]),
        np.asarray([1 + 3j, 2 + 4j]),
        rng.random(size=(2, 4))
        + 1j * rng.random(size=(2, 4)),
        rng.random(size=(3, 6, 9))
        + 1j * rng.random(size=(3, 6, 9)),
        rng.random(size=(3, 2, 4, 5, 1, 6, 7))
        + 1j * rng.random(size=(3, 2, 4, 5, 1, 6, 7)),
    ]

    for i, z in enumerate(zs):
        np.testing.assert_equal(CI.__in_complex_grid_domain__(z), np.ones_like(z))


def test_in_domain_only_reals(rng):
    CI = make_CI()

    zdiam = 2.0
    zshift = -0.9
    zarr = np.linspace(zshift, zdiam + zshift)

    fvals = exfunc(zarr)

    CI.add_real_line_interpolator(zarr, np.real(fvals), np.imag(fvals))

    z2arr = 1.5 * rng.random(size=(3, 4, 2, 5, 1, 6)) - 0.5

    print(zarr)
    print(np.min(z2arr), np.max(z2arr))

    np.testing.assert_equal(CI.in_domain(z2arr), np.ones_like(z2arr))

    z3arr = 4.0 * rng.random(size=((3, 2, 3, 1, 5, 6))) - 1.5

    indomain = (z3arr >= -0.9) * (z3arr <= 1.1)

    np.testing.assert_equal(CI.in_domain(z3arr), indomain)

    z4arr = 5.0 * rng.random(size=(2, 1, 3, 1, 4, 5, 6)) + 8.0

    np.testing.assert_equal(CI.in_domain(z4arr), np.zeros_like(z4arr))


def test_in_domain_only_complexes(rng):
    CI = make_CI()

    xdiam = 5.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)
    zarr_re[0] = xshift
    zarr_re[-1] = xshift + xdiam
    zarr_re = np.sort(zarr_re)

    ydiam = 5.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)
    zarr_im[0] = yshift
    zarr_im[-1] = yshift + ydiam
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))
    print(level)
    # print(CI.grid.scale)

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    xdiam = 4.0
    xshift = -0.5
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)

    ydiam = 4.0
    yshift = -0.5
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    np.testing.assert_equal(CI.in_domain(Zarr), np.ones_like(Zarr))

    xdiam = 5.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)

    ydiam = 8.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    r = CI.grid.scale * 3**level

    inxdomain = np.abs(np.real(Zarr)) <= r
    inydomain = np.abs(np.imag(Zarr)) <= r

    indomain = inxdomain * inydomain

    np.testing.assert_equal(CI.in_domain(Zarr), indomain)

    xdiam = 0.1
    xshift = -10.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)

    ydiam = 0.1
    yshift = -10.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    np.testing.assert_equal(CI.in_domain(Zarr), np.zeros_like(Zarr))


def test_in_domain_mixed(rng):
    CI = make_CI()

    zdiam = 20.0
    zshift = -0.9
    zarr = np.linspace(zshift, zdiam + zshift)

    fvals = exfunc(zarr)

    CI.add_real_line_interpolator(zarr, np.real(fvals), np.imag(fvals))

    xdiam = 5.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)
    zarr_re[0] = xshift
    zarr_re[-1] = xshift + xdiam
    zarr_re = np.sort(zarr_re)

    ydiam = 5.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)
    zarr_im[0] = yshift
    zarr_im[-1] = yshift + ydiam
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    xdiam = 7.0
    xshift = -1.0
    zarr_re = np.sort(xdiam * rng.random(size=(500)) + xshift)
    zarr_re[0] = xshift
    zarr_re[-1] = xshift + xdiam
    zarr_re = np.sort(zarr_re)

    ydiam = 7.0
    yshift = -1.0
    zarr_im = np.sort(ydiam * rng.random(size=(500)) + yshift)
    zarr_im[0] = yshift
    zarr_im[-1] = yshift + ydiam
    zarr_im = np.sort(zarr_im)

    Zarr_re, Zarr_im = np.meshgrid(zarr_re, zarr_im, indexing="ij")

    Zarr = Zarr_re + 1j * Zarr_im

    fvals = exfunc(Zarr)

    level = np.max(CI.grid.determine_level(Zarr))

    CI.add_level_interpolator(
        level,
        Zarr_re,
        Zarr_im,
        np.real(fvals),
        np.imag(fvals),
        impose_schwarz_reflection=True,
    )

    reals = 60.0 * rng.random(size=(500)) - 31.0
    imags = 16.0 * rng.random(size=(500)) - 8.2
    imags[len(imags) :] = 0.0
    np.random.shuffle(imags)

    zarr = reals + 1j * imags

    solution = np.zeros_like(zarr)

    realmask = np.imag(zarr) == 0
    realsol = CI.__in_real_line_domain__(zarr[realmask])
    solution[realmask] = realsol

    complexsol = CI.__in_complex_grid_domain__(zarr[~realmask])
    solution[~realmask] = complexsol

    np.testing.assert_equal(CI.in_domain(zarr), solution)
