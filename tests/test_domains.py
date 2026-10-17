import pytest
import numpy as np
from dispersionrelations.domains import *

from dispersionrelations.complex import schwarz_reflection

@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_within_bounds():
    arrays = [
        [41.],
        [42.],
        [43.],
        np.asarray([-1.5, 0., 3., 2.3, 3.33, 5.]),
        np.asarray([[42., 42.1, 42.3], [42.4, 42.5, 42.6]]),
        np.asarray([[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]], [[13., 14.], [15., 16.], [17., 18.]]])
    ]

    bounds = [
        [41.5, 41.],
        [41., 42.],
        [42., 44.],
        [0., 3.3],
        [42.3, 42.55],
        [5.5, 11.11]
    ]

    solutions = [
        [False],
        [False],
        [True],
        np.asarray([False, False, True, True, False, False]),
        np.asarray([[False, False, False], [True, True, False]]),
        np.asarray([[[False, False], [False, False], [False, True]], [[True, True], [True, True], [True, False]], [[False, False], [False, False], [False, False]]])
    ]

    for i, arr in enumerate(arrays):
        print(rf"Working on array {arr}.")
        np.testing.assert_allclose(within_bounds(arr, bounds[i]), solutions[i])


def test_within_bounds_scalar():
    scalars = [1.0, 42.0, 3.0, 0.0]
    bounds = [0.0, 3.0]

    solutions = [True, False, False, False]

    for i, int in enumerate(scalars):
        np.testing.assert_equal(within_bounds(int, bounds), solutions[i])


def test_within_complex_bounds():
    arrays = [
        [1. + 1j],
        # [-42. - 84j],
        # [101.1 + 4j],
        np.asarray([3. + 4j, -2 + 5j, 16 + 8j]),
        np.asarray([[3. - 12j, 12.1 + 12.2j, 18 + 0j], [4 - 3j, 8 + 9j, np.pi - np.exp(1)*1j]]),
        np.asarray([[[4j, 18.], [13. + 0.1j, 1. + 12j]], [[4. + 0.9j, 5. - 1j], [8 + 7j, 6. + 21j]], [[11 + 5j, 2. + 3j], [10. + 11j, 4. + 5j]]])
    ]

    bounds = [
        [[0.9, 1.1], [0.8, 0.9]],
        [[-3., 10.], [4.5, 9.]],
        [[5., 17], [-1., 21.]],
        [[5., 17], [-1., 21.]]
    ]

    solutions = [
        [False],
        [False, True, False],
        [[False, True, False], [False, True, False]],
        [[[False, False], [True, False]], [[False, False], [True, False]], [[True, False], [True, False]]]
    ]

    for i, arr in enumerate(arrays):
        np.testing.assert_allclose(within_complex_bounds(arr, bounds[i][0], bounds[i][1]), solutions[i])


def test_points_RxR_to_R2():
    arrays1 = [
        [1.],
        np.asarray([2., 3.]),
        np.asarray([[4., 5.], [6., 7.]]),
        np.asarray([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]])
    ]

    arrays2 = [
        [42.],
        np.asarray([43., 44.]),
        np.asarray([[45., 46.], [47., 4.8]]),
        np.asarray([[[42., 43.], [44., 45.]], [[46., 47.], [48., 49.]], [[50, 51.], [52., 53.]]])
    ]

    solutions = [
        np.asarray([[1., 42.]]), 
        np.asarray([[2., 43.], [3., 44.]]),
        np.asarray([[[4., 45.], [5., 46.]], [[6., 47.], [7., 4.8]]]),
        np.asarray([[[[1., 42.], [2., 43.]], [[3., 44.], [4., 45.]]], [[[5., 46.], [6., 47.]], [[7., 48], [8., 49]]], [[[9., 50], [10., 51.]], [[11., 52.], [12., 53.]]]])
    ]

    for i, arr in enumerate(arrays1):
        print(rf"Working on array {arr}.")
        np.testing.assert_allclose(
            points_RxR_to_R2(arr, arrays2[i]), solutions[i]
        )


def test_points_RxR_to_R2_wrong_shape():
    arrays1 = [
        [1.],
        np.asarray([2., 3.])
    ]

    arrays2 = [
        [42., 43.],
        np.asarray([[1., 2.], [3., 4.]])
    ]

    for i, arr in enumerate(arrays1):
        with pytest.raises(ValueError):
            points_RxR_to_R2(arr, arrays2[i])


def test_points_C1_to_R2():

    C1arr = [
        np.asarray([1.0 + 1j * 42.0]),
        np.asarray([2.0 + 43.0j, 3.0 + 44.0j]),
        np.asarray([[4.0 + 45.0j, 5.0 + 46.0j], [6.0 + 47.0j, 7.0 + 4.8j]]),
        np.asarray(
            [
                [[1.0 + 42.0j, 2.0 + 43.0j], [3.0 + 44.0j, 4.0 + 45.0j]],
                [[5.0 + 46.0j, 6.0 + 47.0j], [7.0 + 48j, 8.0 + 49j]],
                [[9.0 + 50j, 10.0 + 51.0j], [11.0 + 52.0j, 12.0 + 53.0j]],
            ]
        ),
    ]

    solutions = [
        np.asarray([[1.0, 42.0]]),
        np.asarray([[2.0, 43.0], [3.0, 44.0]]),
        np.asarray([[[4.0, 45.0], [5.0, 46.0]], [[6.0, 47.0], [7.0, 4.8]]]),
        np.asarray(
            [
                [[[1.0, 42.0], [2.0, 43.0]], [[3.0, 44.0], [4.0, 45.0]]],
                [[[5.0, 46.0], [6.0, 47.0]], [[7.0, 48], [8.0, 49]]],
                [[[9.0, 50], [10.0, 51.0]], [[11.0, 52.0], [12.0, 53.0]]],
            ]
        ),
    ]

    for i, arr in enumerate(C1arr):
        print(rf"Working on array {arr}.")
        np.testing.assert_allclose(points_C1_to_R2(arr), solutions[i])


def test_points_R2_to_C1():

    R2arr = [
        np.asarray([[1.0, 42.0]]),
        np.asarray([[2.0, 43.0], [3.0, 44.0]]),
        np.asarray([[[4.0, 45.0], [5.0, 46.0]], [[6.0, 47.0], [7.0, 4.8]]]),
        np.asarray(
            [
                [[[1.0, 42.0], [2.0, 43.0]], [[3.0, 44.0], [4.0, 45.0]]],
                [[[5.0, 46.0], [6.0, 47.0]], [[7.0, 48], [8.0, 49]]],
                [[[9.0, 50], [10.0, 51.0]], [[11.0, 52.0], [12.0, 53.0]]],
            ]
        ),
    ]

    solutions = [
        np.asarray([1.0 + 1j * 42.0]),
        np.asarray([2.0 + 43.0j, 3.0 + 44.0j]),
        np.asarray([[4.0 + 45.0j, 5.0 + 46.0j], [6.0 + 47.0j, 7.0 + 4.8j]]),
        np.asarray(
            [
                [[1.0 + 42.0j, 2.0 + 43.0j], [3.0 + 44.0j, 4.0 + 45.0j]],
                [[5.0 + 46.0j, 6.0 + 47.0j], [7.0 + 48j, 8.0 + 49j]],
                [[9.0 + 50j, 10.0 + 51.0j], [11.0 + 52.0j, 12.0 + 53.0j]],
            ]
        ),
    ]

    for i, arr in enumerate(R2arr):
        print(rf"Working on array {arr}.")
        np.testing.assert_allclose(points_R2_to_C1(arr), solutions[i])


def test_points_R2_to_C1_wrong_last_dim():

    R2arr = [
        [0.],
        [1., 2., 3.],
        np.asarray([[1.,2., 3.], [4., 5., 6]])
    ]

    for i, arr in enumerate(R2arr):
        with pytest.raises(ValueError):
            points_R2_to_C1(arr)


def test_domain_R2_to_C1():

    def funcR2(z_2d):
        return z_2d**2.

    funcC1 = domain_R2_to_C1(funcR2)

    R2arr = [
        np.asarray([[1.0, 42.0]]),
        np.asarray([[2.0, 43.0], [3.0, 44.0]]),
        np.asarray([[[4.0, 45.0], [5.0, 46.0]], [[6.0, 47.0], [7.0, 4.8]]]),
        np.asarray(
            [
                [[[1.0, 42.0], [2.0, 43.0]], [[3.0, 44.0], [4.0, 45.0]]],
                [[[5.0, 46.0], [6.0, 47.0]], [[7.0, 48], [8.0, 49]]],
                [[[9.0, 50], [10.0, 51.0]], [[11.0, 52.0], [12.0, 53.0]]],
            ]
        ),
    ]

    C1arr = [
        np.asarray([1.0 + 1j * 42.0]),
        np.asarray([2.0 + 43.0j, 3.0 + 44.0j]),
        np.asarray([[4.0 + 45.0j, 5.0 + 46.0j], [6.0 + 47.0j, 7.0 + 4.8j]]),
        np.asarray(
            [
                [[1.0 + 42.0j, 2.0 + 43.0j], [3.0 + 44.0j, 4.0 + 45.0j]],
                [[5.0 + 46.0j, 6.0 + 47.0j], [7.0 + 48j, 8.0 + 49j]],
                [[9.0 + 50j, 10.0 + 51.0j], [11.0 + 52.0j, 12.0 + 53.0j]],
            ]
        ),
    ]

    for i, arr in enumerate(R2arr):
        print(rf"Working on array {arr}.")
        np.testing.assert_allclose(funcC1(C1arr[i]), funcR2(arr))


def test_domain_C1_to_R2():

    def funcC1(z_2d):
        return z_2d**2.0

    funcR2 = domain_C1_to_R2(funcC1)

    R2arr = [
        np.asarray([[1.0, 42.0]]),
        np.asarray([[2.0, 43.0], [3.0, 44.0]]),
        np.asarray([[[4.0, 45.0], [5.0, 46.0]], [[6.0, 47.0], [7.0, 4.8]]]),
        np.asarray(
            [
                [[[1.0, 42.0], [2.0, 43.0]], [[3.0, 44.0], [4.0, 45.0]]],
                [[[5.0, 46.0], [6.0, 47.0]], [[7.0, 48], [8.0, 49]]],
                [[[9.0, 50], [10.0, 51.0]], [[11.0, 52.0], [12.0, 53.0]]],
            ]
        ),
    ]

    C1arr = [
        np.asarray([1.0 + 1j * 42.0]),
        np.asarray([2.0 + 43.0j, 3.0 + 44.0j]),
        np.asarray([[4.0 + 45.0j, 5.0 + 46.0j], [6.0 + 47.0j, 7.0 + 4.8j]]),
        np.asarray(
            [
                [[1.0 + 42.0j, 2.0 + 43.0j], [3.0 + 44.0j, 4.0 + 45.0j]],
                [[5.0 + 46.0j, 6.0 + 47.0j], [7.0 + 48j, 8.0 + 49j]],
                [[9.0 + 50j, 10.0 + 51.0j], [11.0 + 52.0j, 12.0 + 53.0j]],
            ]
        ),
    ]

    for i, arr in enumerate(R2arr):
        print(rf"Working on array {arr}.")
        np.testing.assert_allclose(funcR2(R2arr[i]), funcC1(C1arr[i]))


def test_exclusive_masks():
    masks = [
        [
            np.asarray([1, 1, 1, 1, 1, 0, 1, 0]),
            np.asarray([0, 0, 0, 0, 1, 0, 1, 1]),
            np.asarray([1, 0, 0, 0, 1, 1, 0, 0]),
        ],
        [
            np.asarray([1, 0])
        ],
        [
            np.asarray([0, 0, 1, 1, 0, 0]),
            np.asarray([0, 1, 0, 1, 1, 0]),
            np.asarray([1, 1, 1, 1, 1, 1])
        ],
        [
            np.asarray([[1, 0, 1, 1], [0, 1, 0, 0]]),
            np.asarray([[0, 1, 0, 1], [1, 1, 0, 0]]),
            np.asarray([[1, 0, 1, 1], [0, 1, 1, 0]])
        ]
    ]

    solutions = [
        [ 
            np.asarray([1, 1, 1, 1, 1, 0, 1, 0]),
            np.asarray([0, 0, 0, 0, 0, 0, 0, 1]),
            np.asarray([0, 0, 0, 0, 0, 1, 0, 0])
        ],
        [
            np.asarray([1, 0])
        ],
        [
            np.asarray([0, 0, 1, 1, 0, 0]),
            np.asarray([0, 1, 0, 0, 1, 0]),
            np.asarray([1, 0, 0, 0, 0, 1])
        ],
        [
            np.asarray([[1, 0, 1, 1], [0, 1, 0, 0]]),
            np.asarray([[0, 1, 0, 0], [1, 0, 0, 0]]),
            np.asarray([[0, 0, 0, 0], [0, 0, 1, 0]])
        ]
    ]
    
    for i, mask in enumerate(masks):
        np.testing.assert_allclose(exclusive_masks(mask), solutions[i])


def test_always_in_domain(rng):
    a = 3. + 1j
    b = 42. - 3j
    z_arrays = [
        np.asarray([1 + 1j]),
        (b - a) * rng.random(size=(2)) + a,
        (b - a) * rng.random(size=(3, 4)) + a,
        (b - a) * rng.random(size=((1,3,4,2,6))) + a
    ]

    results = [np.ones(1), np.ones((2)), np.ones((3, 4)), np.ones((1, 3, 4, 2, 6))]

    for i, z in enumerate(z_arrays):
        np.testing.assert_allclose(always_in_domain(z), results[i])


def test_DomainedFunctionCollection_real_input():

    f1 = Domainify(lambda x: np.exp(x), lambda x: x <= -2.)
    f2 = Domainify(lambda x: np.sin(x), lambda x: (x < 0.) * (x > -3.))
    f3 = Domainify(lambda x: np.sqrt(x), lambda x: x > 0)

    DFC = DomainedFunctionCollection([f1, f2, f3])

    xvals = np.linspace(-10, 10, 20)
    result = np.zeros(20)
    result[:] = np.nan
    mask1 = (xvals <= -2.)
    mask2 = (xvals < 0.) * (xvals > -3.)
    mask3 = (xvals > 0.)

    result[mask3] = f3(xvals[mask3])
    result[mask2] = f2(xvals[mask2])
    result[mask1] = f1(xvals[mask1])

    np.testing.assert_allclose(DFC(xvals), result)


def test_Domained_FunctionCollection_real_input_nans():

    f1 = Domainify(lambda x: np.exp(x), lambda x: (x >= -5) * (x <= -2.0))
    f2 = Domainify(lambda x: np.sin(x), lambda x: (x < 0.0) * (x > -3.0))
    f3 = Domainify(lambda x: np.sqrt(x), lambda x: x > 0)

    DFC = DomainedFunctionCollection([f1, f2, f3])

    xvals = np.linspace(-10, 10, 20)
    result = np.zeros(20)
    result[:] = np.nan
    mask1 = (xvals >= -5) * (xvals <= -2.0)
    mask2 = (xvals < 0.0) * (xvals > -3.0)
    mask3 = xvals > 0.0

    result[mask3] = f3(xvals[mask3])
    result[mask2] = f2(xvals[mask2])
    result[mask1] = f1(xvals[mask1])

    print(result)
    print()
    print(DFC(xvals))

    np.testing.assert_allclose(DFC(xvals), result)


def test_Domained_FunctionCollection_complex_input(rng):

    f1 = Domainify(lambda x: np.exp(x), lambda z: (np.abs(z) >= 0.5) * (np.abs(z) <= 2.0))
    f2 = Domainify(
        lambda x: np.sin(x), lambda z: (np.abs(z) > 2.1) * (np.abs(z) < 3.0)
    )
    f3 = Domainify(lambda x: np.sqrt(x), lambda z: np.abs(z) > 2.5)

    DFC = DomainedFunctionCollection([f1, f2, f3])

    size = (2,3,4)
    xvals = 7. * rng.random(size = size)
    yvals = 6. * rng.random(size = size)
    Zvals = xvals + 1j * yvals

    result = np.zeros(size, dtype = complex)
    result[:] = np.nan
    mask1 = (np.abs(Zvals) > 0.5) * (np.abs(Zvals) < 2.0)
    mask2 = (np.abs(Zvals) > 2.1) * (np.abs(Zvals) < 3.0)
    mask3 = np.abs(Zvals) > 2.5

    result[mask3] = f3(Zvals[mask3])
    result[mask2] = f2(Zvals[mask2])
    result[mask1] = f1(Zvals[mask1])

    # print(result)
    # print()
    # print(DFC(Zvals))

    np.testing.assert_allclose(DFC(Zvals), result)


def test_Domained_FunctionCollection_in_domain_complex_input(rng):

    f1 = Domainify(lambda x: np.exp(x), lambda z: (np.abs(z) >= 0.5) * (np.abs(z) <= 1.0))
    f2 = Domainify(
        lambda x: np.sin(x), lambda z: (np.abs(z) > 2.1) * (np.abs(z) < 3.0)
    )
    f3 = Domainify(lambda x: np.sqrt(x), lambda z: np.abs(z) > 2.5)

    DFC = DomainedFunctionCollection([f1, f2, f3])

    size = (2,5,3)
    xvals = 2. * rng.random(size = size)
    yvals = 3.1 * rng.random(size = size)
    Zvals = xvals + 1j * yvals

    mask1 = (np.abs(Zvals) > 0.5) * (np.abs(Zvals) < 1.0)
    mask2 = (np.abs(Zvals) > 2.1) * (np.abs(Zvals) < 3.0)
    mask3 = np.abs(Zvals) > 2.5

    total_mask = mask1 + mask2 + mask3

    # print(result)
    # print()
    # print(DFC(Zvals))

    np.testing.assert_allclose(DFC.in_domain(Zvals), total_mask)


def test_DomainedFunctionCollection_add_function_real_input():

    f1 = Domainify(lambda x: np.exp(x), lambda x: x <= -2.)

    DFC = DomainedFunctionCollection([f1])

    f2 = Domainify(lambda x: np.sin(x), lambda x: (x < 0.) * (x > -3.))
    f3 = Domainify(lambda x: np.sqrt(x), lambda x: x > 0)

    DFC.add_function(f3)
    DFC.add_function(f2, index = 1)

    xvals = np.linspace(-10, 10, 20)
    result = np.zeros(20)
    result[:] = np.nan
    mask1 = (xvals <= -2.)
    mask2 = (xvals < 0.) * (xvals > -3.)
    mask3 = (xvals > 0.)

    result[mask3] = f3(xvals[mask3])
    result[mask2] = f2(xvals[mask2])
    result[mask1] = f1(xvals[mask1])

    np.testing.assert_allclose(DFC(xvals), result)

def test_DomainedFunctionCollection_check_for_empty():
    DFC = DomainedFunctionCollection([])

    with pytest.raises(RuntimeError):
        DFC.__check_for_empty__()


def test_Domain_is_None():
    f1 = Domainify(lambda x: x**2.)

    DFC = DomainedFunctionCollection([f1])

    xvals = np.linspace(-1000., 1000, 20)

    result = np.ones(len(xvals))

    np.testing.assert_equal(DFC.in_domain(xvals), result)


def test_in_domain_scalar_input():
    f1 = Domainify(lambda x: x**2.0)

    DFC = DomainedFunctionCollection([f1])

    xvals = 3.

    result = 1

    np.testing.assert_equal(DFC.in_domain(xvals), result)


def test_DFC__call__scalar_input():
    f1 = Domainify(lambda x: x**2.0)

    DFC = DomainedFunctionCollection([f1])

    xvals = 3.0

    result = 9.

    np.testing.assert_equal(DFC(xvals), result)


def test_DFC_np_any_false():
    f1 = Domainify(lambda x: x**2.0, lambda x: x > 1.)

    DFC = DomainedFunctionCollection([f1])

    xvals = np.asarray([0.1, 0.2, 0.3])

    result = np.ones(3)
    result[:] = np.nan

    np.testing.assert_equal(DFC(xvals), result)
