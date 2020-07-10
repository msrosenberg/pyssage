import pyssage.utils
import numpy
import pytest


def test_check_for_square_matrix():
    # valid test
    test_size = 5
    matrix = numpy.zeros((test_size, test_size))
    n = pyssage.utils.check_for_square_matrix(matrix)
    assert n == test_size

    with pytest.raises(ValueError, match="distance matrix must be two-dimensional"):
        # broken test --> should raise error
        matrix = numpy.zeros((1, 2, 3))
        pyssage.utils.check_for_square_matrix(matrix)

    with pytest.raises(ValueError, match="distance matrix must be square"):
        # broken test --> should raise error
        matrix = numpy.zeros((test_size, test_size * 2))
        pyssage.utils.check_for_square_matrix(matrix)
