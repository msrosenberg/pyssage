import numpy
from math import pi, atan2


def flatten_half(x: numpy.ndarray) -> numpy.ndarray:
    """
    Given a square, symmmetric matrix, return the values from the lower triangle in a single column

    :param x: the input matrix, should be an n x n numpy.ndarray
    :return: a single column numpy.ndarray
    """
    output = []
    for i in range(len(x)):
        for j in range(i):
            output.append(x[i, j])
    return numpy.array(output)


# def euc_dist(x1: float, x2: float, y1: float = 0, y2: float = 0, z1: float = 0, z2: float = 0) -> float:
#     return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def euclidean_angle(x1: float, y1: float, x2: float, y2: float, do360: bool = False) -> float:
    """
    This will calculate the angle between the two points
    The value reported will be between 0 and pi if do360 is False (default) or between 0 and 2pi if do360 is True

    :param x1: x-coordinate of first point
    :param y1: y-coordinate of first point
    :param x2: x-coordinate of second point
    :param y2: y-coordinate of second point
    :param do360: if True, the result will be reported from 0 to 2pi, rather than 0 to pi
    :return: angle as a value between 0 and pi
    """
    angle = atan2(y2-y1, x2-x1)
    if do360:
        f = 2
    else:
        f = 1
    while angle >= f*pi:
        angle -= f*pi
    while angle < 0:
        angle += f*pi
    return angle


def check_for_square_matrix(distances: numpy.ndarray) -> int:
    """
    checks to see that a provided distance matrix is  two-dimensional numpy.ndarray and square

    :param distances: an n x n matrix
    :return: returns the size (length of a size) of the matrix assuming it is two-dimensional and square
    """
    if distances.ndim != 2:
        raise ValueError("distance matrix must be two-dimensional")
    elif distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    else:
        return len(distances)
