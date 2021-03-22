import numpy
from math import pi, atan2
from pyssage.common import OUT_DEC


def flatten_half(square_matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Given a square, symmmetric matrix, return the values from the lower triangle in a single column

    :param square_matrix: the input matrix, should be an n x n numpy.ndarray
    :return: a single column numpy.ndarray
    """
    output = []
    for i in range(len(square_matrix)):
        for j in range(i):
            output.append(square_matrix[i, j])
    return numpy.array(output)


def flatten_without_diagonal(square_matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Given a square matrix, return the values from the non-diagonal elements in a single column

    :param square_matrix: the input matrix, should be an n x n numpy.ndarray
    :return: a single column numpy.ndarray
    """
    output = []
    for i in range(len(square_matrix)):
        for j in range(len(square_matrix)):
            if i != j:
                output.append(square_matrix[i, j])
    return numpy.array(output)


def deflatten_without_diagonal(vector: numpy.ndarray, n: int) -> numpy.ndarray:
    """
    Given a vector, recreate a square-matrix where the diagonal is assumed to be absent

    :param vector: the input vector, should contain n(n-1) elements
    :param n: the size of the desired n x n numpy.ndarray
    :return: a single column numpy.ndarray
    """
    if len(vector) != n*(n-1):
        raise ValueError("length of vector does not match desired output size of square matrix")
    output = []
    c = 0
    for i in range(n):
        row = []
        for j in range(n):
            if i != j:
                row.append(vector[c])
                c += 1
            else:
                row.append(0)
        output.append(row)
    return numpy.array(output)


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


def check_for_square_matrix(test_matrix: numpy.ndarray) -> int:
    """
    checks to see that a provided distance matrix is  two-dimensional numpy.ndarray and square

    :param test_matrix: an n x n matrix
    :return: returns the size (length of a size) of the matrix assuming it is two-dimensional and square
    """
    if test_matrix.ndim != 2:
        raise ValueError("distance matrix must be two-dimensional")
    elif test_matrix.shape[0] != test_matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    else:
        return len(test_matrix)


def create_output_table(output_text: list, table_data: list, col_headers: list, col_formats: list,
                        out_dec: int = OUT_DEC, sbc: int = 3) -> None:
    """
    Create a well-formatted set of strings representing an output table, including headers and computationally
    determined spacing of columns. The table is added to a list provided as input so can be inserted into a
    broader set of output strings.

    :param output_text: a list where the output will be appended; it can be empty or contain strings, but must
                        already have been instantiated
    :param table_data: a list of lists containing the data to appear in the table; each sublist represents a row
                       of the table and must contain the same number of columns
    :param col_headers: a list containing strings representing headers for each column in the table
    :param col_formats: a list containing basic string formatting codes, generally expected to be "f" of "d"
    :param out_dec: the number of decimal places to output floating point numbers in the table
    :param sbc: the number of spaces to use between each column in the table
    """
    # determine maximum width for each column
    max_width = [len(h) for h in col_headers]
    for row in table_data:
        for i, x in enumerate(row):
            if col_formats[i] == "f":
                frmt = "0.{}f".format(out_dec)
            else:
                frmt = col_formats[i]
            max_width[i] = max(max_width[i], len(format(x, frmt)))

    col_spacer = " "*sbc

    # create table header
    cols = [format(h, ">{}".format(max_width[i])) for i, h in enumerate(col_headers)]
    header = col_spacer.join(cols)
    output_text.append(header)
    output_text.append("-"*len(header))

    # create table data template
    cols = []
    for i, f in enumerate(col_formats):
        if f == "f":
            frmt = "{{:>{}.{}f}}".format(max_width[i], out_dec)
        else:
            frmt = "{{:>{}{}}}".format(max_width[i], f)
        cols.append(frmt)
    template = col_spacer.join(cols)

    # create table data
    for row in table_data:
        output_text.append(template.format(*row))


def check_block_size(max_block_size: int, n: int, x: int) -> int:
    """
    Check the maximum block size to be sure it doesn't exceed limits for the particular analysis and input data

    :param max_block_size: the requested largest block size
    :param n: the length of the transect
    :param x: the number of "blocks" that make up the analysis; this affects the maximum allowable size
    :return: the maximum block size that will actually be used in the analysis
    """
    if max_block_size == 0:
        max_block_size = n // x
    if max_block_size < 2:
        max_block_size = 2
    elif max_block_size > n // x:
        max_block_size = n // x
        print("Maximum block size cannot exceed {:0.1f}% of transect length. Reduced to {}.".format(100 / x,
                                                                                                    max_block_size))
    return max_block_size
