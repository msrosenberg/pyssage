import numpy
from math import pi, atan2
from pyssage.common import OUT_DEC


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


def create_output_table(output_text: list, table_data: list, col_headers: tuple, col_formats: tuple,
                        out_dec: int = OUT_DEC, sbc: int = 3) -> None:
    """
    Create a well-formatted set of strings representing an output table, including headers and computationally
    determined spacing of columns. The table is added to a list provided as input so can be inserted into a
    broader set of output strings.

    :param output_text: a list where the output will be appended; it can be empty or contain strings, but must
                        already have been instantiated
    :param table_data: a list of lists containing the data to appear in the table; each sublist represents a row
                       of the table and must contain the same number of columns
    :param col_headers: a tuple containing strings representing headers for each column in the table
    :param col_formats: a tuple containing basic string formatting codes, generally expected to be "f" of "d"
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
