from typing import Tuple
from collections import namedtuple
import numpy
from pyssage.classes import Number
from pyssage.utils import check_block_size
# from datetime import datetime

__all__ = ["ttlqv", "three_tlqv", "pqv", "tqv", "two_nlv", "three_nlv", "four_tlqv", "five_qv", "nine_tlqv",
           "quadrat_variance_randomization"]

_2D_FUNCS = ("four_tlqv", "five_qv", "nine_tlqv")


def ttlqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
          wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Two-Term Local Quadrat Variance analysis (TTLQV) on a transect. Method originally from:

    Hill, M.O. 1973. The intensity of spatial pattern in plant communities. Journal of Ecology 61:225-235.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 2)
    if wrap:
        _transect = numpy.append(transect, transect)
    else:
        _transect = transect
    end_start_pos = n
    for b in range(min_block_size, max_block_size + 1, block_step):
        qv = 0
        if not wrap:
            end_start_pos = n + 1 - 2*b
        for start_pos in range(end_start_pos):
            sum1 = numpy.sum(_transect[start_pos:start_pos + b])
            sum2 = numpy.sum(_transect[start_pos + b:start_pos + 2*b])
            qv += (sum1 - sum2)**2
        try:
            qv /= 2*b*end_start_pos
            output.append([b*unit_scale, qv])
        except ZeroDivisionError:
            pass

    return numpy.array(output)


def three_tlqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
               wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Three-Term Local Quadrat Variance analysis (3TLQV) on a transect. Method originally from:

    Hill, M.O. 1973. The intensity of spatial pattern in plant communities. Journal of Ecology 61:225-235.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 33% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 3)
    if wrap:
        _transect = numpy.append(transect, transect)
    else:
        _transect = transect
    end_start_pos = n
    for b in range(min_block_size, max_block_size + 1, block_step):
        # cnt = 0
        qv = 0
        if not wrap:
            end_start_pos = n + 1 - 3*b
        for start_pos in range(end_start_pos):
            sum1 = numpy.sum(_transect[start_pos:start_pos + b])
            sum2 = numpy.sum(_transect[start_pos + b:start_pos + 2*b])
            sum3 = numpy.sum(_transect[start_pos + 2*b:start_pos + 3*b])
            qv += (sum1 - 2*sum2 + sum3)**2
        try:
            qv /= 8*b*end_start_pos
            output.append([b * unit_scale, qv])
        except ZeroDivisionError:
            pass

    return numpy.array(output)


def pqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
        wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Paired Quadrat Variance analysis (PQV) on a transect.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 2)
    if wrap:
        _transect = numpy.append(transect, transect)
    else:
        _transect = transect
    end_start_pos = n
    for b in range(min_block_size, max_block_size + 1, block_step):
        qv = 0
        if not wrap:
            end_start_pos = n - b
        for start_pos in range(end_start_pos):
            qv += (_transect[start_pos] - _transect[start_pos + b])**2
        try:
            qv /= 2*end_start_pos
            output.append([b*unit_scale, qv])
        except ZeroDivisionError:
            pass

    return numpy.array(output)


def tqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
        wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Triplet Quadrat Variance analysis (tQV) on a transect. Method originally from:

    Dale, M.R T. 1999. Spatial Pattern Analysis in Plant Ecology. Cambridge: Cambridge University Press.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 2)
    if wrap:
        _transect = numpy.append(transect, transect)
    else:
        _transect = transect
    end_start_pos = n
    for b in range(min_block_size, max_block_size + 1, block_step):
        qv = 0
        if not wrap:
            end_start_pos = n - 2*b
        for start_pos in range(end_start_pos):
            qv += (_transect[start_pos] - 2*_transect[start_pos + b] + _transect[start_pos + 2*b])**2
        try:
            qv /= 4*end_start_pos
            output.append([b*unit_scale, qv])
        except ZeroDivisionError:
            pass

    return numpy.array(output)


def two_nlv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
            wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Two-Term New Local Variance analysis (2NLV) on a transect. Method originally from:

    Galiano, E.F. 1982. Détection et mesure de l'hétérogénéité spatiale des espèces dans les pâturages.
    Acta OEcologia / OEcologia Plantarum 3:269-278.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 2)
    if wrap:
        _transect = numpy.append(transect, transect)
    else:
        _transect = transect
    end_start_pos = n
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if not wrap:
            end_start_pos = n - 2*b
        for start_pos in range(end_start_pos):
            sum1 = numpy.sum(_transect[start_pos:start_pos + b])
            sum2 = numpy.sum(_transect[start_pos + b:start_pos + 2*b])
            term1 = (sum1 - sum2)**2
            sum3 = sum1 - _transect[start_pos] + _transect[start_pos + b]
            sum4 = sum2 - _transect[start_pos + b] + _transect[start_pos + 2*b]
            term2 = (sum3 - sum4)**2
            cnt += 1
            qv += abs(term1 - term2)
        try:
            qv /= 2*b*end_start_pos
            output.append([b*unit_scale, qv])
        except ZeroDivisionError:
            pass

    return numpy.array(output)


def three_nlv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
              wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Three-Term New Local Variance analysis (2NLV) on a transect. Method originally from:

    Galiano, E.F. 1982. Détection et mesure de l'hétérogénéité spatiale des espèces dans les pâturages.
    Acta OEcologia / OEcologia Plantarum 3:269-278.

    :param transect: a single dimensional numpy array containing the transect data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 33% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    n = len(transect)
    output = []
    max_block_size = check_block_size(max_block_size, n, 3)
    if wrap:
        _transect = numpy.append(transect, transect)
    else:
        _transect = transect
    end_start_pos = n
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if not wrap:
            end_start_pos = n - 3*b
        for start_pos in range(end_start_pos):
            sum1 = numpy.sum(_transect[start_pos:start_pos + b])
            sum2 = numpy.sum(_transect[start_pos + b:start_pos + 2*b])
            sum3 = numpy.sum(_transect[start_pos + 2*b:start_pos + 3*b])
            term1 = (sum1 - 2*sum2 + sum3)**2
            sum4 = sum1 - _transect[start_pos] + _transect[start_pos + b]
            sum5 = sum2 - _transect[start_pos + b] + _transect[start_pos + 2*b]
            sum6 = sum3 - _transect[start_pos + 2*b] + _transect[start_pos + 3*b]
            term2 = (sum4 - 2*sum5 + sum6)**2

            cnt += 1
            qv += abs(term1 - term2)
        try:
            qv /= 8*b*end_start_pos
            output.append([b*unit_scale, qv])
        except ZeroDivisionError:
            pass

    return numpy.array(output)


def quadrat_variance_randomization(qv_function, nreps: int, data: numpy.ndarray, min_block_size: int = 1,
                                   max_block_size: int = 0, block_step: int = 1, wrap: bool = False,
                                   unit_scale: Number = 1, alpha: float = 0.05) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Uses multiple permutations of the input datato create a confidence limit for the various quadrat variance tests.

    This function can work with any of the quadrat variance methods. The specific method to be used in specified by
    the first parameter which is the function to use

    :param qv_function: the quadrat variance function to use as the basis of the tests
    :param nreps: the number of replicates to perform. By definition the observed data will be treated as the first
                  replicate
    :param data: a one or two-dimensional numpy array containing the transect/surface data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating the maximum possible size)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False) (1D analysis only)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :param alpha: the alpha value to report the confidence limit at [default = 0.05]
    :return: a tuple of two numpy.ndarray's. The first contains three columns: (1) the scaled block size, (2) the
             observed quadrat variance, and (3) the confidence limit determined from the permutated data. The second
             matrix contains the results from all of the individual permutation tests. The first column is the block
             size, the second the observed quadrat variance, and the remaining columns the results from each permuted
             data set.
    """
    if qv_function in _2D_FUNCS:
        base_output = qv_function(data, min_block_size, max_block_size, block_step, unit_scale)
    else:
        base_output = qv_function(data, min_block_size, max_block_size, block_step, wrap, unit_scale)
    all_output = numpy.empty((len(base_output), nreps+1))
    summary_output = numpy.empty((len(base_output), 3))
    all_output[:, 0:2] = base_output[:, 0:2]
    summary_output[:, 0:2] = base_output[:, 0:2]
    shape = data.shape
    rand_data = numpy.copy(data)
    for rep in range(nreps-1):
        rand_data = rand_data.flatten()
        numpy.random.shuffle(rand_data)
        rand_data = rand_data.reshape(shape)
        if qv_function in _2D_FUNCS:
            rand_output = qv_function(rand_data, min_block_size, max_block_size, block_step, unit_scale)
        else:
            rand_output = qv_function(rand_data, min_block_size, max_block_size, block_step, wrap, unit_scale)
        all_output[:, rep+2] = rand_output[:, 1]
    alpha_index = round(nreps * (1-alpha)) - 1
    for r in range(len(all_output)):
        tmp = all_output[r, 1:]  # pull out row, excluding first column
        tmp = numpy.sort(tmp)
        summary_output[r, 2] = tmp[alpha_index]
    output = namedtuple("output", ["summary_output", "all_output"])
    return output(summary_output, all_output)


def check_2d_block_size(max_block_size: int, n: int, x: int) -> int:
    """
    Check the maximum block size to be sure it doesn't exceed limits for the particular analysis and input data

    :param max_block_size: the requested largest block size
    :param n: the smallest side of the surface
    :param x: the number of "blocks" (in 1D) that make up the analysis; this affects the maximum allowable size
    :return: the maximum block size that will actually be used in the analysis
    """
    if max_block_size == 0:
        max_block_size = n // x
    if max_block_size < 2:
        max_block_size = 2
    elif max_block_size > n // x:
        max_block_size = n // x
        print("Maximum block size cannot exceed {:0.1f}% of the smallest side of the surface. "
              "Reduced to {}.".format(100 / x, max_block_size))
    return max_block_size


def four_tlqv(surface: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
              unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Four-Term Local Quadrat Variance analysis (4TLQV) on a surface. Method originally from:

    xxxxx

    :param surface: a two-dimensional numpy array containing the surface data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    nrows, ncols = surface.shape
    max_block_size = check_2d_block_size(max_block_size, min(nrows, ncols), 2)
    output = []
    for b in range(min_block_size, max_block_size + 1, block_step):
        qv = 0
        end_row_start = nrows + 1 - 2*b
        end_col_start = ncols + 1 - 2*b
        for row in range(end_row_start):
            for col in range(end_col_start):
                sum1 = numpy.sum(surface[row:row + b, col:col + b])
                sum2 = numpy.sum(surface[row:row + b, col + b:col + 2*b])
                sum3 = numpy.sum(surface[row + b:row + 2*b, col:col + b])
                sum4 = numpy.sum(surface[row + b:row + 2*b, col + b:col + 2*b])
                # treat each block as a potential focal block, relative to the other three
                qv += (3*sum1 - sum2 - sum3 - sum4)**2 + (3*sum2 - sum1 - sum3 - sum4)**2 + \
                      (3*sum3 - sum1 - sum2 - sum4)**2 + (3*sum4 - sum1 - sum2 - sum3)**2

        qv /= 32 * b**3 * end_row_start * end_col_start
        output.append([b*unit_scale, qv])

    return numpy.array(output)


def nine_tlqv(surface: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
              unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Nine-Term Local Quadrat Variance analysis (9TLQV) on a surface. Method originally from:

    xxxxx

    :param surface: a two-dimensional numpy array containing the surface data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 33% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    nrows, ncols = surface.shape
    max_block_size = check_2d_block_size(max_block_size, min(nrows, ncols), 3)
    output = []
    for b in range(min_block_size, max_block_size + 1, block_step):
        qv = 0
        end_row_start = nrows + 1 - 3*b
        end_col_start = ncols + 1 - 3*b
        for row in range(end_row_start):
            for col in range(end_col_start):
                sum_outer = numpy.sum(surface[row:row + 3*b, col:col + b])  # first column
                sum_outer += numpy.sum(surface[row:row + 3*b, col + 2*b:col + 3*b])  # third column
                sum_outer += numpy.sum(surface[row:row + b, col + b:col + 2*b])  # top row, middle column
                sum_outer += numpy.sum(surface[row + 2*b:row + 3*b, col + b:col + 2*b])  # bottom row, middle column
                sum_middle = numpy.sum(surface[row + b:row + 2*b, col + b:col + 2*b])  # middle cell
                qv += (8*sum_middle - sum_outer)**2

        qv /= 72 * b**3 * end_row_start * end_col_start
        output.append([b*unit_scale, qv])

    return numpy.array(output)


def five_qv(surface: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
            unit_scale: Number = 1) -> numpy.ndarray:
    """
    Performs a Pentuplet Quadrat Variance analysis (5QV) on a surface. Method originally from:

    xxxx

    :param surface: a two-dimensional numpy array containing the surface data
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a two column numpy array, with the first column containing the scaled block size and the second the
             calculated variance
    """
    nrows, ncols = surface.shape
    max_block_size = check_2d_block_size(max_block_size, min(nrows, ncols), 2)
    output = []
    for b in range(min_block_size, max_block_size + 1, block_step):
        qv = 0
        end_row_start = nrows - 2*b
        end_col_start = ncols - 2*b
        for row in range(end_row_start):
            for col in range(end_col_start):
                qv += (4*surface[row + b, col + b] - surface[row + b, col] - surface[row, col + b] -
                       surface[row + 2*b, col + b] - surface[row + b, col + 2*b])**2
        try:
            qv /= 20 * end_row_start * end_col_start
            output.append([b*unit_scale, qv])
        except ZeroDivisionError:
            pass

    return numpy.array(output)
