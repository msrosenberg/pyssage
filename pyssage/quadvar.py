import numpy
from pyssage.classes import Number

__all__ = ["ttlqv"]


def wrap_transect(x: int, n: int) -> int:
    """
    Allows for wrapping an analysis across the ends of a linear transect as if it were a circle.
    Assumes zero-delimited indexing (transect positions are counted from 0 to n-1)

    :param x: the index of the requested position of the transect
    :param n: the length of the transect
    :return: returns the index of the correct position within the transect
    """
    if x >= n:
        return x - n
    elif x < 0:
        return x + n
    else:
        return x


def ttlqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int= 0, block_step: int = 1,
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
    if max_block_size == 0:
        max_block_size = n // 2
        # print("Maximum block size cannot exceed 50% of transect length. Reduced to", max_block_size)
    if max_block_size < 2:
        max_block_size = 2
    for b in range(min_block_size, max_block_size + 1, block_step):
        cnt = 0
        qv = 0
        if wrap:
            end_start_pos = n
        else:
            end_start_pos = n + 1 - 2*b
        for start_pos in range(end_start_pos):
            sum1 = 0
            sum2 = 0
            for i in range(start_pos, start_pos + b):
                j = wrap_transect(i, n)
                sum1 += transect[j]
            for i in range(start_pos + b, start_pos + 2*b):
                j = wrap_transect(i, n)
                sum2 += transect[j]
            cnt += 1
            qv += (sum1 - sum2)**2
        qv /= 2*b*cnt
        output.append([b*unit_scale, qv])
    return numpy.array(output)
