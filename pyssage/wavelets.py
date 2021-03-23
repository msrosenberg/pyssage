from typing import Tuple
from collections import namedtuple
from math import sin, cos, pi, sqrt, exp, log
import numpy
from pyssage.classes import Number
from pyssage.utils import check_block_size


#  ---------------Wavelet Functions---------------
def haar_wavelet(d: float) -> int:
    if -1 <= d < 0:
        return -1
    elif 0 <= d < 1:
        return 1
    else:
        return 0


def french_tophat_wavelet(d: float) -> int:
    if 0.5 <= abs(d) < 1.5:
        return -1
    elif abs(d) < 0.5:
        return 2
    else:
        return 0


def mexican_hat_wavelet(d: float) -> float:
    return (2/sqrt(3)) * pi**(-1/4) * (1 - 4*d**2) * exp(-2*d**2)


def morlet_wavelet(d: float) -> float:
    # returns the real part of the Morlet wavelet
    return pi**(-1/4) * cos((d/2)*pi*sqrt(2/log(2))) * exp(-1 * d**2 / 8)


def sine_wavelet(d: float) -> float:
    if abs(d) > 1:
        return 0
    else:
        return sin(pi*d)


def wavelet_subanalysis(_transect: numpy.ndarray, n: int, win_width: float, wavelet=haar_wavelet,
                        min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1, wrap: bool = False,
                        unit_scale: Number = 1) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    primary wavelet analysis function, separated so can readily be repreated if randomization tests are performed
    """
    start_start_pos = n
    end_start_pos = 2*n
    v_output = []
    w_output = []
    for b in range(min_block_size, max_block_size + 1, block_step):
        w_row = [numpy.nan for _ in range(n)]
        w_output.append(w_row)
        v = 0
        if not wrap:
            start_start_pos = round(win_width*b)
            end_start_pos = round(n + 1 - win_width*b)
        cnt = 0
        for start_pos in range(start_start_pos, end_start_pos):
            if not wrap:
                actual_p = start_pos
            else:
                actual_p = start_pos - n
            startq = start_pos - round(win_width * b)
            endq = start_pos + round(win_width * b) + 1
            if not wrap:
                startq = max(startq, 0)
                endq = min(endq, n)
            tmp_sum = 0
            for i in range(startq, endq):
                d = (i - start_pos) / b
                tmp_sum += _transect[i] * wavelet(d)
            w_row[actual_p] = tmp_sum/b

            v += (tmp_sum/b)**2
            cnt += 1
        try:
            v_output.append([b*unit_scale, v/cnt])
        except ZeroDivisionError:
            v_output.append([b*unit_scale, numpy.nan])
    p_output = []
    for p in range(n):
        pv = 0
        cnt = 0
        for b in range(len(w_output)):
            if w_output[b][p] is not numpy.nan:
                pv += w_output[b][p]**2
                cnt += 1

        try:
            p_output.append([p*unit_scale, pv / cnt])
        except ZeroDivisionError:
            p_output.append([p*unit_scale, numpy.nan])
    return numpy.array(w_output), numpy.array(v_output), numpy.array(p_output)


# ---------------Primary Function---------------
def wavelet_analysis(transect: numpy.ndarray, wavelet=haar_wavelet, min_block_size: int = 1, max_block_size: int = 0,
                     block_step: int = 1, wrap: bool = False, unit_scale: Number = 1, npermutations: int = 0,
                     alpha: float = 0.05) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                                                   numpy.ndarray, numpy.ndarray]:
    """
    Performs a wavelet analysis on a transect

    :param transect: a single dimensional numpy array containing the transect data
    :param wavelet: the name of the wavelet function to use; default is haar_wavelet
    :param min_block_size: the smallest block size of the analysis (default = 1)
    :param max_block_size: the largest block size of the analysis (default = 0, indicating 50% of the transect length)
    :param block_step: the incremental size increase of each block size (default = 1)
    :param wrap: treat the transect as a circle where the ends meet (default = False)
    :param unit_scale: represents the unit scale of a single block (default = 1). Can be used to rescale the units of
           the output, e.g., if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
           output expressed in meters.
    :return: a tuple of three numpy arrays, the first a matrix containing scale x position variances, the second
             a two column array containing variances by scale, and the third a two column array containing variances
             by positions
    """
    win_width = 1
    if wavelet == french_tophat_wavelet:
        win_width = 1.5
    elif wavelet == mexican_hat_wavelet:
        win_width = 2
    elif wavelet == morlet_wavelet:
        win_width = 6
    n = len(transect)
    max_block_size = check_block_size(max_block_size, n, win_width*2)
    if wrap:
        _transect = numpy.append(transect, transect)
        _transect = numpy.append(transect, _transect)
    else:
        _transect = transect

    obs_w, obs_v, obs_p = wavelet_subanalysis(_transect, n, win_width, wavelet, min_block_size, max_block_size,
                                              block_step, wrap, unit_scale)
    if npermutations > 0:
        # create output matrices and store observed data
        v_output = numpy.empty((len(obs_v), 3))
        p_output = numpy.empty((len(obs_p), 3))
        w_output = numpy.empty((len(obs_v), len(obs_p), 3))
        v_all = numpy.empty((len(obs_v), npermutations+1))
        p_all = numpy.empty((len(obs_p), npermutations+1))
        w_all = numpy.empty((len(obs_v), len(obs_p), npermutations))
        v_output[:, 0:2] = obs_v[:, 0:2]
        p_output[:, 0:2] = obs_p[:, 0:2]
        w_output[:, :, 0] = obs_w[:, :]
        v_all[:, 0:2] = obs_v[:, 0:2]
        p_all[:, 0:2] = obs_p[:, 0:2]
        w_all[:, :, 0] = obs_w[:, :]
        rand_transect = numpy.copy(transect)
        for rep in range(npermutations-1):
            numpy.random.shuffle(rand_transect)
            if wrap:
                _transect = numpy.append(rand_transect, rand_transect)
                _transect = numpy.append(rand_transect, _transect)
            else:
                _transect = rand_transect
            rand_w, rand_v, rand_p = wavelet_subanalysis(_transect, n, win_width, wavelet, min_block_size,
                                                         max_block_size, block_step, wrap, unit_scale)
            v_all[:, rep+2] = rand_v[:, 1]
            p_all[:, rep+2] = rand_p[:, 1]
            w_all[:, :, rep+1] = rand_w[:, :]
        # identify random values at alpha level
        alpha_index = round(npermutations * (1-alpha)) - 1
        w_high_alpha_index = round(npermutations * (1-alpha/2)) - 1
        w_low_alpha_index = round(npermutations * (alpha/2)) - 1
        for b in range(len(v_all)):
            tmp = v_all[b, 1:]  # pull out row, excluding first column
            tmp = numpy.sort(tmp)
            v_output[b, 2] = tmp[alpha_index]
        for p in range(len(p_all)):
            tmp = p_all[p, 1:]  # pull out row, excluding first column
            tmp = numpy.sort(tmp)
            p_output[p, 2] = tmp[alpha_index]
            for b in range(len(v_all)):
                tmp = w_all[b, p, :]
                tmp = tmp.flatten()
                tmp = numpy.sort(tmp)
                w_output[b, p, 1] = tmp[w_low_alpha_index]
                w_output[b, p, 2] = tmp[w_high_alpha_index]
    else:
        v_output = obs_v
        p_output = obs_p
        w_output = obs_w
        v_all = None
        p_all = None
        w_all = None

    wavelet_output_tuple = namedtuple("wavelet_output_tuple", ["w_output", "v_output", "p_output", "w_all", "v_all",
                                                               "p_all"])
    return wavelet_output_tuple(w_output, v_output, p_output, w_all, v_all, p_all)
