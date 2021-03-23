import pyssage.wavelets
import pyssage.graph
from tests.test_common import create_test_transect


def test_haar_wavelet_analysis():
    w_out, v_out, p_out = pyssage.wavelets.wavelet_analysis(create_test_transect(), max_block_size=100)
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, figoutput=pyssage.graph.FigOutput(figshow=True))
    # draw all alternative configurations
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, inc_positional_var=False,
                                      figoutput=pyssage.graph.FigOutput(figshow=True))
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, inc_scale_var=False,
                                      figoutput=pyssage.graph.FigOutput(figshow=True))
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, inc_scale_x_pos=False,
                                      figoutput=pyssage.graph.FigOutput(figshow=True))
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, inc_positional_var=False, inc_scale_var=False,
                                      figoutput=pyssage.graph.FigOutput(figshow=True))
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, inc_positional_var=False, inc_scale_x_pos=False,
                                      figoutput=pyssage.graph.FigOutput(figshow=True))
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, inc_scale_x_pos=False, inc_scale_var=False,
                                      figoutput=pyssage.graph.FigOutput(figshow=True))


def test_haar_wavelet_analysis_rescaled():
    w_out, v_out, p_out = pyssage.wavelets.wavelet_analysis(create_test_transect(), max_block_size=100, unit_scale=0.1)
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_french_tophat_wavelet_analysis():
    w_out, v_out, p_out = pyssage.wavelets.wavelet_analysis(create_test_transect(), max_block_size=100,
                                                            wavelet=pyssage.wavelets.french_tophat_wavelet)
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_sine_wavelet_analysis():
    w_out, v_out, p_out = pyssage.wavelets.wavelet_analysis(create_test_transect(), max_block_size=100,
                                                            wavelet=pyssage.wavelets.sine_wavelet)
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_mexican_hat_wavelet_analysis():
    w_out, v_out, p_out = pyssage.wavelets.wavelet_analysis(create_test_transect(), max_block_size=100,
                                                            wavelet=pyssage.wavelets.mexican_hat_wavelet)
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_morlet_wavelet_analysis():
    w_out, v_out, p_out = pyssage.wavelets.wavelet_analysis(create_test_transect(), max_block_size=100,
                                                            wavelet=pyssage.wavelets.morlet_wavelet)
    pyssage.graph.draw_wavelet_result(v_out, p_out, w_out, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_haar_wavelet_template():
    # draw the Haar wavelet
    nsteps = 100
    y = []
    x = []
    for i in range(nsteps):
        p = -2 + 4*i/nsteps
        x.append(p)
        y.append(pyssage.wavelets.haar_wavelet(p))
    pyssage.graph.draw_wavelet_template(x, y, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_french_tophat_wavelet_template():
    # draw the French Tophat wavelet
    nsteps = 100
    y = []
    x = []
    for i in range(nsteps):
        p = -2 + 4*i/nsteps
        x.append(p)
        y.append(pyssage.wavelets.french_tophat_wavelet(p))
    pyssage.graph.draw_wavelet_template(x, y, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_mexican_hat_wavelet_template():
    # draw the Mexican hat wavelet
    nsteps = 100
    y = []
    x = []
    for i in range(nsteps):
        p = -2.5 + 5*i/nsteps
        x.append(p)
        y.append(pyssage.wavelets.mexican_hat_wavelet(p))
    pyssage.graph.draw_wavelet_template(x, y, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_morlet_wavelet_template():
    # draw the Morlet wavelet
    nsteps = 1200
    y = []
    x = []
    for i in range(nsteps):
        p = -6 + 12*i/nsteps
        x.append(p)
        y.append(pyssage.wavelets.morlet_wavelet(p))
    pyssage.graph.draw_wavelet_template(x, y, figoutput=pyssage.graph.FigOutput(figshow=True))


def test_sine_wavelet_template():
    # draw the sine wavelet
    nsteps = 100
    y = []
    x = []
    for i in range(nsteps):
        p = -1.5 + 3*i/nsteps
        x.append(p)
        y.append(pyssage.wavelets.sine_wavelet(p))
    pyssage.graph.draw_wavelet_template(x, y, figoutput=pyssage.graph.FigOutput(figshow=True))
