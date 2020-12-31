import pyssage.graph
from tests.test_common import create_test_transect


def test_draw_transect():
    # formal test not possible for graphical output
    figoutput = pyssage.graph.FigOutput(figshow=True)
    pyssage.graph.draw_transect(create_test_transect(), 1, figoutput=figoutput)
    pyssage.graph.draw_transect(create_test_transect(), 0.1, figoutput=figoutput)
