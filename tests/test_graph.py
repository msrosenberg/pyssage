import pyssage.graph
from tests.test_common import test_transect


def test_draw_transect():
    pyssage.graph.draw_transect(test_transect(), 1)
    pyssage.graph.draw_transect(test_transect(), 0.1)
