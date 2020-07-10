import pyssage.graph
from tests.test_common import create_test_transect


def test_draw_transect():
    # formal test not possible for graphical output
    pyssage.graph.draw_transect(create_test_transect(), 1)
    pyssage.graph.draw_transect(create_test_transect(), 0.1)
