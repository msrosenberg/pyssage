import numpy


def test_transect():
    """
    Creates a transect of 1000 cells, alternating 0 and 0.8 every 40 cells
    """
    transect = []
    for i in range(12):
        for j in range(40):
            transect.append(0)
        for j in range(40):
            transect.append(0.8)
    for j in range(40):  # last 40 cells are 0
        transect.append(0)
    return numpy.array(transect)
