from pyssage.classes import Number
import matplotlib.pyplot as pyplot
import numpy


def draw_transect(transect: numpy.array, unit_scale: Number = 1, title: str = "") -> None:
    x = [i*unit_scale for i in range(len(transect))]
    fix, axs = pyplot.subplots()
    axs.plot(x, transect)
    axs.set_xlabel("Position")
    axs.set_ylabel("Value")
    if title != "":
        axs.set_title(title)
    pyplot.show()


def draw_quadvar_result(quadvar: numpy.array, title: str = "") -> None:
    fix, axs = pyplot.subplots()
    axs.plot(quadvar[:, 0], quadvar[:, 1])
    axs.set_xlabel("Scale")
    axs.set_ylabel("Variance")
    if title != "":
        axs.set_title(title)
    pyplot.show()
