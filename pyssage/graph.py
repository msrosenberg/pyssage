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


# def plot_profiles(data, labels, mask, title):
#     fdata = data[:, mask]
#     tmplab = ""
#     n = len(fdata[0, :])
#     x = [i for i in range(n)]
#     ncols = 6
#     nrows = 6
#     fig, axs = plt.subplots(nrows, ncols)
#     row = 0
#     for rowax in axs:
#         for ax in rowax:
#             ax.get_xaxis().set_visible(False)
#             for i in range(4):
#                 ax.plot(x, fdata[row, :])
#                 tmplab = labels[row]
#                 row += 1
#             ax.set_title(tmplab[:len(tmplab)-1], fontsize=8)
#             ax.set_ylim(-21, 21)
#     fig.suptitle(title)
#     plt.show()
