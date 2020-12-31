from typing import Optional
from math import pi
from pyssage.classes import Number
import pyssage.connections
import pyssage.distances
import pyssage.utils
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D
from matplotlib import collections, colors
# import matplotlib.patches as mpatches
import numpy


class FigOutput:
    def __init__(self, figsize: tuple = (8, 6), dpi: int = 100, figshow: bool = False, figname: str = "",
                 figformat: str = "png"):
        self.figsize = figsize
        self.dpi = dpi
        self.figshow = figshow
        self.figname = figname
        self.figformat = figformat


def check_valid_graph_format(x: str) -> bool:
    valid_formats = pyplot.gcf().canvas.get_supported_filetypes()
    if x in valid_formats:
        return True
    else:
        error_msg = "Invalid Figure Format. Valid formats include:\n"
        for f in valid_formats:
            error_msg += "{:4}    {}\n".format(f, valid_formats[f])
        raise ValueError(error_msg)


def finalize_figure(fig, figoutput: FigOutput) -> None:
    if figoutput.figname != "":
        if check_valid_graph_format(figoutput.figformat):
            fig.savefig(figoutput.figname, format=figoutput.figformat, dpi=figoutput.dpi)
    if figoutput.figshow:
        pyplot.show()


def draw_transect(transect: numpy.array, unit_scale: Number = 1, title: str = "",
                  figoutput: Optional[FigOutput] = None) -> None:
    if figoutput is None:
        figoutput = FigOutput()
    fig, axs = pyplot.subplots(figsize=figoutput.figsize, dpi=figoutput.dpi)
    x = [i*unit_scale for i in range(len(transect))]
    axs.plot(x, transect)
    axs.set_xlabel("Position")
    axs.set_ylabel("Value")
    if title != "":
        axs.set_title(title)
    finalize_figure(fig, figoutput)


# def draw_transect(transect: numpy.array, unit_scale: Number = 1, title: str = "", figsize: tuple = DEFAULT_FIGSIZE,
#                   dpi: int = DEFAULT_DPI, figname: str = "", figformat: str = DEFAULT_FIGFORMAT,
#                   figshow: bool = False) -> None:
#     fig, axs = pyplot.subplots(figsize=figsize, dpi=dpi)
#     x = [i*unit_scale for i in range(len(transect))]
#     axs.plot(x, transect)
#     axs.set_xlabel("Position")
#     axs.set_ylabel("Value")
#     if title != "":
#         axs.set_title(title)
#     finalize_figure(fig, figname, figformat, dpi, figshow)


def draw_quadvar_result(quadvar: numpy.ndarray, rand_ci: Optional[numpy.ndarray] = None, title: str = "",
                        varlabel: str = "", randlabel: str = "", figoutput: Optional[FigOutput] = None) -> None:
    fig, axs = pyplot.subplots(figsize=figoutput.figsize, dpi=figoutput.dpi)
    axs.plot(quadvar[:, 0], quadvar[:, 1], label=varlabel)
    if rand_ci is not None:
        axs.plot(quadvar[:, 0], rand_ci, label=randlabel)
        pyplot.legend(loc="upper right")
    axs.set_xlabel("Scale")
    axs.set_ylabel("Variance")
    if title != "":
        axs.set_title(title)
    finalize_figure(fig, figoutput)


# def draw_triangles(triangles: list, coords, title: str = "") -> None:
#     # this was for debugging purposes only
#     fig, axs = pyplot.subplots()
#     minx = min(coords[:, 0])
#     maxx = max(coords[:, 0])
#     miny = min(coords[:, 1])
#     maxy = max(coords[:, 1])
#     for t in triangles:
#         x = [p.x for p in t.points]
#         x.append(t.points[0].x)
#         y = [p.y for p in t.points]
#         y.append(t.points[0].y)
#         line = Line2D(x, y)
#         axs.add_line(line)
#         # p = mpatches.Circle((t.xc, t.yc), t.radius(), fill=False, color="red")
#         # axs.add_patch(p)
#     pyplot.scatter(coords[:, 0], coords[:, 1], color="black")
#     axs.set_xlim(minx-0.5, maxx+0.5)
#     axs.set_ylim(miny-0.5, maxy+0.5)
#     if title != "":
#         axs.set_title(title)
#     pyplot.show()


def draw_tessellation(tessellation, xcoords: numpy.ndarray, ycoords: numpy.ndarray, title: str = "",
                      figoutput: Optional[FigOutput] = None) -> None:
    fig, axs = pyplot.subplots(figsize=figoutput.figsize, dpi=figoutput.dpi)
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    for e in tessellation.edges:
        x = [e.start_vertex.x, e.end_vertex.x]
        y = [e.start_vertex.y, e.end_vertex.y]
        line = Line2D(x, y)
        axs.add_line(line)
    pyplot.scatter(xcoords, ycoords, color="black")
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    finalize_figure(fig, figoutput)


def check_connection_format(con_frmt: str) -> None:
    valid_formats = ("boolmatrix", "binmatrix", "revbinmatrix", "pairlist")
    if con_frmt not in valid_formats:
        raise ValueError("{} is not a valid connection format".format(con_frmt))


def add_connections_to_plot(axs, connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray):
    if connections.is_symmetric():
        done = []
        for i in range(len(connections)):
            for j in connections.connected_from(i):
                if [i, j] not in done:  # trying to avoid duplicating the reverse line
                    x = [xcoords[i], xcoords[j]]
                    y = [ycoords[i], ycoords[j]]
                    line = Line2D(x, y, zorder=1)
                    axs.add_line(line)
                    done.append([j, i])
    else:
        for i in range(len(connections)):
            for j in connections.connected_from(i):
                if connections[j, i]:
                    if i < j:
                        arrow = "<|-|>"
                    else:
                        arrow = None
                else:
                    arrow = "-|>"
                if arrow is not None:
                    axs.annotate(s="", xytext=(xcoords[i], ycoords[i]), xy=(xcoords[j], ycoords[j]),
                                 arrowprops=dict(arrowstyle=arrow, edgecolor="C0"), zorder=1)


def draw_connections(connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray, title: str = "",
                     figoutput: Optional[FigOutput] = None):
    fig, axs = pyplot.subplots(figsize=figoutput.figsize, dpi=figoutput.dpi)
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    add_connections_to_plot(axs, connections, xcoords, ycoords)
    pyplot.scatter(xcoords, ycoords, color="black", zorder=2)
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    finalize_figure(fig, figoutput)


def draw_shortest_path(connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray, trace_dict: dict,
                       startp: int, endp: int, title: str = "", figoutput: Optional[FigOutput] = None):
    fig, axs = pyplot.subplots(figsize=figoutput.figsize, dpi=figoutput.dpi)
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    add_connections_to_plot(axs, connections, xcoords, ycoords)

    trace_path = pyssage.distances.trace_path(startp, endp, trace_dict)
    for i in range(len(trace_path)-1):
        p1 = trace_path[i]
        p2 = trace_path[i+1]
        x = [xcoords[p1], xcoords[p2]]
        y = [ycoords[p1], ycoords[p2]]
        line = Line2D(x, y, color="red", zorder=3, lw=2)
        axs.add_line(line)

    pyplot.scatter(xcoords, ycoords, color="black", zorder=2)
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    finalize_figure(fig, figoutput)


def draw_distance_class_distribution(dist_matrix: numpy.ndarray, dist_class: numpy.ndarray, title: str = "",
                                     figoutput: Optional[FigOutput] = None):
    fig, axs = pyplot.subplots(figsize=figoutput.figsize, dpi=figoutput.dpi)
    distances = pyssage.utils.flatten_half(dist_matrix)
    distances.sort()
    total = len(distances)
    y = [i+1 for i in range(total)]
    r = []
    i = 0
    c = 0
    while c < len(dist_class) and i < total:
        if distances[i] >= dist_class[c, 1]:
            r.append(i+1)
            c += 1
        i += 1
    r.append(total)

    axs.plot(distances, y, zorder=1)
    for i in range(len(dist_class)):
        c = dist_class[i]
        upper = c[1]
        y = [0, r[i], r[i]]
        x = [upper, upper, 0]
        line = Line2D(x, y, color="red", zorder=2)
        axs.add_line(line)

    axs.set_xlabel("Distance")
    axs.set_ylabel("Cumulative Count")
    if title != "":
        axs.set_title(title)
    finalize_figure(fig, figoutput)


def draw_correlogram(data: numpy.ndarray, metric_title: str = "", title: str = "", alpha: float = 0.05,
                     is_mantel: bool = False, figoutput: Optional[FigOutput] = None):
    fig, axs = pyplot.subplots(figsize=figoutput.figsize, dpi=figoutput.dpi)

    # column order is: min_scale, max_scale, # pairs, expected, observed, sd, z, prob
    # sd is absent from Mantel correlograms
    min_col = 0
    max_col = 1
    exp_col = 3
    obs_col = 4
    if is_mantel:
        p_col = 6
    else:
        p_col = 7

    # plot at midpoint of distance range
    scale = numpy.array([x[min_col] + (x[max_col] - x[min_col])/2 for x in data])

    # draw expected values
    y = [data[0, exp_col], data[0, exp_col]]
    x = [0, scale[len(scale)-1]]
    # x = [0, data[len(data)-1, scale_col]]
    line = Line2D(x, y, color="silver", zorder=1)
    axs.add_line(line)

    # draw base line
    axs.plot(scale, data[:, obs_col], zorder=2)
    # axs.plot(data[:, scale_col], data[:, obs_col], zorder=2)

    # mark significant scales
    sig_mask = [p <= alpha for p in data[:, p_col]]
    # x = data[sig_mask, scale_col]
    x = scale[sig_mask]
    y = data[sig_mask, obs_col]
    pyplot.scatter(x, y, color="black", edgecolors="black", zorder=3, s=25)

    # mark non-significant scales
    ns_mask = numpy.invert(sig_mask)
    # x = data[ns_mask, scale_col]
    x = scale[ns_mask]
    y = data[ns_mask, obs_col]
    pyplot.scatter(x, y, color="white", edgecolors="black", zorder=3, s=15)

    axs.set_xlabel("Scale")
    axs.set_ylabel(metric_title)
    if title != "":
        axs.set_title(title)
    finalize_figure(fig, figoutput)


def draw_bearing_correlogram_old(data: numpy.ndarray, title: str = "", symmetric: bool = True, alpha: float = 0.05,
                                 figoutput: Optional[FigOutput] = None):
    fig = pyplot.figure(figsize=figoutput.figsize, dpi=figoutput.dpi)

    # column order is: min_scale, max_scale, bearing, # pairs, expected, observed, sd, z, prob
    mindist_col = 0
    b_col = 2
    exp_col = 4
    obs_col = 5
    p_col = 8
    dist_classes = sorted(set(data[:, mindist_col]))
    n_dists = len(dist_classes)
    deviation = data[:, obs_col] - data[:, exp_col]
    # one circle for each dist class, by ordinal rank, representing the expected value for that class
    base_circle = numpy.array([dist_classes.index(row[0])+1 for row in data])
    # the radius of each point is its base circle plus its deviation from its expectation
    r = base_circle + deviation
    # the angle (theta) is just the bearing that was tested
    theta = numpy.radians(data[:, b_col])
    drop_lines = [[(theta[i], base_circle[i]), (theta[i], r[i])] for i in range(len(r))]

    # mark positive and negative significant scales and angles
    sig_mask = [p <= alpha for p in data[:, p_col]]
    pos_mask = [i > 0 for i in data[:, obs_col]]
    neg_mask = numpy.invert(pos_mask)
    pos_mask = numpy.logical_and(sig_mask, pos_mask)  # combine to get positive significant
    neg_mask = numpy.logical_and(sig_mask, neg_mask)  # combine to get negative significant

    r_sig_pos = r[pos_mask]
    theta_sig_pos = theta[pos_mask]
    r_sig_neg = r[neg_mask]
    theta_sig_neg = theta[neg_mask]

    # mark non-significant scales and angles
    ns_mask = numpy.invert(sig_mask)
    r_ns = r[ns_mask]
    theta_ns = theta[ns_mask]

    if symmetric:
        # duplicate on opposite side of circle if drawing a full symmetric display
        r_sig_pos = numpy.append(r_sig_pos, r_sig_pos)
        theta_sig_pos = numpy.append(theta_sig_pos, theta_sig_pos + pi)
        r_sig_neg = numpy.append(r_sig_neg, r_sig_neg)
        theta_sig_neg = numpy.append(theta_sig_neg, theta_sig_neg + pi)
        r_ns = numpy.append(r_ns, r_ns)
        theta_ns = numpy.append(theta_ns, theta_ns + pi)
        for i in range(len(r)):
            drop_lines.append([(theta[i] + pi, base_circle[i]), (theta[i] + pi, r[i])])

    axs = fig.add_subplot(projection="polar")
    drop_collection = collections.LineCollection(drop_lines, colors="silver", zorder=1)
    axs.add_collection(drop_collection)

    axs.scatter(theta_sig_pos, r_sig_pos, color="blue", edgecolors="black", zorder=3, s=15)
    axs.scatter(theta_sig_neg, r_sig_neg, color="red", edgecolors="black", zorder=3, s=15)
    axs.scatter(theta_ns, r_ns, color="white", edgecolors="black", zorder=3, s=5)
    pyplot.yticks(numpy.arange(1, n_dists+1))
    axs.set_yticklabels([])
    axs.set_ylim(0, n_dists+1)
    if not symmetric:
        axs.set_xlim(0, pi)

    if title != "":
        axs.set_title(title)
    finalize_figure(fig, figoutput)


def draw_bearing_correlogram(data: numpy.ndarray, title: str = "", symmetric: bool = True, alpha: float = 0.05,
                             figoutput: Optional[FigOutput] = None):
    fig = pyplot.figure(figsize=figoutput.figsize, dpi=figoutput.dpi)

    # column order is: min_scale, max_scale, bearing, # pairs, expected, observed, sd, z, prob
    mindist_col = 0
    b_col = 2
    exp_col = 4
    obs_col = 5
    p_col = 8
    dist_classes = sorted(set(data[:, mindist_col]))
    n_dists = len(dist_classes)
    deviation = data[:, obs_col] - data[:, exp_col]
    # one circle for each dist class, by ordinal rank, representing the expected value for that class
    base_circle = numpy.array([dist_classes.index(row[0])+1 for row in data])
    # the radius of each point is its base circle plus its deviation from its expectation
    r = base_circle + deviation
    # the angle (theta) is just the bearing that was tested
    theta = numpy.radians(data[:, b_col])
    pnt_sizes = []
    edges = []
    for p in data[:, p_col]:
        if p <= alpha:
            pnt_sizes.append(20)
            edges.append("black")
        else:
            pnt_sizes.append(5)
            edges.append("gray")
    # need to normalize values for automatic color-coding
    if data[0, exp_col] == 1:  # Geary's c
        normalize = colors.Normalize(vmin=0, vmax=2)  # technically larger than this, but should suffice
    else:  # Moran's I
        normalize = colors.Normalize(vmin=-1, vmax=1)
    p_colors = pyplot.cm.bwr_r(normalize(data[:, obs_col]))
    print(p_colors)
    if symmetric:
        r = numpy.append(r, r)
        theta = numpy.append(theta, theta + pi)
        pnt_sizes = numpy.append(pnt_sizes, pnt_sizes)
        p_colors = numpy.reshape(numpy.append(p_colors, p_colors), (-1, 4))
        base_circle = numpy.append(base_circle, base_circle)
        edges = edges + edges
    drop_lines = [[(theta[i], base_circle[i]), (theta[i], r[i])] for i in range(len(r))]
    print(p_colors)

    axs = fig.add_subplot(projection="polar")

    drop_collection = collections.LineCollection(drop_lines, colors="silver", zorder=1)
    axs.add_collection(drop_collection)
    axs.scatter(theta, r, facecolor=p_colors, edgecolor=edges, zorder=3, s=pnt_sizes)

    pyplot.yticks(numpy.arange(1, n_dists+1))
    axs.set_yticklabels([])
    axs.set_ylim(0, n_dists+1)
    if not symmetric:
        axs.set_xlim(0, pi)

    if title != "":
        axs.set_title(title)
    pyplot.colorbar(pyplot.cm.ScalarMappable(norm=normalize, cmap=pyplot.cm.bwr_r), ax=axs)
    finalize_figure(fig, figoutput)


def draw_windrose_correlogram(data: numpy.ndarray, title: str = "", symmetric: bool = True, alpha: float = 0.05,
                              show_counts: bool = False, is_mantel: bool = False,
                              figoutput: Optional[FigOutput] = None):
    fig = pyplot.figure(figsize=figoutput.figsize, dpi=figoutput.dpi)

    # pre-determined spacing between sectors in each annulus
    spacer = (14 * pi / 180, 10 * pi / 180, 8 * pi / 180, 6 * pi / 180, 4 * pi / 180, 3 * pi / 180, 2 * pi / 180)
    sig_height = 0.9

    # column order is: min_scale, max_scale, min_angle, max_angle, # pairs, expected, observed, sd, z, prob
    # sd is absent from Mantel correlograms
    mindist_col = 0
    sang_col = 2
    eang_col = 3
    np_col = 4
    exp_col = 5
    obs_col = 6
    if is_mantel:
        p_col = 8
    else:
        p_col = 9
    annuli = set(data[:, mindist_col])
    annuli = sorted(annuli)
    n_annuli = len(annuli)

    sector_widths = numpy.radians(data[:, eang_col] - data[:, sang_col])  # width of each segment in radians
    thetas = numpy.radians(data[:, sang_col]) + sector_widths/2  # angle representing the center of each segment

    # need to normalize values for automatic color-coding
    if data[0, exp_col] == 1:  # Geary's c
        normalize = colors.Normalize(vmin=0, vmax=2)  # technically larger than this, but should suffice
    else:  # Moran's I
        normalize = colors.Normalize(vmin=-1, vmax=1)
    s_colors = pyplot.cm.bwr_r(normalize(data[:, obs_col]))

    axs = fig.add_subplot(projection="polar")
    for annulus in range(n_annuli):
        mask = [annuli[annulus] == row[mindist_col] for row in data]
        annulus_data = data[mask, :]
        annulus_thetas = thetas[mask]
        annulus_widths = sector_widths[mask]
        annulus_colors = s_colors[mask]
        if len(annulus_data) == 1:
            space = 0
            bottom = 0
        else:
            space = spacer[annulus]
            bottom = annulus + (1 - sig_height)

        if show_counts:  # draw sectors with pair counts, rather than correlogram results
            cnt_mask = [np > 0 for np in annulus_data[:, np_col]]
            cnt_annulus_data = annulus_data[cnt_mask, :]
            cnt_annulus_thetas = annulus_thetas[cnt_mask]
            cnt_annulus_widths = annulus_widths[cnt_mask]
            plot_widths = []
            radii = []
            plot_thetas = []
            for i, sector in enumerate(cnt_annulus_data):
                plot_thetas.append(cnt_annulus_thetas[i])
                plot_widths.append(cnt_annulus_widths[i] - space)
                radii.append(sig_height)
            if symmetric:
                for i, sector in enumerate(cnt_annulus_data):
                    plot_thetas.append(cnt_annulus_thetas[i] + pi)
                    plot_widths.append(cnt_annulus_widths[i] - space)
                    radii.append(sig_height)
            axs.bar(plot_thetas, radii, width=plot_widths, bottom=bottom, linestyle="--", color="white",
                    edgecolor="black")
            for i, sector in enumerate(cnt_annulus_data):
                axs.text(plot_thetas[i], bottom + 0.45, str(int(sector[np_col])), horizontalalignment="center",
                         verticalalignment="center", fontdict={"size": 8})
                if symmetric:
                    axs.text(plot_thetas[i] + pi, bottom + 0.45, str(int(sector[np_col])), horizontalalignment="center",
                             verticalalignment="center", fontdict={"size": 8})
        else:  # draw correlogram results
            # significant sectors in this annulus
            sig_mask = [0 <= p <= alpha for p in annulus_data[:, p_col]]
            sig_annulus_thetas = annulus_thetas[sig_mask]
            sig_annulus_widths = annulus_widths[sig_mask]
            sig_annulus_colors = annulus_colors[sig_mask]
            plot_widths = []
            radii = []
            plot_thetas = []
            for i in range(len(sig_annulus_thetas)):
                plot_thetas.append(sig_annulus_thetas[i])
                plot_widths.append(sig_annulus_widths[i] - space)
                radii.append(sig_height)
            if symmetric:
                for i in range(len(sig_annulus_thetas)):
                    plot_thetas.append(sig_annulus_thetas[i] + pi)
                    plot_widths.append(sig_annulus_widths[i] - space)
                    radii.append(sig_height)
            axs.bar(plot_thetas, radii, width=plot_widths, bottom=bottom, color=sig_annulus_colors, edgecolor="black")

            # non-significant sectors in this annulus
            bottom = annulus + (1 - sig_height/2)
            ns_mask = [p > alpha for p in annulus_data[:, p_col]]
            ns_annulus_thetas = annulus_thetas[ns_mask]
            ns_annulus_widths = annulus_widths[ns_mask]
            ns_annulus_colors = annulus_colors[ns_mask]
            plot_widths = []
            radii = []
            plot_thetas = []
            for i in range(len(ns_annulus_thetas)):
                plot_thetas.append(ns_annulus_thetas[i])
                plot_widths.append(ns_annulus_widths[i] - space)
                radii.append(sig_height/2)
            if symmetric:
                for i in range(len(ns_annulus_thetas)):
                    plot_thetas.append(ns_annulus_thetas[i] + pi)
                    plot_widths.append(ns_annulus_widths[i] - space)
                    radii.append(sig_height/2)
            axs.bar(plot_thetas, radii, width=plot_widths, bottom=bottom, color=ns_annulus_colors, edgecolor="black")

            # sectors below the pair threshold in this annulus
            dash_mask = [p == -1 for p in annulus_data[:, p_col]]  # could be no pairs or too few pairs
            dash_annulus_data = annulus_data[dash_mask, :]
            dash_annulus_thetas = annulus_thetas[dash_mask]
            dash_annulus_widths = annulus_widths[dash_mask]
            plot_widths = []
            radii = []
            plot_thetas = []
            for i, sector in enumerate(dash_annulus_data):
                if sector[np_col] > 0:  # only draw if there were some point pairs
                    plot_thetas.append(dash_annulus_thetas[i])
                    plot_widths.append(dash_annulus_widths[i] - space)
                    radii.append(sig_height/2)
            if symmetric:
                for i, sector in enumerate(dash_annulus_data):
                    if sector[np_col] > 0:
                        plot_thetas.append(dash_annulus_thetas[i] + pi)
                        plot_widths.append(dash_annulus_widths[i] - space)
                        radii.append(sig_height/2)
            axs.bar(plot_thetas, radii, width=plot_widths, bottom=bottom, linestyle="--", color="white",
                    edgecolor="black")

    if not symmetric:
        axs.set_xlim(0, pi)
    pyplot.axis("off")
    if title != "":
        axs.set_title(title)
    if not show_counts:
        pyplot.colorbar(pyplot.cm.ScalarMappable(norm=normalize, cmap=pyplot.cm.bwr_r), ax=axs)
    finalize_figure(fig, figoutput)


def draw_bearing(data: numpy.ndarray, alpha: float = 0.05, figoutput: Optional[FigOutput] = None):
    fig, axs = pyplot.subplots(figsize=figoutput.figsize, dpi=figoutput.dpi)

    # # column order is: min_scale, max_scale, # pairs, expected, observed, sd, z, prob
    # min_col = 0
    # max_col = 1
    # exp_col = 3
    # obs_col = 4
    # p_col = 7
    #
    # # plot at midpoint of distance range
    # scale = numpy.array([x[min_col] + (x[max_col] - x[min_col])/2 for x in data])
    #
    n = len(data)

    # draw expected value
    y = [0, 0]
    x = [data[0, 0], data[n-1, 0]]
    line = Line2D(x, y, color="silver", zorder=1)
    axs.add_line(line)

    # draw base line
    axs.plot(data[:, 0], data[:, 1], zorder=2)

    # mark significant bearings
    sig_mask = [p <= alpha for p in data[:, 2]]
    x = data[sig_mask, 0]
    y = data[sig_mask, 1]
    pyplot.scatter(x, y, color="black", edgecolors="black", zorder=3, s=25)

    # mark non-significant scales
    ns_mask = numpy.invert(sig_mask)
    x = data[ns_mask, 0]
    y = data[ns_mask, 1]
    pyplot.scatter(x, y, color="white", edgecolors="black", zorder=3, s=15)

    axs.set_xlabel("Bearing")
    axs.set_ylabel("Mantel Correlation")
    finalize_figure(fig, figoutput)
