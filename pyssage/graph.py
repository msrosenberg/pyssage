from typing import Optional
from math import pi
from pyssage.classes import Number, VoronoiTessellation
import pyssage.connections
import pyssage.distances
import pyssage.utils
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D
from matplotlib import collections, colors, cm
# import matplotlib.patches as mpatches
import numpy

__all__ = ["FigOutput", "draw_angular_correlation", "draw_bearing", "draw_bearing_correlogram", "draw_connections",
           "draw_correlogram", "draw_distance_class_distribution", "draw_histogram", "draw_quadvar_result",
           "draw_shortest_path", "draw_tessellation", "draw_transect", "draw_windrose_correlogram"]


class FigOutput:
    def __init__(self, figsize: tuple = (8, 6), dpi: int = 100, figshow: bool = True, figname: str = "",
                 figformat: str = "png"):
        self.figsize = figsize
        self.dpi = dpi
        self.figshow = figshow
        self.figname = figname
        self.figformat = figformat


class PointStyle:
    def __init__(self, face_color: str = "red", edge_color: str = "black", size: int = 30, edge_width: float = 1,
                 marker: str = "o", alpha: float = 1):
        self.face_color = face_color
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.size = size
        self.marker = marker
        self.alpha = alpha


class GradientPointStyle:
    def __init__(self, colormap: str = "bwr_r", edge_color: str = "black", ns_edge_color: str = "gray",
                 size: int = 20, ns_size: int = 5, edge_width: float = 1, marker: str = "o", alpha: float = 1):
        self.colormap = colormap
        self.edge_color = edge_color
        self.ns_edge_color = ns_edge_color
        self.edge_width = edge_width
        self.size = size
        self.ns_size = ns_size
        self.marker = marker
        self.alpha = alpha


class LineStyle:
    def __init__(self, color: str = "#1f77b4", linestyle: str = "solid", linewidth: float = 1, alpha: float = 1):
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.alpha = alpha


class BarStyle:
    def __init__(self, face_color: str = "#1f77b4", edge_color: str = "black", edge_width: float = 0, alpha: float = 1):
        self.face_color = face_color
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.alpha = alpha


class WindroseStyle:
    def __init__(self, colormap: str = "bwr_r", edge_color: str = "black", edge_width: float = 1,
                 edge_style: str = "solid", alpha: float = 1, ns_edge_color: str = "black", ns_edge_width: float = 1,
                 ns_edge_style: str = "solid", ns_alpha: float = 1, min_face_color: str = "white",
                 min_edge_color: str = "black", min_edge_width: float = 1, min_edge_style: str = "dashed",
                 min_alpha: float = 1):
        self.colormap = colormap
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.edge_style = edge_style
        self.alpha = alpha
        self.ns_edge_color = ns_edge_color
        self.ns_edge_width = ns_edge_width
        self.ns_edge_style = ns_edge_style
        self.ns_alpha = ns_alpha
        self.min_face_color = min_face_color
        self.min_edge_color = min_edge_color
        self.min_edge_width = min_edge_width
        self.min_edge_style = min_edge_style
        self.min_alpha = min_alpha


def check_valid_graph_format(x: str) -> bool:
    """
    checks that the entered graph format is valid

    valid entries are automatically determined as they may depend on back-end, but generally should include
    svg, png, and pdf, among others
    """
    valid_formats = pyplot.gcf().canvas.get_supported_filetypes()
    if x in valid_formats:
        return True
    else:
        error_msg = "Invalid Figure Format. Valid formats include:\n"
        for f in valid_formats:
            error_msg += "{:4}    {}\n".format(f, valid_formats[f])
        raise ValueError(error_msg)


def start_figure(figoutput: Optional[FigOutput] = None, polar: bool = False):
    """
    common function for starting graphs and figures

    sets the size and dpi of the figure and creates the axes
    """
    if figoutput is None:
        figoutput = FigOutput()
    fig = pyplot.figure(figsize=figoutput.figsize, dpi=figoutput.dpi)
    if polar:
        axs = fig.add_subplot(projection="polar")
    else:
        axs = fig.add_subplot()
    return fig, axs


def finalize_figure(fig, axs, figoutput: FigOutput, title: str = "") -> None:
    """
    common function for ending graphs and figures

    sets the title, saves to file (if appropriate), and displays on screen (if appropriate)
    """
    if figoutput is None:
        figoutput = FigOutput()
    if title != "":
        axs.set_title(title)
    if figoutput.figname != "":
        if check_valid_graph_format(figoutput.figformat):
            fig.savefig(figoutput.figname, format=figoutput.figformat, dpi=figoutput.dpi)
    if figoutput.figshow:
        pyplot.show()


def draw_transect(transect: numpy.array, unit_scale: Number = 1, title: str = "",
                  line_style: Optional[LineStyle] = None, figoutput: Optional[FigOutput] = None) -> None:
    fig, axs = start_figure(figoutput)
    x = [i*unit_scale for i in range(len(transect))]
    if line_style is None:
        line_style = LineStyle()
    axs.plot(x, transect, color=line_style.color, linewidth=line_style.linewidth, linestyle=line_style.linestyle,
             alpha=line_style.alpha)
    axs.set_xlabel("Position")
    axs.set_ylabel("Value")
    finalize_figure(fig, axs, figoutput, title)


def draw_quadvar_result(quadvar: numpy.ndarray, inc_random: bool = False, title: str = "",
                        varlabel: str = "", randlabel: str = "",  line_style: Optional[LineStyle] = None,
                        rand_style: Optional[LineStyle] = None, figoutput: Optional[FigOutput] = None) -> None:
    fig, axs = start_figure(figoutput)
    if line_style is None:
        line_style = LineStyle()
    axs.plot(quadvar[:, 0], quadvar[:, 1], label=varlabel, color=line_style.color, linewidth=line_style.linewidth,
             linestyle=line_style.linestyle, alpha=line_style.alpha)
    if inc_random:
        if rand_style is None:
            rand_style = LineStyle(color="red")
        axs.plot(quadvar[:, 0], quadvar[:, 2], label=randlabel, color=rand_style.color, linewidth=rand_style.linewidth,
                 linestyle=rand_style.linestyle, alpha=rand_style.alpha)
        pyplot.legend(loc="upper right")
    axs.set_xlabel("Scale")
    axs.set_ylabel("Variance")
    finalize_figure(fig, axs, figoutput, title)


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


def draw_tessellation(tessellation: VoronoiTessellation, xcoords: numpy.ndarray, ycoords: numpy.ndarray,
                      title: str = "", point_style: Optional[PointStyle] = None, line_style: Optional[LineStyle] = None,
                      figoutput: Optional[FigOutput] = None) -> None:
    fig, axs = start_figure(figoutput)
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    if line_style is None:
        line_style = LineStyle()
    for e in tessellation.edges:
        x = [e.start_vertex.x, e.end_vertex.x]
        y = [e.start_vertex.y, e.end_vertex.y]
        line = Line2D(x, y, color=line_style.color, linewidth=line_style.linewidth, linestyle=line_style.linestyle,
                      alpha=line_style.alpha)
        axs.add_line(line)
    if point_style is None:
        point_style = PointStyle(face_color="black")
    pyplot.scatter(xcoords, ycoords, color=point_style.face_color, edgecolors=point_style.edge_color,
                   s=point_style.size, linewidths=point_style.edge_width, marker=point_style.marker,
                   alpha=point_style.alpha)
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    finalize_figure(fig, axs, figoutput, title)


def check_connection_format(con_frmt: str) -> None:
    valid_formats = ("boolmatrix", "binmatrix", "revbinmatrix", "pairlist")
    if con_frmt not in valid_formats:
        raise ValueError("{} is not a valid connection format".format(con_frmt))


def add_connections_to_plot(axs, connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray,
                            line_style: Optional[LineStyle] = None):
    if line_style is None:
        line_style = LineStyle()
    if connections.is_symmetric():
        done = []
        for i in range(len(connections)):
            for j in connections.connected_from(i):
                if [i, j] not in done:  # trying to avoid duplicating the reverse line
                    x = [xcoords[i], xcoords[j]]
                    y = [ycoords[i], ycoords[j]]
                    line = Line2D(x, y, color=line_style.color, linewidth=line_style.linewidth,
                                  linestyle=line_style.linestyle, alpha=line_style.alpha, zorder=1)
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
                    # axs.annotate(s="", xytext=(xcoords[i], ycoords[i]), xy=(xcoords[j], ycoords[j]),
                    #              arrowprops=dict(arrowstyle=arrow, edgecolor="C0"), zorder=1)
                    axs.annotate(s="", xytext=(xcoords[i], ycoords[i]), xy=(xcoords[j], ycoords[j]), zorder=1,
                                 arrowprops=dict(arrowstyle=arrow, edgecolor=line_style.color, color=line_style.color,
                                                 linewidth=line_style.linewidth, linestyle=line_style.linestyle,
                                                 alpha=line_style.alpha))


def draw_connections(connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray, title: str = "",
                     point_style: Optional[PointStyle] = None, line_style: Optional[LineStyle] = None,
                     figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput)
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    add_connections_to_plot(axs, connections, xcoords, ycoords, line_style)

    if point_style is None:
        point_style = PointStyle(face_color="black")
    pyplot.scatter(xcoords, ycoords, color=point_style.face_color, edgecolors=point_style.edge_color,
                   s=point_style.size, linewidths=point_style.edge_width, marker=point_style.marker,
                   alpha=point_style.alpha, zorder=2)
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    finalize_figure(fig, axs, figoutput, title)


def draw_shortest_path(connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray, trace_dict: dict,
                       startp: int, endp: int, title: str = "", point_style: Optional[PointStyle] = None,
                       line_style: Optional[LineStyle] = None, path_style: Optional[LineStyle] = None,
                       figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput)
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    add_connections_to_plot(axs, connections, xcoords, ycoords, line_style)

    if path_style is None:
        path_style = LineStyle(color="red", linewidth=2)
    trace_path = pyssage.distances.trace_path(startp, endp, trace_dict)
    for i in range(len(trace_path)-1):
        p1 = trace_path[i]
        p2 = trace_path[i+1]
        x = [xcoords[p1], xcoords[p2]]
        y = [ycoords[p1], ycoords[p2]]
        line = Line2D(x, y, color=path_style.color, linewidth=path_style.linewidth, linestyle=path_style.linestyle,
                      alpha=path_style.alpha, zorder=3)
        axs.add_line(line)

    if point_style is None:
        point_style = PointStyle(face_color="black")
    pyplot.scatter(xcoords, ycoords, color=point_style.face_color, edgecolors=point_style.edge_color,
                   s=point_style.size, linewidths=point_style.edge_width, marker=point_style.marker,
                   alpha=point_style.alpha, zorder=2)
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    finalize_figure(fig, axs, figoutput, title)


def draw_distance_class_distribution(dist_matrix: numpy.ndarray, dist_class: numpy.ndarray, title: str = "",
                                     line_style: Optional[LineStyle] = None, drop_style: Optional[LineStyle] = None,
                                     figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput)
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

    if line_style is None:
        line_style = LineStyle()
    axs.plot(distances, y, zorder=1, color=line_style.color, linewidth=line_style.linewidth,
             linestyle=line_style.linestyle, alpha=line_style.alpha)

    if drop_style is None:
        drop_style = LineStyle(color="red")
    for i in range(len(dist_class)):
        c = dist_class[i]
        upper = c[1]
        y = [0, r[i], r[i]]
        x = [upper, upper, 0]
        line = Line2D(x, y, zorder=2, color=drop_style.color, linewidth=drop_style.linewidth,
                      linestyle=drop_style.linestyle, alpha=drop_style.alpha)
        axs.add_line(line)

    axs.set_xlabel("Distance")
    axs.set_ylabel("Cumulative Count")
    finalize_figure(fig, axs, figoutput, title)


def draw_correlogram(data: numpy.ndarray, metric_title: str = "", title: str = "", alpha: float = 0.05,
                     sig_style: Optional[PointStyle] = None, ns_style: Optional[PointStyle] = None,
                     line_style: Optional[LineStyle] = None, exp_style: Optional[LineStyle] = None,
                     figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput)

    # column order is: min_scale, max_scale, # pairs, expected, observed, sd, z, prob
    # sd is absent from Mantel correlograms
    _, ncols = data.shape
    min_col = 0
    max_col = 1
    exp_col = 3
    obs_col = 4
    p_col = ncols - 1

    # plot at midpoint of distance range
    scale = numpy.array([x[min_col] + (x[max_col] - x[min_col])/2 for x in data])

    # draw expected values
    y = [data[0, exp_col], data[0, exp_col]]
    x = [0, scale[len(scale)-1]]
    if exp_style is None:
        exp_style = LineStyle(color="silver")
    line = Line2D(x, y, zorder=1, color=exp_style.color, linewidth=exp_style.linewidth, linestyle=exp_style.linestyle,
                  alpha=exp_style.alpha)
    axs.add_line(line)

    # draw base line
    if line_style is None:
        line_style = LineStyle()
    axs.plot(scale, data[:, obs_col], zorder=2, color=line_style.color, linewidth=line_style.linewidth,
             linestyle=line_style.linestyle, alpha=line_style.alpha)
    # axs.plot(data[:, scale_col], data[:, obs_col], zorder=2)

    # mark significant scales
    sig_mask = [p <= alpha for p in data[:, p_col]]
    x = scale[sig_mask]
    y = data[sig_mask, obs_col]
    if sig_style is None:
        sig_style = PointStyle(face_color="black", edge_color="black", size=25)
    pyplot.scatter(x, y, color=sig_style.face_color, edgecolors=sig_style.edge_color,
                   s=sig_style.size, linewidths=sig_style.edge_width, marker=sig_style.marker,
                   alpha=sig_style.alpha, zorder=3)

    # mark non-significant scales
    ns_mask = numpy.invert(sig_mask)
    # x = data[ns_mask, scale_col]
    x = scale[ns_mask]
    y = data[ns_mask, obs_col]
    if ns_style is None:
        ns_style = PointStyle(face_color="white", edge_color="black", size=15)
    pyplot.scatter(x, y, color=ns_style.face_color, edgecolors=ns_style.edge_color,
                   s=ns_style.size, linewidths=ns_style.edge_width, marker=ns_style.marker,
                   alpha=ns_style.alpha, zorder=3)

    axs.set_xlabel("Scale")
    axs.set_ylabel(metric_title)
    finalize_figure(fig, axs, figoutput, title)


# def draw_bearing_correlogram_old(data: numpy.ndarray, title: str = "", symmetric: bool = True, alpha: float = 0.05,
#                                  figoutput: Optional[FigOutput] = None):
#     fig, axs = start_figure(figoutput, polar=True)
#
#     # column order is: min_scale, max_scale, bearing, # pairs, expected, observed, sd, z, prob
#     mindist_col = 0
#     b_col = 2
#     exp_col = 4
#     obs_col = 5
#     p_col = 8
#     dist_classes = sorted(set(data[:, mindist_col]))
#     n_dists = len(dist_classes)
#     deviation = data[:, obs_col] - data[:, exp_col]
#     # one circle for each dist class, by ordinal rank, representing the expected value for that class
#     base_circle = numpy.array([dist_classes.index(row[0])+1 for row in data])
#     # the radius of each point is its base circle plus its deviation from its expectation
#     r = base_circle + deviation
#     # the angle (theta) is just the bearing that was tested
#     theta = numpy.radians(data[:, b_col])
#     drop_lines = [[(theta[i], base_circle[i]), (theta[i], r[i])] for i in range(len(r))]
#
#     # mark positive and negative significant scales and angles
#     sig_mask = [p <= alpha for p in data[:, p_col]]
#     pos_mask = [i > 0 for i in data[:, obs_col]]
#     neg_mask = numpy.invert(pos_mask)
#     pos_mask = numpy.logical_and(sig_mask, pos_mask)  # combine to get positive significant
#     neg_mask = numpy.logical_and(sig_mask, neg_mask)  # combine to get negative significant
#
#     r_sig_pos = r[pos_mask]
#     theta_sig_pos = theta[pos_mask]
#     r_sig_neg = r[neg_mask]
#     theta_sig_neg = theta[neg_mask]
#
#     # mark non-significant scales and angles
#     ns_mask = numpy.invert(sig_mask)
#     r_ns = r[ns_mask]
#     theta_ns = theta[ns_mask]
#
#     if symmetric:
#         # duplicate on opposite side of circle if drawing a full symmetric display
#         r_sig_pos = numpy.append(r_sig_pos, r_sig_pos)
#         theta_sig_pos = numpy.append(theta_sig_pos, theta_sig_pos + pi)
#         r_sig_neg = numpy.append(r_sig_neg, r_sig_neg)
#         theta_sig_neg = numpy.append(theta_sig_neg, theta_sig_neg + pi)
#         r_ns = numpy.append(r_ns, r_ns)
#         theta_ns = numpy.append(theta_ns, theta_ns + pi)
#         for i in range(len(r)):
#             drop_lines.append([(theta[i] + pi, base_circle[i]), (theta[i] + pi, r[i])])
#
#     drop_collection = collections.LineCollection(drop_lines, colors="silver", zorder=1)
#     axs.add_collection(drop_collection)
#
#     axs.scatter(theta_sig_pos, r_sig_pos, color="blue", edgecolors="black", zorder=3, s=15)
#     axs.scatter(theta_sig_neg, r_sig_neg, color="red", edgecolors="black", zorder=3, s=15)
#     axs.scatter(theta_ns, r_ns, color="white", edgecolors="black", zorder=3, s=5)
#     pyplot.yticks(numpy.arange(1, n_dists+1))
#     axs.set_yticklabels([])
#     axs.set_ylim(0, n_dists+1)
#     if not symmetric:
#         axs.set_xlim(0, pi)
#
#     finalize_figure(fig, axs, figoutput, title)


def draw_bearing_correlogram(data: numpy.ndarray, title: str = "", symmetric: bool = True, alpha: float = 0.05,
                             point_style: Optional[GradientPointStyle] = None, figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput, polar=True)

    # column order is: min_scale, max_scale, bearing, # pairs, expected, observed, sd, z, prob
    _, ncols = data.shape
    mindist_col = 0
    b_col = 2
    exp_col = 4
    obs_col = 5
    p_col = ncols - 1
    dist_classes = sorted(set(data[:, mindist_col]))
    n_dists = len(dist_classes)
    deviation = data[:, obs_col] - data[:, exp_col]
    # one circle for each dist class, by ordinal rank, representing the expected value for that class
    base_circle = numpy.array([dist_classes.index(row[0])+1 for row in data])
    # the radius of each point is its base circle plus its deviation from its expectation
    r = base_circle + deviation
    # the angle (theta) is just the bearing that was tested
    theta = numpy.radians(data[:, b_col])

    if point_style is None:
        point_style = GradientPointStyle()

    pnt_sizes = []
    edges = []
    for p in data[:, p_col]:
        if p <= alpha:
            pnt_sizes.append(point_style.size)
            edges.append(point_style.edge_color)
        else:
            pnt_sizes.append(point_style.ns_size)
            edges.append(point_style.ns_edge_color)

    # need to normalize values for automatic color-coding
    if data[0, exp_col] == 1:  # Geary's c
        normalize = colors.Normalize(vmin=0, vmax=2)  # technically larger than this, but should suffice
    else:  # Moran's I
        normalize = colors.Normalize(vmin=-1, vmax=1)
    cmap = cm.get_cmap(point_style.colormap)
    p_colors = cmap(normalize(data[:, obs_col]))
    if symmetric:
        r = numpy.append(r, r)
        theta = numpy.append(theta, theta + pi)
        pnt_sizes = numpy.append(pnt_sizes, pnt_sizes)
        p_colors = numpy.reshape(numpy.append(p_colors, p_colors), (-1, 4))
        base_circle = numpy.append(base_circle, base_circle)
        edges = edges + edges
    drop_lines = [[(theta[i], base_circle[i]), (theta[i], r[i])] for i in range(len(r))]

    drop_collection = collections.LineCollection(drop_lines, colors="silver", zorder=1)
    axs.add_collection(drop_collection)
    axs.scatter(theta, r, facecolor=p_colors, edgecolor=edges, zorder=3, s=pnt_sizes, alpha=point_style.alpha,
                marker=point_style.marker, linewidths=point_style.edge_width)

    pyplot.yticks(numpy.arange(1, n_dists+1))
    axs.set_yticklabels([])
    axs.set_ylim(0, n_dists+1)
    if not symmetric:
        axs.set_xlim(0, pi)

    pyplot.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axs)
    finalize_figure(fig, axs, figoutput, title)


def draw_windrose_correlogram(data: numpy.ndarray, title: str = "", symmetric: bool = True, alpha: float = 0.05,
                              show_counts: bool = False, windrose_style: Optional[WindroseStyle] = None,
                              figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput, polar=True)

    if windrose_style is None:
        windrose_style = WindroseStyle()

    # pre-determined spacing between sectors in each annulus
    spacer = (14 * pi / 180, 10 * pi / 180, 8 * pi / 180, 6 * pi / 180, 4 * pi / 180, 3 * pi / 180, 2 * pi / 180)
    sig_height = 0.9

    # column order is: min_scale, max_scale, min_angle, max_angle, # pairs, expected, observed, sd, z, prob
    # sd is absent from Mantel correlograms
    _, ncols = data.shape
    mindist_col = 0
    sang_col = 2
    eang_col = 3
    np_col = 4
    exp_col = 5
    obs_col = 6
    p_col = ncols - 1
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
    cmap = cm.get_cmap(windrose_style.colormap)
    s_colors = cmap(normalize(data[:, obs_col]))

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
            axs.bar(plot_thetas, radii, width=plot_widths, bottom=bottom, color=sig_annulus_colors,
                    edgecolor=windrose_style.edge_color, linestyle=windrose_style.edge_style,
                    linewidth=windrose_style.edge_width, alpha=windrose_style.alpha)

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
            axs.bar(plot_thetas, radii, width=plot_widths, bottom=bottom, color=ns_annulus_colors,
                    edgecolor=windrose_style.ns_edge_color, linestyle=windrose_style.ns_edge_style,
                    linewidth=windrose_style.ns_edge_width, alpha=windrose_style.ns_alpha)

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
            axs.bar(plot_thetas, radii, width=plot_widths, bottom=bottom, color=windrose_style.min_face_color,
                    edgecolor=windrose_style.min_edge_color, linestyle=windrose_style.min_edge_style,
                    linewidth=windrose_style.min_edge_width, alpha=windrose_style.min_alpha)

    if not symmetric:
        axs.set_xlim(0, pi)
    pyplot.axis("off")
    if not show_counts:
        pyplot.colorbar(pyplot.cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axs)
    finalize_figure(fig, axs, figoutput, title)


def draw_bearing(data: numpy.ndarray, alpha: float = 0.05, title: str = "", draw_polar: bool = False,
                 sig_style: Optional[PointStyle] = None, ns_style: Optional[PointStyle] = None,
                 line_style: Optional[LineStyle] = None, exp_style: Optional[LineStyle] = None,
                 figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput, draw_polar)
    n = len(data)
    _, ncols = data.shape
    p_col = ncols - 1

    if not draw_polar:
        # draw expected value
        if exp_style is None:
            exp_style = LineStyle(color="silver")
        y = [0, 0]
        x = [data[0, 0], data[n-1, 0]]
        line = Line2D(x, y, zorder=1, color=exp_style.color, linewidth=exp_style.linewidth,
                      linestyle=exp_style.linestyle, alpha=exp_style.alpha)
        axs.add_line(line)

        # draw base line
        if line_style is None:
            line_style = LineStyle()
        axs.plot(data[:, 0], data[:, 1], zorder=2, color=line_style.color, linewidth=line_style.linewidth,
                 linestyle=line_style.linestyle, alpha=line_style.alpha)
        axs.set_xlabel("Bearing")
        axs.set_ylabel("Mantel Correlation")

    # mark significant bearings
    sig_mask = [p <= alpha for p in data[:, p_col]]

    if draw_polar:
        x = numpy.radians(numpy.append(data[sig_mask, 0], data[sig_mask, 0]+180))
        y = numpy.abs(numpy.append(data[sig_mask, 1], data[sig_mask, 1]))
    else:
        x = data[sig_mask, 0]
        y = data[sig_mask, 1]
    if sig_style is None:
        sig_style = PointStyle(face_color="black", edge_color="black", size=25)
    pyplot.scatter(x, y, color=sig_style.face_color, edgecolors=sig_style.edge_color,
                   s=sig_style.size, linewidths=sig_style.edge_width, marker=sig_style.marker,
                   alpha=sig_style.alpha, zorder=3)

    # mark non-significant scales
    ns_mask = numpy.invert(sig_mask)
    if draw_polar:
        x = numpy.radians(numpy.append(data[ns_mask, 0], data[ns_mask, 0]+180))
        y = numpy.abs(numpy.append(data[ns_mask, 1], data[ns_mask, 1]))
    else:
        x = data[ns_mask, 0]
        y = data[ns_mask, 1]
    if ns_style is None:
        ns_style = PointStyle(face_color="white", edge_color="black", size=15)
    pyplot.scatter(x, y, color=ns_style.face_color, edgecolors=ns_style.edge_color,
                   s=ns_style.size, linewidths=ns_style.edge_width, marker=ns_style.marker,
                   alpha=ns_style.alpha, zorder=3)

    finalize_figure(fig, axs, figoutput, title)


def draw_angular_correlation(data: numpy.ndarray, title: str = "", draw_polar: bool = True,
                             point_style: Optional[PointStyle] = None, figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput, polar=draw_polar)
    if point_style is None:
        point_style = PointStyle(face_color="red", edge_color="black")
    # if draw_polar:
    #     pos_mask = [r >= 0 for r in data[:, 1]]
    #     pyplot.scatter(numpy.radians(data[pos_mask, 0]), numpy.abs(data[pos_mask, 1]), color="red",
    #                    edgecolors="black", zorder=3)
    #     neg_mask = numpy.invert(pos_mask)
    #     pyplot.scatter(numpy.radians(data[neg_mask, 0]), numpy.abs(data[neg_mask, 1]), color="blue",
    #                    edgecolors="black" zorder=3)
    #     axs.set_ylim(0, 1)
    # else:
    #     # draw expected value
    #     y = [0, 0]
    #     x = [data[0, 0], data[len(data) - 1, 0]]
    #     line = Line2D(x, y, color="silver", zorder=1)
    #     axs.add_line(line)
    #     pyplot.scatter(data[:, 0], data[:, 1], zorder=2)
    #     axs.set_xlabel("Bearing")
    #     axs.set_ylabel("Correlation")
    #     axs.set_ylim(-1, 1)
    if draw_polar:
        pyplot.scatter(numpy.radians(data[:, 0]), numpy.abs(data[:, 1]), color=point_style.face_color,
                       edgecolors=point_style.edge_color, s=point_style.size, linewidths=point_style.edge_width,
                       marker=point_style.marker, alpha=point_style.alpha, zorder=3)
        axs.set_ylim(0, 1)
    else:
        # draw expected value
        y = [0, 0]
        x = [data[0, 0], data[len(data) - 1, 0]]
        line = Line2D(x, y, color="silver", zorder=1)
        axs.add_line(line)
        pyplot.scatter(data[:, 0], data[:, 1], color=point_style.face_color, edgecolors=point_style.edge_color,
                       s=point_style.size, linewidths=point_style.edge_width, marker=point_style.marker,
                       alpha=point_style.alpha, zorder=2)
        axs.set_xlabel("Bearing")
        axs.set_ylabel("Correlation")
        axs.set_ylim(-1, 1)
    finalize_figure(fig, axs, figoutput, title)


def draw_histogram(data: numpy.ndarray, nbins: int = 20, obs_value: Optional[Number] = None,
                   obs_title: str = "observed", obs_y_adj: int = 5,  title: str = "", xlabel: str = "Bin",
                   ylabel: str = "Frequency", bar_style: Optional[BarStyle] = None,
                   obs_style: Optional[PointStyle] = None, figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput)
    if bar_style is None:
        bar_style = BarStyle()
    n, bins, patches = axs.hist(data, bins=nbins, zorder=1, color=bar_style.face_color, edgecolor=bar_style.edge_color,
                                linewidth=bar_style.edge_width, alpha=bar_style.alpha)
    if obs_value is not None:
        # find bin containing obs_value so we can pick a y-axis value for it
        b = 1
        while obs_value > bins[b]:
            b += 1
        if obs_style is None:
            obs_style = PointStyle(face_color="red", edge_color="black", size=50, edge_width=1)
        axs.scatter(obs_value, n[b-1] + obs_y_adj, color=obs_style.face_color, edgecolors=obs_style.edge_color,
                    s=obs_style.size, linewidths=obs_style.edge_width, marker=obs_style.marker,
                    alpha=obs_style.alpha, label=obs_title, zorder=2)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    finalize_figure(fig, axs, figoutput, title)


def draw_variogram(data: numpy.ndarray, metric_title: str = "g", title: str = "",
                   point_style: Optional[PointStyle] = None, line_style: Optional[LineStyle] = None,
                   figoutput: Optional[FigOutput] = None):
    fig, axs = start_figure(figoutput)

    # column order is: min_scale, max_scale, # pairs, observed
    min_col = 0
    max_col = 1
    obs_col = 3

    # plot at midpoint of distance range
    scale = numpy.array([x[min_col] + (x[max_col] - x[min_col])/2 for x in data])

    # draw base line
    if line_style is None:
        line_style = LineStyle()
    axs.plot(scale, data[:, obs_col], zorder=2, color=line_style.color, linewidth=line_style.linewidth,
             linestyle=line_style.linestyle, alpha=line_style.alpha)

    # draw points
    x = scale
    y = data[:, obs_col]
    if point_style is None:
        point_style = PointStyle(face_color="black", edge_color="black", size=25)
    pyplot.scatter(x, y, color=point_style.face_color, edgecolors=point_style.edge_color,
                   s=point_style.size, linewidths=point_style.edge_width, marker=point_style.marker,
                   alpha=point_style.alpha, zorder=3)

    axs.set_xlabel("Scale")
    axs.set_ylabel(metric_title)
    finalize_figure(fig, axs, figoutput, title)
