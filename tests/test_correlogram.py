import numpy
import pytest
import pyssage.correlogram
import pyssage.distances
import pyssage.connections
import pyssage.graph
from tests.test_common import create_test_scattered, create_test_coords


def test_check_variance_assumption():
    pyssage.correlogram.check_variance_assumption("random")
    pyssage.correlogram.check_variance_assumption("normal")
    pyssage.correlogram.check_variance_assumption(None)
    with pytest.raises(ValueError,
                       match="TYPO is not a valid variance assumption. Valid values are: random, normal, None"):
        # broken test --> should raise error
        pyssage.correlogram.check_variance_assumption("TYPO")


def test_morans_i():
    # answers calculated from PASSaGE 2 and exported to 5 decimals
    # tests both random and normal variance assumption
    answer = [(0.00000, 2.44706, 4187, -0.00282, 0.65306, 0.01474, 44.49577, 0.00000),
              (2.44706, 3.79348, 4190, -0.00282, 0.36807, 0.01472, 25.19434, 0.00000),
              (3.79348, 4.89556, 4189, -0.00282, 0.22621, 0.01470, 15.58188, 0.00000),
              (4.89556, 5.89960, 4189, -0.00282, 0.12944, 0.01476, 8.96358, 0.00000),
              (5.89960, 6.87800, 4189, -0.00282, -0.01105, 0.01474, 0.55805, 0.57681),
              (6.87800, 7.83495, 4188, -0.00282, -0.11995, 0.01475, 7.94297, 0.00000),
              (7.83495, 8.80877, 4190, -0.00282, -0.20705, 0.01474, 13.85548, 0.00000),
              (8.80877, 9.86668, 4189, -0.00282, -0.27155, 0.01478, 18.17685, 0.00000),
              (9.86668, 10.96444, 4189, -0.00282, -0.27587, 0.01479, 18.46680, 0.00000),
              (10.96444, 12.14837, 4189, -0.00282, -0.24051, 0.01477, 16.09118, 0.00000),
              (12.14837, 13.46790, 4189, -0.00282, -0.19934, 0.01480, 13.28110, 0.00000),
              (13.46790, 14.96656, 4189, -0.00282, -0.20647, 0.01476, 13.80134, 0.00000),
              (14.96656, 16.90406, 4189, -0.00282, -0.14810, 0.01469, 9.88984, 0.00000),
              (16.90406, 20.06888, 4189, -0.00282, 0.00519, 0.01441, 0.55641, 0.57793)]

    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    dist_classes = pyssage.distances.create_distance_classes(distances, "determine pair count", 15)
    dc_con = pyssage.connections.distance_classes_to_connections(dist_classes, distances)
    output, output_text = pyssage.correlogram.correlogram(data[:, 0], dc_con, pyssage.correlogram.morans_i)
    pyssage.graph.draw_correlogram(numpy.array(output), "Moran's I", "Correlogram",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
    for line in output_text:
        print(line)

    for i, row in enumerate(answer):
        for j, ans in enumerate(row):
            assert round(output[i][j], 5) == ans

    answer = [(0, 2.44706, 4187, -0.00282, 0.65306, 0.01501, 43.69575, 0),
              (2.44706, 3.79348, 4190, -0.00282, 0.36807, 0.01499, 24.74022, 0),
              (3.79348, 4.89556, 4189, -0.00282, 0.22621, 0.01497, 15.3002, 0),
              (4.89556, 5.8996, 4189, -0.00282, 0.12944, 0.01503, 8.80275, 0),
              (5.8996, 6.878, 4189, -0.00282, -0.01105, 0.01501, 0.54802, 0.58368),
              (6.878, 7.83495, 4188, -0.00282, -0.11995, 0.01502, 7.80026, 0),
              (7.83495, 8.80877, 4190, -0.00282, -0.20705, 0.01501, 13.60635, 0),
              (8.80877, 9.86668, 4189, -0.00282, -0.27155, 0.01505, 17.85189, 0),
              (9.86668, 10.96444, 4189, -0.00282, -0.27587, 0.01505, 18.13673, 0),
              (10.96444, 12.14837, 4189, -0.00282, -0.24051, 0.01504, 15.80303, 0),
              (12.14837, 13.4679, 4189, -0.00282, -0.19934, 0.01507, 13.04404, 0),
              (13.4679, 14.96656, 4189, -0.00282, -0.20647, 0.01503, 13.55369, 0),
              (14.96656, 16.90406, 4189, -0.00282, -0.1481, 0.01496, 9.71083, 0),
              (16.90406, 20.06888, 4189, -0.00282, 0.00519, 0.01468, 0.54596, 0.58509)]

    output, output_text = pyssage.correlogram.correlogram(data[:, 0], dc_con, pyssage.correlogram.morans_i,
                                                          variance="normal")
    for line in output_text:
        print(line)

    for i, row in enumerate(answer):
        for j, ans in enumerate(row):
            assert round(output[i][j], 5) == ans


def test_gearys_c():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    # tests both random and normal variance assumption
    answer = [(0.00000, 2.44706, 4187, 1.00000, 0.33204, 0.03535, 18.89357, 0.00000),
              (2.44706, 3.79348, 4190, 1.00000, 0.82819, 0.03635, 4.72694, 0.00000),
              (3.79348, 4.89556, 4189, 1.00000, 1.01299, 0.03812, 0.34080, 0.73325),
              (4.89556, 5.89960, 4189, 1.00000, 1.08074, 0.03382, 2.38726, 0.01697),
              (5.89960, 6.87800, 4189, 1.00000, 1.17545, 0.03478, 5.04462, 0.00000),
              (6.87800, 7.83495, 4188, 1.00000, 1.24647, 0.03477, 7.08797, 0.00000),
              (7.83495, 8.80877, 4190, 1.00000, 1.25130, 0.03493, 7.19388, 0.00000),
              (8.80877, 9.86668, 4189, 1.00000, 1.32815, 0.03148, 10.42263, 0.00000),
              (9.86668, 10.96444, 4189, 1.00000, 1.31421, 0.03133, 10.02750, 0.00000),
              (10.96444, 12.14837, 4189, 1.00000, 1.21946, 0.03256, 6.73952, 0.00000),
              (12.14837, 13.46790, 4189, 1.00000, 1.12783, 0.03041, 4.20371, 0.00003),
              (13.46790, 14.96656, 4189, 1.00000, 1.12163, 0.03385, 3.59348, 0.00033),
              (14.96656, 16.90406, 4189, 1.00000, 1.00765, 0.03881, 0.19715, 0.84371),
              (16.90406, 20.06888, 4189, 1.00000, 0.68403, 0.05482, 5.76401, 0.00000)]

    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    dist_classes = pyssage.distances.create_distance_classes(distances, "determine pair count", 15)
    dc_con = pyssage.connections.distance_classes_to_connections(dist_classes, distances)
    output, output_text = pyssage.correlogram.correlogram(data[:, 0], dc_con, pyssage.correlogram.gearys_c)
    pyssage.graph.draw_correlogram(numpy.array(output), "Geary's c", "Correlogram",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
    for line in output_text:
        print(line)

    for i, row in enumerate(answer):
        for j, ans in enumerate(row):
            assert round(output[i][j], 5) == ans

    answer = [(0, 2.44706, 4187, 1, 0.33204, 0.03477, 19.20937, 0),
              (2.44706, 3.79348, 4190, 1, 0.82819, 0.03574, 4.80691, 0),
              (3.79348, 4.89556, 4189, 1, 1.01299, 0.03748, 0.34668, 0.72884),
              (4.89556, 5.8996, 4189, 1, 1.08074, 0.03327, 2.42634, 0.01525),
              (5.8996, 6.878, 4189, 1, 1.17545, 0.03421, 5.12833, 0),
              (6.878, 7.83495, 4188, 1, 1.24647, 0.03421, 7.20556, 0),
              (7.83495, 8.80877, 4190, 1, 1.2513, 0.03436, 7.3135, 0),
              (8.80877, 9.86668, 4189, 1, 1.32815, 0.031, 10.58677, 0),
              (9.86668, 10.96444, 4189, 1, 1.31421, 0.03085, 10.18496, 0),
              (10.96444, 12.14837, 4189, 1, 1.21946, 0.03205, 6.8477, 0),
              (12.14837, 13.4679, 4189, 1, 1.12783, 0.02995, 4.2685, 0.00002),
              (13.4679, 14.96656, 4189, 1, 1.12163, 0.0333, 3.65234, 0.00026),
              (14.96656, 16.90406, 4189, 1, 1.00765, 0.03815, 0.20057, 0.84104),
              (16.90406, 20.06888, 4189, 1, 0.68403, 0.0538, 5.87292, 0)]

    output, output_text = pyssage.correlogram.correlogram(data[:, 0], dc_con, pyssage.correlogram.gearys_c,
                                                          variance="normal")
    for line in output_text:
        print(line)

    for i, row in enumerate(answer):
        for j, ans in enumerate(row):
            assert round(output[i][j], 5) == ans


def test_mantel_correl():
    # The Passage 2 Mantel correlogram code appears to be buggy
    # Some manual tests do seem to indiciate this is working correctly
    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    dist_classes = pyssage.distances.create_distance_classes(distances, "determine pair count", 15)
    dc_con = pyssage.connections.distance_classes_to_connections(dist_classes, distances)
    data_distances = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_euclidean)
    output, output_text = pyssage.correlogram.correlogram(data_distances, dc_con, pyssage.correlogram.mantel_correl,
                                                          variance=None)
    pyssage.graph.draw_correlogram(numpy.array(output), "Mantel r", "Correlogram", is_mantel=True,
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
    for line in output_text:
        print(line)


def test_bearing_correlogram():
    # answer calculated from PASSaGE 2 and exported to 5 decimals, but only matches perfectly to 2 decimals
    answer = [(0, 2.44706, 0, 4187, -0.00282, 0.48074, 0.01802, 26.83519, 0),
              (2.44706, 3.79348, 0, 4190, -0.00282, 0.38042, 0.01796, 21.3409, 0),
              (3.79348, 4.89556, 0, 4189, -0.00282, 0.31827, 0.01796, 17.87411, 0),
              (4.89556, 5.8996, 0, 4189, -0.00282, 0.2432, 0.01777, 13.84554, 0),
              (5.8996, 6.878, 0, 4189, -0.00282, 0.1433, 0.01747, 8.36592, 0),
              (6.878, 7.83495, 0, 4188, -0.00282, 0.08647, 0.01738, 5.13659, 0),
              (7.83495, 8.80877, 0, 4190, -0.00282, 0.03599, 0.01738, 2.23299, 0.02555),
              (8.80877, 9.86668, 0, 4189, -0.00282, -0.06002, 0.01699, 3.36671, 0.00076),
              (9.86668, 10.96444, 0, 4189, -0.00282, -0.12693, 0.01679, 7.39314, 0),
              (10.96444, 12.14837, 0, 4189, -0.00282, -0.14255, 0.0167, 8.36824, 0),
              (12.14837, 13.4679, 0, 4189, -0.00282, -0.19782, 0.01641, 11.88437, 0),
              (13.4679, 14.96656, 0, 4189, -0.00282, -0.17445, 0.01598, 10.73789, 0),
              (14.96656, 16.90406, 0, 4189, -0.00282, -0.13395, 0.01576, 8.32041, 0),
              (16.90406, 20.06888, 0, 4189, -0.00282, -0.09655, 0.01513, 6.19295, 0),
              (20.06888, 30.77475, 0, 4190, -0.00282, -0.08202, 0.01338, 5.91971, 0),
              (0, 2.44706, 10, 4187, -0.00282, 0.47307, 0.01807, 26.33563, 0),
              (2.44706, 3.79348, 10, 4190, -0.00282, 0.38604, 0.01814, 21.43293, 0),
              (3.79348, 4.89556, 10, 4189, -0.00282, 0.33896, 0.01822, 18.75573, 0),
              (4.89556, 5.8996, 10, 4189, -0.00282, 0.26948, 0.01813, 15.02122, 0),
              (5.8996, 6.878, 10, 4189, -0.00282, 0.17156, 0.01786, 9.76293, 0),
              (6.878, 7.83495, 10, 4188, -0.00282, 0.11858, 0.01791, 6.77797, 0),
              (7.83495, 8.80877, 10, 4190, -0.00282, 0.07241, 0.018, 4.17932, 0.00003),
              (8.80877, 9.86668, 10, 4189, -0.00282, -0.01655, 0.01761, 0.77927, 0.43582),
              (9.86668, 10.96444, 10, 4189, -0.00282, -0.08255, 0.01739, 4.58481, 0),
              (10.96444, 12.14837, 10, 4189, -0.00282, -0.09822, 0.01722, 5.53833, 0),
              (12.14837, 13.4679, 10, 4189, -0.00282, -0.1476, 0.01698, 8.52601, 0),
              (13.4679, 14.96656, 10, 4189, -0.00282, -0.13423, 0.01651, 7.95905, 0),
              (14.96656, 16.90406, 10, 4189, -0.00282, -0.09395, 0.01631, 5.58802, 0),
              (16.90406, 20.06888, 10, 4189, -0.00282, -0.08283, 0.01564, 5.11522, 0),
              (20.06888, 30.77475, 10, 4190, -0.00282, -0.10641, 0.01371, 7.55398, 0),
              (0, 2.44706, 20, 4187, -0.00282, 0.46412, 0.01816, 25.71993, 0),
              (2.44706, 3.79348, 20, 4190, -0.00282, 0.39231, 0.01836, 21.51603, 0),
              (3.79348, 4.89556, 20, 4189, -0.00282, 0.35983, 0.01849, 19.60887, 0),
              (4.89556, 5.8996, 20, 4189, -0.00282, 0.29332, 0.01853, 15.98136, 0),
              (5.8996, 6.878, 20, 4189, -0.00282, 0.19639, 0.01835, 10.85861, 0),
              (6.878, 7.83495, 20, 4188, -0.00282, 0.14687, 0.01856, 8.06648, 0),
              (7.83495, 8.80877, 20, 4190, -0.00282, 0.10385, 0.01877, 5.68364, 0),
              (8.80877, 9.86668, 20, 4189, -0.00282, 0.02409, 0.01848, 1.45688, 0.14515),
              (9.86668, 10.96444, 20, 4189, -0.00282, -0.03928, 0.01828, 1.99433, 0.04612),
              (10.96444, 12.14837, 20, 4189, -0.00282, -0.0573, 0.018, 3.02626, 0.00248),
              (12.14837, 13.4679, 20, 4189, -0.00282, -0.09885, 0.01788, 5.37144, 0),
              (13.4679, 14.96656, 20, 4189, -0.00282, -0.09371, 0.01743, 5.2138, 0),
              (14.96656, 16.90406, 20, 4189, -0.00282, -0.0545, 0.01727, 2.99216, 0.00277),
              (16.90406, 20.06888, 20, 4189, -0.00282, -0.06657, 0.01654, 3.8536, 0.00012),
              (20.06888, 30.77475, 20, 4190, -0.00282, -0.13482, 0.01438, 9.17691, 0),
              (0, 2.44706, 30, 4187, -0.00282, 0.45484, 0.01827, 25.05425, 0),
              (2.44706, 3.79348, 30, 4190, -0.00282, 0.39849, 0.01859, 21.58777, 0),
              (3.79348, 4.89556, 30, 4189, -0.00282, 0.37807, 0.01875, 20.31387, 0),
              (4.89556, 5.8996, 30, 4189, -0.00282, 0.31096, 0.01894, 16.57148, 0),
              (5.8996, 6.878, 30, 4189, -0.00282, 0.21354, 0.01888, 11.45773, 0),
              (6.878, 7.83495, 30, 4188, -0.00282, 0.16569, 0.01927, 8.746, 0),
              (7.83495, 8.80877, 30, 4190, -0.00282, 0.12295, 0.0196, 6.41713, 0),
              (8.80877, 9.86668, 30, 4189, -0.00282, 0.05402, 0.01955, 2.9079, 0.00364),
              (9.86668, 10.96444, 30, 4189, -0.00282, -0.00478, 0.01947, 0.10036, 0.92006),
              (10.96444, 12.14837, 30, 4189, -0.00282, -0.02791, 0.01907, 1.31546, 0.18836),
              (12.14837, 13.4679, 30, 4189, -0.00282, -0.06017, 0.01919, 2.98788, 0.00281),
              (13.4679, 14.96656, 30, 4189, -0.00282, -0.05904, 0.01897, 2.96275, 0.00305),
              (14.96656, 16.90406, 30, 4189, -0.00282, -0.02338, 0.01895, 1.08516, 0.27785),
              (16.90406, 20.06888, 30, 4189, -0.00282, -0.04755, 0.01817, 2.46153, 0.01383),
              (20.06888, 30.77475, 30, 4190, -0.00282, -0.16708, 0.01586, 10.35664, 0),
              (0, 2.44706, 40, 4187, -0.00282, 0.4463, 0.01839, 24.41848, 0),
              (2.44706, 3.79348, 40, 4190, -0.00282, 0.40373, 0.01878, 21.64978, 0),
              (3.79348, 4.89556, 40, 4189, -0.00282, 0.39061, 0.01896, 20.74913, 0),
              (4.89556, 5.8996, 40, 4189, -0.00282, 0.31829, 0.01929, 16.64843, 0),
              (5.8996, 6.878, 40, 4189, -0.00282, 0.21793, 0.01941, 11.37286, 0),
              (6.878, 7.83495, 40, 4188, -0.00282, 0.1678, 0.01994, 8.55854, 0),
              (7.83495, 8.80877, 40, 4190, -0.00282, 0.12035, 0.02034, 6.05681, 0),
              (8.80877, 9.86668, 40, 4189, -0.00282, 0.06057, 0.02065, 3.07035, 0.00214),
              (9.86668, 10.96444, 40, 4189, -0.00282, 0.00666, 0.02083, 0.45545, 0.64879),
              (10.96444, 12.14837, 40, 4189, -0.00282, -0.02482, 0.02038, 1.07925, 0.28047),
              (12.14837, 13.4679, 40, 4189, -0.00282, -0.05071, 0.0209, 2.29079, 0.02198),
              (13.4679, 14.96656, 40, 4189, -0.00282, -0.04765, 0.02132, 2.1021, 0.03554),
              (14.96656, 16.90406, 40, 4189, -0.00282, -0.02418, 0.02168, 0.98505, 0.3246),
              (16.90406, 20.06888, 40, 4189, -0.00282, -0.02923, 0.0211, 1.2517, 0.21068),
              (20.06888, 30.77475, 40, 4190, -0.00282, -0.18898, 0.01958, 9.50984, 0),
              (0, 2.44706, 50, 4187, -0.00282, 0.43961, 0.01851, 23.8969, 0),
              (2.44706, 3.79348, 50, 4190, -0.00282, 0.40718, 0.01889, 21.70227, 0),
              (3.79348, 4.89556, 50, 4189, -0.00282, 0.39485, 0.0191, 20.82434, 0),
              (4.89556, 5.8996, 50, 4189, -0.00282, 0.31222, 0.01953, 16.13272, 0),
              (5.8996, 6.878, 50, 4189, -0.00282, 0.20547, 0.01983, 10.50219, 0),
              (6.878, 7.83495, 50, 4188, -0.00282, 0.14745, 0.02041, 7.36247, 0),
              (7.83495, 8.80877, 50, 4190, -0.00282, 0.08949, 0.02077, 4.44492, 0.00001),
              (8.80877, 9.86668, 50, 4189, -0.00282, 0.03063, 0.02143, 1.56096, 0.11853),
              (9.86668, 10.96444, 50, 4189, -0.00282, -0.0225, 0.02194, 0.8967, 0.36988),
              (10.96444, 12.14837, 50, 4189, -0.00282, -0.06619, 0.02162, 2.93126, 0.00338),
              (12.14837, 13.4679, 50, 4189, -0.00282, -0.09994, 0.02261, 4.29525, 0.00002),
              (13.4679, 14.96656, 50, 4189, -0.00282, -0.09427, 0.0241, 3.79459, 0.00015),
              (14.96656, 16.90406, 50, 4189, -0.00282, -0.107, 0.02516, 4.1415, 0.00003),
              (16.90406, 20.06888, 50, 4189, -0.00282, -0.03092, 0.02561, 1.09705, 0.27262),
              (20.06888, 30.77475, 50, 4190, -0.00282, -0.0715, 0.02915, 2.3554, 0.0185),
              (0, 2.44706, 60, 4187, -0.00282, 0.43572, 0.01861, 23.56472, 0),
              (2.44706, 3.79348, 60, 4190, -0.00282, 0.40824, 0.01891, 21.73918, 0),
              (3.79348, 4.89556, 60, 4189, -0.00282, 0.38962, 0.01913, 20.51508, 0),
              (4.89556, 5.8996, 60, 4189, -0.00282, 0.29256, 0.01961, 15.06519, 0),
              (5.8996, 6.878, 60, 4189, -0.00282, 0.17599, 0.02005, 8.92038, 0),
              (6.878, 7.83495, 60, 4188, -0.00282, 0.10569, 0.02055, 5.28017, 0),
              (7.83495, 8.80877, 60, 4190, -0.00282, 0.03413, 0.02075, 1.78084, 0.07494),
              (8.80877, 9.86668, 60, 4189, -0.00282, -0.03686, 0.02156, 1.5782, 0.11452),
              (9.86668, 10.96444, 60, 4189, -0.00282, -0.09679, 0.02226, 4.2206, 0.00002),
              (10.96444, 12.14837, 60, 4189, -0.00282, -0.15626, 0.02224, 6.90031, 0),
              (12.14837, 13.4679, 60, 4189, -0.00282, -0.22083, 0.0234, 9.3154, 0),
              (13.4679, 14.96656, 60, 4189, -0.00282, -0.22276, 0.02556, 8.6043, 0),
              (14.96656, 16.90406, 60, 4189, -0.00282, -0.29684, 0.02687, 10.94201, 0),
              (16.90406, 20.06888, 60, 4189, -0.00282, -0.09189, 0.0285, 3.12579, 0.00177),
              (20.06888, 30.77475, 60, 4190, -0.00282, 0.49247, 0.03251, 15.23424, 0),
              (0, 2.44706, 70, 4187, -0.00282, 0.43529, 0.01866, 23.47349, 0),
              (2.44706, 3.79348, 70, 4190, -0.00282, 0.40672, 0.01883, 21.74789, 0),
              (3.79348, 4.89556, 70, 4189, -0.00282, 0.37575, 0.01905, 19.87629, 0),
              (4.89556, 5.8996, 70, 4189, -0.00282, 0.26283, 0.0195, 13.62277, 0),
              (5.8996, 6.878, 70, 4189, -0.00282, 0.13511, 0.01998, 6.90465, 0),
              (6.878, 7.83495, 70, 4188, -0.00282, 0.05265, 0.0203, 2.73215, 0.00629),
              (7.83495, 8.80877, 70, 4190, -0.00282, -0.03029, 0.02031, 1.35272, 0.17615),
              (8.80877, 9.86668, 70, 4189, -0.00282, -0.12205, 0.02101, 5.67541, 0),
              (9.86668, 10.96444, 70, 4189, -0.00282, -0.19225, 0.02161, 8.76695, 0),
              (10.96444, 12.14837, 70, 4189, -0.00282, -0.26658, 0.02185, 12.07043, 0),
              (12.14837, 13.4679, 70, 4189, -0.00282, -0.36851, 0.02269, 16.1172, 0),
              (13.4679, 14.96656, 70, 4189, -0.00282, -0.37466, 0.02421, 15.35972, 0),
              (14.96656, 16.90406, 70, 4189, -0.00282, -0.47438, 0.02472, 19.07718, 0),
              (16.90406, 20.06888, 70, 4189, -0.00282, -0.17217, 0.0254, 6.66709, 0),
              (20.06888, 30.77475, 70, 4190, -0.00282, 0.47201, 0.02011, 23.6116, 0),
              (0, 2.44706, 80, 4187, -0.00282, 0.43845, 0.01867, 23.63945, 0),
              (2.44706, 3.79348, 80, 4190, -0.00282, 0.40288, 0.01868, 21.71519, 0),
              (3.79348, 4.89556, 80, 4189, -0.00282, 0.35586, 0.01885, 19.02364, 0),
              (4.89556, 5.8996, 80, 4189, -0.00282, 0.22916, 0.01923, 12.06622, 0),
              (5.8996, 6.878, 80, 4189, -0.00282, 0.09232, 0.01964, 4.84504, 0),
              (6.878, 7.83495, 80, 4188, -0.00282, 0.00239, 0.01975, 0.26419, 0.79164),
              (7.83495, 8.80877, 80, 4190, -0.00282, -0.08623, 0.0196, 4.25445, 0.00002),
              (8.80877, 9.86668, 80, 4189, -0.00282, -0.197, 0.02007, 9.67494, 0),
              (9.86668, 10.96444, 80, 4189, -0.00282, -0.27336, 0.02039, 13.26693, 0),
              (10.96444, 12.14837, 80, 4189, -0.00282, -0.35474, 0.02072, 16.98631, 0),
              (12.14837, 13.4679, 80, 4189, -0.00282, -0.47453, 0.02105, 22.40755, 0),
              (13.4679, 14.96656, 80, 4189, -0.00282, -0.4621, 0.02147, 21.3919, 0),
              (14.96656, 16.90406, 80, 4189, -0.00282, -0.53191, 0.02127, 24.87785, 0),
              (16.90406, 20.06888, 80, 4189, -0.00282, -0.1995, 0.021, 9.36372, 0),
              (20.06888, 30.77475, 80, 4190, -0.00282, 0.29056, 0.01611, 18.21245, 0),
              (0, 2.44706, 90, 4187, -0.00282, 0.4448, 0.01862, 24.03875, 0),
              (2.44706, 3.79348, 90, 4190, -0.00282, 0.3974, 0.0185, 21.63606, 0),
              (3.79348, 4.89556, 90, 4189, -0.00282, 0.33346, 0.01858, 18.09561, 0),
              (4.89556, 5.8996, 90, 4189, -0.00282, 0.19781, 0.01884, 10.65162, 0),
              (5.8996, 6.878, 90, 4189, -0.00282, 0.05638, 0.01912, 3.0971, 0.00195),
              (6.878, 7.83495, 90, 4188, -0.00282, -0.03508, 0.01905, 1.69383, 0.0903),
              (7.83495, 8.80877, 90, 4190, -0.00282, -0.12398, 0.01883, 6.43473, 0),
              (8.80877, 9.86668, 90, 4189, -0.00282, -0.24591, 0.01909, 12.73616, 0),
              (9.86668, 10.96444, 90, 4189, -0.00282, -0.32249, 0.01914, 16.69749, 0),
              (10.96444, 12.14837, 90, 4189, -0.00282, -0.40141, 0.0194, 20.54101, 0),
              (12.14837, 13.4679, 90, 4189, -0.00282, -0.51842, 0.0194, 26.5754, 0),
              (13.4679, 14.96656, 90, 4189, -0.00282, -0.47944, 0.01919, 24.83787, 0),
              (14.96656, 16.90406, 90, 4189, -0.00282, -0.51009, 0.01877, 27.02716, 0),
              (16.90406, 20.06888, 90, 4189, -0.00282, -0.19556, 0.01825, 10.56136, 0),
              (20.06888, 30.77475, 90, 4190, -0.00282, 0.18576, 0.01466, 12.85951, 0),
              (0, 2.44706, 100, 4187, -0.00282, 0.45342, 0.01854, 24.61181, 0),
              (2.44706, 3.79348, 100, 4190, -0.00282, 0.39113, 0.01831, 21.5192, 0),
              (3.79348, 4.89556, 100, 4189, -0.00282, 0.3119, 0.01828, 17.219, 0),
              (4.89556, 5.8996, 100, 4189, -0.00282, 0.17316, 0.0184, 9.56412, 0),
              (5.8996, 6.878, 100, 4189, -0.00282, 0.03208, 0.01853, 1.88349, 0.05963),
              (6.878, 7.83495, 100, 4188, -0.00282, -0.05668, 0.01834, 2.937, 0.00331),
              (7.83495, 8.80877, 100, 4190, -0.00282, -0.14262, 0.01811, 7.71925, 0),
              (8.80877, 9.86668, 100, 4189, -0.00282, -0.26826, 0.01823, 14.55902, 0),
              (9.86668, 10.96444, 100, 4189, -0.00282, -0.34194, 0.01812, 18.71055, 0),
              (10.96444, 12.14837, 100, 4189, -0.00282, -0.41239, 0.01828, 22.40873, 0),
              (12.14837, 13.4679, 100, 4189, -0.00282, -0.51825, 0.01812, 28.4373, 0),
              (13.4679, 14.96656, 100, 4189, -0.00282, -0.4604, 0.01772, 25.82695, 0),
              (14.96656, 16.90406, 100, 4189, -0.00282, -0.46328, 0.01728, 26.64575, 0),
              (16.90406, 20.06888, 100, 4189, -0.00282, -0.18384, 0.01671, 10.83386, 0),
              (20.06888, 30.77475, 100, 4190, -0.00282, 0.12258, 0.01397, 8.97536, 0),
              (0, 2.44706, 110, 4187, -0.00282, 0.4631, 0.01843, 25.27599, 0),
              (2.44706, 3.79348, 110, 4190, -0.00282, 0.38493, 0.01813, 21.38555, 0),
              (3.79348, 4.89556, 110, 4189, -0.00282, 0.29371, 0.01798, 16.49062, 0),
              (4.89556, 5.8996, 110, 4189, -0.00282, 0.15722, 0.01798, 8.89959, 0),
              (5.8996, 6.878, 110, 4189, -0.00282, 0.02024, 0.01799, 1.28271, 0.19959),
              (6.878, 7.83495, 110, 4188, -0.00282, -0.06389, 0.01772, 3.44647, 0.00057),
              (7.83495, 8.80877, 110, 4190, -0.00282, -0.14554, 0.01751, 8.15027, 0),
              (8.80877, 9.86668, 110, 4189, -0.00282, -0.27032, 0.01755, 15.23956, 0),
              (9.86668, 10.96444, 110, 4189, -0.00282, -0.34063, 0.01737, 19.45055, 0),
              (10.96444, 12.14837, 110, 4189, -0.00282, -0.40052, 0.01743, 22.81334, 0),
              (12.14837, 13.4679, 110, 4189, -0.00282, -0.49483, 0.01723, 28.55793, 0),
              (13.4679, 14.96656, 110, 4189, -0.00282, -0.42801, 0.01681, 25.28987, 0),
              (14.96656, 16.90406, 110, 4189, -0.00282, -0.41375, 0.01641, 25.0478, 0),
              (16.90406, 20.06888, 110, 4189, -0.00282, -0.17149, 0.01582, 10.66167, 0),
              (20.06888, 30.77475, 110, 4190, -0.00282, 0.07985, 0.01359, 6.08499, 0),
              (0, 2.44706, 120, 4187, -0.00282, 0.47254, 0.01832, 25.94233, 0),
              (2.44706, 3.79348, 120, 4190, -0.00282, 0.37948, 0.01798, 21.26104, 0),
              (3.79348, 4.89556, 120, 4189, -0.00282, 0.28047, 0.01773, 15.97381, 0),
              (4.89556, 5.8996, 120, 4189, -0.00282, 0.15023, 0.01763, 8.67995, 0),
              (5.8996, 6.878, 120, 4189, -0.00282, 0.01944, 0.01753, 1.26982, 0.20415),
              (6.878, 7.83495, 120, 4188, -0.00282, -0.05974, 0.01723, 3.30302, 0.00096),
              (7.83495, 8.80877, 120, 4190, -0.00282, -0.1368, 0.01705, 7.85753, 0),
              (8.80877, 9.86668, 120, 4189, -0.00282, -0.25855, 0.01704, 15.00781, 0),
              (9.86668, 10.96444, 120, 4189, -0.00282, -0.32628, 0.01683, 19.21502, 0),
              (10.96444, 12.14837, 120, 4189, -0.00282, -0.37579, 0.01684, 22.14181, 0),
              (12.14837, 13.4679, 120, 4189, -0.00282, -0.46065, 0.01662, 27.54474, 0),
              (13.4679, 14.96656, 120, 4189, -0.00282, -0.3921, 0.01625, 23.95691, 0),
              (14.96656, 16.90406, 120, 4189, -0.00282, -0.36743, 0.01588, 22.96378, 0),
              (16.90406, 20.06888, 120, 4189, -0.00282, -0.15997, 0.01529, 10.28018, 0),
              (20.06888, 30.77475, 120, 4190, -0.00282, 0.048, 0.01336, 3.80467, 0.00014),
              (0, 2.44706, 130, 4187, -0.00282, 0.48059, 0.01822, 26.53073, 0),
              (2.44706, 3.79348, 130, 4190, -0.00282, 0.37533, 0.01786, 21.16844, 0),
              (3.79348, 4.89556, 130, 4189, -0.00282, 0.27296, 0.01756, 15.70279, 0),
              (4.89556, 5.8996, 130, 4189, -0.00282, 0.15147, 0.01738, 8.87842, 0),
              (5.8996, 6.878, 130, 4189, -0.00282, 0.02755, 0.01721, 1.76513, 0.07754),
              (6.878, 7.83495, 130, 4188, -0.00282, -0.0471, 0.01689, 2.62078, 0.00877),
              (7.83495, 8.80877, 130, 4190, -0.00282, -0.11967, 0.01674, 6.98134, 0),
              (8.80877, 9.86668, 130, 4189, -0.00282, -0.23759, 0.01667, 14.07958, 0),
              (9.86668, 10.96444, 130, 4189, -0.00282, -0.30381, 0.01648, 18.26765, 0),
              (10.96444, 12.14837, 130, 4189, -0.00282, -0.34406, 0.01646, 20.7305, 0),
              (12.14837, 13.4679, 130, 4189, -0.00282, -0.4217, 0.01622, 25.8171, 0),
              (13.4679, 14.96656, 130, 4189, -0.00282, -0.356, 0.01589, 22.21982, 0),
              (14.96656, 16.90406, 130, 4189, -0.00282, -0.3249, 0.01556, 20.70374, 0),
              (16.90406, 20.06888, 130, 4189, -0.00282, -0.14931, 0.01496, 9.78971, 0),
              (20.06888, 30.77475, 130, 4190, -0.00282, 0.02227, 0.01322, 1.89801, 0.0577),
              (0, 2.44706, 140, 4187, -0.00282, 0.48638, 0.01813, 26.97921, 0),
              (2.44706, 3.79348, 140, 4190, -0.00282, 0.37279, 0.01778, 21.12189, 0),
              (3.79348, 4.89556, 140, 4189, -0.00282, 0.27139, 0.01748, 15.68917, 0),
              (4.89556, 5.8996, 140, 4189, -0.00282, 0.15989, 0.01724, 9.44098, 0),
              (5.8996, 6.878, 140, 4189, -0.00282, 0.04258, 0.01701, 2.66925, 0.0076),
              (6.878, 7.83495, 140, 4188, -0.00282, -0.02811, 0.0167, 1.51364, 0.13012),
              (7.83495, 8.80877, 140, 4190, -0.00282, -0.09632, 0.01656, 5.6445, 0),
              (8.80877, 9.86668, 140, 4189, -0.00282, -0.21027, 0.01644, 12.61499, 0),
              (9.86668, 10.96444, 140, 4189, -0.00282, -0.27591, 0.01626, 16.79143, 0),
              (10.96444, 12.14837, 140, 4189, -0.00282, -0.30831, 0.01624, 18.81238, 0),
              (12.14837, 13.4679, 140, 4189, -0.00282, -0.38042, 0.01599, 23.6177, 0),
              (13.4679, 14.96656, 140, 4189, -0.00282, -0.32044, 0.01568, 20.25413, 0),
              (14.96656, 16.90406, 140, 4189, -0.00282, -0.28529, 0.01537, 18.37384, 0),
              (16.90406, 20.06888, 140, 4189, -0.00282, -0.13921, 0.01478, 9.22785, 0),
              (20.06888, 30.77475, 140, 4190, -0.00282, 0.00002, 0.01314, 0.21666, 0.82848),
              (0, 2.44706, 150, 4187, -0.00282, 0.48937, 0.01806, 27.24672, 0),
              (2.44706, 3.79348, 150, 4190, -0.00282, 0.37204, 0.01775, 21.125, 0),
              (3.79348, 4.89556, 150, 4189, -0.00282, 0.27566, 0.01748, 15.92723, 0),
              (4.89556, 5.8996, 150, 4189, -0.00282, 0.17441, 0.01721, 10.29966, 0),
              (5.8996, 6.878, 150, 4189, -0.00282, 0.06294, 0.01694, 3.88113, 0.0001),
              (6.878, 7.83495, 150, 4188, -0.00282, -0.00424, 0.01666, 0.08506, 0.93221),
              (7.83495, 8.80877, 150, 4190, -0.00282, -0.06816, 0.01653, 3.95108, 0.00008),
              (8.80877, 9.86668, 150, 4189, -0.00282, -0.17815, 0.01635, 10.72492, 0),
              (9.86668, 10.96444, 150, 4189, -0.00282, -0.24389, 0.01618, 14.89982, 0),
              (10.96444, 12.14837, 150, 4189, -0.00282, -0.26987, 0.01615, 16.53313, 0),
              (12.14837, 13.4679, 150, 4189, -0.00282, -0.33754, 0.01588, 21.07267, 0),
              (13.4679, 14.96656, 150, 4189, -0.00282, -0.28516, 0.01558, 18.12305, 0),
              (14.96656, 16.90406, 150, 4189, -0.00282, -0.24744, 0.0153, 15.99164, 0),
              (16.90406, 20.06888, 150, 4189, -0.00282, -0.1293, 0.0147, 8.60207, 0),
              (20.06888, 30.77475, 150, 4190, -0.00282, -0.02038, 0.01311, 1.33937, 0.18045),
              (0, 2.44706, 160, 4187, -0.00282, 0.48933, 0.01802, 27.31181, 0),
              (2.44706, 3.79348, 160, 4190, -0.00282, 0.37314, 0.01776, 21.17171, 0),
              (3.79348, 4.89556, 160, 4189, -0.00282, 0.28537, 0.01758, 16.39677, 0),
              (4.89556, 5.8996, 160, 4189, -0.00282, 0.19394, 0.01729, 11.37836, 0),
              (5.8996, 6.878, 160, 4189, -0.00282, 0.08733, 0.017, 5.30268, 0),
              (6.878, 7.83495, 160, 4188, -0.00282, 0.02346, 0.01676, 1.56884, 0.11668),
              (7.83495, 8.80877, 160, 4190, -0.00282, -0.03609, 0.01665, 1.99755, 0.04577),
              (8.80877, 9.86668, 160, 4189, -0.00282, -0.14202, 0.01639, 8.49262, 0),
              (9.86668, 10.96444, 160, 4189, -0.00282, -0.20826, 0.01622, 12.66257, 0),
              (10.96444, 12.14837, 160, 4189, -0.00282, -0.22926, 0.01619, 13.98361, 0),
              (12.14837, 13.4679, 160, 4189, -0.00282, -0.29302, 0.01591, 18.24289, 0),
              (13.4679, 14.96656, 160, 4189, -0.00282, -0.24954, 0.01558, 15.83508, 0),
              (14.96656, 16.90406, 160, 4189, -0.00282, -0.21024, 0.01532, 13.53841, 0),
              (16.90406, 20.06888, 160, 4189, -0.00282, -0.11919, 0.01472, 7.90316, 0),
              (20.06888, 30.77475, 160, 4190, -0.00282, -0.04016, 0.01313, 2.84379, 0.00446),
              (0, 2.44706, 170, 4187, -0.00282, 0.48635, 0.018, 27.17048, 0),
              (2.44706, 3.79348, 170, 4190, -0.00282, 0.376, 0.01783, 21.24901, 0),
              (3.79348, 4.89556, 170, 4189, -0.00282, 0.29991, 0.01774, 17.0631, 0),
              (4.89556, 5.8996, 170, 4189, -0.00282, 0.21735, 0.01748, 12.59266, 0),
              (5.8996, 6.878, 170, 4189, -0.00282, 0.11457, 0.01718, 6.83448, 0),
              (6.878, 7.83495, 170, 4188, -0.00282, 0.0541, 0.017, 3.34871, 0.00081),
              (7.83495, 8.80877, 170, 4190, -0.00282, -0.00097, 0.01693, 0.10976, 0.9126),
              (8.80877, 9.86668, 170, 4189, -0.00282, -0.10241, 0.01659, 6.00114, 0),
              (9.86668, 10.96444, 170, 4189, -0.00282, -0.16919, 0.01642, 10.13488, 0),
              (10.96444, 12.14837, 170, 4189, -0.00282, -0.18668, 0.01637, 11.23348, 0),
              (12.14837, 13.4679, 170, 4189, -0.00282, -0.24651, 0.01607, 15.1636, 0),
              (13.4679, 14.96656, 170, 4189, -0.00282, -0.21284, 0.0157, 13.37633, 0),
              (14.96656, 16.90406, 170, 4189, -0.00282, -0.17267, 0.01546, 10.98582, 0),
              (16.90406, 20.06888, 170, 4189, -0.00282, -0.10845, 0.01486, 7.11028, 0),
              (20.06888, 30.77475, 170, 4190, -0.00282, -0.06035, 0.01321, 4.35532, 0.00001)]

    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    dist_classes = pyssage.distances.create_distance_classes(distances, "determine pair count", 15)
    dc_con = pyssage.connections.distance_classes_to_connections(dist_classes, distances)
    output, output_text = pyssage.correlogram.bearing_correlogram(data[:, 22], dc_con, angles)
    # pyssage.graph.draw_bearing_correlogram_old(numpy.array(output), "Moran's I Bearing Correlogram")
    pyssage.graph.draw_bearing_correlogram(numpy.array(output), "Moran's I Bearing Correlogram",
                                           figoutput=pyssage.graph.FigOutput(figshow=True))
    for line in output_text:
        print(line)

    for i, row in enumerate(answer):
        for j, ans in enumerate(row):
            assert round(output[i][j], 2) == round(ans, 2)


def test_windrose_correlogram():
    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    output, output_text, all_output = pyssage.correlogram.windrose_correlogram(data[:, 22], distances, angles,
                                                                               radius_c=3, radius_d=0, radius_e=0)
    pyssage.graph.draw_windrose_correlogram(numpy.array(all_output), title="Moran's I Windrose Correlogram Pair Counts",
                                            show_counts=True, figoutput=pyssage.graph.FigOutput(figshow=True))
    pyssage.graph.draw_windrose_correlogram(numpy.array(all_output), title="Moran's I Windrose Correlogram",
                                            figoutput=pyssage.graph.FigOutput(figshow=True))
    for line in output_text:
        print(line)

    # output, output_text, all_output = pyssage.correlogram.windrose_correlogram(data[:, 22], distances, angles,
    #                                                                            radius_c=3, radius_d=0, radius_e=0,
    #                                                                            segment_param=6)
    # pyssage.graph.draw_windrose_correlogram(numpy.array(all_output), title="Moran's I Windrose Correlogram")
    #
    # output, output_text, all_output = pyssage.correlogram.windrose_correlogram(data[:, 22], distances, angles,
    #                                                                            radius_c=1, radius_d=0, radius_e=0)
    # pyssage.graph.draw_windrose_correlogram(numpy.array(all_output), title="Moran's I Windrose Correlogram")


def test_mantel_windrose_correlogram():
    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    data_distances = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_euclidean)

    output, output_text, all_output = pyssage.correlogram.windrose_correlogram(data_distances, distances, angles,
                                                                               radius_c=3, radius_d=0, radius_e=0,
                                                                               metric=pyssage.correlogram.mantel_correl)
    pyssage.graph.draw_windrose_correlogram(numpy.array(all_output), title="Mantel Windrose Correlogram Pair Counts",
                                            show_counts=True, is_mantel=True,
                                            figoutput=pyssage.graph.FigOutput(figshow=True))
    pyssage.graph.draw_windrose_correlogram(numpy.array(all_output), title="Mantel Windrose Correlogram",
                                            is_mantel=True, figoutput=pyssage.graph.FigOutput(figshow=True))
    for line in output_text:
        print(line)
