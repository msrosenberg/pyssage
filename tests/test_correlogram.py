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
    # answer calculated from PASSaGE 2 and exported to 5 decimals
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
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    dist_classes = pyssage.distances.create_distance_classes(distances, "determine pair count", 15)
    dc_con = pyssage.connections.distance_classes_to_connections(dist_classes, distances)
    output, output_text = pyssage.correlogram.correlogram(data[:, 0], dc_con, pyssage.correlogram.morans_i)
    pyssage.graph.draw_correlogram(numpy.array(output), "Moran's I", "Correlogram")
    for line in output_text:
        print(line)

    for i, row in enumerate(answer):
        for j, ans in enumerate(row):
            assert ans == round(output[i][j], 5)


def test_gearys_c():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
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
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    dist_classes = pyssage.distances.create_distance_classes(distances, "determine pair count", 15)
    dc_con = pyssage.connections.distance_classes_to_connections(dist_classes, distances)
    output, output_text = pyssage.correlogram.correlogram(data[:, 0], dc_con, pyssage.correlogram.gearys_c)
    pyssage.graph.draw_correlogram(numpy.array(output), "Geary's c", "Correlogram")
    for line in output_text:
        print(line)

    for i, row in enumerate(answer):
        for j, ans in enumerate(row):
            assert ans == round(output[i][j], 5)


def test_bearing_correlogram():
    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euc_angle_matrix(coords[:, 0], coords[:, 1])
    dist_classes = pyssage.distances.create_distance_classes(distances, "determine pair count", 15)
    dc_con = pyssage.connections.distance_classes_to_connections(dist_classes, distances)
    output, output_text = pyssage.correlogram.bearing_correlogram(data[:, 22], dc_con, angles)
    pyssage.graph.draw_bearing_correlogram(numpy.array(output), "Moran's I Bearing Correlogram")
    for line in output_text:
        print(line)
