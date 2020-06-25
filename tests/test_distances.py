import numpy
import pytest
import math
import pyssage.connections
import pyssage.distances
import pyssage.graph
from tests.test_common import test_coords, load_answer


def test_euc_dist_matrix():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("answers/euc_distmat_answer.txt")

    coords = test_coords()
    output = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert round(output[i, j], 5) == answer[i, j]


def test_euc_angle_matrix():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("answers/euc_anglemat_answer.txt")

    coords = test_coords()
    output = pyssage.distances.euc_angle_matrix(coords[:, 0], coords[:, 1])
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert round(output[i, j]*180/math.pi, 5) == answer[i, j]


def test_sph_dist_matrix():
    """
    The original code matched the PASSaGE output, but I decided to go away from predetermined constants for
    conversion of angels to radians, meaning the numbers shift a bit due to greater floating-point accuracy. I am
    also be using a better estimate of the radius of the Earth in the calculations

    I have cross-checked the output in a small way using online calculators

    this change might cascade to other tests that relied upon these distances
    """
    coords = test_coords()
    pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])


def test_sph_angle_matrix():
    coords = test_coords()
    pyssage.distances.sph_angle_matrix(coords[:, 0], coords[:, 1])


def test_check_input_distance_matrix():
    # valid test
    test_size = 5
    matrix = numpy.zeros((test_size, test_size))
    n = pyssage.connections.check_input_distance_matrix(matrix)
    assert n == test_size

    with pytest.raises(ValueError, match="distance matrix must be two-dimensional"):
        # broken test --> should raise error
        matrix = numpy.zeros((1, 2, 3))
        pyssage.connections.check_input_distance_matrix(matrix)

    with pytest.raises(ValueError, match="distance matrix must be square"):
        # broken test --> should raise error
        matrix = numpy.zeros((test_size, test_size * 2))
        pyssage.connections.check_input_distance_matrix(matrix)


def test_shortest_path_distances():
    """
    answer provided by PASSaGE 2 did match answer here, although only to nearest integer (rather than 5 decimals)
    due to rounding errors that add up, prior to my change of the spherical distance calculation formula and
    radius of earth used for those calculations
    """
    coords = test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])

    # test a fully connected network
    connections = pyssage.connections.minimum_spanning_tree(distances)
    geodists, trace = pyssage.distances.shortest_path_distances(distances, connections)
    pyssage.graph.draw_shortest_path(connections, coords[:, 0], coords[:, 1], trace, 0, 300)

    # test a partially connected network
    connections = pyssage.connections.nearest_neighbor_connections(distances, 1)
    pyssage.distances.shortest_path_distances(distances, connections)


def test_create_distance_classes():
    coords = test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    dc = pyssage.distances.create_distance_classes(distances, "determine class width", 10)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Ten Equal Width Distance Classes")
    dc = pyssage.distances.create_distance_classes(distances, "determine pair count", 10)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Ten Equal Count Distance Classes")
    dc = pyssage.distances.create_distance_classes(distances, "set class width", 200)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Distance Class Width Set to 200")
    dc = pyssage.distances.create_distance_classes(distances, "set pair count", 5000)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Distance Class Pair Count Set to 5000")
