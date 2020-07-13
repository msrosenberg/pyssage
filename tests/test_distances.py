import math
import pyssage.connections
import pyssage.distances
import pyssage.graph
from tests.test_common import create_test_coords, load_answer


def test_euc_dist_matrix():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("answers/euc_distmat_answer.txt")

    coords = create_test_coords()
    output = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert round(output[i, j], 5) == answer[i, j]


def test_euc_angle_matrix():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("answers/euc_anglemat_answer.txt")

    coords = create_test_coords()
    output = pyssage.distances.euc_angle_matrix(coords[:, 0], coords[:, 1])
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert round(math.degrees(output[i, j]), 5) == answer[i, j]


def test_sph_dist_matrix():
    """
    The original code matched the PASSaGE output, but I decided to go away from predetermined constants for
    conversion of angels to radians, meaning the numbers shift a bit due to greater floating-point accuracy. I am
    also be using a better estimate of the radius of the Earth in the calculations

    I have cross-checked the output in a small way using online calculators

    this change might cascade to other tests that relied upon these distances
    """
    coords = create_test_coords()
    pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])


def test_sph_angle_matrix():
    """
    completely rewritten the way this is done, for a variety of reasons it is difficult to impossible to match
    against PASSaGE 2 output
    """
    coords = create_test_coords()
    pyssage.distances.sph_angle_matrix(coords[:, 0], coords[:, 1])


def test_shortest_path_distances():
    """
    testing with euclidean distances as the spherical estimation procedure is now a bit different
    """
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("answers/shortest_path_minspan_answer.txt")

    # test a fully connected network
    coords = create_test_coords()
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.minimum_spanning_tree(distances)
    geodists, trace = pyssage.distances.shortest_path_distances(distances, connections)
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert round(geodists[i, j], 5) == answer[i, j]
    pyssage.graph.draw_shortest_path(connections, coords[:, 0], coords[:, 1], trace, 0, 300)

    # test a partially connected network
    connections = pyssage.connections.nearest_neighbor_connections(distances, 1)
    pyssage.distances.shortest_path_distances(distances, connections)


def test_create_distance_classes():
    coords = create_test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    dc = pyssage.distances.create_distance_classes(distances, "determine class width", 10)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Ten Equal Width Distance Classes")
    dc = pyssage.distances.create_distance_classes(distances, "determine pair count", 10)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Ten Equal Count Distance Classes")
    dc = pyssage.distances.create_distance_classes(distances, "set class width", 200)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Distance Class Width Set to 200")
    dc = pyssage.distances.create_distance_classes(distances, "set pair count", 5000)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Distance Class Pair Count Set to 5000")
