import numpy
import pyssage.anisotropy
import pyssage.distances
import pyssage.graph
from tests.test_common import create_test_scattered, create_test_coords


def test_bearing_analysis():
    # answer calculated from PASSaGE 2 and exported to 5 decimals, but only matches perfectly to 2 decimals
    answer = [[0.00000, 0.11071, 0.00000],
              [5.00000, 0.09024, 0.00000],
              [10.00000, 0.06913, 0.00001],
              [15.00000, 0.04791, 0.00205],
              [20.00000, 0.02750, 0.08670],
              [25.00000, 0.00930, 0.58459],
              [30.00000, -0.00492, 0.78827],
              [35.00000, -0.01323, 0.50467],
              [40.00000, -0.01393, 0.51275],
              [45.00000, -0.00591, 0.79260],
              [50.00000, 0.01125, 0.62907],
              [55.00000, 0.03736, 0.11396],
              [60.00000, 0.07162, 0.00229],
              [65.00000, 0.11251, 0.00000],
              [70.00000, 0.15754, 0.00000],
              [75.00000, 0.20305, 0.00000],
              [80.00000, 0.24466, 0.00000],
              [85.00000, 0.27850, 0.00000],
              [90.00000, 0.30241, 0.00000],
              [95.00000, 0.31635, 0.00000],
              [100.00000, 0.32181, 0.00000],
              [105.00000, 0.32082, 0.00000],
              [110.00000, 0.31529, 0.00000],
              [115.00000, 0.30672, 0.00000],
              [120.00000, 0.29615, 0.00000],
              [125.00000, 0.28427, 0.00000],
              [130.00000, 0.27152, 0.00000],
              [135.00000, 0.25813, 0.00000],
              [140.00000, 0.24422, 0.00000],
              [145.00000, 0.22984, 0.00000],
              [150.00000, 0.21495, 0.00000],
              [155.00000, 0.19950, 0.00000],
              [160.00000, 0.18341, 0.00000],
              [165.00000, 0.16657, 0.00000],
              [170.00000, 0.14889, 0.00000],
              [175.00000, 0.13028, 0.00000]]

    data, _ = create_test_scattered()
    data_distances = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_euclidean)
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    output, output_text = pyssage.anisotropy.bearing_analysis(data_distances, distances, angles, 36)
    pyssage.graph.draw_bearing(numpy.array(output), figoutput=pyssage.graph.FigOutput(figshow=True))

    for line in output_text:
        print(line)

    for i, row in enumerate(answer):
        for j, ans in enumerate(row):
            assert round(output[i][j], 2) == round(ans, 2)


def test_bearing_analysis_rand():
    # permutation test prevents formal test answer; function used to look for stability
    data, _ = create_test_scattered()
    data_distances = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_euclidean)
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    output, output_text = pyssage.anisotropy.bearing_analysis(data_distances, distances, angles, 36, 100)
    pyssage.graph.draw_bearing(numpy.array(output), figoutput=pyssage.graph.FigOutput(figshow=True))

    for line in output_text:
        print(line)
