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


def test_coords():
    """
    Creates a 355 x 2 array of coordinate locations, based on the 355 western European cancer registration sites
    """
    return numpy.array([[4.33, 50.83],
                        [4.5, 51.17],
                        [4.46, 50.81],
                        [3, 51],
                        [3.67, 51],
                        [5.5, 51],
                        [4.33, 50.6],
                        [4, 50.5],
                        [5.67, 50.5],
                        [5.5, 50],
                        [4.83, 50.33],
                        [9.75, 49],
                        [8.4, 49.02],
                        [7.85, 47.98],
                        [9.05, 48.53],
                        [11.83, 47.83],
                        [12.5, 48.58],
                        [12.17, 49.5],
                        [11.5, 50.02],
                        [10.67, 49.33],
                        [10, 50],
                        [10.5, 48.33],
                        [13.4, 52.52],
                        [8.83, 53.08],
                        [10, 53.58],
                        [8.65, 49.87],
                        [9.5, 51.32],
                        [10.53, 52.27],
                        [9.72, 52.37],
                        [10.25, 53],
                        [8.03, 52.83],
                        [7, 51.25],
                        [6.95, 50.93],
                        [7.63, 51.97],
                        [8.88, 51.93],
                        [8.08, 51.38],
                        [7.6, 50.35],
                        [6.63, 49.75],
                        [8.67, 49.67],
                        [7, 49.33],
                        [10.5, 54],
                        [12.58, 55.67],
                        [12.25, 55.92],
                        [12.17, 55.58],
                        [11.5, 55.5],
                        [11.83, 55],
                        [15, 55.17],
                        [10.5, 55.33],
                        [9, 55],
                        [8.75, 55.58],
                        [9.33, 55.75],
                        [8.75, 56.17],
                        [10.25, 56.17],
                        [9.5, 56.5],
                        [10, 57.25],
                        [7.67, 48.58],
                        [7.22, 47.88],
                        [0.83, 45.17],
                        [-0.58, 44.75],
                        [-1, 44.33],
                        [0.33, 44.33],
                        [-0.83, 43.25],
                        [3, 46.5],
                        [2.67, 45.08],
                        [4, 45.08],
                        [3, 45.67],
                        [-0.5, 49.17],
                        [-1.17, 49],
                        [0.08, 48.67],
                        [4.83, 47.5],
                        [3.5, 47.08],
                        [4.5, 46.67],
                        [3.75, 47.92],
                        [-2.67, 48.42],
                        [-4, 48.33],
                        [-1.5, 48.17],
                        [-2.83, 47.92],
                        [2.5, 47],
                        [1.5, 48.5],
                        [1.67, 46.83],
                        [0.75, 47.25],
                        [1.5, 47.5],
                        [2.33, 47.92],
                        [4.67, 49.67],
                        [4.08, 48.25],
                        [4.17, 48.92],
                        [5.17, 48.08],
                        [9, 42],
                        [6.42, 47.17],
                        [5.83, 46.83],
                        [6.17, 47.67],
                        [6.92, 47.63],
                        [1, 49.17],
                        [1, 49.75],
                        [2.33, 48.87],
                        [3, 48.5],
                        [1.83, 48.83],
                        [2.33, 48.6],
                        [2.18, 48.83],
                        [2.5, 48.92],
                        [2.48, 48.78],
                        [2.17, 49.17],
                        [2.5, 43.08],
                        [4, 44],
                        [3.5, 43.67],
                        [3.5, 44.5],
                        [2.5, 42.75],
                        [1.83, 45.25],
                        [2, 46.08],
                        [1.17, 45.83],
                        [6.17, 48.58],
                        [5.5, 49],
                        [6.5, 49],
                        [6.33, 48.17],
                        [1.5, 43],
                        [2.5, 44.25],
                        [1.5, 43.42],
                        [0.5, 43.67],
                        [1.67, 44.58],
                        [0.17, 43],
                        [2, 43.83],
                        [1.17, 44],
                        [3.67, 50.33],
                        [2.33, 50.5],
                        [-1.83, 47.25],
                        [-0.5, 47.42],
                        [-0.67, 48.08],
                        [0.08, 48],
                        [-1.33, 46.67],
                        [3.5, 49.5],
                        [2.5, 49.5],
                        [2.5, 49.92],
                        [0.17, 45.67],
                        [-0.75, 45.5],
                        [-0.33, 46.5],
                        [0.5, 46.58],
                        [6.15, 44.08],
                        [6.5, 44.67],
                        [7.17, 44],
                        [5, 43.5],
                        [6.33, 43.5],
                        [5.17, 44],
                        [5.33, 46.17],
                        [4.33, 44.67],
                        [5.17, 44.58],
                        [5.83, 45.17],
                        [4, 45.5],
                        [4.5, 46],
                        [6.42, 45.5],
                        [6.33, 46],
                        [0.08, 52.33],
                        [1, 52.58],
                        [1.33, 52.17],
                        [-1.58, 53.17],
                        [-1, 52.63],
                        [-0.37, 52.92],
                        [-0.83, 52.33],
                        [-1, 53],
                        [-1.58, 54.92],
                        [-1.25, 54.58],
                        [-2.75, 54.58],
                        [-1.75, 54.75],
                        [-2.08, 55.25],
                        [-2.17, 53.58],
                        [-3, 53.5],
                        [-2.5, 53.25],
                        [-2.67, 53.92],
                        [-0.17, 51.5],
                        [-0.5, 52.08],
                        [-1.17, 51.5],
                        [-0.8, 51.75],
                        [0.25, 50.92],
                        [0.67, 51.8],
                        [-1.17, 51],
                        [-0.17, 51.83],
                        [-1.25, 50.67],
                        [0.67, 51.25],
                        [-1.33, 51.83],
                        [-0.33, 51.17],
                        [-0.5, 51],
                        [-2.5, 51.5],
                        [-4.5, 50.5],
                        [-3.83, 50.83],
                        [-2.17, 50.83],
                        [-1.92, 51.83],
                        [-3.17, 51.17],
                        [-2, 51.5],
                        [-2, 52.5],
                        [-2.58, 52.17],
                        [-2.83, 52.67],
                        [-2, 52.92],
                        [-1.58, 52.17],
                        [-1.42, 53.5],
                        [-1.5, 53.67],
                        [-0.67, 53.92],
                        [-1.5, 54.25],
                        [-3.25, 53.17],
                        [-4, 52.08],
                        [-2.92, 51.75],
                        [-3.83, 52.83],
                        [-3.58, 51.58],
                        [-3.33, 52.42],
                        [-3.25, 51.5],
                        [-3.92, 51.67],
                        [-5, 57.4],
                        [-2.58, 57.42],
                        [-3.67, 56.5],
                        [-3.25, 56.08],
                        [-3.5, 55.92],
                        [-3, 55.58],
                        [-4.17, 56.25],
                        [-5.25, 56],
                        [-3.58, 55.17],
                        [-3, 59],
                        [-1.5, 60.5],
                        [-7.17, 57.67],
                        [-5.75, 54.5],
                        [-6.25, 54.83],
                        [-6.75, 54.5],
                        [-7.5, 54.58],
                        [13.37, 42.37],
                        [13.7, 42.65],
                        [13.95, 42.33],
                        [14.17, 42.35],
                        [15.8, 40.63],
                        [16.6, 40.67],
                        [16.42, 39.47],
                        [16.6, 38.9],
                        [16, 38.17],
                        [14.33, 41.07],
                        [14.75, 41.13],
                        [14.25, 40.83],
                        [14.78, 40.9],
                        [14.78, 40.68],
                        [9.67, 45.02],
                        [10.33, 44.8],
                        [10.6, 44.72],
                        [10.92, 44.67],
                        [11.33, 44.48],
                        [11.58, 44.83],
                        [11.98, 44.42],
                        [12.05, 44.22],
                        [13.23, 46.05],
                        [13.63, 45.95],
                        [13.77, 45.67],
                        [12.65, 45.95],
                        [12.1, 42.42],
                        [12.85, 42.4],
                        [12.48, 41.9],
                        [13.1, 41.45],
                        [13.32, 41.63],
                        [8.05, 43.88],
                        [8.5, 44.28],
                        [8.95, 44.42],
                        [9.67, 44.22],
                        [8.83, 45.8],
                        [9.08, 45.78],
                        [9.87, 46.17],
                        [9.2, 45.47],
                        [9.72, 45.68],
                        [10.25, 45.55],
                        [9.17, 45.17],
                        [10.03, 45.12],
                        [10.8, 45.15],
                        [12.63, 43.67],
                        [13.17, 43.55],
                        [13.17, 43.2],
                        [13.57, 42.85],
                        [14.65, 41.57],
                        [14.23, 41.6],
                        [7.67, 45.05],
                        [8.42, 45.32],
                        [8.63, 45.47],
                        [7.53, 44.38],
                        [8.2, 44.9],
                        [8.62, 44.9],
                        [15.57, 41.45],
                        [16.85, 41.13],
                        [17.23, 40.47],
                        [17.93, 40.63],
                        [18.18, 40.38],
                        [8.57, 40.72],
                        [9.33, 40.32],
                        [9.12, 39.22],
                        [8.6, 39.9],
                        [12.48, 38.02],
                        [13.37, 38.12],
                        [14.87, 38.05],
                        [13.57, 37.32],
                        [14.07, 37.48],
                        [14.43, 37.58],
                        [15.1, 37.5],
                        [14.73, 36.92],
                        [15.3, 37.07],
                        [10.05, 44.25],
                        [10.48, 43.83],
                        [10.9, 43.92],
                        [11.25, 43.77],
                        [10.58, 43.23],
                        [10.38, 43.72],
                        [11.88, 43.42],
                        [11.35, 43.32],
                        [11.25, 42.83],
                        [11.37, 46.52],
                        [11.13, 46.07],
                        [12.37, 43.13],
                        [12.62, 42.57],
                        [7.33, 45.73],
                        [11, 45.45],
                        [11.55, 45.55],
                        [12.22, 46.15],
                        [12.25, 45.67],
                        [12.35, 45.45],
                        [11.88, 45.42],
                        [11.78, 45.07],
                        [-9, 53.33],
                        [-8.33, 54.33],
                        [-9.5, 53.83],
                        [-8.5, 53.67],
                        [-8.67, 54.17],
                        [-7, 52.83],
                        [-6.25, 53.33],
                        [-6.75, 53.25],
                        [-7.33, 52.67],
                        [-7.5, 53],
                        [-7.67, 53.67],
                        [-6.5, 53.92],
                        [-6.67, 53.58],
                        [-7.5, 53.33],
                        [-7.5, 53.5],
                        [-6.67, 52.33],
                        [-6.5, 53],
                        [-9, 52.83],
                        [-8.5, 52],
                        [-9.5, 52.17],
                        [-9, 52.5],
                        [-8.33, 52.67],
                        [-7.67, 52.17],
                        [-7.5, 53.92],
                        [-8, 54.83],
                        [-7, 54.17],
                        [6.45, 49.68],
                        [6.17, 49.88],
                        [6.08, 49.75],
                        [6.55, 53.22],
                        [5.75, 53.05],
                        [6.5, 52.75],
                        [6.5, 52.42],
                        [5.83, 52.17],
                        [5.13, 52.08],
                        [4.83, 52.67],
                        [4.5, 52],
                        [3.75, 51.45],
                        [5, 51.5],
                        [5.83, 51.23]])


def test_surface():
    """
    Reads a 100 x 100 surface (checkerboard of 1's and 0's in 10 x 10 blocks) from a file
    """
    surface = []
    with open("data/test_surface.txt", "r") as infile:
        for line in infile:
            data = line.strip().split()
            row = []
            for d in data:
                row.append(eval(d))
            surface.append(row)
    return numpy.array(surface)


def load_answer(filename: str) -> numpy.ndarray:
    with open(filename, "r") as infile:
        data = []
        for line in infile:
            if line.strip() != "":
                dat = line.strip().split("\t")
                for i in range(len(dat)):
                    dat[i] = eval(dat[i])  # convert to numbers
            data.append(dat)
    return numpy.array(data)
