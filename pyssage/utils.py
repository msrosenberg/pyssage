import numpy


def flatten_half(x: numpy.ndarray) -> numpy.ndarray:
    output = []
    for i in range(len(x)):
        for j in range(i):
            output.append(x[i, j])
    return numpy.array(output)
