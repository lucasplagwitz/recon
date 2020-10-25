import numpy as np

def two_smooth_squares(size: int=256, small_size: int=200):
    # build image
    image = np.reshape(np.array([(x / size) for x in range(size)] * size), (size, size))
    image[28:small_size + 28, 28:small_size + 28] = \
        np.reshape(np.array([(1 - x / small_size) for x in range(small_size)] * small_size), (small_size, small_size))
    image /= np.max(image)

    assert np.all([0 <= np.min(image), np.max(image) == 1])

    return image


def local_tss():
    image = two_smooth_squares(256, 200)

    some_dots = [(i * 10, i * 10 + 5) for i in range(5, 15)]
    some_dots += [(i * 10, i * 10 + 2) for i in range(15, 21)]
    some_dots += [(i * 10 + 3, i * 10 + 5) for i in range(15, 21)]

    for dot0 in some_dots:
        for dot1 in some_dots:
            image[dot0[0]: dot0[1], dot1[0]: dot1[1]] = 1

    image = image / np.max(image)

    assert np.all([0 <= np.min(image), np.max(image) == 1])

    return image