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

def objects_differ_scales():
    size = 128
    image = np.zeros((size, size))

    off = size//10
    off2 = size//20

    x_bound_bigs = [size//5, 2*size//5+off, 3*size//5, 4*size//5+off, size//5]
    y_bound_bigs = [i*size//6-16 if i%2==1 else i*size//6-4 for i in range(1,7)]

    for j in range(3):
        image[y_bound_bigs[j*2]: y_bound_bigs[1+j*2], x_bound_bigs[0]: x_bound_bigs[1]] = 0.25 + 0 / 4
        image[y_bound_bigs[j*2]: y_bound_bigs[1+j*2], x_bound_bigs[2]: x_bound_bigs[3]] = 0.25 + 0 / 4

    # second detail
    some_dots = [(i * 5, i * 5 + 2) for i in range(0, 7)]
    some_dots2 = [(i * 9, i * 9 + 5) for i in range(0, 4)]
    some_dots3 = [(i * 3, i * 3 + 1) for i in range(0, 12)]

    for dot0 in some_dots[:-1]:
        for dot1 in some_dots:
            image[y_bound_bigs[0]+dot0[0]+2: y_bound_bigs[0]+dot0[1]+2,
                  x_bound_bigs[0]+dot1[0]+2: x_bound_bigs[0]+dot1[1]+2] = np.random.choice([1])*1

    for dot0 in some_dots[:-1]:
        for dot1 in some_dots:
            image[y_bound_bigs[0]+dot0[0]+2: y_bound_bigs[0]+dot0[1]+2,
                  x_bound_bigs[2]+dot1[0]+2: x_bound_bigs[2]+dot1[1]+2] = np.random.choice([1])*1


    for dot0 in some_dots2[:-1]:
        for dot1 in some_dots2:
            image[y_bound_bigs[2]+dot0[0]+2: y_bound_bigs[2]+dot0[1]+2,
                  x_bound_bigs[0]+dot1[0]+2: x_bound_bigs[0]+dot1[1]+2] = np.random.choice([1])*1

    for dot0 in some_dots3[:-2]:
        for dot1 in some_dots3:
            image[y_bound_bigs[2]+dot0[0]+2: y_bound_bigs[2]+dot0[1]+2,
                  x_bound_bigs[2]+dot1[0]+2: x_bound_bigs[2]+dot1[1]+2] = np.random.choice([1])*1

    for dot0 in some_dots3[:-1]:
        for dot1 in some_dots3:
            image[y_bound_bigs[4]+dot0[0]+2: y_bound_bigs[4]+dot0[1]+2,
                  x_bound_bigs[2]+dot1[0]+2: x_bound_bigs[2]+dot1[1]+2] = np.random.choice([1])*1


    return image
