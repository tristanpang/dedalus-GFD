import numpy as np

def square_to_circle(x, y):
    '''Map square to circle via elliptical
    https://arxiv.org/ftp/arxiv/papers/1509/1509.06344.pdf'''
    u = x * np.sqrt(1 - y**2/2)
    v = y * np.sqrt(1 - x**2/2)
    return u, v


def circle_to_square(u, v):
    '''Map circle to square via elliptical
    https://arxiv.org/ftp/arxiv/papers/1509/1509.06344.pdf'''
    x = 1/2 * np.sqrt(2 + u**2 - v**2 + 2*np.sqrt(2) * u) \
        - 1/2 * np.sqrt(2 + u**2 - v**2 - 2*np.sqrt(2) * u)
    y = 1/2 * np.sqrt(2 - u**2 + v**2 + 2*np.sqrt(2) * v) \
        - 1/2 * np.sqrt(2 - u**2 + v**2 - 2*np.sqrt(2) * v)
    return x, y