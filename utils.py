import os
import time

import numpy as np
import requests

np.set_printoptions(linewidth=300)

ROOT = 'data'
URL = "https://adventofcode.com/{0}/day/{1}/input"
SESSION = "53616c7465645f5f654e0ca687caca1ce71c5242d9eafc611d1e3e4f41bac39c" \
          "214d989df66360c8eef709a4ee6f19df47ae6567e02e896c4cab698fa04e0148"


def _download_input(year, day, path):
    os.makedirs(os.path.join(ROOT, year), exist_ok=True)
    response = requests.get(URL.format(year, day), cookies={'session': SESSION})
    if not response:
        raise ConnectionError("Cannot get input from url \"{0}\"".format(response.url))
    with open(path, 'w') as f:
        f.write(response.content.decode())


def get_input(year, day):
    year, day = str(year), str(day)
    path = os.path.join(ROOT, year, day)
    if not os.path.exists(path):
        _download_input(year, day, path)
    return path


def time_me(f):
    def timed_f(*args, **kwargs):
        start_time = time.time()
        out = f(*args, **kwargs)
        print("\nRan in {0}s".format(time.time() - start_time))
        return out

    return timed_f


def print_boolean2d(array, sep=''):
    print('\n'.join([sep.join(['#' if b else '.' for b in line]) for line in array]))


def resize(array, shape_variation):
    shape_1 = np.array(array.shape)
    shape_variation = np.array(shape_variation)
    if (shape_variation > 0).any():
        new_array = np.zeros(shape_1 + np.maximum(0, shape_variation), dtype=array.dtype)
        new_array[tuple(slice(s) for s in shape_1)] = array
        array = new_array
    if (shape_variation < 0).any():
        new_array = np.zeros(shape_1 + np.maximum(0, -shape_variation), dtype=array.dtype)
        new_array[tuple(slice(-s, None) for s in shape_1)] = array
        array = new_array
    return array, np.maximum(0, -shape_variation)
