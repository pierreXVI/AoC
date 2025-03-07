import os
import time

import numpy as np
import requests

np.set_printoptions(linewidth=300)

ROOT = 'data'
URL = "https://adventofcode.com/{0}/day/{1}/input"
SESSION = "53616c7465645f5f00dc0c85d6df97f52142daf781811c0f0381372acc18d0095fb0b0ee1caeb6af1b4e5aa37505d6aaccf0cf5352c69353d3ee3ef7870e8e1a"


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
