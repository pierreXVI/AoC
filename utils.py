import os

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
