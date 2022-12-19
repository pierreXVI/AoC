import os

import requests

ROOT = 'data'
URL = "https://adventofcode.com/{0}/day/{1}/input"
SESSION = "53616c7465645f5f48548cc6057fe5f4163a1a169b2ffd2ef3e66137472d1dec" \
          "948a0e3012c6d2b257de216e9ebc37a2ade9d373e2a4590595c952a94a5bf6ea"


def _download_input(year, day):
    os.makedirs(os.path.join(ROOT, year), exist_ok=True)
    response = requests.get(URL.format(year, day), cookies={'session': SESSION})
    if not response:
        raise ConnectionError("Cannot get input from url \"{0}\"".format(response.url))
    path = os.path.join(ROOT, year, day)
    with open(path, 'w') as f:
        f.write(response.content.decode())
    return path


def get_input(year, day):
    year, day = str(year), str(day)
    path = os.path.join(ROOT, year, day)
    if not os.path.exists(path):
        return _download_input(year, day)
    else:
        return path
