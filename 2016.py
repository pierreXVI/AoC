import utils

YEAR = 2016


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        loc = 0 + 0j
        alpha = 0 + 1j
        visited = set()
        twice = None
        for data in inp.readline().split(', '):
            if data[0] == 'L':
                alpha *= 1j
            else:
                alpha *= -1j
            val = int(data[1:])
            for _ in range(val):
                loc += alpha
                if twice is None and loc in visited:
                    twice = loc
                visited.add(loc)

        print(int(abs(loc.real) + abs(loc.imag)))
        print(int(abs(twice.real) + abs(twice.imag)))


if __name__ == '__main__':
    day1()
