import numpy as np

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


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        table1 = np.array([['1', '2', '3'],
                           ['4', '5', '6'],
                           ['7', '8', '9']])
        table2 = np.array([[' ', ' ', '1', ' ', ' '],
                           [' ', '2', '3', '4', ' '],
                           ['5', '6', '7', '8', '9'],
                           [' ', 'A', 'B', 'C', ' '],
                           [' ', ' ', 'D', ' ', ' ']])
        x1, y1 = 1, 1
        x2, y2 = 2, 0
        code1 = code2 = ''
        for line in inp:
            for move in line[:-1]:
                if move == 'U':
                    x1 = max(0, x1 - 1)
                    x2 = x2 - 1 if x2 > 0 and table2[x2 - 1, y2] != ' ' else x2
                elif move == 'D':
                    x1 = min(2, x1 + 1)
                    x2 = x2 + 1 if x2 < 4 and table2[x2 + 1, y2] != ' ' else x2
                elif move == 'L':
                    y1 = max(0, y1 - 1)
                    y2 = y2 - 1 if y2 > 0 and table2[x2, y2 - 1] != ' ' else y2
                elif move == 'R':
                    y1 = min(2, y1 + 1)
                    y2 = y2 + 1 if y2 < 4 and table2[x2, y2 + 1] != ' ' else y2
            code1 += table1[x1, y1]
            code2 += table2[x2, y2]
        print(code1)
        print(code2)


if __name__ == '__main__':
    day2()
