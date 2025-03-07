import collections
import re

import numpy as np

import utils

YEAR = 2024


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        l1, l2 = np.sort(np.array([list(map(int, l.split())) for l in inp], dtype=int), axis=0).T
    count = collections.Counter(l2)

    print(sum(abs(l1 - l2)))
    print(sum([i * count.get(i, 0) for i in l1]))


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        reports = [list(map(int, l.split())) for l in inp]

    part1 = 0
    dpart2 = 0
    for report in reports:
        diff = np.diff(report)
        if (np.all(diff < 0) or np.all(diff > 0)) and np.all(np.abs(diff) < 4):
            part1 += 1
        else:
            for i in range(len(report)):
                diff2 = np.diff(np.delete(report, i))
                if (np.all(diff2 < 0) or np.all(diff2 > 0)) and np.all(np.abs(diff2) < 4):
                    dpart2 += 1
                    break

    print(part1)
    print(part1 + dpart2)


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        data = ''.join(inp.readlines())

    do = True
    part1 = part2 = 0
    for match in re.findall('mul\(\d{1,3},\d{1,3}\)|do\(\)|don\'t\(\)', data):
        if match == 'do()':
            do = True
        elif match == 'don\'t()':
            do = False
        else:
            prod = np.prod(list(map(int, match[4:-1].split(','))))
            part1 += prod
            if do:
                part2 += prod

    print(part1)
    print(part2)


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        data = np.array([list(l.strip()) for l in inp])

    target1 = list('XMAS')
    target2 = list('MAS')
    n, m = data.shape

    part1 = part2 = 0
    for i in range(n):
        for j in range(m):
            bloc = data[i:i + 4, j:j + 4]

            if bloc[:, 0].tolist() == target1:
                part1 += 1
            if bloc[0, :].tolist() == target1:
                part1 += 1
            if bloc[::-1, 0].tolist() == target1:
                part1 += 1
            if bloc[0, ::-1].tolist() == target1:
                part1 += 1
            if np.diag(bloc).tolist() == target1:
                part1 += 1
            if np.diag(data[max(0, i - 3):i + 1, max(0, j - 3):j + 1])[::-1].tolist() == target1:
                part1 += 1
            if np.diag(data[max(0, i - 3):i + 1, j:j + 4][::-1, :]).tolist() == target1:
                part1 += 1
            if np.diag(data[i:i + 4, max(0, j - 3):j + 1][:, ::-1]).tolist() == target1:
                part1 += 1

            bloc = data[i:i + 3, j:j + 3]
            if np.diag(bloc).tolist() == target2 and np.diag(bloc[::-1, :]).tolist() == target2:
                part2 += 1
            if np.diag(bloc).tolist() == target2 and np.diag(bloc.T[::-1, :]).tolist() == target2:
                part2 += 1
            if np.diag(bloc)[::-1].tolist() == target2 and np.diag(bloc[::-1, :]).tolist() == target2:
                part2 += 1
            if np.diag(bloc)[::-1].tolist() == target2 and np.diag(bloc.T[::-1, :]).tolist() == target2:
                part2 += 1

    print(part1 == 2454)
    print(part2 == 1858)


def day5():
    post = collections.defaultdict(list)
    with open(utils.get_input(YEAR, 5)) as inp:
        line = inp.readline().strip()
        while line:
            a, b = map(int, line.split('|'))
            post[a].append(b)
            line = inp.readline().strip()

        part1 = part2 = 0
        orders = [list(map(int, line.split(','))) for line in inp]
        for order in orders:
            for i in range(len(order) - 1):
                if order[i] in post[order[i + 1]]:
                    fixed = [order[0]]
                    for v in order[1:]:
                        j = len(fixed) - 1
                        while j >= 0 and fixed[j] in post[v]:
                            j -= 1
                        fixed.insert(j + 1, v)
                    part2 += fixed[len(fixed) // 2]
                    break
            else:
                part1 += order[len(order) // 2]
        print(part1)
        print(part2)


def day6():
    with open(utils.get_input(YEAR, 6)) as inp:
        grid = np.array([[0 if c == '.' else 1 if c == '#' else 2 for c in l.strip()] for l in inp])
        augmented_grid = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2))
        augmented_grid[1:-1, 1:-1] = grid
        grid = augmented_grid
    start = list(map(lambda x: int(x[0]), np.where(grid == 2)))

    def explore(oi=None, oj=None):
        dummy = grid.copy()
        if oi is not None and oj is not None:
            dummy[oi, oj] = 1

        i, j = start
        di, dj = -1, 0
        visited = set()
        while 1 <= i < dummy.shape[0] - 1 and 1 <= j < dummy.shape[1] - 1:
            dummy[i, j] = -1
            if (i, j, di, dj) in visited:
                break
            visited.add((i, j, di, dj))

            while dummy[i + di, j + dj] == 1:
                di, dj = dj, -di
            i += di
            j += dj

        return dummy, not (1 <= i < dummy.shape[0] - 1 and 1 <= j < dummy.shape[1] - 1)

    first_explore = explore()[0]
    print(np.count_nonzero(first_explore == -1))

    part2 = 0
    for oi, oj in zip(*np.where(first_explore == -1)):
        if [oi, oj] != start:
            if not explore(oi, oj)[1]:
                part2 += 1
    print(part2)


if __name__ == '__main__':
    utils.time_me(day6)()
