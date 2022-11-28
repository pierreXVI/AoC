import re

import numpy as np

import utils

YEAR = 2021


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        res1 = res2 = 0
        foo = [0, 0, 0]
        i = 0
        for line in inp:
            bar = int(line)
            if i > 0:
                if bar > foo[-1]:
                    res1 += 1
            if i > 2:
                if bar > foo[0]:
                    res2 += 1
            foo = [foo[1], foo[2], bar]
            i += 1
        print(res1)
        print(res2)


def day2():
    x = z1 = z2 = aim = 0
    with open(utils.get_input(YEAR, 2)) as inp:
        for line in inp:
            move, d = line.split()
            if move == 'forward':
                z2 += int(d) * aim
                x += int(d)
            elif move == 'down':
                aim += int(d)
                z1 += int(d)
            elif move == 'up':
                aim -= int(d)
                z1 -= int(d)
        print(x * z1)
        print(x * z2)


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        data = np.array([[2 * int(d) - 1 for d in line[:-1]] for line in inp], dtype=int)

    data1, data2 = data.copy(), data.copy()
    for i in range(data.shape[1]):
        if len(data1) > 1:
            flag = np.sum(data1[:, i]) >= 0
            data1 = data1[np.equal(data1[:, i] > 0, flag)]
        if len(data2) > 1:
            flag = np.sum(data2[:, i]) >= 0
            data2 = data2[np.not_equal(data2[:, i] > 0, flag)]
    foo = int(b''.join(b'0' if d < 0 else b'1' for d in np.sum(data, axis=0)), 2)

    print(foo * ((1 << data.shape[1]) - 1 - foo))
    print(int(b''.join(b'0' if d < 0 else b'1' for d in data1[0]), 2)
          * int(b''.join(b'0' if d < 0 else b'1' for d in data2[0]), 2))


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        draft = [int(d) for d in inp.readline()[:-1].split(',')]

        ranks, scores = [], []
        while True:
            inp.readline()
            grid = np.array([[int(d) for d in inp.readline()[:-1].split()] for _ in range(5)], dtype=int)
            if grid.shape != (5, 5):
                break
            for i, d in enumerate(draft):
                grid[grid == d] = -1
                win = np.any([(grid[k] == -1).all() or (grid[:, k] == -1).all() for k in range(5)])
                if win:
                    ranks.append(i)
                    scores.append(d * np.sum(grid[grid != -1]))
                    break
        print(scores[np.argmin(ranks)])
        print(scores[np.argmax(ranks)])


def day5():
    grid1, grid2 = np.zeros((1, 1), dtype=int), np.zeros((1, 1), dtype=int)
    with open(utils.get_input(YEAR, 5)) as inp:
        for line in inp:
            x1, y1, x2, y2 = [int(d) for d in re.match(r'(\d+),(\d+) -> (\d+),(\d+)', line).groups()]

            if max(x1, x2) >= grid1.shape[0] or max(y1, y2) >= grid1.shape[1]:
                new_grid1 = np.zeros((max(x1 + 1, x2 + 1, grid1.shape[0]), max(y1 + 1, y2 + 1, grid1.shape[1])))
                new_grid2 = np.zeros((max(x1 + 1, x2 + 1, grid1.shape[0]), max(y1 + 1, y2 + 1, grid1.shape[1])))
                new_grid1[:grid1.shape[0], :grid1.shape[1]] = grid1
                new_grid2[:grid1.shape[0], :grid1.shape[1]] = grid2
                grid1 = new_grid1
                grid2 = new_grid2

            if x1 == x2:
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    grid1[x1, y] += 1
                    grid2[x1, y] += 1
            elif y1 == y2:
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    grid1[x, y1] += 1
                    grid2[x, y1] += 1
            else:
                x, y = x1, y1
                for t in range(max(x1, x2) - min(x1, x2) + 1):
                    grid2[x, y] += 1
                    x += 1 if x2 > x1 else -1
                    y += 1 if y2 > y1 else -1

    grid1[grid1 == 1] = 0
    grid2[grid2 == 1] = 0
    print(np.count_nonzero(grid1))
    print(np.count_nonzero(grid2))


if __name__ == '__main__':
    day5()
