import collections

import numpy as np

import utils
import re

np.set_printoptions(linewidth=300)

YEAR = 2018


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        data = [int(line) for line in inp]

        count = 0
        val = None
        reached = set()
        for d in data:
            count += d
            if val is None and count in reached:
                val = count
            reached.add(count)
        print(count)

        while val is None:
            for d in data:
                count += d
                if val is None and count in reached:
                    val = count
                reached.add(count)
        print(val)


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        n2 = n3 = 0
        letters = []
        for line in inp:
            letters.append(np.array([ord(d) for d in line[:-1]]))
            count = collections.Counter(line[:-1])
            i2 = i3 = False
            for c in count:
                if not i2 and count[c] == 2:
                    i2 = True
                elif not i3 and count[c] == 3:
                    i3 = True
            if i2:
                n2 += 1
            if i3:
                n3 += 1
        print(n2 * n3)

        for i in range(len(letters)):
            for j in range(i + 1, len(letters)):
                foo = letters[i] - letters[j]
                if len(foo[foo != 0]) == 1:
                    print(''.join([chr(d) for d in letters[i][foo == 0]]))


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:

        grid = np.empty((0, 0), dtype=list)
        claims = set()
        for line in inp:
            line = line.split()
            x, y = [int(d) for d in line[2][:-1].split(',')]
            w, h = [int(d) for d in line[3].split('x')]
            if grid.shape[0] <= x + w or grid.shape[1] <= y + h:
                new_grid = np.empty((max(grid.shape[0], x + w), max(grid.shape[1], y + h)), dtype=list)
                for i in range(new_grid.shape[0]):
                    for j in range(new_grid.shape[1]):
                        new_grid[i, j] = []
                new_grid[:grid.shape[0], :grid.shape[1]] = grid
                grid = new_grid
            for i in range(x, x + w):
                for j in range(y, y + h):
                    grid[i, j].append(line[0][1:])
            claims.add(line[0][1:])
        count = 0
        overlap = set()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if len(grid[i, j]) > 1:
                    count += 1
                    for c in grid[i, j]:
                        overlap.add(c)
        print(count)
        print(claims.difference(overlap).pop())


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        data = sorted([line for line in inp])

        asleep = collections.defaultdict(lambda: np.zeros((60,), dtype=int))
        for line in data:
            if '#' in line:
                guard = int(re.search(r'Guard #(\d+) begins shift', line).groups()[0])
            elif 'falls asleep' in line:
                start = int(re.match(r'\[\d{4}-\d{2}-\d{2} \d{2}:(\d{2})]', line).groups()[0])
            else:
                m = int(re.match(r'\[\d{4}-\d{2}-\d{2} \d{2}:(\d{2})]', line).groups()[0])
                asleep[guard][start: m] += 1

        best1 = sleepy1 = best2 = sleepy2 = 0
        for guard in asleep:
            score = np.sum(asleep[guard])
            if score > best1:
                best1 = score
                sleepy1 = guard

            score = np.max(asleep[guard])
            if score > best2:
                best2 = score
                sleepy2 = guard

        print(sleepy1 * np.argmax(asleep[sleepy1]))
        print(sleepy2 * np.argmax(asleep[sleepy2]))


if __name__ == '__main__':
    day4()
