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
        while True:
            if (i, j, di, dj) in visited:
                break
            visited.add((i, j, di, dj))

            while 1 <= i < dummy.shape[0] - 1 and 1 <= j < dummy.shape[1] - 1 and dummy[i + di, j + dj] != 1:
                dummy[i, j] = -1
                i += di
                j += dj

            if not (1 <= i < dummy.shape[0] - 1 and 1 <= j < dummy.shape[1] - 1):
                break

            while dummy[i + di, j + dj] == 1:
                di, dj = dj, -di

        return dummy, not (1 <= i < dummy.shape[0] - 1 and 1 <= j < dummy.shape[1] - 1)

    first_explore = explore()[0]
    print(np.count_nonzero(first_explore == -1))

    part2 = 0
    for obstacle in zip(*np.where(first_explore == -1)):
        if obstacle != start:
            if not explore(*obstacle)[1]:
                part2 += 1
    print(part2)


def day7():
    def count_valid1(n0, n_current, available):
        if not available:
            return n0 == n_current
        return count_valid1(n0, n_current + available[0], available[1:]) \
            or count_valid1(n0, n_current * available[0], available[1:])

    def count_valid2(n0, n_current, available):
        if not available:
            return n0 == n_current
        return count_valid2(n0, n_current + available[0], available[1:]) \
            or count_valid2(n0, n_current * available[0], available[1:]) \
            or count_valid2(n0, int(str(n_current) + str(available[0])), available[1:])

    part1 = dpart2 = 0
    with open(utils.get_input(YEAR, 7)) as inp:
        for line in inp:
            target, values = line.strip().split(':')
            target = int(target)
            values = list(map(int, values.split()))
            if count_valid1(target, 0, values):
                part1 += target
            elif count_valid2(target, 0, values):
                dpart2 += target

    print(part1)
    print(part1 + dpart2)


def day8():
    with open(utils.get_input(YEAR, 8)) as inp:
        grid = np.array([[c for c in line.strip()] for line in inp])

    antinodes1 = set()
    antinodes2 = set()
    for label in np.unique(grid):
        if label == '.':
            continue

        locations = list(zip(*np.where(grid == label)))
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                dx = locations[i][0] - locations[j][0]
                dy = locations[i][1] - locations[j][1]

                x = locations[i][0]
                y = locations[i][1]
                if 0 <= x + dx < grid.shape[0] and 0 <= y + dy < grid.shape[1]:
                    antinodes1.add((x + dx, y + dy))
                while 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    antinodes2.add((x, y))
                    x += dx
                    y += dy

                x = locations[j][0]
                y = locations[j][1]
                if 0 <= x - dx < grid.shape[0] and 0 <= y - dy < grid.shape[1]:
                    antinodes1.add((x - dx, y - dy))
                while 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    antinodes2.add((x, y))
                    x -= dx
                    y -= dy
    print(len(antinodes1))
    print(len(antinodes2))


def day9():
    with open(utils.get_input(YEAR, 9)) as inp:
        data = list(map(int, list(inp.readline().strip())))
        data1 = data.copy()

    part1 = 0
    pos = 0
    j = len(data1) - 1
    for i in range(len(data1)):
        if i % 2 == 0:
            part1 += (i // 2) * sum(range(pos, pos + data1[i]))
            pos += data1[i]
        else:
            while data1[i] and j > i:
                if data1[j] <= data1[i]:
                    part1 += (j // 2) * sum(range(pos, pos + data1[j]))
                    pos += data1[j]
                    data1[i] -= data1[j]
                    j -= 2
                else:
                    part1 += (j // 2) * sum(range(pos, pos + data1[i]))
                    pos += data1[i]
                    data1[j] -= data1[i]
                    break
        if i >= j:
            break

    memory = []
    file_id = 0
    for i in range(len(data)):
        if i % 2 == 0:
            memory.append((file_id if i % 2 == 0 else None, data[i]))
            file_id += 1
        else:
            memory.append((None, data[i]))

    i = len(memory) - 1
    while i >= 0:
        if memory[i][0] is None:
            i -= 1
            continue

        for j in range(i):
            if memory[j][0] is None and memory[j][1] >= memory[i][1]:
                if memory[j][1] == memory[i][1]:
                    memory[j] = (memory[i][0], memory[i][1])
                else:
                    memory.insert(j, (memory[i][0], memory[i][1]))
                    i += 1
                    j += 1
                    memory[j] = (None, memory[j][1] - memory[i][1])

                memory[i] = (None, memory[i][1])
                break
        i -= 1

    part2 = 0
    pos = 0
    for file_id, size in memory:
        if file_id is not None:
            part2 += file_id * sum(range(pos, pos + size))
        pos += size

    print(part1)
    print(part2)


def day10():
    with open(utils.get_input(YEAR, 10)) as inp:
        grid = np.array([list(l.strip()) for l in inp], dtype=int)

    part1 = [[{(i, j)} if grid[i, j] == 9 else set() for j in range(grid.shape[1])] for i in range(grid.shape[0])]
    part2 = np.zeros_like(grid)
    part2[grid == 9] = 1
    for level in range(8, -1, -1):
        for i, j in zip(*np.where(grid == level)):
            if i > 0 and grid[i - 1, j] == level + 1:
                part1[i][j].update(part1[i - 1][j])
                part2[i, j] += part2[i - 1, j]
            if i < grid.shape[0] - 1 and grid[i + 1, j] == level + 1:
                part1[i][j].update(part1[i + 1][j])
                part2[i, j] += part2[i + 1, j]
            if j > 0 and grid[i, j - 1] == level + 1:
                part1[i][j].update(part1[i][j - 1])
                part2[i, j] += part2[i, j - 1]
            if j < grid.shape[1] - 1 and grid[i, j + 1] == level + 1:
                part1[i][j].update(part1[i][j + 1])
                part2[i, j] += part2[i, j + 1]

    print(sum([len(part1[i][j]) for i, j in zip(*np.where(grid == 0))]))
    print(sum(part2[grid == 0]))

def day11():
    with open(utils.get_input(YEAR, 11)) as inp:
        stones = inp.readline().strip().split()

    for _ in range(25):
        new_stones = []
        for s in stones:
            if s == '0':
                new_stones.append('1')
            elif len(s) % 2 == 0:
                new_stones.append(s[:len(s) // 2])
                new_stones.append(s[len(s) // 2:].lstrip('0'))
                if new_stones[-1] == '':
                    new_stones[-1] = '0'
            else:
                new_stones.append(str(int(s) * 2024))
        stones = new_stones
    print(len(stones))

if __name__ == '__main__':
    utils.time_me(day11)()
