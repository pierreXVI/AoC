import functools
import heapq
import json
import re

import numpy as np

import utils

YEAR = 2022
np.set_printoptions(linewidth=300)


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        count = 0
        max_count = [0, 0, 0]
        for line in inp:
            if line == '\n':
                max_count = sorted([count, *max_count])[1:]
                count = 0
            else:
                count += int(line)
        max_count = sorted([count, *max_count])[1:]

        print(max_count[-1])
        print(sum(max_count))


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        score1 = score2 = 0
        value1 = {'X': 1, 'Y': 2, 'Z': 3}
        value2 = {'X': 0, 'Y': 3, 'Z': 6}
        order = {'X': {'A': 3, 'B': 0, 'C': 6}, 'Y': {'A': 6, 'B': 3, 'C': 0}, 'Z': {'A': 0, 'B': 6, 'C': 3}}
        reverse = {'X': {'A': 3, 'B': 1, 'C': 2}, 'Y': {'A': 1, 'B': 2, 'C': 3}, 'Z': {'A': 2, 'B': 3, 'C': 1}}
        for line in inp:
            you, me = line[:-1].split()
            score1 += value1[me] + order[me][you]
            score2 += value2[me] + reverse[me][you]
        print(score1)
        print(score2)


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        count1 = count2 = 0
        group = []
        for line in inp:
            n = len(line) // 2
            common = set(line[:n]).intersection(line[n:-1]).pop()
            if common.islower():
                count1 += ord(common) - ord('a') + 1
            else:
                count1 += ord(common) - ord('A') + 27
            group.append(set(line[:-1]))
            if len(group) == 3:
                common = group[0].intersection(group[1]).intersection(group[2]).pop()
                if common.islower():
                    count2 += ord(common) - ord('a') + 1
                else:
                    count2 += ord(common) - ord('A') + 27
                group = []
        print(count1)
        print(count2)


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        count1 = count2 = 0
        for line in inp:
            min1, max1, min2, max2 = [int(d) for ranges in line.split(',') for d in ranges.split('-')]
            if (min1 <= min2 and max2 <= max1) or (min2 <= min1 and max1 <= max2):
                count1 += 1
            if (min1 <= max2 and min2 <= max1) or (min2 <= max1 and min1 <= max2):
                count2 += 1
        print(count1)
        print(count2)


def day5():
    with open(utils.get_input(YEAR, 5)) as inp:
        data = []
        for line in inp:
            if line == '\n':
                break
            data.append(line)

        loc = {}
        line = data.pop()
        for i in range(len(line) - 1):
            if line[i] != ' ':
                loc[int(line[i])] = i

        piles1 = {i: [] for i in loc}
        piles2 = {i: [] for i in loc}
        for line in data[::-1]:
            for i in loc:
                if line[loc[i]] != ' ':
                    piles1[i].append(line[loc[i]])
                    piles2[i].append(line[loc[i]])

        for line in inp:
            line = line.split()
            n = int(line[1])
            i = int(line[3])
            j = int(line[5])
            for _ in range(n):
                piles1[j].append(piles1[i].pop())
            piles2[j] = piles2[j] + piles2[i][-n:]
            piles2[i] = piles2[i][:-n]
        print(''.join([piles1[i].pop() for i in piles1]))
        print(''.join([piles2[i].pop() for i in piles2]))


def day6():
    with open(utils.get_input(YEAR, 6)) as inp:
        datastream = inp.readline().strip()

        for i in range(4, len(datastream)):
            if len(set(datastream[i - 4: i])) == 4:
                print(i)
                break

        for i in range(14, len(datastream)):
            if len(set(datastream[i - 14: i])) == 14:
                print(i)
                break


def day7():
    with open(utils.get_input(YEAR, 7)) as inp:
        data = {'/': {}}
        depth = ['/']
        current = data['/']
        line = True
        while line:
            line = inp.readline()
            if line.startswith('$ cd'):
                foo = line.split()[2]
                if foo == '/':
                    depth = ['/']
                    current = data['/']
                elif foo == '..':
                    depth = depth[:-1]
                    current = data
                    for d in depth:
                        current = current[d]
                else:
                    depth.append(foo)
                    current = current[foo]
            else:
                new_line = True
                while new_line:
                    last = inp.tell()
                    new_line = inp.readline()
                    if new_line.startswith('$'):
                        inp.seek(last)
                        break
                    if new_line.startswith('dir'):
                        current[new_line.split()[1]] = {}
                    elif new_line:
                        foo, bar = new_line.split()
                        current[bar] = int(foo)

    all_sizes = []

    def recursive_shit(structure):
        size = score = 0
        for sub in structure:
            if isinstance(structure[sub], dict):
                sub_size, sub_score = recursive_shit(structure[sub])
                score += sub_score
                if sub_size <= 100000:
                    score += sub_size
                size += sub_size
            else:
                size += structure[sub]
        all_sizes.append(size)
        return size, score

    result = recursive_shit(data)
    all_sizes = np.array(all_sizes)
    print(result[1])
    print(min(all_sizes[all_sizes >= result[0] - 40000000]))


def day8():
    with open(utils.get_input(YEAR, 8)) as inp:
        grid = np.array([[int(d) for d in line[:-1]] for line in inp])

    visible = np.zeros_like(grid, dtype=bool)
    visible[0] = visible[-1] = visible[:, 0] = visible[:, -1] = True
    for i in range(1, visible.shape[0] - 1):
        for j in range(1, visible.shape[1] - 1):
            if grid[i, j] > max(grid[i, :j]) or grid[i, j] > max(grid[i, j + 1:]):
                visible[i, j] = True
    for j in range(1, visible.shape[1] - 1):
        for i in range(1, visible.shape[0] - 1):
            if grid[i, j] > max(grid[:i, j]) or grid[i, j] > max(grid[i + 1:, j]):
                visible[i, j] = True
    print(np.sum(visible))

    score = np.ones_like(grid)
    for i in range(visible.shape[0]):
        for j in range(visible.shape[1]):
            k = i
            for k in range(i + 1, visible.shape[0]):
                if grid[i, j] <= grid[k, j]:
                    break
            score[i, j] *= k - i
            k = j
            for k in range(j + 1, visible.shape[1]):
                if grid[i, j] <= grid[i, k]:
                    break
            score[i, j] *= k - j
            k = 0
            for k in range(1, i + 1):
                if grid[i, j] <= grid[i - k, j]:
                    break
            score[i, j] *= k
            k = 0
            for k in range(1, j + 1):
                if grid[i, j] <= grid[i, j - k]:
                    break
            score[i, j] *= k
    print(np.max(score))


def day9():
    def move(master, slave):
        dist = master - slave
        if abs(master - slave) >= 2:
            slave += dist.real / max(1, abs(dist.real)) + 1j * (dist.imag / max(1, abs(dist.imag)))
        return slave

    with open(utils.get_input(YEAR, 9)) as inp:
        table = {'R': 1, 'L': -1, 'U': 1j, 'D': -1j}
        rope1 = 2 * [0 + 0j]
        rope2 = 10 * [0 + 0j]
        visited1 = {rope1[-1]}
        visited2 = {rope2[-1]}
        for line in inp:
            where, n = line.split()
            for _ in range(int(n)):
                rope1[0] += table[where]
                rope2[0] += table[where]
                for i in range(1, len(rope1)):
                    rope1[i] = move(rope1[i - 1], rope1[i])
                for i in range(1, len(rope2)):
                    rope2[i] = move(rope2[i - 1], rope2[i])
                visited1.add(rope1[-1])
                visited2.add(rope2[-1])

        print(len(visited1))
        print(len(visited2))


def day10():
    with open(utils.get_input(YEAR, 10)) as inp:
        x = 1
        n = 1
        score = 0
        display = np.zeros((6, 40), dtype=bool)
        for line in inp:
            if n in (20, 60, 100, 140, 180, 220):
                score += n * x
            if x <= (n - 1) % 40 + 1 <= x + 2:
                display[(n - 1) // 40, (n - 1) % 40] = True
            if line.startswith('noop'):
                n += 1
            else:
                n += 1
                if n in (20, 60, 100, 140, 180, 220):
                    score += n * x
                if x <= (n - 1) % 40 + 1 <= x + 2:
                    display[(n - 1) // 40, (n - 1) % 40] = True
                x += int(line.split()[1])
                n += 1

        print(score)
        print('\n'.join(''.join('##' if p else '  ' for p in d) for d in display))


def day12():
    def djikstra(costs, start, end):
        score = {(x, y): np.inf for x in range(costs.shape[0]) for y in range(costs.shape[1])}
        score[start] = 0
        done = set()
        hq = [(score[start], start[0], start[1]), ]

        while hq:
            c, x, y = heapq.heappop(hq)
            # assert c == score[x, y]
            # assert (x, y) not in done
            if (x, y) == end:
                break
            done.add((x, y))
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= x + dx < costs.shape[0] and 0 <= y + dy < costs.shape[1] and (x + dx, y + dy) not in done:
                    val = c + 1
                    if val < score[x + dx, y + dy] and costs[x + dx, y + dy] <= costs[x, y] + 1:
                        score[x + dx, y + dy] = val
                        heapq.heappush(hq, (score[x + dx, y + dy], x + dx, y + dy))

        return score[end]

    def reverse_djikstra(costs, start, end):
        score = {(x, y): np.inf for x in range(costs.shape[0]) for y in range(costs.shape[1])}
        score[start] = 0
        done = set()
        hq = [(score[start], start[0], start[1]), ]

        while hq:
            c, x, y = heapq.heappop(hq)
            # assert c == score[x, y]
            # assert (x, y) not in done
            if costs[x, y] == end:
                break
            done.add((x, y))
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= x + dx < costs.shape[0] and 0 <= y + dy < costs.shape[1] and (x + dx, y + dy) not in done:
                    val = c + 1
                    if val < score[x + dx, y + dy] and costs[x + dx, y + dy] <= costs[x, y] + 1:
                        score[x + dx, y + dy] = val
                        heapq.heappush(hq, (score[x + dx, y + dy], x + dx, y + dy))

        return score[end]

    with open(utils.get_input(YEAR, 12)) as inp:
        grid = np.array([[ord(d) for d in line[:-1]] for line in inp])
        xs, ys = np.argwhere(grid == ord('S'))[0]
        xe, ye = np.argwhere(grid == ord('E'))[0]
        grid[xs, ys] = ord('a')
        grid[xe, ye] = ord('z')

        print(djikstra(grid, (xs, ys), (xe, ye)))

        foo = np.inf
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == ord('a'):
                    foo = min(foo, djikstra(grid, (i, j), (xe, ye)))
        print(foo)


def day13():
    def compare(l1, l2):
        for x1, x2 in zip(l1, l2):
            if isinstance(x1, int) and isinstance(x2, int):
                if x1 != x2:
                    return 2 * (x1 < x2) - 1
            else:
                x1 = x1 if isinstance(x1, list) else [x1]
                x2 = x2 if isinstance(x2, list) else [x2]
                res = compare(x1, x2)
                if res:
                    return res
        return 1 if len(l1) < len(l2) else -1 if len(l2) < len(l1) else 0

    with open(utils.get_input(YEAR, 13)) as inp:
        count = n = 0
        packets = [[[2]], [[6]]]
        while True:
            p1 = json.loads(inp.readline())
            p2 = json.loads(inp.readline())
            n += 1
            if compare(p1, p2) == 1:
                count += n
            packets.append(p1)
            packets.append(p2)
            if not inp.readline():
                break
        print(count)
        packets.sort(key=functools.cmp_to_key(compare), reverse=True)
        print((packets.index([[2]]) + 1) * (packets.index([[6]]) + 1))


def day14():
    def fall():
        x, y = 500, 0
        while True:
            move = False
            for dx in (0, -1, 1):
                if (0 <= x + dx < grid.shape[0]) and grid[x + dx, y + 1] == 0:
                    x += dx
                    y += 1
                    move = True
                    break
            if y == grid.shape[1] - 1:
                return True
            if not move:
                grid[x, y] = 2
                return y == 0

    with open(utils.get_input(YEAR, 14)) as inp:
        grid = np.zeros((0, 0), dtype=int)
        for line in inp:
            points = np.array([[int(d) for d in p.split(',')] for p in line.split('->')])
            new_x, new_y = np.max(points, axis=0)
            if new_x >= grid.shape[0] or new_y >= grid.shape[1]:
                new_grid = np.zeros((max(new_x + 1, grid.shape[0]), max(new_y + 1, grid.shape[1])), dtype=int)
                new_grid[:grid.shape[0], :grid.shape[1]] = grid
                grid = new_grid
            for i in range(points.shape[0] - 1):
                x1, x2 = sorted([points[i, 0], points[i + 1, 0]])
                y1, y2 = sorted([points[i, 1], points[i + 1, 1]])
                grid[x1:x2 + 1, y1:y2 + 1] = 1

        n = 0
        while not fall():
            n += 1
        print(n)

        nx, ny = grid.shape
        points = np.array([[500 - ny - 1, ny + 1], [500 + ny + 1, ny + 1]])
        new_x, new_y = np.max(points, axis=0)
        if new_x >= grid.shape[0] or new_y >= grid.shape[1]:
            new_grid = np.zeros((max(new_x + 1, grid.shape[0]), max(new_y + 1, grid.shape[1])), dtype=int)
            new_grid[:grid.shape[0], :grid.shape[1]] = grid
            grid = new_grid
        for i in range(points.shape[0] - 1):
            x1, x2 = sorted([points[i, 0], points[i + 1, 0]])
            y1, y2 = sorted([points[i, 1], points[i + 1, 1]])
            grid[x1:x2 + 1, y1:y2 + 1] = 1

        n = grid.shape[1] ** 2
        grid[grid == 2] = 0
        for j in range(1, grid.shape[1]):
            for i in range(500 - j, 500 + j + 1):
                if grid[i, j] == 1 or np.sum(grid[i - 1:i + 2, j - 1]) == 3:
                    grid[i, j] = 1
                    n -= 1
        print(n)


def day15():
    with open(utils.get_input(YEAR, 15)) as inp:
        data = np.array([list(map(int, re.match(
            r'Sensor at x=(-?\d+), y=(-?\d+): closest beacon is at x=(-?\d+), y=(-?\d+)', line).groups()
                                  )) for line in inp])

    intervals = []
    beacons = set()
    y = 2000000
    for xs, ys, xb, yb in data:
        if yb == y:
            beacons.add(xb)
        d = abs(xb - xs) + abs(yb - ys) - abs(y - ys)
        if d > 0:
            intervals.append([xs - d, xs + d])
    intervals.sort()
    x, y = intervals[0]
    for bar in intervals[1:]:
        y = max(y, bar[1])
    print(y - x + 1 - len(beacons))

    n = 4000000
    y = 0
    while y < n:
        intervals = []
        for xs, ys, xb, yb in data:
            d = abs(xb - xs) + abs(yb - ys) - abs(y - ys)
            if d > 0:
                intervals.append([xs - d, xs + d])
        x = res = 0
        overlap = np.inf
        for x_min, x_max in sorted(intervals):
            if x < x_min:
                res = n * (x + 1) + y
                break
            overlap = min(overlap, x - x_min + 1)
            x = max(x, x_max)
            if x >= n:
                break

        if res:
            print(res)
            break
        y += max(1, overlap // 2)


if __name__ == '__main__':
    day15()
