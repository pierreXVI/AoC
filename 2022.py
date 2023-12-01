import collections
import functools
import heapq
import json
import re

import numpy as np

import utils

YEAR = 2022


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


def day11():
    with open(utils.get_input(YEAR, 11)) as inp:
        monkeys_0 = {}
        while True:
            try:
                monkey_id = int(inp.readline().split()[1][:-1])
                items = [int(d) for d in inp.readline().split(':')[1].split(',')]
                operation = inp.readline().split('=')[1].replace('old', '{0}')
                test = int(inp.readline().split('by')[1])
                true = int(inp.readline().split('monkey')[1])
                false = int(inp.readline().split('monkey')[1])
                monkeys_0[monkey_id] = {'items': items, 'oper': operation, 'test': (test, true, false)}
                inp.readline()
            except IndexError:
                break

        monkeys = {
            m: {'items': monkeys_0[m]['items'].copy(), 'oper': monkeys_0[m]['oper'], 'test': monkeys_0[m]['test']} for m
            in monkeys_0}
        monkey_ids = sorted(list(monkeys))
        scores = collections.defaultdict(int)
        for _ in range(20):
            for m in monkey_ids:
                for item in monkeys[m]['items']:
                    scores[m] += 1
                    level = eval(monkeys[m]['oper'].format(item)) // 3
                    if level % monkeys[m]['test'][0]:
                        monkeys[monkeys[m]['test'][2]]['items'].append(level)
                    else:
                        monkeys[monkeys[m]['test'][1]]['items'].append(level)
                monkeys[m]['items'] = []
        print(np.prod(sorted(scores.values())[-2:]))

        monkeys = {
            m: {'items': monkeys_0[m]['items'].copy(), 'oper': monkeys_0[m]['oper'], 'test': monkeys_0[m]['test']} for m
            in monkeys_0}
        monkey_ids = sorted(list(monkeys))
        scores = collections.defaultdict(int)
        ref = np.prod([monkeys[m]['test'][0] for m in monkeys])
        for _ in range(10000):
            for m in monkey_ids:
                for item in monkeys[m]['items']:
                    scores[m] += 1
                    level = eval(monkeys[m]['oper'].format(item))
                    level = level % ref
                    if level % monkeys[m]['test'][0]:
                        monkeys[monkeys[m]['test'][2]]['items'].append(level)
                    else:
                        monkeys[monkeys[m]['test'][1]]['items'].append(level)
                monkeys[m]['items'] = []
        print(np.prod(sorted(scores.values())[-2:]))


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
    data[:, 2] = np.abs(data[:, 2] - data[:, 0]) + np.abs(data[:, 3] - data[:, 1])
    data = data[:, :3]
    for x1, y1, d1 in data:
        foo1 = x1 + y1 + d1
        for x2, y2, d2 in data:
            foo2 = -x2 + y2 + d2
            x = (foo1 - foo2) // 2
            y = (foo1 + foo2 + 1) // 2 + 1
            if 0 <= x < n and 0 <= y < n and np.all(np.abs(x - data[:, 0]) + np.abs(y - data[:, 1]) > data[:, 2]):
                print(n * x + y)


def day16():
    def djikstra(costs, start, end):
        score = {x: np.inf for x in costs}
        score[start] = 0
        done = set()
        hq = [(score[start], start), ]

        while hq:
            c, x = heapq.heappop(hq)
            if x == end:
                break
            done.add(x)
            for y in costs[x]:
                if y not in done:
                    val = c + costs[x][y]
                    if val < score[y]:
                        score[y] = val
                        heapq.heappush(hq, (score[y], y))
        return score[end]

    with open(utils.get_input(YEAR, 16)) as inp:
        cave = {}
        for line in inp:
            foo = re.search(r'Valve (\w+) has flow rate=(\d+); tunnels? leads? to valves? (.*)', line).groups()
            cave[foo[0]] = (int(foo[1]), {bar: 1 for bar in foo[2].split(', ')})

        for d in list(cave.keys()):
            if cave[d][0] == 0 and len(cave[d][1]) == 2:
                a, b = cave[d][1].keys()
                cave[a][1][b] = cave[b][1][a] = cave[d][1][a] + cave[d][1][b]
                del cave[a][1][d], cave[b][1][d], cave[d]

        for d in cave:
            for dd in cave:
                if dd not in cave[d][1] and dd != d:
                    cave[dd][1][d] = cave[d][1][dd] = djikstra({d: cave[d][1] for d in cave}, d, dd)

        def recursive_shit(rest, c, n):
            if n <= 0:
                return 0
            return n * cave[c][0] + max((recursive_shit(rest.difference({v}), v, n - cave[c][1][v] - 1) for v in rest),
                                        default=0)

        print(recursive_shit({d for d in cave if cave[d][0]}, 'AA', 30))

        def explore(rest, c1, c2, n1, n2):
            if n1 <= 0:
                return 0

            return n1 * cave[c1][0] + max(recursive_shit(rest, c2, n2), max(
                (explore(rest.difference({v}), v, c2, n1 - cave[c1][1][v] - 1, n2) for v in rest),
                default=0))

        print(explore({d for d in cave if cave[d][0]}, 'AA', 'AA', 26, 26))


def day17():
    class Rock:
        rocks = [
            np.array([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=int),
            np.array([[1, 0, 1, 2, 1], [0, 1, 1, 1, 2]], dtype=int),
            np.array([[0, 1, 2, 2, 2], [0, 0, 0, 1, 2]], dtype=int),
            np.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=int),
            np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=int),
        ]

        def __init__(self, i, x0, y0):
            self.x = Rock.rocks[i][0] + x0
            self.y = Rock.rocks[i][1] + y0

        def move(self, grid, dx):
            if np.all(0 <= self.x + dx) and np.all(self.x + dx < grid.shape[0]) and not self.intersect(grid, dx=dx):
                self.x += dx
            if np.all(0 < self.y) and not self.intersect(grid, dy=-1):
                self.y -= 1
                return True
            return False

        def intersect(self, grid, dx=0, dy=0):
            for x, y in zip(self.x + dx, self.y + dy):
                if grid[x, y]:
                    return True
            return False

        def apply(self, grid):
            for x, y in zip(self.x, self.y):
                grid[x, y] = True
            return np.max(self.y) + 1

    with open(utils.get_input(YEAR, 17)) as inp:
        wind = inp.readline().strip()

    def simulate(n_iter):
        grid = np.zeros((7, 2 * n_iter), dtype=bool)
        n = 0
        i_wind = -1
        for i_rocks in range(n_iter):
            rock = Rock(i_rocks % 5, 2, n + 3)
            while True:
                i_wind += 1
                if not rock.move(grid, 1 if wind[i_wind % len(wind)] == '>' else -1):
                    n = max(n, rock.apply(grid))
                    break
        # print('\n'.join([''.join(['#' if x else ' ' for x in g]) for g in grid[:, ::-1].T]))
        return n

    def find_cycle(n_iter=3000):
        grid = np.zeros((7, 2 * n_iter), dtype=bool)
        n = 0
        i_wind = -1
        visited = []
        for i_rocks in range(n_iter):
            rock = Rock(i_rocks % 5, 2, n + 3)
            while True:
                i_wind += 1
                if not rock.move(grid, 1 if wind[i_wind % len(wind)] == '>' else -1):
                    n = max(n, rock.apply(grid))
                    entry = (i_rocks % 5, i_wind % len(wind), *(grid[:, n - 20:n].flatten()))
                    if entry in visited:
                        n = visited.index(entry) + 1
                        return i_rocks - n, n
                    visited.insert(0, entry)
                    break

    print(simulate(2022))

    start, period = find_cycle()
    a, b = divmod(1000000000000 - start, period)
    print(a * (simulate(start + period) - simulate(start)) + simulate(start + b))


def day18():
    with open(utils.get_input(YEAR, 18)) as inp:
        grid = np.zeros((0, 0, 0), dtype=int)
        for line in inp:
            x, y, z = [int(d) for d in line.split(',')]
            if grid.shape[0] <= x or grid.shape[1] <= y or grid.shape[2] <= z:
                new_grid = np.zeros((max(x + 1, grid.shape[0]), max(y + 1, grid.shape[1]), max(z + 1, grid.shape[2])),
                                    dtype=int)
                new_grid[:grid.shape[0], :grid.shape[1], :grid.shape[2]] = grid
                grid = new_grid
            grid[x, y, z] = 1
        print(
            np.sum(abs(np.diff(grid, axis=0))) + np.sum(abs(np.diff(grid, axis=1))) + np.sum(abs(np.diff(grid, axis=2)))
            + np.sum(grid[0, :, :] + grid[-1, :, :])
            + np.sum(grid[:, 0, :] + grid[:, -1, :])
            + np.sum(grid[:, :, 0] + grid[:, :, -1])
        )

        fill = np.ones_like(grid, dtype=int)
        to_visit = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    to_visit.add((x, y, 0))
                    to_visit.add((x, y, grid.shape[2] - 1))
                    to_visit.add((x, 0, z))
                    to_visit.add((x, grid.shape[1] - 1, z))
                    to_visit.add((0, y, z))
                    to_visit.add((grid.shape[0] - 1, y, z))
        while to_visit:
            x, y, z = to_visit.pop()
            if not grid[x, y, z] and fill[x, y, z]:
                fill[x, y, z] = 0
                if 0 <= x + 1 < grid.shape[0]:
                    to_visit.add((x + 1, y, z))
                if 0 <= x - 1 < grid.shape[0]:
                    to_visit.add((x - 1, y, z))
                if 0 <= y + 1 < grid.shape[1]:
                    to_visit.add((x, y + 1, z))
                if 0 <= x - 1 < grid.shape[1]:
                    to_visit.add((x, y - 1, z))
                if 0 <= z + 1 < grid.shape[2]:
                    to_visit.add((x, y, z + 1))
                if 0 <= z - 1 < grid.shape[2]:
                    to_visit.add((x, y, z - 1))
        print(
            np.sum(abs(np.diff(fill, axis=0))) + np.sum(abs(np.diff(fill, axis=1))) + np.sum(abs(np.diff(fill, axis=2)))
            + np.sum(fill[0, :, :] + fill[-1, :, :])
            + np.sum(fill[:, 0, :] + fill[:, -1, :])
            + np.sum(fill[:, :, 0] + fill[:, :, -1])
        )


def day19():
    def damn_this_was_hard(blueprint, n_max):
        costs = np.array([[blueprint[1], 0, 0, 0],
                          [blueprint[2], 0, 0, 0],
                          [blueprint[3], blueprint[4], 0, 0],
                          [blueprint[5], 0, blueprint[6], 0]])
        eye = np.eye(4, dtype=int)
        maximums = np.max(costs.astype(float), axis=0)
        maximums[3] = np.inf

        best = 0
        queue = collections.deque()
        queue.append((1, 0, 0, 0, 0, 0, 0, 0, n_max))
        memo = set()
        while queue:
            state = queue.pop()
            if state in memo:
                continue
            memo.add(state)
            workers = np.array(state[:4])
            resources = np.array(state[4:8])
            n = state[8]

            if n == 0:
                best = max(best, resources[3])
                continue
            if resources[3] + n * workers[3] + n * (n - 1) / 2 <= best:
                continue

            queue.append((*workers, *(resources + n * workers), 0))
            for i in range(4):
                if workers[i] < maximums[i] and np.all(costs[i] <= resources + n * workers):
                    j = int(max(1, *np.ceil((costs[i] - resources) / workers) + 1))
                    new_state = (*(workers + eye[i]), *(resources + j * workers - costs[i]), n - j)
                    queue.append(new_state)
        return best

    with open(utils.get_input(YEAR, 19)) as inp:
        blueprints = [[int(d) for d in re.findall(r'(\d+)', line)] for line in inp]

        result = 0
        for data in blueprints:
            result += data[0] * damn_this_was_hard(data, 24)
        print(result)

        print(damn_this_was_hard(blueprints[0], 32)
              * damn_this_was_hard(blueprints[1], 32)
              * damn_this_was_hard(blueprints[2], 32))


def day20():
    def decode(msg, key, repeat):
        n = len(msg)
        msg *= key
        indices = np.arange(n, dtype=int)

        for _ in range(repeat):
            for i in range(n):
                k = (indices[i] + data[i]) % (n - 1)
                # Not needed as the answer use position in the circular list relative to 0 but needed to match example
                # if k == 0 and data[i] < 0:
                #     k = n - 1
                if k > indices[i]:
                    indices[np.logical_and(indices[i] < indices, indices <= k)] -= 1
                elif k < indices[i]:
                    indices[np.logical_and(k <= indices, indices < indices[i])] += 1
                indices[i] = k
        res = np.zeros(n, dtype=int)
        for i in range(n):
            res[indices[i]] = data[i]
        i0 = np.where(res == 0)[0]
        return (res[(i0 + 1000) % n] + res[(i0 + 2000) % n] + res[(i0 + 3000) % n])[0]

    with open(utils.get_input(YEAR, 20)) as inp:
        data = np.array([int(line) for line in inp], dtype=int)
        print(decode(data, 1, 1))
        print(decode(data, 811589153, 10))


def day21():
    def forward(instructions):
        rest = {}
        values = {}
        for line in instructions:
            if match := re.match(r'(\w{4})= (\d+)', line):
                values[match.groups()[0]] = int(match.groups()[1])
            else:
                match = re.match(r'(\w{4})= (\w{4}) (.*) (\w{4})', line)
                rest[match.groups()[0]] = [match.groups()[1], match.groups()[2], match.groups()[3]]
        while rest:
            advance = False
            for v in list(rest.keys()):
                if rest[v][0] in values:
                    rest[v][0] = values[rest[v][0]]
                if rest[v][2] in values:
                    rest[v][2] = values[rest[v][2]]
                if isinstance(rest[v][0], (int, float)) and isinstance(rest[v][2], (int, float)):
                    if rest[v][1] == '+':
                        values[v] = rest[v][0] + rest[v][2]
                    elif rest[v][1] == '*':
                        values[v] = rest[v][0] * rest[v][2]
                    elif rest[v][1] == '-':
                        values[v] = rest[v][0] - rest[v][2]
                    else:
                        values[v] = rest[v][0] / rest[v][2]
                    del rest[v]
                    advance = True
            if not advance:
                return rest
        return int(values['root'])

    def backward(rest):
        a, b = rest['root'][0], rest['root'][2]
        if b in rest:
            a, b = b, a

        while True:
            if a == 'humn':
                return int(b)
            v1, oper, v2 = rest[a]
            if oper == '+':
                if v1 in rest:
                    a = v1
                    b = b - v2
                else:
                    a = v2
                    b = b - v1
            elif oper == '-':
                if v1 in rest:
                    a = v1
                    b = b + v2
                else:
                    b = v1 - b
                    a = v2
            elif oper == '*':
                if v1 in rest:
                    a = v1
                    b = b / v2
                else:
                    a = v2
                    b = b / v1
            elif oper == '/':
                if v1 in rest:
                    a = v1
                    b = b * v2
                else:
                    a = v2
                    b = v1 / b

    with open(utils.get_input(YEAR, 21)) as inp:
        data = [line.strip().replace(':', '=') for line in inp]
        print(forward(data))

        for i in range(len(data)):
            if data[i].startswith('root'):
                data[i] = data[i].replace('+', '==').replace('*', '==').replace('-', '==').replace('/', '==')
            if data[i].startswith('humn'):
                data[i] = 'humn= humn + humn'
        print(backward(forward(data)))


def day22():
    with open(utils.get_input(YEAR, 22)) as inp:
        data = []
        while True:
            line = inp.readline()
            if line == '\n':
                break
            data.append(line[:-1])
        grid = np.zeros((len(data), max(len(d) for d in data)), dtype=int)
        for i in range(len(data)):
            for j in range(len(data[i])):
                grid[i, j] = 1 if data[i][j] == '.' else '-1' if data[i][j] == '#' else 0
        grid = grid.T
        nx, ny = grid.shape

        data = inp.readline()
        moves = []
        rotations = []
        i0 = 0
        for i in range(len(data)):
            if data[i] in ('L', 'R'):
                moves.append(int(data[i0:i]))
                rotations.append(data[i])
                i0 = i + 1
        moves.append(int(data[i0:]))
        rotations.append('')

        def move1(loc, a):
            new_loc = int(loc.real + a.real) % nx + 1j * (int(loc.imag + a.imag) % ny)
            while grid[int(new_loc.real), int(new_loc.imag)] == 0:
                new_loc = int(new_loc.real + a.real) % nx + 1j * (int(new_loc.imag + a.imag) % ny)
            if grid[int(new_loc.real), int(new_loc.imag)] == 1:
                loc = new_loc
            return loc

        def move2(loc, a):
            new_loc = int(loc.real + a.real) + 1j * int(loc.imag + a.imag)
            new_a = a
            if new_loc.real == 49 and 0 <= new_loc.imag < 50:
                new_loc = 0 + 1j * (149 - new_loc.imag)
                new_a = 1
            elif new_loc.real == -1 and 100 <= new_loc.imag < 150:
                new_loc = 50 + 1j * (149 - new_loc.imag)
                new_a = 1

            elif 50 <= new_loc.real < 100 and new_loc.imag == -1:
                new_loc = 0 + 1j * (100 + new_loc.real)
                new_a = 1
            elif new_loc.real == -1 and 150 <= new_loc.imag < 200:
                new_loc = (new_loc.imag - 100) + 1j * 0
                new_a = 1j

            elif new_loc.real == 49 and 50 <= new_loc.imag < 100:
                new_loc = (new_loc.imag - 50) + 1j * 100
                new_a = 1j
            elif 0 <= new_loc.real < 50 and new_loc.imag == 99:
                new_loc = 50 + 1j * (new_loc.real + 50)
                new_a = 1

            elif new_loc.real == 50 and 150 <= new_loc.imag < 200:
                new_loc = (new_loc.imag - 100) + 1j * 149
                new_a = -1j
            elif 50 <= new_loc.real < 100 and new_loc.imag == 150:
                new_loc = 49 + 1j * (new_loc.real + 100)
                new_a = -1

            elif new_loc.real == 100 and 50 <= new_loc.imag < 100:
                new_loc = (new_loc.imag + 50) + 1j * 49
                new_a = -1j
            elif 100 <= new_loc.real < 150 and new_loc.imag == 50:
                new_loc = 99 + 1j * (new_loc.real - 50)
                new_a = -1

            elif new_loc.real == 150 and 0 <= new_loc.imag < 50:
                new_loc = 99 + 1j * (149 - new_loc.imag)
                new_a = -1
            elif new_loc.real == 100 and 100 <= new_loc.imag < 150:
                new_loc = 149 + 1j * (149 - new_loc.imag)
                new_a = -1

            elif 100 <= new_loc.real < 150 and new_loc.imag == -1:
                new_loc = (new_loc.real - 100) + 1j * 199
                new_a = -1j
            elif 0 <= new_loc.real < 50 and new_loc.imag == 200:
                new_loc = (new_loc.real + 100) + 1j * 0
                new_a = 1j

            if grid[int(new_loc.real), int(new_loc.imag)] == 1:
                loc = new_loc
                a = new_a
            return loc, a

        position = 0
        angle = 1
        while grid[int(position.real), int(position.imag)] == 0:
            position = int(position.real + 1) + 1j * position.imag
        for i in range(len(moves)):
            for _ in range(moves[i]):
                nposition = move1(position, angle)
                if nposition == position:
                    break
                position = nposition
            angle *= 1j if rotations[i] == 'R' else -1j if rotations[i] == 'L' else 1
        score_facing = 0 if angle == 1 else 1 if angle == 1j else 2 if angle == -1 else 3
        print(1000 * int(position.imag + 1) + 4 * int(position.real + 1) + score_facing)

        position = 0
        angle = 1
        while grid[int(position.real), int(position.imag)] == 0:
            position = int(position.real + 1) + 1j * position.imag
        for i in range(len(moves)):
            for _ in range(moves[i]):
                nposition, nangle = move2(position, angle)
                if nposition == position:
                    break
                position = nposition
                angle = nangle
            angle *= 1j if rotations[i] == 'R' else -1j if rotations[i] == 'L' else 1
        score_facing = 0 if angle == 1 else 1 if angle == 1j else 2 if angle == -1 else 3
        print(1000 * int(position.imag + 1) + 4 * int(position.real + 1) + score_facing)


def day23():
    def move(cur):
        new = np.zeros_like(cur)
        n = 0
        for i in range(cur.shape[1]):
            x, y = cur[:, i]
            we = cur[:, np.logical_and(np.logical_and(x - 1 <= cur[0], cur[0] <= x + 1), cur[1] == y - 1)]
            ea = cur[:, np.logical_and(np.logical_and(x - 1 <= cur[0], cur[0] <= x + 1), cur[1] == y + 1)]
            no = cur[:, np.logical_and(cur[0] == x - 1, np.logical_and(y - 1 <= cur[1], cur[1] <= y + 1))]
            so = cur[:, np.logical_and(cur[0] == x + 1, np.logical_and(y - 1 <= cur[1], cur[1] <= y + 1))]
            if not (we.size or ea.size or no.size or so.size):
                new[:, i] = x, y
            else:
                direction = (no, so, we, ea)
                results = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                new[:, i] = x, y
                for k in range(4):
                    if not direction[(k + roll) % 4].size:
                        new[:, i] = results[(k + roll) % 4]
                        n += 1
                        break
        reset = []
        for i in range(cur.shape[1]):
            x, y = new[:, i]
            if np.sum(np.logical_and(new[0] == x, new[1] == y)) > 1:
                reset.append(i)
        n -= len(reset)
        for i in reset:
            new[:, i] = cur[:, i]
        return new, n

    with open(utils.get_input(YEAR, 23)) as inp:
        grid = np.array([[1 if c == '#' else 0 for c in line.strip()] for line in inp], dtype=int)
        coords = np.array(np.where(grid), dtype=int)
        roll = 0
        for _ in range(10):
            coords, _ = move(coords)
            roll += 1
        print(np.prod(np.max(coords, axis=1) - np.min(coords, axis=1) + 1) - coords.shape[1])
        while True:
            coords, n_moved = move(coords)
            roll += 1
            if not n_moved:
                break
        print(roll)


if __name__ == '__main__':
    day23()
