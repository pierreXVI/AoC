import collections
import heapq
import itertools
import re

import numpy as np

import utils

YEAR = 2025


def day1():
    part1 = part2 = 0

    v = 50
    with open(utils.get_input(YEAR, 1)) as inp:
        for line in inp:
            delta = int(line[1:-1])

            if line[0] == 'R':
                v += delta
                part2 += v // 100
            else:
                part2 += (((100 - v) % 100) + delta) // 100
                v -= delta
            v %= 100
            if v == 0:
                part1 += 1
    print(part1)
    print(part2)


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        ids = [list(map(int, r.split('-'))) for r in inp.readline().split(',')]

    def is_invalid_part1(i_str):
        return i_str[:(n := len(i_str) // 2)] == i_str[n:]

    def is_invalid_part2(i_str):
        for n in range(1, len(i_str) // 2 + 1):
            k = len(i_str) // n
            if n * k == len(i_str) and i_str[:n] * k == i_str:
                return True
        return False

    print(sum(i for (a, b) in ids for i in range(a, b + 1) if is_invalid_part1(str(i))))
    print(sum(i for (a, b) in ids for i in range(a, b + 1) if is_invalid_part2(str(i))))

    # print(sum(i for (a, b) in ids for i in range(a, b + 1) if re.match(r'^(\d+)\1$', str(i))))
    # print(sum(i for (a, b) in ids for i in range(a, b + 1) if re.match(r'^(\d+)\1+$', str(i))))


def day3():
    def get_v(digits):
        return sum(digits[-(i + 1)] * (10 ** i) for i in range(len(digits)))

    with open(utils.get_input(YEAR, 3)) as inp:
        part1 = part2 = 0
        for line in inp:
            bank = list(map(int, line.strip()))

            max_battery = bank[0]
            opt = 0
            for i in range(1, len(bank)):
                opt = max(opt, 10 * max_battery + bank[i])
                max_battery = max(max_battery, bank[i])
            part1 += opt

            values = bank[-12:]
            v = get_v(values)
            for i in range(12, len(bank)):
                best_values, best_v = values[:], v
                for j in range(12):
                    new_values = [bank[-(i + 1)]] + values[:j] + values[j + 1:]
                    new_v = get_v(new_values)
                    if new_v > best_v:
                        best_values, best_v = new_values, new_v
                values, v = best_values, best_v
            part2 += v

        print(part1)
        print(part2)


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        diagram = np.array([list(map(lambda c: c == '@', line[:-1])) for line in inp])

    def remove_rolls(diagram_):
        removed_ = 0
        new_diagram_ = np.zeros_like(diagram_)
        for i_ in range(diagram.shape[0]):
            for j_ in range(diagram.shape[1]):
                if diagram[i_, j_]:
                    if np.sum(diagram[max(0, i_ - 1):i_ + 2, max(0, j_ - 1):j_ + 2]) < 5:
                        removed_ += 1
                    else:
                        new_diagram_[i_, j_] = True
        return removed_, new_diagram_

    print(remove_rolls(diagram)[0])

    def remove_rolls(diagram_):
        removed_ = 0
        for i_ in range(diagram.shape[0]):
            for j_ in range(diagram.shape[1]):
                if diagram[i_, j_] and np.sum(diagram[max(0, i_ - 1):i_ + 2, max(0, j_ - 1):j_ + 2]) < 5:
                    removed_ += 1
                    diagram_[i_, j_] = False
        return removed_

    part2 = 0
    while (result := remove_rolls(diagram)) > 0:
        part2 += result
    print(part2)


def day5():
    with open(utils.get_input(YEAR, 5)) as inp:
        intervals = collections.defaultdict(int)
        while line := inp.readline().strip():
            a, b = map(int, line.split('-'))
            intervals[a] += 1
            intervals[b + 1] -= 1
        ingredients = sorted([int(line) for line in inp])

    part1 = 0
    part2 = 0
    start = None
    n_ranges = 0
    for v in sorted(intervals.keys()):
        if start is None:
            start = v
        n_ranges += intervals[v]
        if n_ranges == 0:
            part2 += v - start
            while ingredients and ingredients[0] < start:
                ingredients.pop(0)
            while ingredients and ingredients[0] < v:
                ingredients.pop(0)
                part1 += 1
            start = None
    print(part1)
    print(part2)


def day6():
    with open(utils.get_input(YEAR, 6)) as inp:
        lines = inp.readlines()

    part1 = 0
    values = np.array([list(map(int, line.split())) for line in lines[:-1]])
    for i, operation in enumerate(lines[-1].split()):
        part1 += np.sum(values[:, i]) if operation == '+' else np.prod(values[:, i])
    print(part1)

    text = np.array([list(line) for line in lines])
    text[:, -1] = ' '

    operation = None
    numbers = []
    part2 = 0
    for i in range(text.shape[1]):
        if (text[:, i] == ' ').all():
            part2 += np.sum(numbers) if operation == '+' else np.prod(numbers)
            operation = None
            numbers = []
        else:
            if operation is None:
                operation = text[-1, i]
            numbers.append(int(''.join(text[:-1, i])))
    print(part2)


def day7():
    with open(utils.get_input(YEAR, 7)) as inp:
        grid = np.array([list(line[:-1]) for line in inp])

    particles = collections.defaultdict(int)
    particles[np.where(grid[0] == 'S')[0][0]] = 1

    part1 = 0
    for i in range(grid.shape[0] - 1):
        new_particles = collections.defaultdict(int)
        for j in range(grid.shape[1]):
            if grid[i, j] == 'S' or grid[i, j] == '|':
                if grid[i + 1, j] == '^':
                    part1 += 1
                    if j > 0:
                        grid[i + 1, j - 1] = '|'
                        new_particles[j - 1] += particles[j]
                    if j < grid.shape[0]:
                        grid[i + 1, j + 1] = '|'
                        new_particles[j + 1] += particles[j]
                else:
                    grid[i + 1, j] = '|'
                    new_particles[j] += particles[j]
        particles = new_particles
    print(part1)
    print(sum(particles.values()))


def day8():
    with open(utils.get_input(YEAR, 8)) as inp:
        boxes = np.loadtxt(inp, dtype=int, delimiter=',')

    heap = []
    for i in range(boxes.shape[0]):
        for j in range(i + 1, boxes.shape[0]):
            d = boxes[i] - boxes[j]
            heapq.heappush(heap, (d @ d, i, j))

    circuits = {i: {i} for i in range(boxes.shape[0])}
    for _ in range(1000):
        _, i, j = heapq.heappop(heap)
        circuit = circuits[i].union(circuits[j])
        for k in circuit:
            circuits[k] = circuit
    print(np.prod(sorted([len(circuit) for circuit in set(frozenset(circuit) for circuit in circuits.values())])[-3:]))

    while True:
        _, i, j = heapq.heappop(heap)
        circuit = circuits[i].union(circuits[j])
        if len(circuit) == boxes.shape[0]:
            print(boxes[i, 0] * boxes[j, 0])
            break
        for k in circuit:
            circuits[k] = circuit


def day9():
    with open(utils.get_input(YEAR, 9)) as inp:
        tiles = np.loadtxt(inp, dtype=int, delimiter=',')

    print(max([np.prod(abs(tiles[i] - tiles[j]) + 1) for i in range(len(tiles)) for j in range(i + 1, len(tiles))]))

    coords_x = {x: 2 * i + 1 for i, x in enumerate(sorted(np.unique(tiles[:, 0])))}
    coords_y = {y: 2 * i + 1 for i, y in enumerate(sorted(np.unique(tiles[:, 1])))}

    grid = np.zeros((2 * len(coords_x) + 1, 2 * len(coords_y) + 1), dtype=bool)
    for i in range(len(tiles)):
        range_x = coords_x[tiles[i - 1, 0]], coords_x[tiles[i, 0]]
        range_y = coords_y[tiles[i - 1, 1]], coords_y[tiles[i, 1]]
        grid[min(range_x):max(range_x) + 1, min(range_y):max(range_y) + 1] = True

    interior = np.ones_like(grid)
    to_fill = [(grid.shape[0] - 1, 0)]
    while to_fill:
        i, j = to_fill.pop()
        interior[i, j] = False
        if i > 0 and interior[i - 1, j] and not grid[i - 1, j]:
            to_fill.append((i - 1, j))
        if i < grid.shape[0] - 1 and interior[i + 1, j] and not grid[i + 1, j]:
            to_fill.append((i + 1, j))
        if j > 0 and interior[i, j - 1] and not grid[i, j - 1]:
            to_fill.append((i, j - 1))
        if j < grid.shape[1] - 1 and interior[i, j + 1] and not grid[i, j + 1]:
            to_fill.append((i, j + 1))

    print(max([np.prod(abs(tiles[i] - tiles[j]) + 1) for i in range(len(tiles)) for j in range(i + 1, len(tiles))
               if np.all(interior[
                         min(range_x := (coords_x[tiles[i, 0]], coords_x[tiles[j, 0]])):max(range_x) + 1,
                         min(range_y := (coords_y[tiles[i, 1]], coords_y[tiles[j, 1]])):max(range_y) + 1])
               ]))


def day10():
    with open(utils.get_input(YEAR, 10)) as inp:
        part1 = part2 = 0
        for line in inp:
            data = line.split()
            lights = tuple(1 if c == '#' else 0 for c in data[0][1:-1])
            joltages = tuple(map(int, data[-1][1:-1].split(',')))
            buttons = [list(map(int, data[i + 1][1:-1].split(','))) for i in range(len(data) - 2)]

            patterns = {}
            for length in range(len(buttons) + 1):
                for used in itertools.combinations(range(len(buttons)), length):
                    p = [0 for _ in joltages]
                    for b in used:
                        for i in buttons[b]:
                            p[i] += 1
                    patterns.setdefault(tuple(p), length)

            cache = {tuple(0 for _ in joltages): 0}

            def solve_p2(sub_target):
                if sub_target not in cache:
                    cache[sub_target] = min(
                        [cost + 2 * solve_p2(tuple((j - i) // 2 for i, j in zip(pattern, sub_target)))
                         for pattern, cost in patterns.items()
                         if all(i <= j and i % 2 == j % 2 for i, j in zip(pattern, sub_target))],
                        default=np.inf)
                return cache[sub_target]

            part1 += min(
                [cost for pattern, cost in patterns.items() if all(i % 2 == j for i, j in zip(pattern, lights))]
            )
            part2 += solve_p2(joltages)
        print(part1)
        print(part2)


def day11():
    def count_paths(graph_, start, end):
        accessible = {start, end}
        todo = [start]
        while todo:
            device = todo.pop()
            for d in graph_[device]:
                if not d in accessible:
                    accessible.add(d)
                    todo.append(d)

        reverse_graph = collections.defaultdict(list)
        for device, outputs in graph_.items():
            if not device in accessible:
                continue
            for d in outputs:
                reverse_graph[d].append(device)

        done = collections.defaultdict(bool)
        n_paths = collections.defaultdict(int)

        todo = [start]
        n_paths[start] = 1
        while todo:
            device = todo.pop()
            done[device] = True
            if device == end:
                break
            for d in graph_[device]:
                n_paths[d] += n_paths[device]
                if all(done[inp_device] for inp_device in reverse_graph[d]):
                    todo.append(d)
        return n_paths[end]

    with open(utils.get_input(YEAR, 11)) as inp:
        graph = {'out': []}
        for line in inp:
            data = line[:-1].split()
            graph[data[0][:-1]] = data[1:]

    print(count_paths(graph, 'you', 'out'))
    print(count_paths(graph, 'svr', 'fft') * count_paths(graph, 'fft', 'dac') * count_paths(graph, 'dac', 'out'))


def day12():
    with open(utils.get_input(YEAR, 12)) as inp:
        blocks = []
        configs = []

        while line := inp.readline():
            if re.match(r'^\d+:$', line):
                blocks.append(
                    np.array([[c == '#' for c in inp.readline()[:-1]] for _ in range(3)])
                )
            elif match := re.match(r'(\d+)x(\d+):((?: \d+)+)', line):
                configs.append((int(match.group(1)), int(match.group(2)), list(map(int, match.group(3).split()))))

        weights_min = np.array([np.sum(b) for b in blocks])
        weights_max = np.array([b.size for b in blocks])

        if any(weights_min.dot(values) < x * y < weights_max.dot(values) for x, y, values in configs):
            raise ValueError("Must do the work")

        print(len([None for (x, y, values) in configs if x * y > weights_min.dot(values)]))


if __name__ == '__main__':
    utils.time_me(day10)()
