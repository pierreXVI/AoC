import collections
import functools
import heapq
import math
import queue
import re

import numpy as np

import utils

YEAR = 2023


def day1():
    def parse_line(line, digits):
        a = b = -1
        for i in range(len(line)):
            for d in digits:
                if line[i:i + len(d)] == d:
                    a = digits[d]
                    break
            if not a == -1:
                break
        for i in range(len(line)):
            for d in digits:
                if line[-len(d) - i:-i] == d:
                    b = digits[d]
                    break
            if not b == -1:
                break

        return 10 * a + b

    part1 = {str(i): i for i in range(10)}
    part2 = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
             'nine': 9, **part1}

    count1 = count2 = 0
    with open(utils.get_input(YEAR, 1)) as inp:
        for _line in inp:
            count1 += parse_line(_line, part1)
            count2 += parse_line(_line, part2)

    print(count1)
    print(count2)


def day2():
    def parse_line(line):
        game, line = line.split(':', maxsplit=1)
        out = collections.defaultdict(int)

        for pick in line.split(';'):
            for color_pick in pick.split(','):
                n, color = color_pick.split()
                out[color] = max(out[color], int(n))
        return int(game.split()[-1]), out

    with open(utils.get_input(YEAR, 2)) as inp:
        part1 = part2 = 0
        for _line in inp:
            game_id, max_cubes = parse_line(_line)
            if max_cubes['red'] <= 12 and max_cubes['green'] <= 13 and max_cubes['blue'] <= 14:
                part1 += game_id
            part2 += max_cubes['red'] * max_cubes['green'] * max_cubes['blue']
        print(part1)
        print(part2)


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        data = np.array([list(line) for line in inp.read().splitlines()])

    def is_symbol(c):
        return c != '.' and not '0' <= c <= '9'

    part1 = 0
    gears: collections.defaultdict[tuple[int, int], list] = collections.defaultdict(list)
    for i in range(data.shape[0]):
        j = 0
        while j < data.shape[1]:
            if '0' <= data[i, j] <= '9':
                j0 = j1 = j
                while j1 < data.shape[1] and '0' <= data[i, j1] <= '9':
                    j1 += 1
                n = int(''.join(data[i, j0:j1]))
                for j in range(max(0, j0 - 1), min(data.shape[1], j1 + 1)):
                    if i > 0 and is_symbol(data[i - 1, j]):
                        part1 += n
                        gears[(i - 1, j)].append(n)
                        break
                    if i + 1 < data.shape[0] and is_symbol(data[i + 1, j]):
                        part1 += n
                        gears[(i + 1, j)].append(n)
                        break

                else:
                    if j0 > 0 and is_symbol(data[i, j0 - 1]):
                        part1 += n
                        gears[(i, j0 - 1)].append(n)
                    elif j1 < data.shape[1] and is_symbol(data[i, j1]):
                        part1 += n
                        gears[(i, j1)].append(n)
                j = j1
            j += 1
    print(part1)

    part2 = 0
    for k in gears:
        if len(gears[k]) == 2:
            part2 += gears[k][0] * gears[k][1]
    print(part2)


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        cards = inp.read().splitlines()

    part1 = 0
    part2 = np.ones((len(cards),), dtype=int)
    i_card = 0
    for card in cards:
        winning, numbers = card.split(':')[-1].split('|')
        winning = {int(n) for n in winning.split()}
        numbers = {int(n) for n in numbers.split()}
        score = len(winning.intersection(numbers))
        if score > 0:
            part1 += 2 ** (score - 1)
        part2[i_card + 1: i_card + 1 + score] += part2[i_card]
        i_card += 1

    print(part1)
    print(part2.sum())


def day5():
    with open(utils.get_input(YEAR, 5)) as inp:
        seeds = [int(s) for s in inp.readline().split(':')[-1].split()]
        inp.readline()

        part_1 = [(s, 1) for s in seeds]
        part_2 = [(seeds[2 * i], seeds[2 * i + 1]) for i in range(len(seeds) // 2)]

        def read_map():
            parameters = []
            _line = inp.readline()
            while _line:
                parameters.append(tuple(map(int, _line.split())))
                _line = inp.readline().strip()
            parameters.sort(key=lambda p: p[1])

            def f(interval):
                out = []
                a, b = interval[0], interval[0] + interval[1] - 1
                for dst_start, src_start, length in parameters:
                    c, d = src_start, src_start + length - 1
                    if a <= d and c <= b:
                        out.append((a, max(a, c) - a))
                        out.append((max(a, c) + dst_start - src_start, min(b, d) - max(a, c) + 1))
                        a = min(b, d) + 1
                    if a > b:
                        return out
                out.append((a, b - a + 1))
                return out

            return lambda intervals: [i for interval in intervals for i in f(interval) if i[1] > 0]

        inp.readline()
        seed2soil = read_map()
        inp.readline()
        soil2fertilizer = read_map()
        inp.readline()
        fertilizer2water = read_map()
        inp.readline()
        water2light = read_map()
        inp.readline()
        light2temperature = read_map()
        inp.readline()
        temperature2humidity = read_map()
        inp.readline()
        humidity2location = read_map()

        print(min(humidity2location(temperature2humidity(
            light2temperature(water2light(fertilizer2water(soil2fertilizer(seed2soil(part_1))))))))[0])

        print(min(humidity2location(temperature2humidity(
            light2temperature(water2light(fertilizer2water(soil2fertilizer(seed2soil(part_2))))))))[0])


def day6():
    with open(utils.get_input(YEAR, 6)) as inp:
        times = list(map(int, inp.readline().strip().split(':')[-1].split()))
        distances = list(map(int, inp.readline().strip().split(':')[-1].split()))

    def solve(_d, _t):
        n1 = int(np.ceil(t / 2 - np.sqrt((_t / 2) ** 2 - _d)))
        n2 = int(np.floor(t / 2 + np.sqrt((_t / 2) ** 2 - _d)))
        if n1 * (_t - n1) == _d:
            n1 += 1
        if n2 * (_t - n2) == _d:
            n2 -= 1
        return n2 - n1 + 1

    part_1 = 1
    for t, d in zip(times, distances):
        part_1 *= solve(d, t)
    print(part_1)

    print(solve(int(''.join(map(str, distances))), int(''.join(map(str, times)))))


def day7():
    cards_1 = 'AKQJT98765432'
    cards_2 = 'AKQT98765432J'

    def kind(hand):
        count = collections.Counter(hand)
        s: collections.defaultdict[int, list] = collections.defaultdict(list)
        for k in count:
            s[count[k]].append(k)

        if 5 in s:
            return 6
        elif 4 in s:
            return 5
        elif 3 in s and 2 in s:
            return 4
        elif 3 in s:
            return 3
        else:
            return len(s[2])

    def comp_1(player_a, player_b):
        kind_a = kind(player_a[0])
        kind_b = kind(player_b[0])
        if kind_a == kind_b:
            score_a = [cards_1.index(card) for card in player_a[0]]
            score_b = [cards_1.index(card) for card in player_b[0]]
            return 1 if score_a < score_b else -1 if score_b < score_a else 0
        return 1 if kind_b < kind_a else -1

    def best_hand(hand):
        if 'J' not in hand:
            return hand
        i = hand.index('J')
        possible_cards = set([c for c in hand if c != 'J'])
        if not possible_cards:
            possible_cards = ('A',)
        return sorted([(best_hand(hand[:i] + c + hand[i + 1:]), 0) for c in possible_cards],
                      key=functools.cmp_to_key(comp_1))[-1][0]

    def comp_2(player_a, player_b):
        kind_a = kind(player_a[1])
        kind_b = kind(player_b[1])
        if kind_a == kind_b:
            score_a = [cards_2.index(card) for card in player_a[0]]
            score_b = [cards_2.index(card) for card in player_b[0]]
            return 1 if score_a < score_b else -1 if score_b < score_a else 0
        return 1 if kind_b < kind_a else -1

    with open(utils.get_input(YEAR, 7)) as inp:
        players = []
        for line in inp:
            players.append(line.split())

    part_1 = 0
    for idx, player in enumerate(sorted(players, key=functools.cmp_to_key(comp_1))):
        part_1 += (idx + 1) * int(player[1])
    print(part_1)

    part_2 = 0
    players_jacked = [(player[0], best_hand(player[0]), player[1]) for player in players]
    for idx, player in enumerate(sorted(players_jacked, key=functools.cmp_to_key(comp_2))):
        part_2 += (idx + 1) * int(player[2])
    print(part_2)


def day8():
    with open(utils.get_input(YEAR, 8)) as inp:
        directions = inp.readline().strip()

        inp.readline()
        graph = {}
        for line in inp:
            line = re.match(r'([A-Z1-9]{3}) = \(([A-Z1-9]{3}), ([A-Z1-9]{3})\)', line)
            graph[line.group(1)] = (line.group(2), line.group(3))

    def solve(start, is_end):
        i = 0
        n_directions = len(directions)
        while not is_end(start):
            start = graph[start][0 if directions[i % n_directions] == 'L' else 1]
            i += 1
        return i

    print(solve('AAA', lambda here: here == 'ZZZ'))
    print(math.lcm(*[solve(place, lambda here: here[2] == 'Z') for place in graph if place[2] == 'A']))


def day9():
    with open(utils.get_input(YEAR, 9)) as inp:
        part1 = part2 = 0
        for line in inp:
            data = np.array(line.split(), dtype=int)
            foo = []
            bar = []
            while (data != 0).any():
                foo.append(data[-1])
                bar.append(data[0])
                data = np.diff(data)
            part1 += sum(foo)
            part2 += sum([-bar[i] if i % 2 else bar[i] for i in range(len(bar))])
        print(part1)
        print(part2)


def day10():
    with open(utils.get_input(YEAR, 10)) as inp:
        data = np.array(list(map(list, inp.read().splitlines())))

    directions = {
        '|': {(1, 0): '|LJ', (-1, 0): '|7F'},
        '-': {(0, 1): '-J7', (0, -1): '-LF'},
        'L': {(0, 1): '-J7', (-1, 0): '|7F'},
        'J': {(0, -1): '-LF', (-1, 0): '|7F'},
        '7': {(1, 0): '|LJ', (0, -1): '-LF'},
        'F': {(0, 1): '-J7', (1, 0): '|LJ'},
        'S': {(0, 1): '-J7', (1, 0): '|LJ', (0, -1): '-LF', (-1, 0): '|7F'}}

    def find_next(here, prev):
        dirs = directions[data[here[0], here[1]]]
        for k in dirs:
            if 0 <= here[0] + k[0] < data.shape[0] and 0 <= here[1] + k[1] < data.shape[1] \
                    and (k[0] + prev[0] != 0 or k[1] + prev[1] != 0) \
                    and data[here[0] + k[0], here[1] + k[1]] in dirs[k]:
                return k

    loop = [tuple(np.argwhere(data == 'S')[0])]

    previous = (0, 0)
    while True:
        previous = find_next(loop[-1], previous)
        if previous is None:
            break
        loop.append((loop[-1][0] + previous[0], loop[-1][1] + previous[1]))
    print(len(loop) // 2)

    data[tuple(loop[0])] = set(directions[data[tuple(loop[1])]][loop[0][0] - loop[1][0], loop[0][1] - loop[1][1]]) \
        .intersection(directions[data[tuple(loop[-1])]][loop[0][0] - loop[-1][0], loop[0][1] - loop[-1][1]]).pop()

    part2 = 0
    loop = set(loop)
    for i in range(data.shape[0]):
        inside = False
        foo = ''
        for j in range(data.shape[1]):
            if (i, j) in loop:
                if data[i, j] == '|':
                    inside = not inside
                    foo = ''
                elif data[i, j] == 'J':
                    inside = not inside if foo == 'F' else inside
                    foo = ''
                elif data[i, j] == '7':
                    inside = not inside if foo == 'L' else inside
                    foo = ''
                elif not foo:
                    foo = data[i, j]
            elif inside:
                part2 += 1
    print(part2)


def day11():
    with open(utils.get_input(YEAR, 11)) as inp:
        data = np.array([[c == '#' for c in line.strip()] for line in inp.readlines()])

    def expand(_galaxies, scale):
        def expand_axis(array):
            expanded = np.zeros_like(array)
            indices = np.argsort(array)
            line = -1
            expansion = 0
            for i in indices:
                expansion += max(0, array[i] + expansion - line - 1) * (scale - 1)
                expanded[i] = array[i] + expansion
                line = expanded[i]
            return expanded

        return expand_axis(_galaxies[:, 0]), expand_axis(_galaxies[:, 1])

    def get_score(rows, cols):
        score = 0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                score += abs(rows[i] - rows[j]) + abs(cols[i] - cols[j])
        return score

    galaxies = np.argwhere(data)
    print(get_score(*expand(galaxies, 2)))
    print(get_score(*expand(galaxies, 1000000)))


def day12():
    mem = {}

    def validate(springs, groups):
        if (springs, *groups) in mem:
            return mem[(springs, *groups)]

        if not springs:
            mem[(springs, *groups)] = int(not groups)
            return int(not groups)
        if not groups:
            mem[(springs, *groups)] = int('#' not in springs)
            return int('#' not in springs)

        i = 0
        while i < len(springs) and springs[i] == '.':
            i += 1
        if groups and i == len(springs):
            mem[(springs, *groups)] = 0
            return 0
        j = i
        while j < len(springs) and springs[j] != '.':
            j += 1

        if springs[i] == '?':
            if j - i < groups[0] or (i + groups[0] < len(springs) and springs[i + groups[0]] == '#'):
                mem[(springs, *groups)] = validate(springs[i + 1:], groups)
            else:
                mem[(springs, *groups)] = validate(springs[i + 1:], groups) + validate(springs[i + groups[0] + 1:],
                                                                                       groups[1:])
        else:
            if j - i < groups[0] or (i + groups[0] != len(springs) and springs[i + groups[0]] == '#'):
                mem[(springs, *groups)] = 0
            else:
                mem[(springs, *groups)] = validate(springs[i + groups[0] + 1:], groups[1:])
        return mem[(springs, *groups)]

    with open(utils.get_input(YEAR, 12)) as inp:
        part1 = part2 = 0
        for line in inp:
            record = line.strip().split()
            part1 += validate(record[0], [int(c) for c in record[1].split(',')])
            part2 += validate('?'.join(5 * [record[0]]), 5 * [int(c) for c in record[1].split(',')])
        print(part1)
        print(part2)


def day13():
    def find_score(pattern, ndiff):
        for i in range(1, pattern.shape[0]):
            n = min(i, pattern.shape[0] - i)
            if np.sum(pattern[i - n:i][::-1] != pattern[i:i + n]) == ndiff:
                return 100 * i
        for i in range(1, pattern.shape[1]):
            n = min(i, pattern.shape[1] - i)
            if np.sum(pattern[:, i - n:i][:, ::-1] != pattern[:, i:i + n]) == ndiff:
                return i
        return 0

    with open(utils.get_input(YEAR, 13)) as inp:
        part1 = part2 = 0
        while True:
            data = []
            line = inp.readline().strip()
            while line:
                data.append([c == '#' for c in line])
                line = inp.readline().strip()
            if not data:
                break
            part1 += find_score(np.array(data), 0)
            part2 += find_score(np.array(data), 1)
        print(part1)
        print(part2)


def day14():
    with open(utils.get_input(YEAR, 14)) as inp:
        data = np.array([[1 if c == 'O' else -1 if c == '#' else 0 for c in line.strip()] for line in inp])

    def score(rocks):
        return np.sum((rocks == 1).T * np.arange(rocks.shape[0], 0, -1))

    def tilt_north(rocks):
        for i in range(rocks.shape[0]):
            for j in range(rocks.shape[1]):
                if rocks[i, j] == 1:
                    k = i - 1
                    while k >= 0 and rocks[k, j] == 0:
                        k -= 1
                    rocks[i, j] = 0
                    rocks[k + 1, j] = 1

    def tilt_west(rocks):
        for j in range(rocks.shape[1]):
            for i in range(rocks.shape[0]):
                if rocks[i, j] == 1:
                    k = j - 1
                    while k >= 0 and rocks[i, k] == 0:
                        k -= 1
                    rocks[i, j] = 0
                    rocks[i, k + 1] = 1

    def tilt_south(rocks):
        for i in range(rocks.shape[0] - 1, -1, -1):
            for j in range(rocks.shape[1]):
                if rocks[i, j] == 1:
                    k = i + 1
                    while k < rocks.shape[0] and rocks[k, j] == 0:
                        k += 1
                    rocks[i, j] = 0
                    rocks[k - 1, j] = 1

    def tilt_east(rocks):
        for j in range(rocks.shape[1] - 1, -1, -1):
            for i in range(rocks.shape[0]):
                if rocks[i, j] == 1:
                    k = j + 1
                    while k < rocks.shape[1] and rocks[i, k] == 0:
                        k += 1
                    rocks[i, j] = 0
                    rocks[i, k - 1] = 1

    def tilt_cycle(rocks):
        tilt_north(rocks)
        tilt_west(rocks)
        tilt_south(rocks)
        tilt_east(rocks)

    part1 = data.copy()
    tilt_north(part1)
    print(score(part1))

    part2 = data.copy()
    history = [part2.flatten().tolist()]
    for n in range(1000000000):
        tilt_cycle(part2)
        part2_flat = part2.flatten().tolist()
        if part2_flat in history:
            n0 = history.index(part2_flat)
            part2 = np.array(history[((1000000000 - n0) % (n + 1 - n0)) + n0]).reshape(data.shape)
            break
        history.append(part2_flat)
    print(score(part2))


def day15():
    with open(utils.get_input(YEAR, 15)) as inp:
        data = inp.readline().strip().split(',')

    def hash_algorithm(_step):
        score = 0
        for c in _step:
            score = ((score + ord(c)) * 17) % 256
        return score

    print(sum(hash_algorithm(step) for step in data))

    boxes = [[] for _ in range(256)]
    for step in data:
        if step[-1] == '-':
            label = step[:-1]
            box = boxes[hash_algorithm(label)]
            for i in range(len(box)):
                if box[i][0] == label:
                    box.pop(i)
                    break
        else:
            label = step[:-2]
            box = boxes[hash_algorithm(label)]
            for i in range(len(box)):
                if box[i][0] == label:
                    box[i][1] = int(step[-1])
                    break
            else:
                box.append([label, int(step[-1])])

    part2 = 0
    for i in range(len(boxes)):
        for j in range(len(boxes[i])):
            part2 += (i + 1) * (j + 1) * boxes[i][j][1]
    print(part2)


def day16():
    with open(utils.get_input(YEAR, 16)) as inp:
        data = np.array([list(line.strip()) for line in inp])

    def get_energized(start):
        energized = np.zeros((*data.shape, 2, 2), dtype=bool)
        rays = [start]
        while rays:
            new_rays = []
            for i, j, direction in rays:

                if not 0 <= i < data.shape[0] or not 0 <= j < data.shape[1] \
                        or energized[i, j, direction % 2, (direction // abs(direction) + 1) // 2]:
                    continue

                energized[i, j, direction % 2, (direction // abs(direction) + 1) // 2] = True

                if data[i, j] == '-':
                    energized[i, j, 0, :] = True
                    if direction == 2:
                        new_rays.append((i, j + 1, 2))
                    elif direction == -2:
                        new_rays.append((i, j - 1, -2))
                    else:
                        new_rays.append((i, j - 1, -2))
                        new_rays.append((i, j + 1, 2))
                elif data[i, j] == '|':
                    energized[i, j, 1, :] = True
                    if direction == 1:
                        new_rays.append((i + 1, j, 1))
                    elif direction == -1:
                        new_rays.append((i - 1, j, -1))
                    else:
                        new_rays.append((i - 1, j, -1))
                        new_rays.append((i + 1, j, 1))
                elif data[i, j] == '/':
                    if direction == 2:
                        new_rays.append((i - 1, j, -1))
                    elif direction == -2:
                        new_rays.append((i + 1, j, 1))
                    elif direction == 1:
                        new_rays.append((i, j - 1, -2))
                    elif direction == -1:
                        new_rays.append((i, j + 1, 2))
                elif data[i, j] == '\\':
                    if direction == 2:
                        new_rays.append((i + 1, j, 1))
                    elif direction == -2:
                        new_rays.append((i - 1, j, -1))
                    elif direction == 1:
                        new_rays.append((i, j + 1, 2))
                    elif direction == -1:
                        new_rays.append((i, j - 1, -2))
                else:
                    energized[i, j, direction % 2, :] = True
                    if direction == 2:
                        new_rays.append((i, j + 1, 2))
                    elif direction == -2:
                        new_rays.append((i, j - 1, -2))
                    elif direction == 1:
                        new_rays.append((i + 1, j, 1))
                    elif direction == -1:
                        new_rays.append((i - 1, j, -1))

            rays = new_rays
        return energized

    def count_energized(energized):
        return np.sum(energized, axis=(2, 3), dtype=bool).sum()

    print(count_energized(get_energized((0, 0, 2))))

    part2 = 0
    all_energized = np.zeros((*data.shape, 2, 2), dtype=bool)
    for ii in range(data.shape[0]):
        if not all_energized[ii, 0, 0, :].any():
            current_energized = get_energized((ii, 0, 2))
            all_energized += current_energized
            part2 = max(part2, count_energized(current_energized))

        if not all_energized[ii, data.shape[1] - 1, 0, :].any():
            current_energized = get_energized((ii, data.shape[1] - 1, -2))
            all_energized += current_energized
            part2 = max(part2, count_energized(current_energized))

    for jj in range(data.shape[1]):
        if not all_energized[0, jj, 1, :].any():
            current_energized = get_energized((0, jj, 1))
            all_energized += current_energized
            part2 = max(part2, count_energized(current_energized))

        if not all_energized[data.shape[0] - 1, jj, :].any():
            current_energized = get_energized((data.shape[0] - 1, jj, -1))
            all_energized += current_energized
            part2 = max(part2, count_energized(current_energized))
    print(part2)


def day17():
    with open(utils.get_input(YEAR, 17)) as inp:
        data = np.array([list(line.strip()) for line in inp], dtype=int)

    def dijkstra(min_consecutive, max_consecutive):
        score = np.inf * np.ones((*data.shape, 2))
        score[0, 0] = 0
        done = set()
        hq = [(0, (0, 0, 0)), (0, (0, 0, 1))]

        while hq:
            c0, (x, y, axis) = heapq.heappop(hq)
            if (x, y) == (data.shape[0] - 1, data.shape[1] - 1):
                break
            done.add((x, axis))
            for direction in (+1, -1):
                c = c0
                for i in range(1, min_consecutive):
                    new = (x + axis * direction * i, y + (1 - axis) * direction * i)
                    if not 0 <= new[1 - axis] < data.shape[1 - axis]:
                        break
                    c += data[new]
                for i in range(min_consecutive, max_consecutive):
                    new = (x + axis * direction * i, y + (1 - axis) * direction * i)
                    if not 0 <= new[1 - axis] < data.shape[1 - axis]:
                        break
                    c += data[new]
                    if c < score[new[0], new[1], axis] and (*new, 1 - axis) not in done:
                        score[new[0], new[1], axis] = c
                        heapq.heappush(hq, (c, (new[0], new[1], 1 - axis)))

        return int(min(score[-1, -1]))

    print(dijkstra(1, 4))
    print(dijkstra(4, 11))


def day18():
    with open(utils.get_input(YEAR, 18)) as inp:
        part1 = part2 = 0
        pos1 = pos2 = (0, 0)
        for line in inp:
            direction, n, color = line.split()
            dx = int(n) if direction == 'D' else - int(n) if direction == 'U' else 0
            dy = int(n) if direction == 'R' else - int(n) if direction == 'L' else 0
            part1 += dx * (2 * pos1[1] + dy) + abs(dx) + abs(dy)
            pos1 = (pos1[0] + dx, pos1[1] + dy)

            n = int(color[2:-2], 16)
            direction = {'0': 'R', '1': 'D', '2': 'L', '3': 'U'}[color[-2]]
            dx = int(n) if direction == 'D' else - int(n) if direction == 'U' else 0
            dy = int(n) if direction == 'R' else - int(n) if direction == 'L' else 0
            part2 += dx * (2 * pos2[1] + dy) + abs(dx) + abs(dy)
            pos2 = (pos2[0] + dx, pos2[1] + dy)

        print(part1 // 2 + 1)
        print(part2 // 2 + 1)


def day19():
    def apply_rule(r, p):
        if r[1]:
            return r[3] if p[r[0]] > r[2] else None
        else:
            return r[3] if p[r[0]] < r[2] else None

    def apply_workflow(w, p):
        for r in w:
            if apply_rule(r, p):
                return apply_rule(r, p)

    workflows = {}
    with open(utils.get_input(YEAR, 19)) as inp:
        for line in inp:
            if not line.strip():
                break

            _name, _rules = re.match(r'(\w+)\{(.*)}', line).groups()
            workflows[_name] = tuple()
            for _rule in _rules.split(','):
                rule_detail = re.match(r'([xmas])([<>])(\d+):(\w+)', _rule)
                if rule_detail:
                    workflows[_name] += ((rule_detail.group(1), rule_detail.group(2) == '>', int(rule_detail.group(3)),
                                          rule_detail.group(4)),)
                else:
                    workflows[_name] += (('x', True, 0, _rule,),)

        part1 = 0
        for line in inp:
            part = {attribute.split('=')[0]: int(attribute.split('=')[1]) for attribute in line[1:-2].split(',')}

            _name = 'in'
            while _name not in ('A', 'R'):
                _name = apply_workflow(workflows[_name], part)
            if _name == 'A':
                part1 += part['x'] + part['m'] + part['a'] + part['s']
        print(part1)

    def back_propagate(name, valid_range):
        new_valid_ranges = []
        for workflow in workflows:
            valid = valid_range.copy()
            for rule in workflows[workflow]:
                if rule[3] == name:
                    new_valid = valid.copy()
                    if rule[1]:
                        new_valid[rule[0]] = (max(valid[rule[0]][0], rule[2] + 1), valid[rule[0]][1])
                    else:
                        new_valid[rule[0]] = (valid[rule[0]][0], min(valid[rule[0]][1], rule[2] - 1))
                    new_valid_ranges.append((workflow, new_valid))

                if rule[1]:
                    valid[rule[0]] = (valid[rule[0]][0], min(valid[rule[0]][1], rule[2]))
                else:
                    valid[rule[0]] = (max(valid[rule[0]][0], rule[2]), valid[rule[0]][1])
        return new_valid_ranges

    part2 = 0
    all_valid = [('A', {'x': (1, 4000), 'm': (1, 4000), 'a': (1, 4000), 's': (1, 4000)})]
    while all_valid:
        _name, _valid = all_valid.pop()
        if _name == 'in':
            part2 += math.prod(max(0, _valid[cat][1] - _valid[cat][0] + 1) for cat in _valid)
            continue
        all_valid += back_propagate(_name, _valid)
    print(part2)


def day20():
    class Module:
        def __init__(self):
            self.outputs = []
            self.inputs = {}
            self.mode = 0
            self.value = False

        def add_input(self, _input):
            self.inputs[_input] = False

        def add_output(self, _name):
            self.outputs.append(_name)

        def set_mode(self, _mode):
            self.mode = _mode

        def send_pulse(self, _input, value):
            if _input:
                self.inputs[_input] = value
            if self.mode == 1:
                if not value:
                    self.value = not self.value
                    return self.outputs, self.value
            elif self.mode == 2:
                self.value = not all(self.inputs[_i] for _i in self.inputs)
                return self.outputs, self.value
            elif self.mode == 3:
                self.value = value
                return self.outputs, self.value
            return [], None

        def __repr__(self):
            return "mode {0}, inputs = {1}, outputs = {2}".format(self.mode, self.inputs, self.outputs)

        def __copy__(self):
            copy = Module()
            copy.outputs = self.outputs.copy()
            copy.inputs = self.inputs.copy()
            copy.mode = self.mode
            copy.value = self.value
            return copy

    def press_button(conf):
        pulse_count = [0, 0]
        pulses = queue.Queue()
        pulses.put((None, 'broadcaster', False))
        while not pulses.empty():
            sender, receiver, val = pulses.get()
            pulse_count[val] += 1
            new_receivers, val = conf[receiver].send_pulse(sender, val)
            for new_receiver in new_receivers:
                pulses.put((receiver, new_receiver, val))
        return pulse_count

    def get_first_activation(conf, label):
        activations = {_name: 0 for _name in conf[label].inputs}

        i = 0
        while True:
            i += 1
            pulses = queue.Queue()
            pulses.put((None, 'broadcaster', False))
            while not pulses.empty():
                sender, receiver, val = pulses.get()
                if sender in conf[label].inputs and val:
                    activations[sender] = i
                if all(activations[_name] > 0 for _name in activations):
                    return math.lcm(*activations.values())
                new_receivers, val = conf[receiver].send_pulse(sender, val)
                for new_receiver in new_receivers:
                    pulses.put((receiver, new_receiver, val))

    configuration = collections.defaultdict(Module)
    with open(utils.get_input(YEAR, 20)) as inp:
        for line in inp:
            line = line.strip().split(' -> ')
            if line[0][0] == '%':
                name = line[0][1:]
                configuration[name].set_mode(1)
            elif line[0][0] == '&':
                name = line[0][1:]
                configuration[name].set_mode(2)
                pass
            else:
                name = line[0]
                configuration[name].set_mode(3)
            for o in line[1].split(', '):
                configuration[name].add_output(o)
                configuration[o].add_input(name)

    part1 = np.array([0, 0])
    config = {c: configuration[c].__copy__() for c in configuration}
    for _ in range(1000):
        part1 += press_button(config)
    print(part1.prod())

    print(get_first_activation({c: configuration[c].__copy__() for c in configuration}, 'qb'))


def day21():
    with open(utils.get_input(YEAR, 21)) as inp:
        data = []
        n = 0
        for line in inp:
            data.append([c != '#' for c in line.strip()])
            if 'S' in line:
                start = (n, line.index('S'))
            n += 1
        data = np.array(data)

    visited = {start: 0}
    positions = {start}
    n_total = 26501365
    vals = []
    n = 0
    while True:
        if n % data.shape[0] == n_total % data.shape[0]:
            vals.append(len([None for d in visited.values() if d % 2 == n % 2]))
            if len(vals) == 3:
                break

        n += 1
        new_positions = set()
        for x, y in positions:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if data[(x + dx) % data.shape[0], (y + dy) % data.shape[1]] and (x + dx, y + dy) not in visited:
                    new_positions.add((x + dx, y + dy))
                    visited[(x + dx, y + dy)] = n
        positions = new_positions

    print(len([None for d in visited.values() if d % 2 == 0 and d < 65]))

    a = (vals[2] + vals[0] - 2 * vals[1]) // 2
    b = (4 * vals[1] - 3 * vals[0] - vals[2]) // 2
    c = vals[0]
    x = n_total // data.shape[0]
    print(a * x * x + b * x + c)


def day22():
    def intersect(brick1, brick2):
        return brick1[0][0] <= brick2[0][1] and brick2[0][0] <= brick1[0][1] \
            and brick1[1][0] <= brick2[1][1] and brick2[1][0] <= brick1[1][1]

    bricks = []
    with open(utils.get_input(YEAR, 22)) as inp:
        for line in inp:
            start, end = line.strip().split('~')
            start = start.split(',')
            end = end.split(',')
            bricks.append([(int(start[i]), int(end[i])) for i in range(3)])
    bricks.sort(key=lambda brick: brick[2][0])

    for i in range(len(bricks)):
        max_z = 1
        for j in range(i):
            if intersect(bricks[i], bricks[j]):
                max_z = max(max_z, bricks[j][2][1] + 1)
        bricks[i][2] = (max_z, max_z + bricks[i][2][1] - bricks[i][2][0])

    bases = [[] for _ in range(len(bricks))]
    tops: list[list[int]] = [[] for _ in range(len(bricks))]
    unique_bases = set()
    for i in range(len(bricks)):
        for j in range(i):
            if intersect(bricks[i], bricks[j]) and bricks[j][2][1] + 1 == bricks[i][2][0]:
                bases[i].append(j)
                tops[j].append(i)
        if len(bases[i]) == 1:
            unique_bases.add(bases[i][0])
    print(len(bricks) - len(unique_bases))

    part2 = 0
    for unique_base in unique_bases:
        new_bases = [base.copy() for base in bases]
        to_remove = [unique_base]
        while to_remove:
            base = to_remove.pop()
            for top in tops[base]:
                new_bases[top].remove(base)
                if not new_bases[top]:
                    to_remove.append(top)
                    part2 += 1
    print(part2)


def day23():
    with open(utils.get_input(YEAR, 23)) as inp:
        data = np.array([list(line.strip()) for line in inp])

    def move_part1(i, j, visited):
        if i == data.shape[0] - 1:
            return

        if data[i, j] == '^':
            available_moves = ((-1, 0),)
        elif data[i, j] == 'v':
            available_moves = ((1, 0),)
        elif data[i, j] == '<':
            available_moves = ((0, -1),)
        elif data[i, j] == '>':
            available_moves = ((0, 1),)
        else:
            available_moves = ((1, 0), (-1, 0), (0, 1), (0, -1))

        new_moves = []
        for di, dj in available_moves:
            if 0 <= i + di < data.shape[0] and 0 <= j + dj < data.shape[1] \
                    and not data[i + di, j + dj] == '#' \
                    and not (di == 1 and data[i + di, j + dj] == '<') \
                    and not (di == -1 and data[i + di, j + dj] == '>') \
                    and not (dj == 1 and data[i + di, j + dj] == '^') \
                    and not (dj == -1 and data[i + di, j + dj] == 'v') \
                    and not (i + di, j + dj) in visited:
                new_moves.append((i + di, j + dj, visited.union({(i + di, j + dj)})))
        return new_moves

    def move_part2(i, j, visited):
        if i == data.shape[0] - 1:
            return

        new_moves = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= i + di < data.shape[0] and 0 <= j + dj < data.shape[1] \
                    and not data[i + di, j + dj] == '#' \
                    and not (i + di, j + dj) in visited:
                new_moves.append((i + di, j + dj, visited.union({(i + di, j + dj)})))
        print(len(new_moves))
        return new_moves

    def get_max_path_lengths(j_start, move):
        to_check = [(0, j_start, {(0, j_start)})]
        max_path_lengths = 0
        while to_check:
            i, j, visited = to_check.pop()
            if i == data.shape[0] - 1:
                max_path_lengths = max(max_path_lengths, len(visited) - 1)
            else:
                to_check += move(i, j, visited)
        return max_path_lengths

    print(get_max_path_lengths(np.argwhere(data[0] == '.')[0, 0], move_part1))
    # print(get_max_path_lengths(np.argwhere(data[0] == '.')[0, 0], move_part2))


if __name__ == '__main__':
    utils.time_me(day23)()
