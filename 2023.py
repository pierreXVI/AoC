import collections
import functools
import math
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
    gears = collections.defaultdict(list)
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
        s = collections.defaultdict(list)
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
    def count_valid(_pipes, _count):
        count = 0
        indices = np.argwhere(_pipes == '?').flatten()
        for i in range(2 ** len(indices)):
            substitution = _pipes.copy()
            for j in range(len(indices)):
                substitution[indices[j]] = '#' if (i // 2 ** j) % 2 else '.'
            if [len(broken) for broken in ''.join(substitution).split('.') if broken] == _count:
                count += 1
        return count

    with open(utils.get_input(YEAR, 12)) as inp:
        part1 = 0
        for line in inp:
            pipes, data = line.strip().split()
            pipes = np.array(list(pipes))
            data = np.array(data.split(','), dtype=int).tolist()
            part1 += count_valid(pipes, data)
        print(part1)


if __name__ == '__main__':
    day12()
