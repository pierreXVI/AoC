import collections
import heapq
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
        draft = [int(d) for d in inp.readline().split(',')]

        ranks, scores = [], []
        while True:
            inp.readline()
            grid = np.array([[int(d) for d in inp.readline().split()] for _ in range(5)], dtype=int)
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
                new_grid1 = np.zeros((max(x1 + 1, x2 + 1, grid1.shape[0]), max(y1 + 1, y2 + 1, grid1.shape[1])),
                                     dtype=int)
                new_grid2 = np.zeros((max(x1 + 1, x2 + 1, grid1.shape[0]), max(y1 + 1, y2 + 1, grid1.shape[1])),
                                     dtype=int)
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


def day6():
    with open(utils.get_input(YEAR, 6)) as inp:
        seed = [int(d) for d in inp.readline().split(',')]

    state = np.zeros((9,), dtype=int)
    for s in seed:
        state[s] += 1

    def iterate():
        births = state[0]
        state[:8] = state[1:]
        state[6] += births
        state[8] = births

    for n in range(80):
        iterate()
    print(sum(state))
    for n in range(256 - 80):
        iterate()
    print(sum(state))


def day7():
    with open(utils.get_input(YEAR, 7)) as inp:
        state = np.array([int(d) for d in inp.readline().split(',')], dtype=int)

    def cost2(x):
        return x * (x + 1) // 2

    print(np.sum(np.abs(state - int(np.median(state)))))
    print(np.sum(cost2(np.abs(state - int(np.mean(state))))))


def day8():
    with open(utils.get_input(YEAR, 8)) as inp:
        count_1478 = count = 0
        for line in inp:
            patterns, output = line.split('|')
            patterns = [''.join(sorted(list(pattern))) for pattern in patterns.split()]
            remainder = []
            encode, decode = {}, {}
            for pattern in patterns:
                if len(pattern) == 2:
                    encode['1'] = pattern
                    decode[pattern] = '1'
                elif len(pattern) == 4:
                    encode['4'] = pattern
                    decode[pattern] = '4'
                elif len(pattern) == 3:
                    encode['7'] = pattern
                    decode[pattern] = '7'
                elif len(pattern) == 7:
                    encode['8'] = pattern
                    decode[pattern] = '8'
                else:
                    remainder.append(pattern)
            for pattern in remainder:
                if len(pattern) == 5:
                    if encode['1'][0] in pattern and encode['1'][1] in pattern:
                        encode['3'] = pattern
                        decode[pattern] = '3'
                    elif len(set(pattern).intersection(set(encode['4']))) == 3:
                        encode['5'] = pattern
                        decode[pattern] = '5'
                    else:
                        encode['2'] = pattern
                        decode[pattern] = '2'
                else:
                    if encode['1'][0] not in pattern or encode['1'][1] not in pattern:
                        encode['6'] = pattern
                        decode[pattern] = '6'
                    elif len(set(pattern).intersection(set(encode['4']))) == 4:
                        encode['9'] = pattern
                        decode[pattern] = '9'
                    else:
                        encode['0'] = pattern
                        decode[pattern] = '0'

            output = output.split()
            for i in range(len(output)):
                output[i] = decode[''.join(sorted(list(output[i])))]
                if output[i] in '1478':
                    count_1478 += 1
            count += int(''.join(output))

        print(count_1478)
        print(count)


def day9():
    with open(utils.get_input(YEAR, 9)) as inp:
        map_cave = np.array([[int(d) for d in line[:-1]] for line in inp], dtype=int)
    n_x, n_y = map_cave.shape
    map_dx = np.diff(map_cave, axis=0)
    map_dy = np.diff(map_cave, axis=1)
    low_points = []
    for i in range(n_x):
        for j in range(n_y):
            if i == 0:
                if not map_dx[i, j] > 0:
                    continue
            elif i == n_x - 1:
                if not map_dx[i - 1, j] < 0:
                    continue
            elif not map_dx[i - 1, j] < 0 < map_dx[i, j]:
                continue
            if j == 0:
                if not map_dy[i, j] > 0:
                    continue
            elif j == n_y - 1:
                if not map_dy[i, j - 1] < 0:
                    continue
            elif not map_dy[i, j - 1] < 0 < map_dy[i, j]:
                continue

            low_points.append((i, j))

    def scan_bassin(_i, _j, _bassin=None):
        if _bassin is None:
            _bassin = np.zeros_like(map_cave, dtype=bool)
        if not (0 <= _i < n_x and 0 <= _j < n_y) or _bassin[_i, _j] or map_cave[_i, _j] == 9:
            return 0
        _bassin[_i, _j] = True
        return 1 + scan_bassin(_i - 1, _j, _bassin) + scan_bassin(_i, _j - 1, _bassin) \
            + scan_bassin(_i + 1, _j, _bassin) + scan_bassin(_i, _j + 1, _bassin)

    level = 0
    sizes = [0, 0, 0]
    for low_point in low_points:
        level += map_cave[low_point] + 1

        s = scan_bassin(*low_point)
        if s > sizes[2]:
            sizes = [sizes[1], sizes[2], s]
        elif s > sizes[1]:
            sizes = [sizes[1], s, sizes[2]]
        elif s > sizes[0]:
            sizes = [s, sizes[1], sizes[2]]

    print(level)
    print(sizes[0] * sizes[1] * sizes[2])


def day10():
    match = {')': '(', ']': '[', '}': '{', '>': '<'}
    costs = {')': 3, ']': 57, '}': 1197, '>': 25137}
    score = {'(': 1, '[': 2, '{': 3, '<': 4}

    with open(utils.get_input(YEAR, 10)) as inp:
        error_score = 0
        completion_scores = []
        for line in inp:
            word = ''
            illegal = False
            for c in line[:-1]:
                if c in match:
                    if word[-1] == match[c]:
                        word = word[:-1]
                    else:
                        error_score += costs[c]
                        illegal = True
                        break
                else:
                    word = word + c
            if illegal:
                continue

            count = 0
            for c in word[::-1]:
                count = 5 * count + score[c]
            completion_scores.append(count)

        completion_scores.sort()
        print(error_score)
        print(completion_scores[len(completion_scores) // 2])


def day11():
    with open(utils.get_input(YEAR, 11)) as inp:
        levels = np.array([[int(d) for d in line[:-1]] for line in inp], dtype=int)

    def resolve_step(state):
        flash = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] > 9:
                    state[max(0, i - 1):i + 2, max(0, j - 1):j + 2] += 1
                    state[i, j] = -10
                    flash += 1
        if flash > 0:
            flash += resolve_step(state)
        state[state < 0] = 0
        return flash

    count = step = 0
    synchronised = -1
    for step in range(100):
        levels += 1
        inc = resolve_step(levels)
        count += inc
        if synchronised < 0 and inc == levels.size:
            synchronised = step + 1
    print(count)
    while synchronised < 0:
        step += 1
        levels += 1
        inc = resolve_step(levels)
        if inc == levels.size:
            synchronised = step + 1
    print(synchronised)


def day12():
    with open(utils.get_input(YEAR, 12)) as inp:
        cave = collections.defaultdict(list)
        for line in inp:
            v1, v2 = line[:-1].split('-')
            if v2 != 'start':
                cave[v1].append(v2)
            if v1 != 'start':
                cave[v2].append(v1)

    def explore(loc, tree, path, twice=False):
        if loc == 'end':
            return 1

        n_paths = 0
        for o in tree[loc]:
            if o.isupper():
                n_paths += explore(o, tree, path + [loc], twice)
            else:
                if o in path:
                    if not twice:
                        n_paths += explore(o, tree, path + [loc], True)
                else:
                    n_paths += explore(o, tree, path + [loc], twice)
        return n_paths

    print(explore('start', cave, [], True))
    print(explore('start', cave, [], False))


def day13():
    with open(utils.get_input(YEAR, 13)) as inp:
        paper = np.zeros((0, 0), dtype=bool)

        for line in inp:
            if line == '\n':
                break
            x, y = [int(d) for d in line[:-1].split(',')]
            if x >= paper.shape[0] or y > paper.shape[1]:
                new_paper = np.zeros((max(x + 1, paper.shape[0]), max(y + 1, paper.shape[1])), dtype=bool)
                new_paper[:paper.shape[0], :paper.shape[1]] = paper
                paper = new_paper
            paper[x, y] = True

        first = True
        for line in inp:
            axis, val = line.split()[2].split('=')
            val = int(val)
            if axis == 'x':
                left, right = paper[:val, :], np.flip(paper[val + 1:, :], axis=0)
                paper = np.zeros((max(left.shape[0], right.shape[0]), paper.shape[1]), dtype=bool)
                paper[:min(paper.shape[0], left.shape[0]), :] += left[:min(paper.shape[0], left.shape[0]), :]
                paper[-min(paper.shape[0], right.shape[0]):, :] += right[-min(paper.shape[0], right.shape[0]):, :]
            elif axis == 'y':
                left, right = paper[:, :val], np.flip(paper[:, val + 1:], axis=1)
                paper = np.zeros((paper.shape[0], max(left.shape[1], right.shape[1])), dtype=bool)
                paper[:, :min(paper.shape[1], left.shape[1])] += left[:, :min(paper.shape[1], left.shape[1])]
                paper[:, -min(paper.shape[1], right.shape[1]):] += right[:, -min(paper.shape[1], right.shape[1]):]

            if first:
                print(np.sum(paper))
                first = False

        for line in paper.T:
            for c in line:
                print('##' if c else '  ', end='')
            print()


def day14():
    with open(utils.get_input(YEAR, 14)) as inp:
        start = inp.readline()[:-1]
        inp.readline()

        rules = {}
        for line in inp:
            foo, bar = line[:-1].split(' -> ')
            rules[foo] = bar

    polymer = collections.defaultdict(int)
    for i in range(len(start) - 1):
        polymer[start[i:i + 2]] += 1

    def get_result(values):
        freq = collections.defaultdict(int)
        for _c in polymer:
            freq[_c[0]] += values[_c]
            freq[_c[1]] += values[_c]
        for _c in freq:
            freq[_c] = freq[_c] // 2 + freq[_c] % 2
        freq_sort = sorted(freq, key=freq.get)
        return freq[freq_sort[-1]] - freq[freq_sort[0]]

    for _ in range(10):
        new_polymer = collections.defaultdict(int)
        for c in polymer:
            new_polymer[c[0] + rules[c]] += polymer[c]
            new_polymer[rules[c] + c[1]] += polymer[c]
        polymer = new_polymer
    print(get_result(polymer))
    for _ in range(30):
        new_polymer = collections.defaultdict(int)
        for c in polymer:
            new_polymer[c[0] + rules[c]] += polymer[c]
            new_polymer[rules[c] + c[1]] += polymer[c]
        polymer = new_polymer
    print(get_result(polymer))


def day15():
    with open(utils.get_input(YEAR, 15)) as inp:
        data = np.array([[int(d) for d in line[:-1]] for line in inp], dtype=int)

    def djikstra(costs, start, end):
        score = {(x, y): np.inf for x in range(costs.shape[0]) for y in range(costs.shape[1])}
        score[start] = 0
        done = set()
        hq = [(0, start[0], start[1]), ]

        while hq:
            c, x, y = heapq.heappop(hq)
            # assert c == score[x, y]
            # assert (x, y) not in done
            if (x, y) == end:
                break
            done.add((x, y))
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= x + dx < costs.shape[0] and 0 <= y + dy < costs.shape[1] and (x + dx, y + dy) not in done:
                    val = c + costs[x + dx, y + dy]
                    if val < score[x + dx, y + dy]:
                        score[x + dx, y + dy] = val
                        heapq.heappush(hq, (val, x + dx, y + dy))

        return score[end]

    print(djikstra(data, (0, 0), (data.shape[0] - 1, data.shape[1] - 1)))

    nx, ny = data.shape
    new_data = np.zeros((5 * nx, 5 * ny), dtype=int)
    for i in range(5):
        for j in range(5):
            new_data[i * nx:(i + 1) * nx, j * ny:(j + 1) * ny] = data + i + j
    data = ((new_data - 1) % 9) + 1
    print(djikstra(data, (0, 0), (data.shape[0] - 1, data.shape[1] - 1)))


def day16():
    with open(utils.get_input(YEAR, 16)) as inp:
        data = inp.readline()[:-1]
        data = bin(int(data, 16))[2:].zfill(4 * len(data))

    def parse_msg(msg):
        version = int(msg[0:3], 2)
        type_id = int(msg[3:6], 2)
        remainder = msg[6:]

        if type_id == 4:
            value = ''
            while True:
                value += remainder[1:5]
                if remainder[0] == '0':
                    remainder = remainder[5:]
                    break
                remainder = remainder[5:]
            value = int(value, 2)

        else:
            values = []
            if remainder[0] == '0':
                bits_packet = remainder[16:16 + int(remainder[1:16], 2)]
                while len(bits_packet) > 8:
                    subversion, bits_packet, value = parse_msg(bits_packet)
                    version += subversion
                    values.append(value)
                remainder = remainder[16 + int(remainder[1:16], 2):]
            else:
                n_subs = int(remainder[1:12], 2)
                remainder = remainder[12:]
                for _ in range(n_subs):
                    subversion, remainder, value = parse_msg(remainder)
                    version += subversion
                    values.append(value)

            if type_id == 0:
                value = sum(values)
            elif type_id == 1:
                value = np.prod(values)
            elif type_id == 2:
                value = min(values)
            elif type_id == 3:
                value = max(values)
            elif type_id == 5:
                value = 1 if values[0] > values[1] else 0
            elif type_id == 6:
                value = 1 if values[0] < values[1] else 0
            elif type_id == 7:
                value = 1 if values[0] == values[1] else 0

        return version, remainder, value

    total = 0
    result = ''
    while len(data) > 8:
        ver, data, val = parse_msg(data)
        total += ver
        result += str(val)
    print(total)
    print(result)


def day17():
    with open(utils.get_input(YEAR, 17)) as inp:
        data = inp.readline().split(':')[1].split(',')
        x_min, x_max = [int(d) for d in data[0].split('=')[1].split('..')]
        y_min, y_max = [int(d) for d in data[1].split('=')[1].split('..')]

    def shoot(v_x, v_y):
        x, y = 0, 0
        while x <= x_max and y >= y_min:
            x += v_x
            y += v_y
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
            v_x = max(0, v_x - 1)
            v_y -= 1
        return False

    print(-y_min * (-y_min - 1) // 2)

    count = 0
    for i in range(x_max + 1):
        for j in range(y_min - 1, -y_min):
            if shoot(i, j):
                count += 1
            pass
    print(count)


def day18():
    class SnailfishNumber:
        def __init__(self, pair, lvl=0):
            if type(pair) == int:
                self.left = self.right = pair
                self.regular = True
            else:
                self.left = pair[0] if type(pair[0]) is SnailfishNumber else SnailfishNumber(pair[0], lvl + 1)
                self.right = pair[1] if type(pair[1]) is SnailfishNumber else SnailfishNumber(pair[1], lvl + 1)
                self.regular = False

            self.lvl = lvl
            if self.lvl == 0:
                self.update_lvl()
                self.reduce()

        def update_lvl(self, lvl=0):
            self.lvl = lvl
            if self.regular:
                return
            self.left.update_lvl(self.lvl + 1)
            self.right.update_lvl(self.lvl + 1)

        def split(self):
            if self.regular:
                if self.left >= 10:
                    self.left = SnailfishNumber(self.left // 2, self.lvl + 1)
                    self.right = SnailfishNumber(self.right - self.right // 2, self.lvl + 1)
                    self.regular = False
                    return True
                return False
            else:
                return self.left.split() or self.right.split()

        def explode(self):
            if self.regular:
                return ()

            if self.lvl > 3:
                my_explode = (self.left, self.right)
                self.left = self.right = 0
                self.regular = True
                return my_explode

            explode_l = self.left.explode()
            if explode_l:
                self.right.first_add(explode_l[1], True)
                return explode_l[0], 0
            explode_r = self.right.explode()
            if explode_r:
                self.left.first_add(explode_r[0], False)
                return 0, explode_r[1]

            return ()

        def reduce(self):
            while self.explode() or self.split():
                pass

        def first_add(self, val, left):
            if self.regular:
                self.left = self.right = self.left + val
            else:
                which = self.left if left else self.right
                which.first_add(val, left)

        def magnitude(self):
            if self.regular:
                return self.left
            return 3 * self.left.magnitude() + 2 * self.right.magnitude()

        def __add__(self, other):
            if self.regular and type(other) == int:
                return self.left + other
            return SnailfishNumber([self.copy(), other.copy()])

        def __radd__(self, other):
            return self + other

        def __str__(self):
            if self.regular:
                return str(self.left)
            else:
                return '[' + str(self.left) + ',' + str(self.right) + ']'

        def to_list(self):
            if self.regular:
                return self.left
            return [self.left.to_list(), self.right.to_list()]

        def copy(self):
            return SnailfishNumber(self.to_list())

    numbers = []
    with open(utils.get_input(YEAR, 18)) as inp:
        for line in inp:
            numbers.append(SnailfishNumber(eval(line[:-1])))

    total_sum = numbers[0].copy()
    for i in range(1, len(numbers)):
        total_sum = total_sum + numbers[i]
    print(total_sum.magnitude())

    best = 0
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j:
                mag = (numbers[i] + numbers[j]).magnitude()
                if mag > best:
                    best = mag
    print(best)


if __name__ == '__main__':
    day18()
