import collections
import re

import numpy as np

import utils

YEAR = 2020
np.set_printoptions(linewidth=300)


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        data = {}
        res1 = res2 = 0
        for line in inp:
            i = int(line)
            if 2020 - i in data:
                res1 = i * (2020 - i)
            for j in data:
                if i in data[j]:
                    res2 = i * j * (2020 - i - j)
                else:
                    data[j].append(2020 - i - j)
            data[i] = []
        print(res1)
        print(res2)


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        count1 = count2 = 0
        for line in inp:
            foo = line.split()
            i, j = [int(d) - 1 for d in foo[0].split('-')]
            if i + 1 <= foo[2].count(foo[1][0]) <= j + 1:
                count1 += 1
            if (len(foo[2]) > i and foo[2][i] == foo[1][0]) != (len(foo[2]) > j and foo[2][j] == foo[1][0]):
                count2 += 1
        print(count1)
        print(count2)


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        data = np.array([[c == '#' for c in line[:-1]] for line in inp], dtype=bool)
        print(sum(data[i, (3 * i) % data.shape[1]] for i in range(data.shape[0])))

        res = 1
        for k in ((1, 1), (1, 3), (1, 5), (1, 7), (2, 1)):
            res *= sum(data[k[0] * i, (k[1] * i) % data.shape[1]] for i in range(data.shape[0] // k[0]))
        print(res)


def day4():
    def valid_passport(byr, iyr, eyr, hgt, hcl, ecl, pid, cid=''):
        if not (re.match('[0-9]{4}$', byr) and 1920 <= int(byr) <= 2002):
            return False
        if not (re.match('[0-9]{4}$', iyr) and 2010 <= int(iyr) <= 2020):
            return False
        if not (re.match('[0-9]{4}$', eyr) and 2020 <= int(eyr) <= 2030):
            return False
        if hgt[-2:] == 'cm':
            if not 150 <= int(hgt[:-2]) <= 193:
                return False
        elif hgt[-2:] == 'in':
            if not 56 <= int(hgt[:-2]) <= 76:
                return False
        else:
            return False
        if not re.match('#[0-9a-f]{6}$', hcl):
            return False
        if ecl not in ('amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth'):
            return False
        if not re.match('[0-9]{9}$', pid):
            return False
        return True

    with open(utils.get_input(YEAR, 4)) as inp:
        reset = True
        passport = {}
        count1 = count2 = 0
        for line in inp:
            if reset:
                try:
                    if valid_passport(**passport):
                        count2 += 1
                    count1 += 1
                except TypeError:
                    pass

                passport = {}
                reset = False
            if line == '\n':
                reset = True
                continue
            for p in line.split():
                k, d = p.split(':')
                passport[k] = d
        try:
            if valid_passport(**passport):
                count2 += 1
            count1 += 1
        except TypeError:
            pass
        print(count1)
        print(count2)


def day5():
    with open(utils.get_input(YEAR, 5)) as inp:
        res = 0
        data = set()
        for line in inp:
            row = int(line[:7].replace('B', '1').replace('F', '0'), 2)
            col = int(line[7:10].replace('R', '1').replace('L', '0'), 2)
            res = max(res, row * 8 + col)
            data.add(row * 8 + col)
        print(res)

        data = sorted(data)
        for i in range(len(data) - 1):
            if data[i] + 1 != data[i + 1]:
                print(data[i] + 1)


def day6():
    with open(utils.get_input(YEAR, 6)) as inp:
        count1 = count2 = 0
        reset = first = True
        union = intersection = set()
        for line in inp:
            if reset:
                count1 += len(union)
                count2 += len(intersection)
                union = set()
                intersection = set()
                reset = False
                first = True
            if line == '\n':
                reset = True
                continue
            new_set = set(c for c in line[:-1])
            union.update(new_set)
            if first:
                intersection = new_set
                first = False
            else:
                intersection = intersection.intersection(new_set)

        count1 += len(union)
        count2 += len(intersection)
        print(count1)
        print(count2)


def day7():
    with open(utils.get_input(YEAR, 7)) as inp:
        rules = collections.defaultdict(set)
        rrules = collections.defaultdict(set)
        for line in inp:
            foo = line.split(' bags contain ')
            for bar in foo[1].split(','):
                if bar == 'no other bags.\n':
                    continue
                bar = bar.split()
                n = int(bar[0])
                c = ' '.join(bar[1:-1])
                rules[foo[0]].add((c, n))
                rrules[c].add(foo[0])

    colors = {'shiny gold'}
    done = set()
    while colors:
        new_colors = set()
        for c in colors:
            done.add(c)
            rrules[c] -= done
            for cc in rrules[c]:
                new_colors.add(cc)
        colors = new_colors
    print(len(done) - 1)

    def get_cost(_c):
        out = 1
        for _cc in rules[_c]:
            out += _cc[1] * get_cost(_cc[0])
        return out

    print(get_cost('shiny gold') - 1)


def day8():
    def run(program):
        acc = i = 0
        table = [False] * len(program)
        while True:
            if i >= len(program):
                return acc, True
            if table[i]:
                return acc, False
            table[i] = True
            if program[i][0] == 'acc':
                acc += program[i][1]
                i += 1
            elif program[i][0] == 'jmp':
                i += program[i][1]
            elif program[i][0] == 'nop':
                i += 1

    with open(utils.get_input(YEAR, 8)) as inp:
        instructions = []
        for line in inp:
            act, val = line.split()
            instructions.append([act, int(val)])

    print(run(instructions)[0])

    for p in instructions:
        if p[0] == 'jmp' or p[0] == 'nop':
            p[0] = 'jmp' if p[0] == 'nop' else 'nop'
            out = run(instructions)
            if out[1]:
                print(out[0])
            p[0] = 'jmp' if p[0] == 'nop' else 'nop'


def day9():
    with open(utils.get_input(YEAR, 9)) as inp:
        base = [int(inp.readline()) for _ in range(25)]
        for line in inp:
            val = int(line)
            found = False
            for i in range(len(base)):
                if val - base[i] in base[:i] + base[i + 1:]:
                    found = True
                    break
            if not found:
                break
            base = base[1:] + [val]

        print(val)

    with open(utils.get_input(YEAR, 9)) as inp:
        base = []
        for line in inp:
            if len(base) > 1 and sum(base) == val:
                break
            base.append(int(line))
            while base and sum(base) > val:
                base = base[1:]

        print(max(base) + min(base))


def day10():
    with open(utils.get_input(YEAR, 10)) as inp:
        data = np.array([int(line) for line in inp])

    data.sort()
    diff = np.diff(np.array([0, *data, data[-1] + 3]))
    print(len(diff[diff == 1]) * len(diff[diff == 3]))

    count = 1
    n = 0
    for i in range(len(diff)):
        if diff[i] == 1:
            n += 1
        else:
            if n > 1:
                foo = 2 ** (n - 1) + (n - 1) * n // 2 - n * (n - 3) - 3
                count *= foo
            n = 0
    print(count)


def day11():
    with open(utils.get_input(YEAR, 11)) as inp:
        state0 = np.array([[0 if c == '.' else 1 if c == '#' else 2 for c in line[:-1]] for line in inp], dtype=int)

    def update1(grid):
        new_grid = grid.copy()
        change = False
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 2:
                    if np.sum(grid[max(0, i - 1):i + 2, max(0, j - 1):j + 2] == 1) == 0:
                        new_grid[i, j] = 1
                        change = True
                elif grid[i, j] == 1:
                    if np.sum(grid[max(0, i - 1):i + 2, max(0, j - 1):j + 2] == 1) > 4:
                        new_grid[i, j] = 2
                        change = True
        return new_grid, change

    state = state0.copy()
    run = True
    while run:
        state, run = update1(state)
    print(np.sum(state == 1))

    def update2(grid):
        new_grid = grid.copy()
        change = False
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 2:
                    k = i - 1
                    while k >= 0 and grid[k, j] == 0:
                        k -= 1
                    if k >= 0 and grid[k, j] == 1:
                        continue
                    k = j - 1
                    while k >= 0 and grid[i, k] == 0:
                        k -= 1
                    if k >= 0 and grid[i, k] == 1:
                        continue
                    k = i + 1
                    while k < grid.shape[0] and grid[k, j] == 0:
                        k += 1
                    if k < grid.shape[0] and grid[k, j] == 1:
                        continue
                    k = j + 1
                    while k < grid.shape[1] and grid[i, k] == 0:
                        k += 1
                    if k < grid.shape[1] and grid[i, k] == 1:
                        continue

                    k = 1
                    while i - k >= 0 and j - k >= 0 and grid[i - k, j - k] == 0:
                        k += 1
                    if i - k >= 0 and j - k >= 0 and grid[i - k, j - k] == 1:
                        continue
                    k = 1
                    while i + k < grid.shape[0] and j + k < grid.shape[1] and grid[i + k, j + k] == 0:
                        k += 1
                    if i + k < grid.shape[0] and j + k < grid.shape[1] and grid[i + k, j + k] == 1:
                        continue
                    k = 1
                    while i + k < grid.shape[0] and j - k >= 0 and grid[i + k, j - k] == 0:
                        k += 1
                    if i + k < grid.shape[0] and j - k >= 0 and grid[i + k, j - k] == 1:
                        continue
                    k = 1
                    while i - k >= 0 and j + k < grid.shape[1] and grid[i - k, j + k] == 0:
                        k += 1
                    if i - k >= 0 and j + k < grid.shape[1] and grid[i - k, j + k] == 1:
                        continue

                    new_grid[i, j] = 1
                    change = True
                elif grid[i, j] == 1:
                    count = 0

                    k = i - 1
                    while k >= 0 and grid[k, j] == 0:
                        k -= 1
                    if k >= 0 and grid[k, j] == 1:
                        count += 1
                    k = j - 1
                    while k >= 0 and grid[i, k] == 0:
                        k -= 1
                    if k >= 0 and grid[i, k] == 1:
                        count += 1
                    k = i + 1
                    while k < grid.shape[0] and grid[k, j] == 0:
                        k += 1
                    if k < grid.shape[0] and grid[k, j] == 1:
                        count += 1
                    k = j + 1
                    while k < grid.shape[1] and grid[i, k] == 0:
                        k += 1
                    if k < grid.shape[1] and grid[i, k] == 1:
                        count += 1

                    k = 1
                    while i - k >= 0 and j - k >= 0 and grid[i - k, j - k] == 0:
                        k += 1
                    if i - k >= 0 and j - k >= 0 and grid[i - k, j - k] == 1:
                        count += 1
                    k = 1
                    while i + k < grid.shape[0] and j + k < grid.shape[1] and grid[i + k, j + k] == 0:
                        k += 1
                    if i + k < grid.shape[0] and j + k < grid.shape[1] and grid[i + k, j + k] == 1:
                        count += 1
                    k = 1
                    while i + k < grid.shape[0] and j - k >= 0 and grid[i + k, j - k] == 0:
                        k += 1
                    if i + k < grid.shape[0] and j - k >= 0 and grid[i + k, j - k] == 1:
                        count += 1
                    k = 1
                    while i - k >= 0 and j + k < grid.shape[1] and grid[i - k, j + k] == 0:
                        k += 1
                    if i - k >= 0 and j + k < grid.shape[1] and grid[i - k, j + k] == 1:
                        count += 1

                    if count > 4:
                        new_grid[i, j] = 2
                        change = True
        return new_grid, change

    state = state0.copy()
    run = True
    iii = 0
    while run:
        iii += 1
        state, run = update2(state)
    print(np.sum(state == 1))


def day12():
    with open(utils.get_input(YEAR, 12)) as inp:
        alpha1 = x1 = y1 = x2 = y2 = 0
        wx = 10
        wy = 1
        for line in inp:
            val = int(line[1:])
            if line[0] == 'N':
                y1 += val
                wy += val
            elif line[0] == 'S':
                y1 -= val
                wy -= val
            elif line[0] == 'E':
                x1 += val
                wx += val
            elif line[0] == 'W':
                x1 -= val
                wx -= val
            elif line[0] == 'L':
                alpha1 += val
                foo = (val // 90) % 4
                if foo == 1:
                    wx, wy = -wy, wx
                elif foo == 2:
                    wx, wy = -wx, -wy
                elif foo == 3:
                    wx, wy = wy, -wx
            elif line[0] == 'R':
                alpha1 -= val
                foo = (val // 90) % 4
                if foo == 1:
                    wx, wy = wy, -wx
                elif foo == 2:
                    wx, wy = -wx, -wy
                elif foo == 3:
                    wx, wy = -wy, wx
            elif line[0] == 'F':
                foo = (alpha1 // 90) % 4
                if foo == 0:
                    x1 += val
                elif foo == 1:
                    y1 += val
                elif foo == 2:
                    x1 -= val
                elif foo == 3:
                    y1 -= val
                x2 += val * wx
                y2 += val * wy
        print(abs(x1) + abs(y1))
        print(abs(x2) + abs(y2))


def day13():
    with open(utils.get_input(YEAR, 13)) as inp:
        start = int(inp.readline())
        data = inp.readline().split(',')
        table = np.array([int(d) for d in data if d != 'x'])
        rank = np.array([i for i in range(len(data)) if data[i] != 'x'])
        wait = table - start % table
        best = np.argmin(wait)

        # print(wait[best] * table[best])

        def dioph(a, b):
            q, r = divmod(a, b)
            if r == 0:
                return 0, b
            else:
                x, y = dioph(b, r)
                return y, x - q * y

        for k in range(100):
            print(k, (8 + k * 17) * 13 - (6 + k * 13) * 17)

        # table[i] * n[i] = t + rank[i]
        # table[j] * n[j] = t + rank[j]

        # table[j] * a[i, j] - table[i] * b[i, j] = rank[j] - rank[i]
        # ==> n[j] = a[i, j] + k * table[i]
        # ==> n[i] = b[i, j] + k * table[j]

        # 17 * n0 = t
        # 13 * n1 = t + 2
        # 19 * n2 = t + 3
        #
        # 13 * n1 - 17 * n0 = 2


def day13():
    def euclide(a, b):
        r, u, v, rp, up, vp = a, 1, 0, b, 0, 1
        while rp != 0:
            q = r // rp
            r, u, v, rp, up, vp = rp, up, vp, r - q * rp, u - q * up, v - q * vp
        return r, u, v

    with open(utils.get_input(YEAR, 13)) as inp:
        start = int(inp.readline())
        line = inp.readline().split(',')
        data = np.array([int(d) for d in line if d != 'x'])
        rank = np.array([i for i in range(len(line)) if line[i] != 'x'])

    p = np.prod(data)
    x = 0
    for i in range(len(data)):
        n = p // data[i]
        x -= rank[i] * euclide(n, data[i])[1] * n
    while x < 0:
        x += p
    while True:
        if x - p < 0:
            break
        x -= p
    print(x)


if __name__ == '__main__':
    day13()
