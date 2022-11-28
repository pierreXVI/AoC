import numpy as np
import re
import utils
import collections

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


if __name__ == '__main__':
    day10()
