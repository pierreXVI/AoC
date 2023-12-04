import collections

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
        part1 = 0
        part2 = 0
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


if __name__ == '__main__':
    day4()
