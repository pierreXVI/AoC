import numpy as np

import utils

YEAR = 2019


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        count1 = count2 = 0
        for line in inp:
            val = int(line) // 3 - 2
            count1 += val
            while val > 0:
                count2 += val
                val = val // 3 - 2

        print(count1)
        print(count2)


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        tape = [int(d) for d in inp.readline().split(',')]

        def run(program, noun, verb):
            head = 0
            program = program.copy()
            program[1] = noun
            program[2] = verb
            while program[head] != 99:
                if program[head] == 1:
                    program[program[head + 3]] = program[program[head + 1]] + program[program[head + 2]]
                elif program[head] == 2:
                    program[program[head + 3]] = program[program[head + 1]] * program[program[head + 2]]
                head += 4
            return program[0]

        print(run(tape, 12, 2))

        for i in range(100):
            for j in range(100):
                if run(tape, i, j) == 19690720:
                    print(100 * i + j)
                    break


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        direction = {'U': 1j, 'D': -1j, 'L': -1, 'R': 1}

        wire1 = {}
        loc = n = 0
        for move in inp.readline().split(','):
            for _ in range(int(move[1:])):
                loc += direction[move[0]]
                n += 1
                if loc not in wire1:
                    wire1[loc] = n

        best1 = best2 = np.inf
        loc = n = 0
        for move in inp.readline().split(','):
            for _ in range(int(move[1:])):
                loc += direction[move[0]]
                n += 1
                if loc in wire1:
                    best1 = min(best1, int(abs(loc.real) + abs(loc.imag)))
                    best2 = min(best2, n + wire1[loc])
        print(best1)
        print(best2)


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        start, stop = [int(d) for d in inp.readline().split('-')]

        count1 = count2 = 0
        for n in range(start, stop + 1):
            diff = np.diff(np.array([int(d) for d in str(n)]))
            if (diff >= 0).all() and (diff == 0).any():
                count1 += 1

                ok = False
                for i in range(len(diff)):
                    if diff[i] == 0:
                        if (i == 0 or diff[i - 1] != 0) and (i == 4 or diff[i + 1] != 0):
                            ok = True
                            break
                if ok:
                    count2 += 1

        print(count1)
        print(count2)


if __name__ == '__main__':
    day4()
