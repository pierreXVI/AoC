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


if __name__ == '__main__':
    day2()
