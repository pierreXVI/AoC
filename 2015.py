import utils

YEAR = 2015


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        data = inp.readline().strip()
    print(len(data) - 2 * data.count(')'))

    i = j = 0
    while i != -1:
        i += 1 if data[j] == '(' else -1
        j += 1
    print(j)


def day2():
    paper = ribbon = 0
    with open(utils.get_input(YEAR, 2)) as inp:
        for line in inp:
            lengths = sorted([int(c) for c in line.split('x')])
            paper += 3 * (lengths[0] * lengths[1]) + 2 * (lengths[0] + lengths[1]) * lengths[2]
            ribbon += 2 * (lengths[0] + lengths[1]) + lengths[0] * lengths[1] * lengths[2]
    print(paper)
    print(ribbon)


if __name__ == '__main__':
    day2()
