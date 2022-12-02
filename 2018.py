import collections

import numpy as np

import utils

np.set_printoptions(linewidth=300)

YEAR = 2018


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        data = [int(line) for line in inp]

        count = 0
        val = None
        reached = set()
        for d in data:
            count += d
            if val is None and count in reached:
                val = count
            reached.add(count)
        print(count)

        while val is None:
            for d in data:
                count += d
                if val is None and count in reached:
                    val = count
                reached.add(count)
        print(val)


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        n2 = n3 = 0
        letters = []
        for line in inp:
            letters.append(np.array([ord(d) for d in line[:-1]]))
            count = collections.Counter(line[:-1])
            i2 = i3 = False
            for c in count:
                if not i2 and count[c] == 2:
                    i2 = True
                elif not i3 and count[c] == 3:
                    i3 = True
            if i2:
                n2 += 1
            if i3:
                n3 += 1
        print(n2 * n3)

        for i in range(len(letters)):
            for j in range(i + 1, len(letters)):
                foo = letters[i] - letters[j]
                if len(foo[foo != 0]) == 1:
                    print(''.join([chr(d) for d in letters[i][foo == 0]]))


if __name__ == '__main__':
    day2()
