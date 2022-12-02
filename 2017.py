import numpy as np

import utils

np.set_printoptions(linewidth=300)

YEAR = 2017


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        data = [int(d) for d in inp.readline()[:-1]]
        count1 = count2 = 0
        n = len(data)
        for i in range(n):
            if data[i] == data[(i + 1) % n]:
                count1 += data[i]
            if data[i] == data[(i + n // 2) % n]:
                count2 += data[i]
        print(count1)
        print(count2)


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        count1 = count2 = 0
        for line in inp:
            nums = np.array([int(d) for d in line.split()], dtype=int)
            count1 += np.max(nums) - np.min(nums)
            foo = np.tensordot(nums, 1 / nums, axes=0)
            count2 += int(foo[np.bitwise_and(foo == foo.astype(int), foo != 1)])
        print(count1)
        print(count2)


if __name__ == '__main__':
    day2()
