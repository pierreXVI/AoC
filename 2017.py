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


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        n_max = int(inp.readline())
        i = 1
        while i * i < n_max:
            i += 2
        grid = np.zeros((i, i), dtype=int)
        alpha = 1 + 0j
        loc = (1 + 1j) * (i // 2)
        length = 1
        grid[int(loc.real), int(loc.imag)] = 1
        new_grid = grid.copy()

        n = 1
        while True:
            for i in range(length):
                n += 1
                loc += alpha
                x, y = int(loc.real), int(loc.imag)
                new_grid[x, y] = np.sum(new_grid[max(0, x - 1):x + 2, max(0, y - 1):y + 2])
                grid[x, y] = n
                if n == n_max:
                    break
            if n == n_max:
                break
            alpha *= 1j
            if alpha == alpha.real:
                length += 1
        loc -= (1 + 1j) * (i // 2)
        print(int(abs(loc.real) + abs(loc.imag)))
        print(min(new_grid[new_grid > n_max]))


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        count1 = count2 = 0
        for line in inp:
            words1 = set()
            words2 = set()
            flag1 = flag2 = True
            for w in line.split():
                if w in words1:
                    flag1 = False
                    break
                words1.add(w)

                w = ''.join(sorted(list(w)))
                if w in words2:
                    flag2 = False
                    break
                words2.add(w)

            if flag1:
                count1 += 1
                if flag2:
                    count2 += 1
        print(count1)
        print(count2)


if __name__ == '__main__':
    day4()
