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


if __name__ == '__main__':
    day1()
