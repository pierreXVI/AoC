import utils

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


if __name__ == '__main__':
    day1()
