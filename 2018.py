import utils

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


if __name__ == '__main__':
    day1()
