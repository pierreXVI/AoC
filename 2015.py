import utils
import hashlib

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


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        x1 = y1 = 0
        x2 = [0, 0]
        y2 = [0, 0]
        which = 0
        visited1 = {(x1, y1)}
        visited2 = {(x2[which], y2[which])}
        for c in inp.readline():
            if c == '>':
                x1 += 1
                x2[which] += 1
            elif c == '<':
                x1 -= 1
                x2[which] -= 1
            if c == '^':
                y1 += 1
                y2[which] += 1
            if c == 'v':
                y1 -= 1
                y2[which] -= 1
            visited1.add((x1, y1))
            visited2.add((x2[which], y2[which]))
            which = 1 - which
        print(len(visited1))
        print(len(visited2))


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        key = inp.readline()[:-1]
    i = 0
    for n in (5, 6):
        prefix = '0' * n
        while hashlib.md5("{0}{1}".format(key, i).encode('utf-8')).hexdigest()[:n] != prefix:
            i += 1
        print(i)


#
#
# def day5():
#     nice_1 = nice_2 = 0
#
#     with open('input/5.txt') as inp:
#         for line in inp:
#             if not re.search(r'(ab|cd|pq|xy)', line) and re.search(r'(\w)\1', line) and len(
#                     re.findall(r'[aeiou]', line)) > 2:
#                 nice_1 += 1
#             if re.search(r'(\w).\1', line) and re.search(r'(\w{2}).*?\1', line):
#                 nice_2 += 1
#
#     print("There are {0} nice lines with rules 1 and {1} with rules 2".format(nice_1, nice_2))
#
#
# def day6(mode=1):
#     # x, y = [], []
#     grid = np.zeros((1000, 1000), dtype=int)
#     with open('input/6.txt') as inp:
#         for line in inp:
#             if line[:7] == 'turn on':
#                 action = 1
#                 line = line[8:-1]
#             elif line[:8] == 'turn off':
#                 action = -1
#                 line = line[9:-1]
#             else:
#                 action = 0
#                 line = line[7:-1]
#             x_range, y_range = line.split(' through ')
#             x1, y1 = [int(d) for d in x_range.split(',')]
#             x2, y2 = [int(d) for d in y_range.split(',')]
#             x1, x2 = min(x1, x2), max(x1, x2) + 1
#             y1, y2 = min(y1, y2), max(y1, y2) + 1
#             if mode == 1:
#                 grid[x1:x2, y1:y2] = 1 if action == 1 else 0 if action == -1 else 1 - grid[x1:x2, y1:y2]
#             else:
#                 grid[x1:x2, y1:y2] = grid[x1:x2, y1:y2] + (1 if action == 1 else -1 if action == -1 else 2)
#                 grid[grid < 0] = 0
#
#     print("The total brightness is {0}".format(np.sum(grid)))
#
#
# def day7():
#     pending = {}
#     with open('input/7.txt') as inp:
#         for line in inp:
#             oper, out = line[:-1].split(' -> ')
#             pending[out] = oper.split()
#             for i in range(len(pending[out])):
#                 try:
#                     pending[out][i] = int(pending[out][i])
#                 except ValueError:
#                     pass
#
#     connected = {}
#     while pending:
#         for wire in list(pending.keys()):
#             if len(pending[wire]) == 1:
#                 if type(pending[wire][0]) is int:
#                     connected[wire] = pending[wire][0]
#                     del pending[wire]
#                 elif pending[wire][0] in connected:
#                     connected[wire] = connected[pending[wire][0]]
#                     del pending[wire]
#             elif pending[wire][0] == 'NOT':
#                 if type(pending[wire][1]) is int:
#                     connected[wire] = ~pending[wire][1]
#                     del pending[wire]
#                 elif pending[wire][1] in connected:
#                     connected[wire] = ~connected[pending[wire][1]]
#                     del pending[wire]
#             else:
#                 a = b = None
#                 if type(pending[wire][0]) is int:
#                     a = pending[wire][0]
#                 elif pending[wire][0] in connected:
#                     a = connected[pending[wire][0]]
#                 if type(pending[wire][2]) is int:
#                     b = pending[wire][2]
#                 elif pending[wire][2] in connected:
#                     b = connected[pending[wire][2]]
#                 if a is not None and b is not None:
#                     if pending[wire][1] == 'OR':
#                         connected[wire] = a | b
#                     elif pending[wire][1] == 'AND':
#                         connected[wire] = a & b
#                     elif pending[wire][1] == 'RSHIFT':
#                         connected[wire] = a >> b
#                     elif pending[wire][1] == 'LSHIFT':
#                         connected[wire] = a << b
#                     del pending[wire]
#
#     print("Wire 'a' = {0}".format(connected['a']))
#
#     pending = {}
#     with open('input/7.txt') as inp:
#         for line in inp:
#             oper, out = line[:-1].split(' -> ')
#             pending[out] = oper.split()
#             for i in range(len(pending[out])):
#                 try:
#                     pending[out][i] = int(pending[out][i])
#                 except ValueError:
#                     pass
#
#     connected = {'b': connected['a']}
#     del pending['b']
#     while pending:
#         for wire in list(pending.keys()):
#             if len(pending[wire]) == 1:
#                 if type(pending[wire][0]) is int:
#                     connected[wire] = pending[wire][0]
#                     del pending[wire]
#                 elif pending[wire][0] in connected:
#                     connected[wire] = connected[pending[wire][0]]
#                     del pending[wire]
#             elif pending[wire][0] == 'NOT':
#                 if type(pending[wire][1]) is int:
#                     connected[wire] = ~pending[wire][1]
#                     del pending[wire]
#                 elif pending[wire][1] in connected:
#                     connected[wire] = ~connected[pending[wire][1]]
#                     del pending[wire]
#             else:
#                 a = b = None
#                 if type(pending[wire][0]) is int:
#                     a = pending[wire][0]
#                 elif pending[wire][0] in connected:
#                     a = connected[pending[wire][0]]
#                 if type(pending[wire][2]) is int:
#                     b = pending[wire][2]
#                 elif pending[wire][2] in connected:
#                     b = connected[pending[wire][2]]
#                 if a is not None and b is not None:
#                     if pending[wire][1] == 'OR':
#                         connected[wire] = a | b
#                     elif pending[wire][1] == 'AND':
#                         connected[wire] = a & b
#                     elif pending[wire][1] == 'RSHIFT':
#                         connected[wire] = a >> b
#                     elif pending[wire][1] == 'LSHIFT':
#                         connected[wire] = a << b
#                     del pending[wire]
#     print("Wire 'a' = {0} after override a_old -> b".format(connected['a']))
#
#
# def day8():
#     length = length_escaped = 0
#     with open('input/8.txt') as inp:
#         for line in inp:
#             length += len(line[:-1]) - len(eval(line[:-1]))
#             length_escaped += 2 + line.count('\\') + line.count('"')
#     print("Answer 1 is {0}, 2 is {1}".format(length, length_escaped))


if __name__ == '__main__':
    day4()
