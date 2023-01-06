import re

import numpy as np
import collections
import utils
import hashlib
import collections

YEAR = 2016


def day1():
    with open(utils.get_input(YEAR, 1)) as inp:
        loc = 0 + 0j
        alpha = 0 + 1j
        visited = set()
        twice = None
        for data in inp.readline().split(', '):
            if data[0] == 'L':
                alpha *= 1j
            else:
                alpha *= -1j
            val = int(data[1:])
            for _ in range(val):
                loc += alpha
                if twice is None and loc in visited:
                    twice = loc
                visited.add(loc)

        print(int(abs(loc.real) + abs(loc.imag)))
        print(int(abs(twice.real) + abs(twice.imag)))


def day2():
    with open(utils.get_input(YEAR, 2)) as inp:
        table1 = np.array([['1', '2', '3'],
                           ['4', '5', '6'],
                           ['7', '8', '9']])
        table2 = np.array([[' ', ' ', '1', ' ', ' '],
                           [' ', '2', '3', '4', ' '],
                           ['5', '6', '7', '8', '9'],
                           [' ', 'A', 'B', 'C', ' '],
                           [' ', ' ', 'D', ' ', ' ']])
        x1, y1 = 1, 1
        x2, y2 = 2, 0
        code1 = code2 = ''
        for line in inp:
            for move in line[:-1]:
                if move == 'U':
                    x1 = max(0, x1 - 1)
                    x2 = x2 - 1 if x2 > 0 and table2[x2 - 1, y2] != ' ' else x2
                elif move == 'D':
                    x1 = min(2, x1 + 1)
                    x2 = x2 + 1 if x2 < 4 and table2[x2 + 1, y2] != ' ' else x2
                elif move == 'L':
                    y1 = max(0, y1 - 1)
                    y2 = y2 - 1 if y2 > 0 and table2[x2, y2 - 1] != ' ' else y2
                elif move == 'R':
                    y1 = min(2, y1 + 1)
                    y2 = y2 + 1 if y2 < 4 and table2[x2, y2 + 1] != ' ' else y2
            code1 += table1[x1, y1]
            code2 += table2[x2, y2]
        print(code1)
        print(code2)


def day3():
    with open(utils.get_input(YEAR, 3)) as inp:
        buffer = np.zeros((3, 3))
        count1 = count2 = i = 0
        for line in inp:
            buffer[i] = [int(d) for d in line.split()]
            sides = sorted(buffer[i])
            if sides[0] + sides[1] > sides[2]:
                count1 += 1

            i += 1
            if i == 3:
                for j in range(3):
                    sides = sorted(buffer[:, j])
                    if sides[0] + sides[1] > sides[2]:
                        count2 += 1
                i = 0
        print(count1)
        print(count2)


def day4():
    with open(utils.get_input(YEAR, 4)) as inp:
        count = 0
        res2 = 0
        for line in inp:
            data = line.split('-')
            name = list('-'.join(data[:-1]))
            letters = collections.Counter(''.join(name))
            sorted_letters = sorted(letters, key=lambda x: (-letters[x], x))
            data = data[-1].split('[')
            sector_id = int(data[0])
            checksum = data[1][:-2]
            if ''.join(sorted_letters[:5]) == checksum:
                count += sector_id
            for i in range(len(name)):
                if name[i] == '-':
                    name[i] = ' '
                else:
                    name[i] = chr((ord(name[i]) - ord('a') + sector_id) % 26 + ord('a'))
            name = ''.join(name)
            if 'northpole' in name:
                res2 = sector_id
        print(count)
        print(res2)


def day5():
    with open(utils.get_input(YEAR, 5)) as inp:
        password1 = []
        password2 = {}

        door = inp.readline().strip()
        md5 = hashlib.md5()
        md5.update(door.encode('utf8'))
        n = 0
        while len(password2) < 8:
            md5_copy = md5.copy()
            md5_copy.update(str(n).encode('utf8'))
            val = md5_copy.hexdigest()

            if val.startswith('00000'):
                if len(password1) < 8:
                    password1.append(str(val[5]))

                if '0' <= val[5] < '8' and val[5] not in password2:
                    password2[val[5]] = val[6]

            n += 1
        print(''.join(password1))
        print(''.join([password2[str(i)] for i in range(8)]))


def day6():
    with open(utils.get_input(YEAR, 6)) as inp:
        code1 = []
        code2 = []
        lines = [line[:-1] for line in inp]
        for i in range(len(lines[0])):
            counter = collections.Counter([line[i] for line in lines])
            code1.append(max(counter, key=lambda x: counter[x]))
            code2.append(min(counter, key=lambda x: counter[x]))
        print(''.join(code1))
        print(''.join(code2))


def day7():
    def abba(string):
        for j in range(1, len(string) - 2):
            if string[j] == string[j + 1] and string[j] != string[j - 1] and string[j - 1] == string[j + 2]:
                return True
        return False

    def aba(strings, bab=False):
        out = set()
        for string in strings:
            for j in range(1, len(string) - 1):
                if string[j] != string[j - 1] and string[j - 1] == string[j + 1]:
                    if bab:
                        out.add((string[j - 1], string[j]))
                    else:
                        out.add((string[j], string[j - 1]))
        return out

    with open(utils.get_input(YEAR, 7)) as inp:
        count1 = count2 = 0
        for line in inp:
            supernet, hypernet = [], []
            i = 0
            line = line.strip()
            for match in re.finditer(r'\[(\w+)]', line):
                supernet.append(line[i:match.span()[0]])
                hypernet.append(match.groups()[0])
                i = match.span()[1]
            supernet.append(line[i:])

            if True in [abba(word) for word in supernet] and True not in [abba(word) for word in hypernet]:
                count1 += 1

            if aba(supernet).intersection(aba(hypernet, bab=True)):
                count2 += 1

        print(count1)
        print(count2)


def day8():
    with open(utils.get_input(YEAR, 8)) as inp:
        display = np.zeros((50, 6), dtype=bool)
        for line in inp:
            if line.startswith('rotate row'):
                line = line.split()
                j = int(line[2][2:])
                n = int(line[4])
                display[:, j] = np.roll(display[:, j], n)
            elif line.startswith('rotate column'):
                line = line.split()
                i = int(line[2][2:])
                n = int(line[4])
                display[i] = np.roll(display[i], n)
            else:
                i, j = [int(d) for d in line.split()[1].split('x')]
                display[:i, :j] = True
        print(np.sum(display))

        for j in range(6):
            for i in range(50):
                print('##' if display[i, j] else '  ', end='')
            print()


def day9():
    def get_len(string):
        if '(' not in string:
            return len(string)
        k = string.index('(')
        match = re.match(r'\((\d+)x(\d+)\)', string[k:])
        i, j = [int(d) for d in match.groups()]
        return k + j * get_len(line[match.span()[1]:])

    with open(utils.get_input(YEAR, 9)) as inp:
        line = inp.readline().strip()
        out = ''
        while line:
            if line[0] == '(':
                match = re.match(r'\((\d+)x(\d+)\)', line)
                i, j = [int(d) for d in match.groups()]
                out = out + line[match.span()[1]:match.span()[1] + i] * j
                line = line[match.span()[1] + i:]
            else:
                out = out + line[0]
                line = line[1:]
        print(len(out))


if __name__ == '__main__':
    day9()
