#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #
#  This file is part of SCAMP (Suite for Computer-Assisted Music in Python)                      #
#  Copyright © 2020 Marc Evanstein <marc@marcevanstein.com>.                                     #
#                                                                                                #
#  This program is free software: you can redistribute it and/or modify it under the terms of    #
#  the GNU General Public License as published by the Free Software Foundation, either version   #
#  3 of the License, or (at your option) any later version.                                      #
#                                                                                                #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;     #
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.     #
#  See the GNU General Public License for more details.                                          #
#                                                                                                #
#  You should have received a copy of the GNU General Public License along with this program.    #
#  If not, see <http://www.gnu.org/licenses/>.                                                   #
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #

import math
from xml.etree import ElementTree


def _least_common_multiple(*args):
    # utility for getting the least_common_multiple of a list of numbers
    if len(args) == 0:
        return 1
    elif len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return args[0] * args[1] // math.gcd(args[0], args[1])
    else:
        return _least_common_multiple(args[0], _least_common_multiple(*args[1:]))


def _is_power_of_two(x):
    # utility for checking if x is a power of two
    log2_x = math.log2(x)
    return log2_x == int(log2_x)


def _escape_split(s, delimiter):
    # Borrowed from https://stackoverflow.com/questions/18092354/python-split-string-without-splitting-escaped-character
    i, res, buf = 0, [], ''
    while True:
        j, e = s.find(delimiter, i), 0
        if j < 0:  # end reached
            return res + [buf + s[i:]]  # add remainder
        while j - e and s[j - e - 1] == '\\':
            e += 1  # number of escapes
        d = e // 2  # number of double escapes
        if e != d * 2:  # odd number of escapes
            buf += s[i:j - d - 1] + s[j]  # add the escaped char
            i = j + 1  # and skip it
            continue  # add more to buf
        res.append(buf + s[i:j - d])
        i, buf = j + len(delimiter), ''  # start after delim


def get_average_square_correlation(test_string: str, template_string: str) -> float:
    """
    A test of the similarity of two strings via a (squared) cross-correlation of their characters.
    (Scaled down to compensate for the influence of string lengths.)

    :param test_string: string we are testing
    :param template_string: template string we are testing against
    """
    square_correlation_sum = 0
    test_length, template_length = len(test_string), len(template_string)
    for offset in range(-test_length + 1, template_length):
        test_string_segment = test_string[max(0, -offset): template_length - offset]
        template_string_segment = template_string[max(0, offset): max(0, offset) + len(test_string_segment)]
        correlation_score = sum(a == b for a, b in zip(test_string_segment, template_string_segment))
        square_correlation_sum += correlation_score ** 2
    return square_correlation_sum / (test_length + template_length)
