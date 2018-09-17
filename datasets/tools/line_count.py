import os
import sys

from typing import Tuple

"""Functions to get line length statistics from file.
Accepts path to file from standard input. Prints line count,
longest line length, and average line length to console. 
"""


def line_utils(data_file: str) -> Tuple[int, int, float]:
    with open(data_file) as f:
        line_length = []
        for i, l in enumerate(f):
            line_length.append(len(l.split(' ')))
    line_cnt = i + 1
    max_ll = max(line_length)
    average_ll = sum(line_length) / line_cnt
    return (line_cnt, max_ll, average_ll)


if __name__ == '__main__':
    datapath = os.path.abspath('..')
    dataset, datatype = sys.argv[1], sys.argv[2]
    path = "{}/{}/{}".format(datapath, dataset, datatype)
    files = os.listdir(path)

    for df in files:
        print("{}: {}".format(df, line_utils(os.path.join(path, df))))
