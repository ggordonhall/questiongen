#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from os.path import join, exists, abspath

from typing import List, TextIO

"""
Functions to combine parsed text with the same text
split with byte pair encoding. Based on a script by
Rico Sennrich: 
https://github.com/rsennrich/wmt16-scripts

Args: 
    data_dir {str}: name of data containing dir
    out_dir {str}: name of dir for combined data
    bpe_dir {str}: name of dir for BPE split data
    parsed_dir {str}: name of dir for parsed data

Writes result to files in `out_dir'
"""


def get_factors(sentence: str, idx: int) -> List[str]:
    """Get linguistic annotations for a word at `idx`"""
    word = sentence[idx]
    factors = word.split(u"￨")[2]
    return [factors]


def filter_sentence(sentence: str) -> str:
    """Strip sentence of special characters"""
    sentence = list(filter(lambda x: x != u"￨"u"￨", sentence))
    return list(filter(lambda x: x != "", sentence))


def factor_file(bpe_lines: List[str], parse_lines: List[str], write_file: TextIO):
    """Combine linguistic tags with BPE split text.
    Write the annotated data to files in `out_dir`.

    Arguments:
        bpe_lines {List[str]} -- list of BPE encoded lines
        parse_lines {List[str]} -- list of parsed lines
        write_file {TextIO} -- file to write to
    """
    for line in bpe_lines:
        state = "O"
        i = 0
        next_parsed = next(parse_lines).strip().split()
        sentence = filter_sentence(next_parsed)

        for word in line.split():
            factors = get_factors(sentence, i)
            if word.endswith('@@'):
                if state == "O" or state == "E":
                    state = "B"
                elif state == "B" or state == "I":
                    state = "I"
            else:
                i += 1
                if state == "B" or state == "I":
                    state = "E"
                else:
                    state = "O"
            out_str = u"￨".join([word, state] + factors) + ' '
            write_file.write(out_str)
        write_file.write('\n')


def main(data_dir: str, out_dir: str, bpe_dir: str, parsed_dir: str):
    """Factor files in a data directory.

    Args: data_dir {str}: path to data
          out_dir {str}: name of output dir for combined data
          bpe_dir {str}: path to BPE split data
          parsed_data {str}: path to parsed data
    """
    if not exists(out_dir):
        os.mkdir(out_dir)

    split = ['train', 'dev', 'test']
    for s in split:
        filename = "{}.src".format(s)
        out_file = open(join(out_dir, filename),
                        'w+', encoding='utf-8')
        bpe_file = open(join(bpe_dir, filename),
                        encoding='utf-8')
        parse_file = open(join(
            parsed_dir, filename), encoding='utf-8')

        bpe_lines = iter(bpe_file.readlines())
        parse_lines = iter(parse_file.readlines())
        factor_file(bpe_lines, parse_lines, out_file)

        out_file.close(), bpe_file.close(), parse_file.close()


if __name__ == '__main__':
    base = abspath('..')
    data_dir = join(base, sys.argv[1])
    out_dir = join(data_dir, sys.argv[2])
    bpe_dir = join(data_dir, sys.argv[3])
    parsed_dir = join(data_dir, sys.argv[4])

    main(data_dir, out_dir, bpe_dir, parsed_dir)
