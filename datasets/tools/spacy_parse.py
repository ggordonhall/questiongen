# -*- coding: utf-8 -*-
import sys
import os
import re
import spacy

from typing import List


class SpacyParser:
    """Class to parse and tokenise files with SpaCy.

    Raises:
        LookupError -- If entered filepath does not exist.

    Args:
        data_dir {str}: path to dir containing text data
        data_type {str}: whether file is source or target
        parse_path {str}: where to save parsed files
        tok_path {str}: where to save tokenised files
    """

    def __init__(self, data_dir: str, data_type: str, parse_path: str, tok_path: str):
        self.nlp = spacy.load('en')
        # parent directory (open_mnt/datasets)
        self.base = os.path.join(os.path.abspath('..'), data_dir)
        self.input_dir = "{}/{}".format(self.base, data_type)
        self.parse_dir = "{}/{}".format(self.base, parse_path)
        self.tok_dir = "{}/{}".format(self.base, tok_path)

        mkdir(self.parse_dir)
        mkdir(self.tok_dir)

        self.split = ['train', 'dev', 'test']
        self.type = ['src', 'tgt']
        for s, t in [(s, t) for s in self.split for t in self.type]:
            print("PARSING {}.{}\n\n".format(s, t))
            self._parse(s, t)

    def _parse(self, split: str, type: str):
        """Returns a parsed file in str form"""
        input = '{}/{}.{}'.format(self.input_dir, split, type)

        if not os.path.isfile(input):
            raise LookupError("Invalid filename! {}".format(input))

        with open(input, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                if idx % 50 == 0:
                    print("Parsing line {}...\n\n".format(idx))
                parsed_line = self._pipeline(line, type)
                self._save(parsed_line, split, type)

    def _save(self, parse: List[str], split: str, type: str):
        """Write the parse to a txt file"""
        files = ['{}/{}.{}'.format(self.tok_dir, split,
                                   type)]  # Always tokenise
        if len(parse) > 1:
            files.append('{}/{}.{}'.format(self.parse_dir, split, type))

        for i, file in enumerate(files):
            mode = write_mode(file)
            with open(file, mode, encoding='utf-8') as f:
                f.write(parse[i] + '\n')

    def _pipeline(self, line: str, type: str) -> List[str]:
        """Spacy parse a line"""
        doc = self.nlp(line.strip())
        if type == 'src':
            return [self._tokenise_line(doc), self._parse_line(doc)]
        return [self._tokenise_line(doc)]

    def _parse_line(self, doc: spacy.tokens.doc) -> str:
        """Format a parsed line"""
        parsed_line = ''
        for sent in doc.sents:
            for word in sent:
                parsed_line += u"￨".join(
                    [str(word), str(word.tag_), str(word.dep_)]) + ' '
        return parsed_line

    def _tokenise_line(self, doc: spacy.tokens.Doc) -> str:
        """Format a tokenised line"""
        tok_line = ''
        for sent in doc.sents:
            for word in sent:
                tok_line += str(word) + ' '
        return tok_line


def write_mode(filename: str) -> str:
    """Write or append"""
    if os.path.exists(filename):
        return 'a'
    return 'w'


def mkdir(dir: str):
    """Make directory if needed"""
    if not os.path.exists(dir):
        os.mkdir(dir)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    data_type = sys.argv[2]
    parse_path = sys.argv[3]
    tok_path = sys.argv[4]
    SpacyParser(data_dir, data_type, parse_path, tok_path)
