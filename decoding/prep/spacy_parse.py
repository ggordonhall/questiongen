# -*- coding: utf-8 -*-
import spacy

"""
Functions to tokenise a string with the SpaCy tokeniser.
"""

nlp = spacy.load('en')


def tokenise(inp: str):
    """Tokenise input string"""
    return '\n'.join(list(map(_tokenise_line, inp.splitlines())))


def _tokenise_line(line: str) -> str:
    """Format a tokenised line"""
    doc = nlp(line.strip())
    tok_line = ''
    for sent in doc.sents:
        for word in sent:
            tok_line += str(word) + ' '
    return tok_line
