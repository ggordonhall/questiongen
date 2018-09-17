# -*- coding: utf-8 -*-
import re
import sys
import json
import subprocess

from os import mkdir, getcwd
from os.path import join, exists, isdir

from typing import Tuple, Union
from collections import namedtuple

from decoding.prep.spacy_parse import tokenise


class DecodingPipeline:
    """Decodes a QG model with the input passed by the user. 
    Users can specify the diversity factor and whether to run a 
    paragraph or sentence level model.

    Args:
        model_type: str -- paragraph or sentence model
        text_inp: str -- the text to pass to the model
        div_factor: float -- diversity delta for DBS

    Returns:
        question: Union[Tuple[str, str]], str] -- 
            a question and answer, or just a question
    """

    def __init__(self, model_type: str, text_inp: str, div_factor: float):
        self._cwd = join(getcwd(), 'decoding')
        temp_dir = join(self._cwd, 'tmp')
        make_tmp(temp_dir)

        self._prep_dir = join(self._cwd, 'prep')
        self._model_dir = join(self._cwd, 'models')

        self._bpe_codes_path = join(self._prep_dir, 'bpe-codes.src')
        check_bpe(self._bpe_codes_path)

        self._bpe_inp_path = join(temp_dir, 'bpe-inp.src')
        self._inp_path = join(temp_dir, 'inp.src')
        self._pred_path = join(temp_dir, 'pred.bpe')

        self._opts = self._load_args(model_type)
        self._div_factor = val_div(div_factor)
        self._write_input(text_inp, self._opts['char_lim'])

        self._pred = self._run()

    def _load_args(self, model_type: str):
        """Load model configuration"""
        with open(join(self._cwd, 'config.json'), 'r') as j:
            json_opts = json.load(j)
        try:
            return json_opts[model_type]
        except ValueError:
            print("Invalid model type!")

    def _write_input(self, txt: str, lim: int):
        """Write input text to tmp file"""
        txt = txt.strip()
        if len(list(txt)) > lim:
            txt = txt[:lim]

        with open(self._inp_path, 'w+') as f:
            f.write(tokenise(txt.lower()))

    def _bpe_input(self):
        """Byte pair encode user input"""
        args = ['-c', self._bpe_codes_path, '-i',
                self._inp_path, '-o', self._bpe_inp_path]

        subprocess.call([join(self._prep_dir, 'apply_bpe.py')] + args)

    def _remove_bpe(self) -> str:
        """Remove byte pair encoding"""
        pred_bpe = open(self._pred_path, 'r').read()
        pred = re.sub(r'(@@ )|(@@ ?$)', '', pred_bpe, flags=re.MULTILINE)
        return pred

    def _decode(self, model_path: str, div_factor: float):
        """Decode with DBS with a given diversity"""
        args = ['-model', model_path, '-src', self._bpe_inp_path, '-output',
                self._pred_path, '-beam_size', '8', '-group_size', '2',
                '-length_penalty', 'wu', '-diversity_penalty', 'cum',
                '-diversity_factor', str(div_factor),
                '-coverage_penalty', 'summary', '-stepwise_penalty', '-replace_unk',
                '-alpha', '0.9', '-beta', '5', '-block_ngram_repeat', '3',
                '-ignore_when_blocking', "'.' ' < /t > ' ' < t >'"]

        subprocess.call([join(self._cwd, 'translate.py')] + args)

    def _run(self) -> Union[Tuple[str, str], str]:
        """Run the decoding pipeline"""
        model_path = join(self._model_dir, self._opts['path'])
        self._bpe_input()
        self._decode(model_path, self._div_factor)

        pred = uppercase(self._remove_bpe())
        return split_pred(pred) if self._opts['name'] == 'paragraph' else pred

    @property
    def pred(self):
        """Get prediction"""
        return self._pred


def uppercase(sentence: str):
    """Uppercase first word in sentence"""
    tokens = sentence.split()
    try:
        first = tokens[0]
    except IndexError:
        raise IndexError
    cap_first = [first[0].upper() + first[1:]]
    return ' '.join(cap_first + tokens[1:])


def check_bpe(bpe_codes_path: str):
    """Check bpe codes"""
    if not exists(bpe_codes_path):
        raise FileNotFoundError("BPE codes not found!")


def make_tmp(tmp_dir: str):
    """Create temporary directory"""
    if not isdir(tmp_dir):
        mkdir(tmp_dir)


def val_div(div: float) -> int:
    """Validate diversity factor, default = 0.6"""
    try:
        div = float(div)
    except ValueError:
        return 0.6
    return 0.6 if div > 1.0 or div < 0 else div


def split_pred(pred: str) -> Tuple[str, str]:
    """Split prediction into question and answer.
    Raises an error if the model has not predicted an
    answer. This indicates that the model failed to work."""
    try:
        question, answer = [part.strip() for part in pred.split('--')]
        answer = uppercase(answer)
    except IndexError:
        print("""Sorry, the system didn't work!
                Try again with a different input""")
        sys.exit()
    return question, answer
