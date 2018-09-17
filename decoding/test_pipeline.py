import pytest
from decoding.pipeline import *


def test_pipeline():
    pipe_one = DecodingPipeline('s', 'This is a sentence.', 0.6)
    pipe_two = DecodingPipeline('s', 'This is a sentence.', -0.6)
    pipe_three = DecodingPipeline('p', 'This is a sentence.', 0.6)
    assert pipe_one.pred == pipe_two.pred
    assert pipe_three != pipe_one


def test_val_div():
    assert val_div(1.0) == 1.0
    assert val_div(-1.0) == 0.6
    assert val_div('0.6') == 0.6
    assert val_div(0.45) == 0.45


def test_split_pred():
    assert split_pred("Question -- answer") == ("Question", "Answer")
    assert split_pred("Question     --    answer") == ("Question", "Answer")
    assert split_pred(
        "Is this a question? -- Yes it is.") == ("Is this a question?", "Yes it is.")
    with pytest.raises(SystemExit):
        split_pred("John -- ")
        split_pred("John and Mary")
        split_pred("")
        split_pred(1.0)


def test_uppercase():
    assert uppercase("carl is my name") == "Carl is my name"
    assert uppercase("tom") == "Tom"
    with pytest.raises(IndexError):
        uppercase("")
