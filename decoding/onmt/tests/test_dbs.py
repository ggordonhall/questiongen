import torch
import pytest

import onmt


PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


def test_dbs():
    """Assert that function does not throw an error"""
    beam_scores = torch.Tensor([0.3, 0.7])
    beam = onmt.translate.Beam(2, PAD_IDX, BOS_IDX, EOS_IDX, group_size=1)
    beam.diverse_beam_search(beam_scores, 2)


def test_invalid_group():
    """Assert that indivisible group size raises an error"""
    with pytest.raises(ValueError):
        onmt.translate.Beam(2, PAD_IDX, BOS_IDX, EOS_IDX,
                            group_size=3)


def test_diversity_function():
    """Assert more similar group has greater penalty"""
    hypothesis = torch.Tensor([2, 1])
    prev_hyps_one = torch.Tensor([2, 1], [3, 6])
    prev_hyps_two = torch.Tensor([2, 1], [2, 1])
    penalty_one = onmt.translate.penalties(hypothesis, prev_hyps_one, 1)
    penalty_two = onmt.translate.penalties(hypothesis, prev_hyps_two, 1)
    assert penalty_two > penalty_one
