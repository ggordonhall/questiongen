from __future__ import division
import math
import torch
from onmt.translate import penalties

from typing import Tuple


class Beam(object):
    """
    Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.
    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 group_size=-1,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # If sub-beam size is specified and between 1 and the beam size,
        # run the diverse beam search algorithm
        self.diverse = False
        if group_size in range(1, self.size):
            self.diverse = True
            self.group_size = group_size
            if self.size % self.group_size != 0:
                raise ValueError(
                    "Beam must be perfectly divisible by group size")

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.
        Parameters:
        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step
        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            # Block ngram repeats
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram + [hyp[i]])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20
        else:
            beam_scores = word_probs[0]

        # run diverse beam search if not the first iteration
        if self.diverse and len(self.prev_ks) > 0:
            best_scores, prev_k, next_y = self.diverse_beam_search(
                beam_scores, num_words)
        else:
            flat_beam_scores = beam_scores.view(-1)
            best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                                True, True)
            # best_scores_id is flattened beam x word array, so calculate which
            # word and beam each score came from
            prev_k = best_scores_id / num_words
            next_y = (best_scores_id - prev_k * num_words)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])

    def diverse_beam_search(self, beam_scores: torch.Tensor, num_words: int) ->
                                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform diverse beam search as described in:
        https://arxiv.org/abs/1610.02424

        Args:
            beam_scores {torch.Tensor}: beam probabilities
            num_words {int}: the magnitude of vocab |V| 

        Returns:
            best_scores {torch.Tensor}: the top beam scores
            prev_k {torch.Tensor}: indices of top beams
            next_y {torch.Tensor}: indices of predicted words
        """
        # split beam into groups of (group_size * |V|)
        beam_groups = torch.split(beam_scores, self.group_size)
        # list beam hypotheses
        beam_hypotheses = torch.stack(self.next_ys, 1)

        prev_hyps, best_scores = [], []
        prev_k, next_y = [], []
        for idx, group in enumerate(beam_groups):
            # top group predictions
            group_scores, group_scores_id = group.view(
                -1).topk(self.group_size, 0, True, True)
            # beam that each prediction came from: (1 * group_size)
            beam_ids = (group_scores_id / num_words) + \
                (idx * self.group_size)
            # indicies of predicted words: (1 * group_size)
            word_ids = ((group_scores_id -
                         (group_scores_id / num_words) * num_words)).unsqueeze(1)
            # full group hypotheses
            hyps = torch.index_select(beam_hypotheses, 0, beam_ids)
            hyps = torch.cat((hyps, word_ids), 1)
            # amend scores by diversity factor
            if idx > 0:
                group_scores = self.global_scorer.add_diversity(
                    group_scores, hyps, prev_hyps)
            # add scores to best scores
            best_scores.append(group_scores)
            prev_k.append(beam_ids)
            next_y.append(word_ids.squeeze(1))
            # add group hypothesis
            prev_hyps.append(hyps)

        return torch.cat(best_scores), torch.cat(prev_k), torch.cat(next_y)


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`
    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
       diversity_factor (float): diversity parameter
    """

    def __init__(self, alpha, beta, cov_penalty, length_penalty, diversity_penalty="cum", diversity_factor=0.):
        self.alpha = alpha
        self.beta = beta
        self.diversity_factor = diversity_factor
        penalty_builder = penalties.PenaltyBuilder(cov_penalty,
                                                   length_penalty,
                                                   diversity_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()
        # Term will be subtracted from probability
        self.diversity_penalty = penalty_builder.diversity_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def add_diversity(self, sub_beam, hypothesis, prev_hyps):
        """
        Function to update scores of a Beam based on diversity
        """
        penalty = self.diversity_penalty(
            hypothesis, prev_hyps, self.diversity_factor)
        return sub_beam - penalty

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attentions"
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty
