from sys import argv
import operator

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

from typing import Optional, List

"""
Functions to extract the highest similarity sentence from a
paragraph of text. Uses the WordNet similarity metric (i.e. the
distance between two word in the WordNet graph).

Compares source and target files in a directory. The directory path
is passed via standard input.
"""


def penn_to_wn(tag: str) -> str:
    """Convert between a Penn Treebank tag to a simplified Wordnet tag """
    d = {'N': 'n', 'V': 'v', 'J': 'a', 'R': 'r'}
    return d.get(tag[0])


def tagged_to_synset(word: str, tag: str) -> Optional[wn.Synset]:
    """Get WordNet synsets of word"""
    wn_tag = penn_to_wn(tag)
    if not wn_tag:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1: str, sentence2: str):
    """Compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = word_tokenize(sentence1)
    sentence2 = word_tokenize(sentence2)

    sentence1 = pos_tag(sentence1)
    sentence2 = pos_tag(sentence2)

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        scores = [synset.wup_similarity(ss) for ss in synsets2]
        scores = [ss for ss in scores if ss]
        best_score = max(scores) if scores else None

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    score = (score / count) if count > 0 else 0
    return score


def run(base, split):
    """Run sentence comparison pipeline"""
    print("WORKING ON {}".format(split))

    src_exs = open('{}/{}.src'.format(base, split)).readlines()
    tgt_exs = open('{}/{}.tgt'.format(base, split)).readlines()

    src_out = open('{}/{}_sent.src'.format(base, split), 'w+')
    tgt_out = open('{}/{}_sent.tgt'.format(base, split), 'w+')

    for idx, src in enumerate(src_exs):
        if idx % 50 == 0:
            print("Working on step {}...\n".format(idx))

        src = src.split('.')
        tgt = tgt_exs[idx].split('--')[0]

        similarity = {}
        for idx, sent in enumerate(src):
            similarity[idx] = sentence_similarity(tgt, sent)

        if any(similarity.values()):
            top_idx = max(similarity.items(), key=operator.itemgetter(1))[0]
            top_sent = src[top_idx]

            src_out.write(top_sent + '\n')
            tgt_out.write(tgt + '\n')

    src_out.close()
    tgt_out.close()


if __name__ == '__main__':
    base = sys.argv[1]
    splits = ['train', 'test', 'dev']
    for split in splits:
        run(base, split)
