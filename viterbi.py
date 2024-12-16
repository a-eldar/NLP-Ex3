from typing import Callable, TypeVar
from itertools import product

Word = str
Label = str
STOP_KEYWORD = 'STOP'

def viterbi_algorithm_trigram(sentence: list[Word], labels: list[Label],
                      q: Callable[[Label, Label, Label], float],
                      e: Callable[[Word, Label], float]) -> list[Label]:
    """The Viterbi algorithm uses dynamic programming to find the best
    path of labels.\nHere specifically using trigram.

    Args:
        sentence (list[Word]): 
        labels (list[Label]): All labels
        q (Callable[[Label, Label, Label], float]): probability function q(label_k | label_k-2, label_k-1), \
            Should also accept 'STOP' as label!
        e (Callable[[Word, Label], float]): probability function e(word_k | label_k)

    Returns:
        list[Label]: The best path of labels.
    """

    pi: dict[tuple[int, Label, Label], float] = {}
    bp: dict[tuple[int, Label, Label], Label] = {}
    n = len(sentence)
    x = sentence
    pi_of = lambda k, u, v: pi[k, u, v] if k > 0 else 1
    criterion = lambda k, u, v, w: pi_of(k-1, u, v) * q(v, w, u) * e(x[k-1], v)

    for k in range(1, n+1):
        __fill_pi_one_index(labels, pi, bp, criterion, k)
    
    # Extract solution:
    best_pair = ()
    best_pair_value = 0
    for v in labels:
        for u in labels:
            if best_pair_value < pi[(n, u, v)]:
                best_pair = (v, u)
                best_pair_value = pi[(n, u, v)]

    result = [*best_pair] # We insert elements in reverse
    for k in range(n-2, 0, -1):
        result.append(bp[(k+2, result[-1], result[-2])])
    
    return list(reversed(result))


def viterbi_algorithm_bigram(sentence: list[Word], labels: list[Label],
                            q: Callable[[Label, Label], float],
                            e: Callable[[Word, Label], float]) -> list[Label]:
    """The Viterbi algorithm uses dynamic programming to find the best
    path of labels.\nHere specifically using bigram.

    Args:
        sentence (list[Word]):
        labels (list[Label]): All labels
        q (Callable[[Label, Label], float]): probability function q(label_k | label_k-1), \
            Should also accept 'STOP' as label!
        e (Callable[[Word, Label], float]): probability function e(word_k | label_k)

    Returns:
        list[Label]: The best path of labels
    """
    BIGRAM = 2
    pi: dict[tuple[int, Label], float] = {}
    bp: dict[tuple[int, Label], Label] = {}
    n = len(sentence)
    x = sentence
    pi_of = lambda k, v: pi[k, v] if k > 0 else 1
    criterion = lambda k, v, u: pi_of(k-1, v) * q(v, u) * e(x[k-1], v)

    for k in range(1, n+1):
        __fill_pi_one_index(labels, pi, bp, criterion, k, BIGRAM, 'H')
    
    # Extract solution:
    best_label = None
    best_label_value = 0
    for v in labels:
        if best_label_value < pi[(n, v)]:
            best_label = v
            best_label_value = pi[(n, v)]

    print(pi)
    print()
    print(bp)
    result = [best_label] # We insert elements in reverse
    for k in range(n-1, 0, -1):
        result.append(bp[(k+1, result[-1])])
    
    return list(reversed(result))


def __fill_pi_one_index(labels, pi, bp, criterion, k, gram=3, initial_label=None):
    if k == 1 and initial_label:
        for comb in product(labels, repeat=gram-1):
            bp[(k, *comb)] = initial_label
            pi[(k, *comb)] = criterion(k, *comb, initial_label)
    else:    
        for comb in product(labels, repeat=gram-1):
            bp[(k, *comb)] = max(labels, key=lambda w: criterion(k, *comb, w))
            pi[(k, *comb)] = criterion(k, *comb, bp[(k, *comb)])




if __name__ == "__main__":
    q = lambda v, u: 0.5 if u == 'H' else 0.4 + 0.2 * int(v == 'L')
    def e(x, v):
        h_prob = {
            'A': 0.2, 'C': 0.3, 'G': 0.3, 'T': 0.2
        }
        l_prob = {
            'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3
        }
        if v == 'H':
            return h_prob[x]
        return l_prob[x]
    print(viterbi_algorithm_bigram("ACCGTGCA", ['H', 'L'], q, e))