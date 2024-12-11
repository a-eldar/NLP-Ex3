from typing import Callable, TypeVar

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
    criterion = lambda k, u, v, w: pi_of(k, u, v) * q(v, w, u) * e(x[k], v)

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




def __fill_pi_one_index(labels, pi, bp, criterion, k):
    for v in labels:
        for u in labels:
            bp[(k, u, v)] = max(labels, key=lambda w: criterion(k, u, v, w))
            pi[(k, u, v)] = criterion(k, u, v, bp[(k, u, v)])

