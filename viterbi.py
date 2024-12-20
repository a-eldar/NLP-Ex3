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

def viterbi_bigram(observations, states, start_prob, transition_prob, emission_prob):
    """
    Perform the Viterbi algorithm for a bigram HMM.

    Parameters:
        observations (list): The sequence of observed words.
        states (list): The set of possible tags.
        start_prob (dict): Start probabilities, P(t0).
                           Example: {'NOUN': 0.5, 'VERB': 0.5}
        transition_prob (dict): Transition probabilities, P(t|u).
                                Example: {('NOUN', 'VERB'): 0.1, ('VERB', 'NOUN'): 0.2}
        emission_prob (dict): Emission probabilities, P(x|t).
                              Example: {('NOUN', 'dog'): 0.5, ('VERB', 'runs'): 0.3}

    Returns:
        tuple: The most probable sequence of states (tags) and its probability.
    """
    n = len(observations)
    viterbi = [{} for _ in range(n)]  # DP table for probabilities
    backpointer = [{} for _ in range(n)]  # DP table for backpointers

    # Initialization step
    for state in states:
        viterbi[0][state] = start_prob.get(state) * emission_prob(state, observations[0])
        backpointer[0][state] = None

    # Recursion step
    for t in range(1, n):
        for state in states:
            max_prob, best_prev_state = max(
                (
                    viterbi[t - 1][prev_state] * transition_prob.get((prev_state, state), 0)
                    * emission_prob(state, observations[t]),
                    prev_state,
                )
                for prev_state in states
            )
            viterbi[t][state] = max_prob
            backpointer[t][state] = best_prev_state

    # Termination step
    max_prob, best_last_state = max((viterbi[n - 1][state], state) for state in states)

    # Backtracking
    best_path = [best_last_state]
    for t in range(n - 1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])

    return best_path, max_prob


def viterbi(obs, states, start_p, trans_p, emit_p, known_words):
    V = [{}]
    path = {}

    default_emit = lambda y, x: 1 if y == 'NN' and x.lower() not in known_words else 0
    emit_f = lambda y, x: emit_p[y][x.lower()] if x.lower() in emit_p[y] else default_emit(y, x)
    start_f = lambda y: start_p[y] if y in start_p else 0
    trans_f = lambda y0, y: trans_p[y0][y] if y in trans_p[y0] else 0

    for y in states:
        V[0][y] = start_f(y) * emit_f(y, obs[0])
        path[y] = [y]

    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(
                [(V[t-1][y0] * trans_f(y0, y) * emit_f(y, obs[t]), y0) for y0 in states]
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        path = newpath

    (prob, state) = max([(V[-1][y], y) for y in states])
    return (prob, path[state])

if __name__ == "__main__":
    #q = lambda v, u: 0.5 if u == 'H' else 0.4 + 0.2 * int(v == 'L')
    q = {'H':{'H':0.5, 'L':0.5},'L':{'H':0.4, 'L':0.6}}
    #{('H','H'): 0.5, ('H','L'): 0.4, ('L','H'): 0.5, ('L','L'): 0.6}
    def e(v,x):
        h_prob = {
            'A': 0.2, 'C': 0.3, 'G': 0.3, 'T': 0.2
        }
        l_prob = {
            'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3
        }
        if v == 'H':
            return h_prob[x]
        return l_prob[x]
    emit = {'H':{'A': 0.2, 'C': 0.3, 'G': 0.3, 'T': 0.2},
            'L':{'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3}}
    start = {'H': 1, 'L':0}
    print(viterbi("ACCGTGCA", ['H', 'L'], start, q, emit))