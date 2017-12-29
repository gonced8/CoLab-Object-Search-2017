import numpy as np


def search(calc_prob, seq_max_size, vocab, mem_size):
    empty = float('inf')

    seqs = np.array([[]])

    for _ in range(seq_max_size):

        # TODO split the ones that end in `empty` from the rest
        children = np.array([np.append(seq, word) for seq in seqs for word in vocab])

        probs = calc_prob(children, seq_max_size, empty)

        max_seq_ind = np.argpartition(probs, -min(probs.size, mem_size))[-min(probs.size, mem_size):]

        seqs = children[max_seq_ind]

    # TODO return only the max
    return seqs
