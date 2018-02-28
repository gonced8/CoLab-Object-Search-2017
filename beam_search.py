'''
Beam search function

The function "search" generates the highest probability sequenece
'''


import numpy as np

#mem_size = beam search number of steps
def search(calc_prob, seq_max_size, vocab, mem_size):
    empty = float('inf')
    vocab_with_empty = np.append(np.array(vocab), empty)

    #sequence vector (empty init)
    seqs = np.array([[]])
    #sequence probability vector (empty init)
    seqs_prob = np.array([[]])

    for current_size in range(1, seq_max_size + 1):

        empty_ind = [seq.size > 0 and seq[-1] == empty for seq in seqs]

        seqs_empty = seqs[empty_ind]
        seqs_not_empty = seqs[np.invert(empty_ind)]

        children_empty = np.array([np.append(seq, empty) for seq in seqs_empty])
        children_not_empty = np.array([np.append(seq, word) for seq in seqs_not_empty for word in vocab_with_empty])

        children = np.concatenate((children_not_empty.reshape(children_not_empty.shape[0], current_size),
                                   children_empty.reshape(children_empty.shape[0], current_size)))

        probs = calc_prob(children, seq_max_size, empty)

        max_seq_ind = np.argpartition(probs, -min(probs.size, mem_size))[-min(probs.size, mem_size):]

        seqs = children[max_seq_ind]

        seqs_prob = probs[max_seq_ind]

    return seqs[np.argmax(seqs_prob)]
