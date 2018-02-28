import numpy as np


def search(calc_prob, seq_max_size, vocab, mem_size):
    """
    Beam search function

    The function "search" generates the highest probability sequence

<<<<<<< HEAD
#mem_size = beam search number of steps
def search(calc_prob, seq_max_size, vocab, mem_size):
=======
    :param calc_prob:  function that takes as arguments sequences, the maximum size a sequence can have and the empty value, and returns the probability of each sequence
    :param seq_max_size: the maximum size a generated sequence can have
    :param vocab: a list of the values that can be used to generate the sequences
    :param mem_size: the size of the memory used in the beam search
    :return: the generated sequence with highest probability
    """

    # the empty element used to fill the sequences
>>>>>>> 74c77916c4aa91469962acec9c59e3d1495d94f8
    empty = float('inf')
    # append the empty element to the vocabulary
    vocab_with_empty = np.append(np.array(vocab), empty)

    #sequence vector (empty init)
    seqs = np.array([[]])
    #sequence probability vector (empty init)
    seqs_prob = np.array([[]])

    for current_size in range(1, seq_max_size + 1):

        # find the sequences where an empty element was already inserted, since we do not want
        # to "increase" those sequences
        empty_ind = [seq.size > 0 and seq[-1] == empty for seq in seqs]

        # the sequences with empty elements
        seqs_empty = seqs[empty_ind]
        # the sequences without empty elements
        seqs_not_empty = seqs[np.invert(empty_ind)]

        # append only empty elements to sequences that already contain empty elements
        children_empty = np.array([np.append(seq, empty) for seq in seqs_empty])

        # from the "non empty" sequences generate all possible sequences appending the one element from the vocabulary
        children_not_empty = np.array([np.append(seq, word) for seq in seqs_not_empty for word in vocab_with_empty])

        # join 'children_empty'
        children = np.concatenate((children_not_empty.reshape(children_not_empty.shape[0], current_size),
                                   children_empty.reshape(children_empty.shape[0], current_size)))

        # calculate the probabilities of the generated sequences
        probs = calc_prob(children, seq_max_size, empty)

        num_seqs_to_save = min(probs.size, mem_size)

        # find the indices of the sequences with higher probability
        max_seq_ind = np.argpartition(probs, -num_seqs_to_save)[-num_seqs_to_save:]

        # store the sequences with highest probability
        seqs = children[max_seq_ind]
        seqs_prob = probs[max_seq_ind]

    return seqs[np.argmax(seqs_prob)]
