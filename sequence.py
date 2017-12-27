import numpy as np



def correct_sequence (features, saliency, n, size):

    seq = np.empty((0, n, size, features.shape[-1]), dtype=features.dtype)
    ind = np.empty((0, n, size), dtype=int)

    for i in range (features.shape[0]):

        step = np.empty((0, features.shape[-1]), dtype=features.dtype)
        n_step = np.empty((0))

        for j in range (size*n):
            pick = np.argmax(saliency[i])
            saliency[i][int(pick/saliency.shape[1])][pick%saliency.shape[2]] = 0
            step = np.append(step, [features[i][int(pick/features.shape[1])][pick%features.shape[2]]], axis=0)
            n_step = np.append(n_step, pick)

        step = np.flip(step, axis=0)
        n_step = np.flip(n_step, axis=0)

        step_ordered = np.empty((0, features.shape[-1]), dtype=features.dtype)
        n_step_ordered = np.empty((0))

        for j in range (n):

            for k in range (j, size*n, n):

                step_ordered = np.append(step_ordered, [step[k]], axis=0)
                n_step_ordered = np.append(n_step_ordered, [n_step[k]], axis=0)

        seq = np.append(seq, [np.split(step_ordered, n)], axis=0)
        ind = np.append(ind, [np.split(n_step_ordered, n)], axis=0)

    y=np.ones((seq.shape[0], seq.shape[1]))

    return (seq, ind, y)



def split_sequence (seq, ind, y):

    count = seq.shape[1]*seq.shape[2]

    new_seq = np.empty((0, count, seq.shape[2], seq.shape[3]), dtype=seq.dtype)
    new_ind = np.empty((0, count, ind.shape[2]), dtype=ind.dtype)
    new_y = np.empty((0, count), dtype=y.dtype)

    for i in range (seq.shape[0]):

        seq_i = np.empty((0, seq.shape[2], seq.shape[3]), dtype=seq.dtype)
        ind_i = np.empty((0, ind.shape[2]), dtype=ind.dtype)
        y_i = np.empty((0), dtype=y.dtype)

        for j in range (seq.shape[1]):

            step = np.zeros((seq.shape[2], seq.shape[3]), dtype=seq.dtype)
            n_step = np.zeros((ind.shape[2]), dtype=ind.dtype)

            for k in range (seq.shape[2]):

                step[k] = seq[i][j][k]
                seq_i = np.append(seq_i, [step], axis=0)

                n_step[k] = ind[i][j][k]
                ind_i = np.append(ind_i, [n_step], axis=0)

                y_i = np.append(y_i, [y[i][j]], axis=0)

        new_seq = np.append(new_seq, [seq_i], axis=0)
        new_ind = np. append(new_ind, [ind_i], axis=0)
        new_y = np.append(new_y, [y_i], axis=0)

    return (new_seq, new_ind, new_y)



def wrong_sequence (seq, ind, features, y, n=0):

    total = features.shape[1] * features.shape[2]

    order = np.arange(total)

    if n:

        new_seq = np.empty((0, n, seq.shape[2], seq.shape[3]), dtype=seq.dtype)
        new_ind = np.empty((0, n, ind.shape[2]), dtype=ind.dtype)

    for i in range (seq.shape[0]):

        i_seq = np.empty((0, seq.shape[2], seq.shape[3]), dtype=seq.dtype)
        i_ind = np.empty((0, ind.shape[2]), dtype=ind.dtype)

        select = np.empty((0, total-seq.shape[1]), dtype=int)

        for j in range (seq.shape[2]):

            order_updated = np.copy(order)

            for k in range (seq.shape[1]):

                order_updated = np.delete(order_updated, np.argwhere(order_updated==ind[i][k][j]))

            select = np.append(select, [order_updated], axis=0)

        if n:

            for j in range(n):

                step = np.empty((0, seq.shape[-1]), dtype=features.dtype)
                n_step = np.empty((0))

                for k in range (seq.shape[2]):

                    pick = np.random.choice(select[k])
                    step = np.append(step, [features[i][int(pick/features.shape[1])][pick%features.shape[2]]], axis=0)
                    n_step = np.append(n_step, pick)

                i_seq = np.append(i_seq, [step], axis=0)
                i_ind = np.append(i_ind, [n_step], axis=0)

            new_seq = np.append(new_seq, [i_seq], axis=0)
            new_ind = np.append(new_ind, [i_ind], axis=0)

        else:

            select = cartesian(select)

            for j in range(select.shape[0]):

                step = np.empty((0, seq.shape[-1]), dtype=features.dtype)
                n_step = np.empty((0))

                for k in range (select.shape[1]):

                    step = np.append(step, [features[i][int(select[j][k]/features.shape[1])][select[j][k]%features.shape[2]]], axis=0)
                    n_step = np.append(n_step, select[j][k])

                i_seq = np.append(i_seq, [step], axis=0)
                i_ind = np.append(i_ind, [n_step], axis=0)

            if i<=0:

                new_seq = np.expand_dims(i_seq, axis=0)
                new_ind = np.expand_dims(i_ind, axis=0)

            else:

                new_seq = np.append(new_seq, [i_seq], axis=0)
                new_ind = np.append(new_ind, [i_ind], axis=0)

    new_y = np.concatenate([y, np.zeros((new_seq.shape[0], new_seq.shape[1]))], axis=1)
    new_seq = np.concatenate([seq, new_seq], axis=1)
    new_ind = np.concatenate([ind, new_ind], axis=1)

    return (new_seq, new_ind, new_y)



def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out



def generate_sequence (x_features, x_saliency, step, correct, wrong):

    (x_sequence, x_sequence_index, y) = correct_sequence(x_features, x_saliency, correct, step) # gets "correct" correct sequences of size "step" for each image
    print(x_sequence.shape, x_sequence_index.shape, y.shape)
    # print(x_sequence_index, y)

    (x_sequence2, x_sequence_index2, y2) = wrong_sequence(x_sequence, x_sequence_index, x_features, y, wrong)   # gets "wrong" incorrect sequences for each image
    print(x_sequence2.shape, x_sequence_index2.shape, y2.shape)
    # print(x_sequence_index2, y2)

    (x_sequence3, x_sequence_index3, y3) = split_sequence(x_sequence2, x_sequence_index2, y2)
    print(x_sequence3.shape, x_sequence_index3.shape, y3.shape)
    # print(x_sequence_index3, y3)

    return (x_sequence3, x_sequence_index3, y3)
