import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    max_l_seq = max(len(seq) for seq in seqs)
    L = max_len if max_len is not None else max_l_seq or 0
        

    res = np.full((N,L), pad_value)
    for i in range(N):
        i_len = len(seqs[i])
        length = min(i_len, L)
        res[i][0:length] = seqs[i][:length]
    return res