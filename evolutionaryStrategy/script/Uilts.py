import numpy as np

def feature_construct(board):
    """convert board state to 16 features

    Args:
        board (_type_): board state

    Returns:
        features: 16 input features
    """
    row = board.sum(axis=0)
    col = board.sum(axis=1)

    a_row = row + 1
    b_row = row - 1
    a_col = col + 1
    b_col = col - 1

    dig = board.trace()
    flip_dig = np.fliplr(board).trace()

    a_dig = dig + 1
    a_flip_dig = flip_dig + 1
    b_dig = dig - 1
    b_flip_dig = flip_dig - 1

    features = np.concatenate([a_row,b_row,a_col,b_col,[a_dig,b_dig,a_flip_dig,b_flip_dig]])
    features = 1/(1 + np.exp(-features))

    return features