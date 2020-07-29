import numpy as np


# markov chain
# converge as n -> infinity
def markov():
    """markov chain example with matrix power"""
    initial_state_1 = np.array([0.1, 0.3, 0.6])  # 1*3 vector
    initial_state_2 = np.array([0.2, 0.4, 0.4])
    transfer_matrix = np.array([[0.9, 0.075, 0.025],
                                [0.15, 0.8, 0.05],
                                [0.25, 0.25, 0.5]])  # 3*3 matrix
    markov_test(initial_state_1, transfer_matrix)
    # print("\n")
    # markov_test(initial_state_2, transfer_matrix)
    print("\n")
    markov_test(transfer_matrix, transfer_matrix)  # matrix power


def markov_test(initial_state, transfer_matrix):
    next_state = initial_state
    for i in range(30):
        res = np.dot(next_state, transfer_matrix)
        print(i, "\t", res)
        next_state = res



markov()
