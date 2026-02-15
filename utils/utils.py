#!/usr/bin/env python

import matplotlib
import numpy as np

import matplotlib.pyplot as plt

matplotlib.use('agg')


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)

    ### END YOUR CODE

    return s


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x


def get_negative_samples(output_word_idx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """
    neg_sample_word_indices = [None] * K
    for k in range(K):
        new_idx = dataset.sample_token_idx()
        while new_idx == output_word_idx:
            new_idx = dataset.sample_token_idx()
        neg_sample_word_indices[k] = new_idx
    return neg_sample_word_indices


def visualize_function(tokens, wordVectors, savePath):
    """visualize the given visualize words"""
    n_words = len(tokens)
    print("n_words:   ", n_words)
    visualizeWords = [
        "great", "cool", "brilliant", "perfectly", "well", "good",
        "worth", "bad", "poor", "down",
        "european", "australian", "chinese", "american",
        "female", "man", "men", "woman", "women", "king", "queen"
    ]
    # concatenate the input and output word vectors
    words = np.concatenate(
        (wordVectors[:n_words, :], wordVectors[n_words:, :]), axis=0)
    visualize_idx = [tokens[word] for word in visualizeWords]
    visualize_vecs = words[visualize_idx, :]
    temp = (visualize_vecs - np.mean(visualize_vecs, axis=0))
    covariance = 1.0 / len(visualize_idx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])

    for i in range(len(visualizeWords)):
        plt.text(coord[i, 0], coord[i, 1], visualizeWords[i],
                 bbox=dict(facecolor='blue', alpha=0.1))

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    plt.savefig(savePath)
    plt.cla()


def normalize_rows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x ** 2, axis=1)).reshape((N, 1)) + 1e-30
    return x


if __name__ == "__main__":
    pass
