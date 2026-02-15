#!/usr/bin/env python

import numpy as np
import random

from utils.utils import sigmoid, softmax, get_negative_samples


def naive_softmax_loss_and_gradient(
        input_vector,
        output_word_idx,
        output_vectors,
        dataset):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between an input word's
    embedding and an output word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    input_vector -- numpy ndarray, input word's embedding
                    in shape (word vector length, )
                    (v_i in the pdf handout)
    output_word_idx -- integer, the index of the output word
                    (o of u_o in the pdf handout)
    output_vectors -- output vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    grad_input_vec -- the gradient with respect to the input word vector
                    in shape (word vector length, )
                    (dL / dv_i in the pdf handout)
    grad_output_vecs -- the gradient with respect to all the output word vectors
                    in shape (num words in vocab, word vector length) 
                    (dL / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow.

    # Compute scores and softmax probabilities: y_hat = softmax(U @ v_i)
    scores = output_vectors @ input_vector                  # (vocab_size,)
    y_hat = softmax(scores)                                 # (vocab_size,)

    # Cross-entropy loss: L = -log(y_hat_o)
    loss = -np.log(y_hat[output_word_idx])

    # Gradient w.r.t. input vector: dL/dv_i = U^T (y_hat - y)
    delta = y_hat.copy()
    delta[output_word_idx] -= 1                             # (y_hat - y)
    grad_input_vec = output_vectors.T @ delta               # (dim,)

    # Gradient w.r.t. all output vectors: dL/dU = (y_hat - y) v_i^T
    grad_output_vecs = np.outer(delta, input_vector)        # (vocab_size, dim)

    ### END YOUR CODE

    return loss, grad_input_vec, grad_output_vecs


def neg_sampling_loss_and_gradient(
        input_vector,
        output_word_idx,
        output_vectors,
        dataset,
        K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a inputWordVec
    and a outputWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an output word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
    """

    # Negative sampling of words is done for you.
    neg_sample_word_indices = get_negative_samples(output_word_idx, dataset, K)
    indices = [output_word_idx] + neg_sample_word_indices

    ### YOUR CODE HERE (~10 Lines)
    ### Please use your implementation of sigmoid in here.

    # Retrieve the true output vector and negative sample vectors
    u_o = output_vectors[output_word_idx]                       # (dim,)
    u_neg = output_vectors[neg_sample_word_indices]              # (K, dim)

    # Compute sigmoid scores
    pos_score = sigmoid(u_o @ input_vector)                     # scalar
    neg_scores = sigmoid(-u_neg @ input_vector)                 # (K,)

    # Loss: -log(sigma(u_o^T v_i)) - sum_k log(sigma(-u_k^T v_i))
    loss = -np.log(pos_score) - np.sum(np.log(neg_scores))

    # Gradient w.r.t. input vector v_i
    grad_input_vec = -(1 - pos_score) * u_o + ((1 - neg_scores) @ u_neg)   # (dim,)

    # Gradient w.r.t. all output vectors (sparse: only o and negative indices)
    grad_output_vecs = np.zeros(output_vectors.shape)
    grad_output_vecs[output_word_idx] = -(1 - pos_score) * input_vector

    # Accumulate gradients for negative samples (handles duplicates via np.add.at)
    np.add.at(grad_output_vecs, neg_sample_word_indices,
              np.outer(1 - neg_scores, input_vector))

    ### END YOUR CODE

    return loss, grad_input_vec, grad_output_vecs


def skipgram(current_input_word, window_size, output_words, word2_ind,
             input_vectors, output_vectors, dataset,
             word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    current_input_word -- a string of the current center word
    window_size -- integer, context window size
    output_words -- list of no more than 2*windowSize strings, the outside words
    word2_ind -- a dictionary that maps words to their indices in
              the word vector list
    input_vectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    output_vectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    dataset -- dataset for generating negative samples
    word2vec_loss_and_gradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.
    Return:
    loss -- the loss function value for the skip-gram model
            (L in the pdf handout)
    grad_input_vecs -- the gradient with respect to the center word vectors
            (dL / dV, this should have the same shape with V)
    grad_output_vecs -- the gradient with respect to the outside word vectors
                        (dL / dU)
    """

    loss = 0.0
    grad_input_vecs = np.zeros(input_vectors.shape)
    grad_output_vecs = np.zeros(output_vectors.shape)

    ### YOUR CODE HERE (~8 Lines)

    # Look up the center (input) word index and its embedding vector
    center_word_idx = word2_ind[current_input_word]
    v_i = input_vectors[center_word_idx]                        # (dim,)

    # Sum loss and gradients over each context (output) word in the window
    for output_word in output_words:
        output_word_idx = word2_ind[output_word]
        l, g_in, g_out = word2vec_loss_and_gradient(
            v_i, output_word_idx, output_vectors, dataset
        )
        loss += l
        grad_input_vecs[center_word_idx] += g_in
        grad_output_vecs += g_out

    ### END YOUR CODE

    return loss, grad_input_vecs, grad_output_vecs


def word2vec_sgd_wrapper(word2vec_model, word2_ind, word_vectors, dataset,
                         window_size,
                         word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    batch_size = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]

    input_vectors = word_vectors[:int(N / 2), :]
    output_vectors = word_vectors[int(N / 2):, :]

    for i in range(batch_size):
        window_size1 = random.randint(1, window_size)
        input_word, context = dataset.get_random_context(window_size1)

        c, gin, gout = word2vec_model(
            input_word, window_size1, context, word2_ind, input_vectors,
            output_vectors, dataset, word2vec_loss_and_gradient
        )
        loss += c / batch_size
        grad[:int(N / 2), :] += gin / batch_size
        grad[int(N / 2):, :] += gout / batch_size

    return loss, grad
