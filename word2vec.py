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

    # loss

    # grad_input_vec, grad_output_vecs

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
