from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        scores = X[i].dot(W)
        scores_exp = np.exp(scores)
        probs = scores_exp / np.sum(scores_exp)
        
        correct_class_prob = probs[y[i]]
        loss -= np.log(correct_class_prob)
        
        for j in range(num_classes):
            if j != y[i]:
                dW[:, j] += X[i] * probs[j]
            else:
                dW[:, j] += X[i] * (probs[j] - 1)
        
    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    scores = X.dot(W) # N*C
    
    exp_scores = np.exp(scores) # N*C
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # N*C
    
    correct_probs = probs[np.arange(num_train), y] # N*1
    correct_log_probs = -np.log(correct_probs) # N*1
    
    data_loss = np.sum(correct_log_probs) / num_train # scalar
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss
    
    # 由 N*1 训练标签y构建 N*C 标签矩阵
    label = np.zeros((num_train, num_classes))
    label[np.arange(num_train), y] = 1
    
    dW = X.T.dot(probs - label) / num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
