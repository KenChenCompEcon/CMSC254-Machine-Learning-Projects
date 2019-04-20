# -*- coding: utf-8 -*-
"""
Project: Neural Network @CMSC 25400
Created on Sat Mar  2 14:21:37 2019
@author: Ken CHEN
"""
import numpy as np
import matplotlib.pyplot as plt

def layer_dims(input_size, output_size, N, L):
    '''
    return: the structure of the neural network
    '''
    layers = [input_size] + [N]*(L-1) + [output_size]
    return layers

def initialize(layers, seed = 25400):
    np.random.seed(seed)
    N = layers
    L = len(layers)
    W = {}; B ={}
    for l in range(1, L):
        W[l] = np.random.normal(size=(N[l-1], N[l]))*0.1
        B[l] = np.random.randn(N[l])*0.1
        # these parameters were multiplied by 0.1, in case the activations 
        # are so large that the sigmoid function can generate very wek gradients
    return W, B

# Foward propagation
def linear_forward(a, w, b):
    z = np.dot(a, w) + b
    return z

def sigmoid_forward(z):
    a = 1/(1 + np.exp(-z))
    return a

def softmax_output(a):
    out = np.exp(a)/np.sum(np.exp(a), axis=1, keepdims = True)
    return out

def forward_propagate(a_0, W, B):
    L = len(W)
    a = a_0
    A = [a]
    for l in range(1, L):
        z = linear_forward(a, W[l], B[l])
        a = sigmoid_forward(z)
        A.append(a)
    zL = linear_forward(a, W[L], B[L])
    aL = sigmoid_forward(zL)
    A.append(aL)
    return aL, A

def label_mat_gen(labels):
    m = len(labels)
    label_mat = np.zeros((m, 10))
    for coor in zip(range(m), labels):
        label_mat[coor] = 1
    return label_mat

def comp_loss(aL, label_mat):
    m = label_mat.shape[0]
    loss = 1/m * np.sum((label_mat - aL)**2)
    return loss

#Backward Propagation
def sigmoid_backward(da, a):
    dz = da*a*(1-a)
    return dz
    
def output_backward(label_mat, aL):
    daL = 2*(aL - label_mat)
    dz = sigmoid_backward(daL, aL)
    return dz

def linear_backward(dz, w, b, aLag):
    m = aLag.shape[0]
    dw = 1/m*np.dot(aLag.T, dz)
    db = 1/m*np.sum(dz, axis = 0)
    daLag = np.dot(dz, w.T)
    return daLag, dw, db

def backward_propagate(aL, label_mat, A, W, B):
    dz = output_backward(label_mat, aL)
    dW, dB = [], []
    for l in range(len(W),0,-1):
        w = W[l]; b = B[l]; aLag = A[l-1]
        da, dw, db = linear_backward(dz, w, b, aLag)
        dW.insert(0, dw); dB.insert(0, db)
        dz = sigmoid_backward(da, aLag)
    return dW, dB

def predict(x, W, B):
    aL, _ = forward_propagate(x, W, B)
    pred = softmax_output(aL)
    return pred

def test(W, B, x_test, y_test):
    pred = predict(x_test, W, B)
    error_rate = 1*(np.argmax(pred, 1) != y_test).sum()/len(y_test)
    return error_rate

def train(x_train, y_train, output_size, epoch, N, L, lr, minibatch_size):
#    x_holdout = x_train[-5000:]; y_holdout = y_train[-5000:]
#    errors = list(range(50, 0, -1))
    input_size = x_train.shape[1]
    layers = layer_dims(input_size, output_size, N, L)
    W, B = initialize(layers)
    for i in range(epoch):
        select = np.random.choice(50000, minibatch_size, replace = False)
        a_0 = x_train[select]
        labels = y_train[select]  
        label_mat = label_mat_gen(labels)
        aL, A = forward_propagate(a_0, W, B)
        dW, dB = backward_propagate(aL, label_mat, A, W, B)
        for j in range(1, L+1):
            W[j] -= lr*dW[j-1]
            B[j] -= lr*dB[j-1]
#        errors.append(test(W, B, x_holdout, y_holdout))
#        if errors[-1] > sum(errors[-50:]): break
    return W, B

def experiment(x_train, y_train, output_size, epoch, experiment_var, lst):
    param_dict = {'N': 0, 'L': 1, 'lr': 2, 'minibatch_size': 3}
    idx = param_dict[experiment_var]
    param_lst = [256, 3, 0.1, 100]
    error_rates = []
    for var in lst:
        param_lst[idx] = var
        W, B = train(x_train, y_train, output_size, epoch, *param_lst)
        error_rate = test(W, B, x_test, y_test)
        error_rates.append(error_rate)
    print("For {}, I select the number to be {}"
          .format(experiment_var, lst[np.argmin(error_rates)]))
    return error_rates
        
if __name__ == '__main__':
    x_train = np.loadtxt("TrainDigitX.csv", delimiter = ',')
    y_train = np.loadtxt("TrainDigitY.csv", delimiter = ',', dtype=int)
    x_test = np.loadtxt("TestDigitX.csv", delimiter = ',')
    y_test = np.loadtxt("TestDigitY.csv", delimiter = ',', dtype=int)
    x_test2 = np.loadtxt("TestDigitX2.csv", delimiter = ',')
    print("data loaded")
    print("--------------------------------------------------------------------")
    print("Let's start experimenting with the parameters")
    output_size = 10
    """
    Experimenting the the N, L, lr, and minibatch_size parameters
    """
    errors_N = experiment(x_train, y_train, output_size, 8000, 'N', [32, 64, 128, 256])
    errors_L = experiment(x_train, y_train, output_size, 8000, 'L', [2, 3, 4, 5])
    errors_lr = experiment(x_train, y_train, output_size, 8000, 'lr', [0.01, 0.05, 0.1, 0.2])
    errors_batch = experiment(x_train, y_train, output_size, 8000, 'minibatch_size', [50, 100, 200, 500])
    fig, axes = plt.subplots(2,2)
    fig.set_size_inches((10, 6))
    ax = axes[0,0]
    ax.plot([32, 64, 128, 256], errors_N); ax.set_xlabel("Number of neurons at hidden layers")
    ax = axes[0,1]
    ax.plot([2, 3, 4, 5], errors_L); ax.set_xlabel("Number of layers")
    ax = axes[1,0]
    ax.plot([0.01, 0.05, 0.1, 0.2], errors_lr); ax.set_xlabel("Learning rate")
    ax = axes[1,1]
    ax.plot([50, 100, 200, 500], errors_N); ax.set_xlabel("Size of the minibatches")
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Error rate")
    plt.show()
    """
    Experimenting with the number of epoches
    """
    error_epoch = []
    for epoch in [8000, 10000, 20000, 30000, 40000, 50000, 60000]:
        W,B = train(x_train, y_train, output_size, epoch, 256, 3, 0.2, 500)
        error = test(W, B, x_test, y_test)
        error_epoch.append(error)
    plt.plot([8000, 10000, 20000, 30000, 40000, 50000, 60000], error_epoch)
    plt.xlabel("Number of epoches"); plt.ylabel("Error rate")
    plt.show()
    print("--------------------------------------------------------------------")
    print("Let's start testing with the optimal parameters")
    N = 256; L = 3; lr = 0.2; minibatch_size = 500
    W,B = train(x_train, y_train, output_size, 60000, N, L, lr, minibatch_size)
    error = test(W, B, x_test, y_test)
    print("The test error rate is: {:.2%}".format(error))
    print("--------------------------------------------------------------------")
    print("Now we get the prediction on the test set 1")
    W_opt = W; B_opt = B
    pred1 = predict(x_test, W_opt, B_opt); pred1_label = np.argmax(pred1, 1)
    np.savetxt("TestDigitX.predict", pred1_label, fmt = "%i")
    fig, axes = plt.subplots(1, 9)
    for i in range(9):
        ax = axes[i]
        idx = (pred1_label==i)
        ax.imshow(x_test[idx].mean(axis=0).reshape(28,28))
    plt.show()
    print("Now we get the prediction on the test set 2")
    pred2 = predict(x_test, W_opt, B_opt); pred2_label = np.argmax(pred2, 1)
    np.savetxt("TestDigitX2.predict", pred2_label, fmt = "%i")
    fig, axes = plt.subplots(1, 9)
    for i in range(9):
        ax = axes[i]
        idx = (pred2_label==i)
        ax.imshow(x_test[idx].mean(axis=0).reshape(28,28))
    plt.show()