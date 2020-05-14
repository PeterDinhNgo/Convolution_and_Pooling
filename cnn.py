from mxnet import autograd, np, npx
from mxnet.gluon import nn
import numpy as np
import math
npx.set_np()

H = np.zeros((5,5))
H[:, 0] = 1
H[:, 4] = 1
H[2,: ] = 1

K = np.zeros((2,2))
K[0,1] = 1
K[1,0] = 1

pool = np.zeros((3,3))

H = np.ones((5,5))
J = np.diag(np.diag(H))
L = np.fliplr(J)
X = J + L
X[2,2] = 1

# Average Pooling
def pool2d(X, pool, mode):
    p_h, p_w = pool.shape
    Y = np.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = np.max(X[i: i + p_h, j: j + p_w])
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

avg_pool = pool2d(H, pool, 'avg')
print(avg_pool)


H = np.ones((5,5))
J = np.diag(np.diag(H))
L = np.fliplr(J)
X = J + L
X[2,2] = 1
R = np.zeros((1,5))
newX = np.vstack([X, R])
T = np.zeros((6,1))
newerX = np.hstack([newX, T])

# [[1. 0. 0. 0. 1. 0.]
#  [0. 1. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0.]
#  [0. 1. 0. 1. 0. 0.]
#  [1. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0.]]

K = np.zeros((2,2))
K[0,1] = 1
K[1,0] = 1

# [[0. 1.]
#  [1. 0.]]

def convolution(newerX, X, K):
     h, w = K.shape
     Y_Height = (X.shape[0] - h + 1 + 2)/2
     Y_Width = (X.shape[1] - w + 1 + 2)/2
     Y = np.zeros((math.floor(Y_Height), math.floor(Y_Width)))
     for i in range(Y.shape[0],2):
        for j in range(Y.shape[1],2):
             Y[i,j] = (newerX[i: i + h , j: j + w ] * K).sum()  
     return Y

cnn = convolution(newerX, X, K)
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]



