# notes

## finding outputs to the next layer using one sample
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

weights is 2-d and inputs is 1-d, so np.dot is a sum product over the last axis (4) of weights and inputs
[[0.2, 0.8, -0.5, 1]
 [0.5, -0.91, 0.26, -0.5]           [1, 2, 3, 2.5]
 [-0.26, -0.27, 0.17, 0.87]]
3,4                                 4,
4 is the last axis, dot product of each of the 3 4by4 vectors is calculated and returned as a list
layer = np.dot(weights, inputs) + biases 

## matmul practice
rv = np.array([[1, 2, 3]]) # this is a row vector, a row vector has double angle brackets because that signifies its a part of a bigger matrix (i think)
cv = np.array([[4, 5, 6]]).T # this is a col vector, its the same as a row vector but transposed
np.dot(rv, cv)
np.matmul(rv, cv)
rv @ cv
