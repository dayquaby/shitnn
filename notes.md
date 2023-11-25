# notes

## finding outputs to the next layer using one sample
 - inputs = [1, 2, 3, 2.5]

 - weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

 - biases = [2.0, 3.0, 0.5]

 - weights is 2-d and inputs is 1-d, so np.dot is a sum product over the last axis (4) of weights and inputs

[[0.2, 0.8, -0.5, 1]

 [0.5, -0.91, 0.26, -0.5]

 [-0.26, -0.27, 0.17, 0.87]] (size (3,4))

 multiplied by

 [1, 2, 3, 2.5] (size (4,))

4 is the last axis, dot product of each of the 3 4by4 vectors is calculated and returned as a list

- layer = np.dot(weights, inputs) + biases

## matmul practice
- rv = np.array([[1, 2, 3]]) # this is a row vector, a row vector has double angle brackets because that signifies its a part of a bigger matrix (i think)

- cv = np.array([[4, 5, 6]]).T # this is a col vector, its the same as a row vector but transposed

- different ways of doing matmul

np.dot(rv, cv)

np.matmul(rv, cv)

rv @ cv

## example forward pass with weights and biases written by hand

- inputs contains multiple samples to be ran at once, each row is a sample

inputs = np.array([
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
])

- weights contains weight values corresponding to each exit node, is of size (n_neurons, n_inputs) - each row is for one neuron on next layer

weights = np.array([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
])

weights2 = np.array([
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13],
])

- biases contain bias for each neuron in the layer, stays the same across different samples and weights

biases = [2.0, 3.0, 0.5]

biases2 = [-1, 2, -0.5]

this is a 2-D vector adding a 1-D vector, which means the biases vector gets added to each row of the 2-D vector (broadcasting)

for example: adding a (3,3) matrix to a (1,3) array makes the array automatically broadcast to fit the size of the matrix (it becomes a (3,3) matrix with all the rows being the exact same)

- layer1_outputs = inputs @ weights.T + biases
- layer2_outputs = layer1_outputs @ weights2.T + biases2

## finding the loss for categorical crossentropy

one-hot encoding is a way to show the target values for the true prediction of a categorical neural network

- for a nn that has 3 classes - dog, cat, and person, the one-hot encoding for 3 samples could look like this: [[0, 1, 0], [1, 0, 0], [0, 1, 0]], where each row describes a sample, and the index of the row that contains a 1 shows which of the 3 classes is the correct prediction for that sample (cat, dog, cat)

scarce encoding is another way to show which class is the true prediction of a categorical neural network

- instead of having either a 0 or 1 value for each output node, scarce encoding is just a 1-dimensional array of the correct indexes for each sample. for example, a scarce encoding of the above example would be [1, 0, 1]