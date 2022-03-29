class LayerDense:

    # Initialize the layer with the correct orders
    # for the weights matrix and the biases vector.
    def init(n_inputs, n_outputs):
        this.Weights = Matrix.Random(n_inputs, n_outputs)
        this.Biases = Vector.Zeros(n_outputs)

        # Perform forward propagation through this layer
        # by taking the dot product of the inputs and weights
        # with an added bias.
    def forward(X):
        this.Inputs = X
        return X.dot(this.Weights) + this.Biases

        # Perform backpropagation through this layer
        # and set the derived matrices of the
        # weights, biases and inputs, so that they
        # may be used by an optimizer later.
    def backward(d_values):
        d_weights = d_values.dot(this.Inputs)
        d_inputs = d_values.dot(this.Weights)
        d_biases = d_values.sum(axis=1)
        return d_inputs
