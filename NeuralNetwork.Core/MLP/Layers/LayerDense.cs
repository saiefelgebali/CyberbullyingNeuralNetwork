using System;
using Accord.Math;

namespace NeuralNetwork.Core.MLP.Layers
{
    // This class inherits from the LayerMLP base class
    // therefore it gains access to all the methods and properties of the base class
    public class LayerDense : LayerMLP
    {
        // Shape
        // These are definitions for properties of the LayerDense class
        // 'public' means the property can be accessed from outside the layer object
        public int NumInputs { get; private set; } // 'private set;' means this property can only be set from within the class
        public int NumNeurons { get; private set; }

        // Network
        public double[][] Weights { get; set; } // this is a matrix of weights, where each row is an input and each column is a neuron
        public double[] Biases { get; set; }

        // Regularization is a technique that helps prevent overfitting by penalizing large weights
        // These are properties of the LayerDense class that are used to control the regularization of the layer parameters
        public double WeightsL1 { get; set; }
        public double BiasesL1 { get; set; }
        public double WeightsL2 { get; set; }
        public double BiasesL2 { get; set; }

        // Derivatives
        public double[][] DWeights { get; set; } // this is a matrix of the derivatives of the weights
        public double[] DBiases { get; set; }

        // Optimizer properties
        // These are properties used by the Adam optimizer to update the weights and biases
        public double[][] WeightMomentums { get; set; }
        public double[] BiasMomentums { get; set; }
        public double[][] WeightCache { get; set; }
        public double[] BiasCache { get; set; }

        /// <summary>
        /// Create a new dense layer with a random weights.
        /// - Weights are uniformly distributed between -1.0 and 1.0
        /// - Biases are all initialized to 0
        /// </summary>
        /// <param name="numInputs">Number of inputs to layer</param>
        /// <param name="numNeurons">Number of neurons in layer</param>
        public LayerDense(int numInputs, int numNeurons, // these are the required arugments to this constructor method
            double weightsL1 = 0, double biasesL1 = 0, // these arguments are optional, and the default values are 0
            double weightsL2 = 0, double biasesL2 = 0)
        {
            // Set layer shape
            NumInputs = numInputs;
            NumNeurons = numNeurons;

            // Initialize weights and biases

            // This initializes the weights to a random matrix of values between -0.01 and 0.01
            Weights = Jagged.Random(numInputs, numNeurons, -1.0, 1.0).Multiply(0.01);

            // This initializes the biases to a vector of zeros
            Biases = Vector.Zeros(numNeurons);

            // Set regularization strength
            WeightsL1 = weightsL1;
            WeightsL2 = weightsL2;
            BiasesL1 = biasesL1;
            BiasesL2 = biasesL2;
        }

        /// <summary>
        /// Create a new dense layer with predefined weights and biases.
        /// </summary>
        /// <param name="numInputs">Number of inputs to layer</param>
        /// <param name="numNeurons">Number of neurons in layer</param>
        /// <param name="initialWeights">Weights must be a matrix with shape (numInputs x numNeurons)</param>
        /// <param name="initialBiases">Weights must be an array with length (numNeurons)</param>
        /// <exception cref="ArgumentException"></exception>
        public LayerDense(
            int numInputs, int numNeurons,
            double[][] initialWeights, // take the initial weights matrix as input
            double[] initialBiases,
            double weightsL1 = 0, double biasesL1 = 0, 
            double weightsL2 = 0, double biasesL2 = 0)
        {
            // Set layer shape
            NumInputs = numInputs;
            NumNeurons = numNeurons;

            // Validate weights shape
            if (initialWeights.Rows() != numInputs 
                || initialWeights.Columns() != numNeurons)
            {
                // If the initialWeights matrix is not the correct shape, throw an exception
                // the 'throw' keyword is used to raise an error and stop the program, when something goes wrong
                throw new ArgumentException("initialWeights is of incorrect shape. Must be (numInputs x numNeurons)");
            }

            // Validate biases shape
            if (initialBiases.Length != numNeurons)
            {
                throw new ArgumentException("initialBiases is of incorrect length. Must be same length as numNeurons");
            }

            // Initialize weights and biases
            Weights = initialWeights;
            Biases = initialBiases;

            // Set regularization strength
            WeightsL1 = weightsL1;
            WeightsL2 = weightsL2;
            BiasesL1 = biasesL1;
            BiasesL2 = biasesL2;
        }

        /// <summary>
        /// Perform a forward pass through the layer.
        /// - Takes the weighted sum of inputs and adds a bias.
        /// </summary>
        /// <param name="inputs">
        /// Must be a matrix with shape (N x numInputs) where N is the number of samples.
        /// </param>
        /// <exception cref="ArgumentException"></exception>
        public override void Forward(double[][] inputs, bool training = false)
        {
            if (inputs.Columns() != NumInputs)
            {
                throw new ArgumentException("Inputs to layer must be of shape (N x NumInputs)");
            }

            Inputs = inputs;

            // Weighted Sum of inputs
            Output = Inputs.Dot(Weights);

            // Add bias
            // Loop through all rows in the output matrix
            for (int i = 0; i < Output.Rows(); i++)
            {
                // Add the bias vector to each row vector
                Output.SetRow(i, Output.GetRow(i).Add(Biases));
            }
        }

        /// <summary>
        /// Perform a backward pass through layer.
        /// - Calculate the gradient with respect to Weights, Biases and Inputs.
        /// </summary>
        /// <param name="dValues">
        /// The gradient with respect to the inputs of the next layer in the model.
        /// Chain rule is used to calculate the gradient with respect to this layer's parameters.
        /// </param>
        public override void Backward(double[][] dValues)
        {
            // Gradients on params
            DWeights = Inputs.Transpose().Dot(dValues); // Transpose flips the matrix so that the rows are columns and columns are rows
            DBiases = dValues.Sum(dimension: 0); // 'dimension: 0' tells the Sum function to sum the rows of the matrix, and not the columns

            // Gradients on regularization
            // Weights Regularization
            if (WeightsL1 != 0)
            {
                // Find the derivative of the L1 regularization with respect to the weights
                double[][] dL1 = Jagged.Ones(Weights.Rows(), Weights.Columns());
                for (int i = 0; i < Weights.Rows(); i++)
                {
                    for (int j = 0; j < Weights.Columns(); j++)
                    {
                        if (Weights[i][j] < 0) dL1[i][j] = -1;
                    }
                }
                // Add the L1 regularization derivative to the gradient of the weights
                DWeights = DWeights.Add(dL1.Multiply(WeightsL1));
            }
            if (WeightsL2 != 0)
            {
                // Add the L2 regularization derivative to the gradient of the weights
                DWeights = DWeights.Add(Weights.Multiply(2 * WeightsL2));
            }

            // Biases Regularization
            if (BiasesL1 != 0)
            {
                // Perform L1 regularization on the biases
                double[] dL1 = Vector.Ones(Biases.Length);
                for (int i = 0; i < Biases.Length; i++)
                {
                    if (Biases[i] < 0) dL1[i] = -1;
                }
                DBiases = DBiases.Add(dL1.Multiply(WeightsL1));
            }
            if (BiasesL2 != 0)
            {
                DBiases = DBiases.Add(Biases.Multiply(2 * BiasesL2));
            }

            // Gradient of the input values
            DInputs = dValues.Dot(Weights.Transpose());
        }

        /// <summary>
        /// This method outputs the layer's parameters as an object,
        /// which can be converted to JSON.
        /// </summary>
        public LayerDenseParams GetParameters()
        {
            // the 'return' keyword declares the output of the method
            return new LayerDenseParams(Weights, Biases);
        }

        /// <summary>
        /// This method sets the layer's parameters from an object,
        /// which can be converted from JSON.
        /// - The layer parameters are the weights and biases.
        /// </summary>
        public void SetParameters(LayerDenseParams parameters) // take the paramters object as an argument
        {
            Weights = parameters.Weights;
            Biases = parameters.Biases;
        }
    }
}
