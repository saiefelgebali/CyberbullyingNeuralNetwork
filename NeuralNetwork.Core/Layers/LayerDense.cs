using System;
using Accord.Math;

namespace NeuralNetwork.Core.Layers
{
    public class LayerDense : NetworkLayer
    {
        // Shape
        public int NumInputs { get; private set; }
        public int NumNeurons { get; private set; }
        // Network
        public double[][] Weights { get; set; }
        public double[] Biases { get; set; }

        // Regularization
        public double WeightsL1 { get; set; }
        public double BiasesL1 { get; set; }
        public double WeightsL2 { get; set; }
        public double BiasesL2 { get; set; }

        // Derivatives
        public double[][] DWeights { get; set; }
        public double[] DBiases { get; set; }

        // Optimizer properties
        public double[][] WeightMomentums { get; set; }
        public double[] BiasMomentums { get; set; }
        public double[][] WeightCache { get; set; }
        public double[] BiasCache { get; set; }

        public LayerDenseParams GetParameters()
        {
            return new LayerDenseParams(Weights, Biases);
        }

        public void SetParameters(LayerDenseParams parameters)
        {
            Weights = parameters.Weights;
            Biases = parameters.Biases;
        }

        /// <summary>
        /// Create a new dense layer with a random weights.
        /// - Weights are uniformly distributed between -1.0 and 1.0
        /// - Biases are all initialized to 0
        /// </summary>
        /// <param name="numInputs">Number of inputs to layer</param>
        /// <param name="numNeurons">Number of neurons in layer</param>
        public LayerDense(int numInputs, int numNeurons, 
            double weightsL1 = 0, double biasesL1 = 0, 
            double weightsL2 = 0, double biasesL2 = 0)
        {
            // Set layer shape
            NumInputs = numInputs;
            NumNeurons = numNeurons;

            // Initialize weights and biases
            Weights = Jagged.Random(numInputs, numNeurons, -1.0, 1.0).Multiply(0.01);
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
            double[][] initialWeights, 
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
            for (int i = 0; i < Output.Rows(); i++)
            {
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
            DWeights = Inputs.Transpose().Dot(dValues);
            DBiases = dValues.Sum(dimension: 0);

            // Gradients on regularization
            // Weights Regularization
            if (WeightsL1 != 0)
            {
                double[][] dL1 = Jagged.Ones(Weights.Rows(), Weights.Columns());
                for (int i = 0; i < Weights.Rows(); i++)
                {
                    for (int j = 0; j < Weights.Columns(); j++)
                    {
                        if (Weights[i][j] < 0) dL1[i][j] = -1;
                    }
                }
                DWeights = DWeights.Add(dL1.Multiply(WeightsL1));
            }
            if (WeightsL2 != 0)
            {
                DWeights = DWeights.Add(Weights.Multiply(2 * WeightsL2));
            }

            // Biases Regularization
            if (BiasesL1 != 0)
            {
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

            // Gradient on values
            DInputs = dValues.Dot(Weights.Transpose());
        }
    }
}
