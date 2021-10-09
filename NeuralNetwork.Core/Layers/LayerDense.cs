using System;
using Accord.Math;

namespace NeuralNetwork.Core
{
    public class LayerDense : NetworkLayer
    {
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

        // Init a new layer with dense connections
        public LayerDense(int numInputs, int numNeurons, 
            double weightsL1 = 0, double biasesL1 = 0, 
            double weightsL2 = 0, double biasesL2 = 0)
        {
            // Initialize weights and biases
            Weights = Jagged.Random(numInputs, numNeurons, -1.0, 1.0).Multiply(0.01);
            Biases = Vector.Zeros(numNeurons);

            // Set regularization strength
            WeightsL1 = weightsL1;
            WeightsL2 = weightsL2;
            BiasesL1 = biasesL1;
            BiasesL2 = biasesL2;
        }

        public override void Forward(double[][] inputs, bool training = false)
        {
            Inputs = inputs;

            // Weighted Sum of inputs
            Output = Inputs.Dot(Weights);

            // Add bias
            for (int i = 0; i < Output.Rows(); i++)
            {
                Output.SetRow(i, Output.GetRow(i).Add(Biases));
            }
        }

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
