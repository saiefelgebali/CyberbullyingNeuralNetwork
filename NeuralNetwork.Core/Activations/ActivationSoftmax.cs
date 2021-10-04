using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Core.Activations
{
    public class ActivationSoftmax
    {
        public double[,] Inputs { get; set; }
        public double[,] DInputs { get; set; }
        public double[,] Output { get; set; }

        public void Forward(double[,] inputs)
        {
            Inputs = inputs;

            // Get unnormalised probabilities
            double[,] expValues = Inputs.Copy();
            for (int i = 0; i < expValues.Rows(); i++)
            {
                double maxValue = expValues.GetRow(i).Max();
                for (int j = 0; j < expValues.Columns(); j++)
                {
                    expValues[i, j] = Math.Exp(expValues[i, j] - maxValue);
                }
            }

            // Normalise them for each sample
            double[,] probabilities = expValues.Copy();
            for (int i = 0; i < probabilities.Rows(); i++)
            {
                double sampleSum = probabilities.GetRow(i).Sum();
                for (int j = 0; j < probabilities.Columns(); j++)
                {
                    probabilities[i, j] = probabilities[i, j] / sampleSum;
                }
            }

            Output = probabilities;
        }

        public void Backward(double[,] dValues)
        {
            // Create an uninitialized array
            DInputs = Matrix.Zeros(dValues.Rows(), dValues.Columns());

            for (int i = 0; i < dValues.Rows(); i++)
            {
                double[] outputRow = Output.GetRow(i);

                // Flatten output array
                double[,] singleOutput = outputRow.Transpose();

                // Calculate Jacobian matrix of the output
                double[,] a = Matrix.Diagonal(outputRow);
                double[,] b = singleOutput.Dot(singleOutput.Transpose());
                double[,] jacobianMatrix = a.Subtract(b);

                // Calculate sample-wise gradient and add it to the array of sample gradients
                DInputs.SetRow(i, jacobianMatrix.Dot(dValues.GetRow(i)));
            }
        }
    }
}
