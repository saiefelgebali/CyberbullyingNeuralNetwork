using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Core.MLP.Activations
{
    public class ActivationSoftmax : Activation
    {
        public override void Forward(double[][] inputs, bool training = false )
        {
            Inputs = inputs;

            // Get unnormalised probabilities
            double[][] expValues = Inputs.Copy();
            for (int i = 0; i < expValues.Rows(); i++)
            {
                double maxValue = expValues.GetRow(i).Max();
                for (int j = 0; j < expValues.Columns(); j++)
                {
                    expValues[i][j] = Math.Exp(expValues[i][j] - maxValue);
                }
            }

            // Normalise them for each sample
            double[][] probabilities = expValues.Copy();
            for (int i = 0; i < probabilities.Rows(); i++)
            {
                double sampleSum = probabilities.GetRow(i).Sum();
                for (int j = 0; j < probabilities.Columns(); j++)
                {
                    probabilities[i][j] = probabilities[i][j] / sampleSum;
                }
            }

            Output = probabilities;
        }

        public override void Backward(double[][] dValues)
        {
            // Create an uninitialized array
            DInputs = Jagged.Zeros(dValues.Rows(), dValues.Columns());

            for (int i = 0; i < dValues.Rows(); i++)
            {
                double[] outputRow = Output.GetRow(i);

                // Flatten output array
                double[][] singleOutput = outputRow.Transpose().ToJagged();

                // Calculate Jacobian matrix of the output
                double[][] a = Jagged.Diagonal(outputRow);
                double[][] b = singleOutput.Dot(singleOutput.Transpose());
                double[][] jacobianMatrix = a.Subtract(b);

                // Calculate sample-wise gradient and add it to the array of sample gradients
                DInputs.SetRow(i, jacobianMatrix.Dot(dValues.GetRow(i)));
            }
        }

        public override int[] Predictions()
        {
            int[] predictions = new int[Output.Rows()];

            // Get predictions for all samples
            for (int i = 0; i < Output.Rows(); i++)
            {
                int predIndex = Output.GetRow(i).ArgMax();
                predictions[i] = predIndex;
            }

            return predictions;
        }
    }
}
