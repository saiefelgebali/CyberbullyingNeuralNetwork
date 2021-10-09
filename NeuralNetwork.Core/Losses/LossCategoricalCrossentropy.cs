using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.Losses
{
    public class LossCategoricalCrossentropy : Loss
    {
        protected override double[] Forward(double[][] yPred, int[] yTrue)
        {
            // Number of samples in batch
            int samplesLength = yPred.Rows();

            // Clip data to prevent division by 0
            for (int i = 0; i < yPred.Rows(); i++)
            {
                for (int j = 0; j < yPred.Columns(); j++)
                {
                    yPred[i][j] = Math.Clamp(yPred[i][j], 1e-7, 1 - 1e-7);
                }
            }

            // Probabilities for target values
            double[] correctConfidences = new double[samplesLength];

            for (int i = 0; i < samplesLength; i++)
            {
                correctConfidences[i] = yPred[i][yTrue[i]];
            }

            // Calculate losses - negative log function
            double[] losses = correctConfidences.Log().Multiply(-1);

            return losses;
        }

        public override void Backward(double[][] dValues, int[] yTrue)
        {
            // Number of samples in batch
            int samplesLength = dValues.Rows();

            // Number of labels in every sample
            int labelsLength = dValues.Columns();

            // Calculate one-hot vectors of labels
            double[][] oneHotLabels = Jagged.OneHot(yTrue, labelsLength);

            // Caclulate gradient
            DInputs = oneHotLabels.Multiply(-1).Divide(dValues);

            // Normalize gradient
            DInputs = DInputs.Divide(samplesLength);
        }
    }
}
