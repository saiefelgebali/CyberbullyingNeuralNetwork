using Accord.Math;
using Accord.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.MLP.Losses
{
    public class LossBinaryCrossentropy : Loss
    {
        protected override double[] Forward(double[][] yPred, int[] yTrue)
        {
            // Clip data to prevent log by 0
            for (int i = 0; i < yPred.Rows(); i++)
            {
                for (int j = 0; j < yPred.Columns(); j++)
                {
                    yPred[i][j] = Math.Clamp(yPred[i][j], 1e-7, 1 - 1e-7);
                }
            }

            // Calculate sample losses
            double[][] yTrueTransposed = yTrue.ToDouble().Transpose().ToJagged();

            // yTrue * log(yPred)
            double[][] lossPositive = yTrueTransposed.Multiply(yPred.Log());

            // (1-yTrue) * log(1-yPred)
            double[][] lossNegative = yTrueTransposed.Multiply(-1).Add(1).Multiply(yPred.Multiply(-1).Add(1).Log());

            // -(yTrue* log(yPred) + (1 - yTrue) * log(1 - yPred))
            double[][] sampleLossesSplit = lossPositive.Add(lossNegative).Multiply(-1);

            // Calculate mean of +/- sample losses
            double[] sampleLosses = Measures.Mean(sampleLossesSplit, dimension: 1);

            return sampleLosses;
        }

        public override void Backward(double[][] dValues, int[] yTrue)
        {
            // Number of samples in batch
            int samplesLength = dValues.Rows();

            // Number of outputs in every sample
            int outputLength = dValues.Columns();

            // Clip data to prevent division by 0
            for (int i = 0; i < dValues.Rows(); i++)
            {
                for (int j = 0; j < dValues.Columns(); j++)
                {
                    dValues[i][j] = Math.Clamp(dValues[i][j], 1e-7, 1 - 1e-7);
                }
            }

            // Caclulate gradient
            double[][] yTrueTransposed = yTrue.ToDouble().Transpose().ToJagged();

            // yTrue / dValues
            double[][] gradPositive = yTrueTransposed.Divide(dValues);

            // (1 - yTrue) / (1 - dValues)
            double[][] gradNegative = yTrueTransposed.Multiply(-1).Add(1).Divide(dValues.Multiply(-1).Add(1));

            // -((yTrue / dValues) - ((1 - yTrue) / (1 - dValues))) / outputLength
            DInputs = gradPositive.Subtract(gradNegative).Multiply(-1).Divide(outputLength);

            // Normalize gradient
            DInputs = DInputs.Divide(samplesLength);
        }
    }
}
