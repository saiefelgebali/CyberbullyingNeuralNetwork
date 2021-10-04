using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Statistics;
using Accord.Math;

namespace NeuralNetwork.Core.Loss
{
    public abstract class Loss
    {
        // Calculates the data and regularization losses
        // given model output and ground truth values
        public double Calculate(double[,] output, int[] y)
        {
            // Calculate sample losses
            double[] sampleLosses = Forward(output, y);

            // Calculate mean loss
            double dataLoss = Measures.Mean(sampleLosses);

            // Return data loss value
            return dataLoss;
        }

        public static double RegularizationLoss(LayerDense layer)
        {
            double regularizationLoss = 0;

            // Weights Regularization
            if (layer.WeightsL1 != 0)
            {
                regularizationLoss += layer.Weights.Abs().Sum() * layer.WeightsL1;
            }
            if (layer.WeightsL2 != 0)
            {
                regularizationLoss += layer.Weights.Pow(2).Sum() * layer.WeightsL2;
            }
            
            // Bias Regularization
            if (layer.BiasesL1 != 0)
            {
                regularizationLoss += layer.Biases.Abs().Sum() * layer.BiasesL1;
            }
            if (layer.BiasesL2 != 0)
            {
                regularizationLoss += layer.Biases.Pow(2).Sum() * layer.BiasesL2;
            }

            return regularizationLoss;
        }

        public static double Accuracy(double[,] yPred, int[] yTrue)
        {
            double[] correct = new double[yPred.Rows()];
            for (int i = 0; i < correct.Rows(); i++)
            {
                int predicted = yPred.GetRow(i).ArgMax();
                correct[i] = predicted == yTrue[i] ? 1 : 0;
            }

            // Return % of accurate results
            return Measures.Mean(correct);
        }

        protected abstract double[] Forward(double[,] yPred, int[] yTrue);
    }
}
