using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Statistics;
using Accord.Math;
using NeuralNetwork.Core.MLP.Layers;

namespace NeuralNetwork.Core.MLP.Losses
{
    public abstract class Loss : NetworkMLP
    {
        public double AccumulatedSum { get; private set; }
        public int AccumulatedCount { get; private set; }

        public IEnumerable<LayerDense> TrainableLayers { get; set; }

        // Calculates the data and regularization losses
        // given model output and ground truth values
        public double Calculate(double[][] output, int[] y)
        {
            // Calculate sample losses
            double[] sampleLosses = Forward(output, y);

            // Calculate mean loss
            double dataLoss = Measures.Mean(sampleLosses);

            // Add accumulated sum of losses and sample count
            AccumulatedSum += sampleLosses.Sum();
            AccumulatedCount += sampleLosses.Length;

            // Return data loss value
            return dataLoss;
        }

        public (double, double) Calculate(double[][] output, int[] y, bool regularization = true)
        {
            var dataLoss = Calculate(output, y);
            var regLoss = regularization ? RegularizationLoss() : 0;

            // Return data loss value, and reg loss in a tuple
            return (dataLoss, regLoss);
        }

        // Calculates the accumulated loss
        public double CalculateAccumulated()
        {
            // Calculate mean loss
            double dataLoss = AccumulatedSum / AccumulatedCount;

            return dataLoss;
        }

        public (double, double) CalculateAccumulated(bool regularization = true)
        {
            // Include reg loss in output
            var dataLoss = CalculateAccumulated();
            var regLoss = regularization ? RegularizationLoss() : 0;

            return (dataLoss, regLoss);
        }

        // If new network pass
        public void NewPass()
        {
            AccumulatedSum = 0;
            AccumulatedCount = 0;
        }

        private double RegularizationLoss()
        {
            double regularizationLoss = 0;

            // Calculate reg loss for all trainable layers in model
            for (int i = 0; i < TrainableLayers.Count(); i++)
            {
                double layerRegLoss = RegularizationLossLayer(TrainableLayers.ElementAt(i));

                regularizationLoss += layerRegLoss;
            }

            return regularizationLoss;
        }

        private static double RegularizationLossLayer(LayerDense layer)
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

        protected abstract double[] Forward(double[][] yPred, int[] yTrue);
        public abstract void Backward(double[][] dValues, int[] yTrue);
    }
}
