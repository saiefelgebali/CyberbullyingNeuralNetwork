using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.Optimizers
{
    public class OptimizerSGD : Optimizer
    {
        public double Decay { get; set; }
        public double Momentum { get; set; }

        // Init optimizer with hyper-paramaters
        public OptimizerSGD(double learningRate = 1, double decay = 0, double momentum = 0)
        {
            LearningRate = learningRate;
            CurrentLearningRate = LearningRate;
            Decay = decay;
            Momentum = momentum;
        }

        // Call once before any param updates
        public override void PreUpdateParams()
        {
            // Decay learning rate
            if (Decay != 0)
            {
                CurrentLearningRate = LearningRate / (1 + Decay * Iterations);
            }
        }

        // Param updates
        public override void UpdateParams(LayerDense layer)
        {
            double[][] weightUpdates;
            double[] biasUpdates;

            // Use momentum
            if (Momentum != 0)
            {
                // Create layer momentum arrays if not already created
                if (layer.WeightMomentums == null)
                {
                    layer.WeightMomentums = Jagged.Zeros(layer.Weights.Rows(), layer.Weights.Columns());
                    layer.BiasMomentums = Vector.Zeros(layer.Biases.Length);
                }

                // Build weight updates
                weightUpdates = layer.WeightMomentums.Multiply(Momentum).Subtract(layer.DWeights.Multiply(CurrentLearningRate));
                layer.WeightMomentums = weightUpdates;

                // Build bias updates
                biasUpdates = layer.BiasMomentums.Multiply(Momentum).Subtract(layer.DBiases.Multiply(CurrentLearningRate));
                layer.BiasMomentums = biasUpdates;
            }

            // Vanilla SGD
            else
            {
                weightUpdates = layer.DWeights.Multiply(-CurrentLearningRate);
                biasUpdates = layer.DBiases.Multiply(-CurrentLearningRate);
            }

            // Update weights and biases
            layer.Weights = layer.Weights.Add(weightUpdates);
            layer.Biases = layer.Biases.Add(biasUpdates);
        }

        // Call once after any param updates
        public override void PostUpdateParams()
        {
            Iterations++;
        }
    }
}
