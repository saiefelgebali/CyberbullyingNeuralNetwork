using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using NeuralNetwork.Core.Layers;

namespace NeuralNetwork.Core.Optimizers
{
    public class OptimizerRMSProp : Optimizer
    {
        public double Decay { get; set; }
        public double Epsilon { get; set; }
        public double Rho { get; set; }

        // Root Mean Square Propogation Optimizer
        // Init optimizer with hyper-paramaters
        public OptimizerRMSProp(double learningRate = 0.001, double decay = 0, double epsilon = 1e-7, double rho = 0.9)
        {
            LearningRate = learningRate;
            CurrentLearningRate = LearningRate;
            Decay = decay;
            Epsilon = epsilon;
            Rho = rho;            
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
            // Create layer cache arrays if not already created
            if (layer.WeightCache == null)
            {
                layer.WeightCache = Jagged.Zeros(layer.Weights.Rows(), layer.Weights.Columns());
                layer.BiasCache = Vector.Zeros(layer.Biases.Length);
            }

            // Update cache with squared current gradient
            layer.WeightCache = layer.WeightCache.Multiply(Rho).Add(layer.DWeights.Pow(2).Multiply(1 - Rho));
            layer.BiasCache = layer.BiasCache.Multiply(Rho).Add(layer.DBiases.Pow(2).Multiply(1 - Rho));

            // Update weights and biases
            var weightsDenominator = layer.WeightCache.Sqrt().Add(Epsilon);
            var biasDenominator = layer.BiasCache.Sqrt().Add(Epsilon);

            // Vanilla SGD parameter update + normalization
            // with square rooted cache
            layer.Weights = layer.Weights.Add(layer.DWeights.Multiply(-CurrentLearningRate).Divide(weightsDenominator));
            layer.Biases = layer.Biases.Add(layer.DBiases.Multiply(-CurrentLearningRate).Divide(biasDenominator));
        }

        // Call once after any param updates
        public override void PostUpdateParams()
        {
            Iterations++;
        }
    }
}
