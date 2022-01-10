using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using NeuralNetwork.Core.MLP.Layers;

namespace NeuralNetwork.Core.Optimizers
{
    public class OptimizerAdam : Optimizer
    {
        public double Decay { get; set; }
        public double Epsilon { get; set; }
        public double Beta1 { get; set; }
        public double Beta2 { get; set; }

        // Init optimizer with hyper-paramaters
        public OptimizerAdam(double learningRate = 0.001, double decay = 0, double epsilon = 1e-7, double beta1 = 0.9, double beta2 = 0.999)
        {
            LearningRate = learningRate;
            CurrentLearningRate = LearningRate;
            Decay = decay;
            Epsilon = epsilon;
            Beta1 = beta1;
            Beta2 = beta2;
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
            // Create layer cache and momentum arrays if not already created
            if (layer.WeightCache == null)
            {
                layer.WeightMomentums = Jagged.Zeros(layer.Weights.Rows(), layer.Weights.Columns());
                layer.WeightCache = Jagged.Zeros(layer.Weights.Rows(), layer.Weights.Columns());

                layer.BiasMomentums = Vector.Zeros(layer.Biases.Length);
                layer.BiasCache = Vector.Zeros(layer.Biases.Length);
            }

            // Update momentum with current gradients
            layer.WeightMomentums = layer.WeightMomentums.Multiply(Beta1).Add(layer.DWeights.Multiply(1 - Beta1));
            layer.BiasMomentums = layer.BiasMomentums.Multiply(Beta1).Add(layer.DBiases.Multiply(1 - Beta1));

            // Get corrected momentum
            double[][] weightMomentumsCorrected = layer.WeightMomentums.Divide(1 - Math.Pow(Beta1, Iterations + 1));
            double[] biasMomentumsCorrected = layer.BiasMomentums.Divide(1 - Math.Pow(Beta1, Iterations + 1));

            // Update cache with squared gradients
            layer.WeightCache = layer.WeightCache.Multiply(Beta2).Add(layer.DWeights.Pow(2).Multiply(1 - Beta2));
            layer.BiasCache = layer.BiasCache.Multiply(Beta2).Add(layer.DBiases.Pow(2).Multiply(1 - Beta2));

            // Get corrected cache
            double[][] weightCacheCorrected = layer.WeightCache.Divide(1 - Math.Pow(Beta2, Iterations + 1));
            double[] biasCacheCorrected = layer.BiasCache.Divide(1 - Math.Pow(Beta2, Iterations + 1));

            // Vanilla SGD parameter update + normalization
            // with square rooted cache
            double[][] weightDemoninator = weightCacheCorrected.Sqrt().Add(Epsilon);
            double[] biasDemoninator = biasCacheCorrected.Sqrt().Add(Epsilon);
            layer.Weights = layer.Weights.Add(weightMomentumsCorrected.Multiply(-CurrentLearningRate).Divide(weightDemoninator));
            layer.Biases = layer.Biases.Add(biasMomentumsCorrected.Multiply(-CurrentLearningRate).Divide(biasDemoninator));
        }

        // Call once after any param updates
        public override void PostUpdateParams()
        {
            Iterations++;
        }
    }
}
