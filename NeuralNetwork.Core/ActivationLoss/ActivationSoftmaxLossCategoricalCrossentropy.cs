﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using NeuralNetwork.Core.Activations;
using NeuralNetwork.Core.Loss;

namespace NeuralNetwork.Core.ActivationLoss
{
    public class ActivationSoftmaxLossCategoricalCrossentropy
    {
        public ActivationSoftmax Activation { get; set; }
        public LossCategoricalCrossentropy Loss { get; set; }
        public double[,] Output { get; set; }
        public double[,] DInputs { get; set; }

        public ActivationSoftmaxLossCategoricalCrossentropy()
        {
            Activation = new();
            Loss = new();
        }

        public double Forward(double[,] inputs, int[] yTrue)
        {
            Activation.Forward(inputs);

            Output = Activation.Output;

            return Loss.Calculate(Output, yTrue);
        }

        public void Backward(double[,] dValues, int[] yTrue)
        {
            int samplesLength = dValues.Rows();

            // Copy so we can safely modify
            DInputs = dValues.Copy();

            for (int i = 0; i < DInputs.Rows(); i++)
            {
                // Calculate gradient
                DInputs[i, yTrue[i]] -= 1;
            }

            // Normalize gradient
            DInputs = DInputs.Divide(samplesLength);
        }
    }
}
