using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.MLP.Activations
{
    public class ActivationSigmoid : Activation
    {
        public override void Forward(double[][] input, bool training = false)
        {
            Inputs = input;

            // 1 / (1 + e^-input)
            Output = Jagged.Ones(input.Rows(), input.Columns()).Divide(input.Multiply(-1).Exp().Add(1));
        }

        public override void Backward(double[][] dValues)
        {
            DInputs = dValues.Multiply(Output.Multiply(-1).Add(1)).Multiply(Output);
        }

        public override int[] Predictions()
        {
            int[] predictions = new int[Output.Rows()];

            // Get predictions for all samples
            for (int i = 0; i < Output.Rows(); i++)
            {
                int prediction = (Output[i][0] > 0.5) ? 1 : 0;
                predictions[i] = prediction;
            }

            return predictions;
        }
    }
}
