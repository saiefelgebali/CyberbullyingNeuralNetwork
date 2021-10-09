using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Core.Activations
{
    public class ActivationReLU : Activation
    {

        public override void Forward(double[][] inputs, bool training = false)
        {
            Inputs = inputs;

            Output = inputs.Copy();

            // Clip output to 0
            for (int i = 0; i < Inputs.Rows(); i++)
            {
                for (int j = 0; j < Inputs.Columns(); j++)
                {
                    if (Output[i][j] < 0) Output[i][j] = 0;
                }
            }
        }

        public override void Backward(double[][] dValues)
        {
            DInputs = dValues.Copy();

            // Clip gradients to 0
            for (int i = 0; i < DInputs.Rows(); i++)
            {
                for (int j = 0; j < DInputs.Columns(); j++)
                {
                    if (Inputs[i][j] < 0) DInputs[i][j] = 0;
                }
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
