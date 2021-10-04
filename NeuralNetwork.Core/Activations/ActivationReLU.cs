using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Core.Activations
{
    public class ActivationReLU
    {
        public double[,] Inputs { get; set; }
        public double[,] DInputs { get; set; }
        public double[,] Output { get; set; }

        public void Forward(double[,] inputs)
        {
            Inputs = inputs;

            Output = inputs.Copy();

            // Clip output to 0
            for (int i = 0; i < Inputs.Rows(); i++)
            {
                for (int j = 0; j < Inputs.Columns(); j++)
                {
                    if (Output[i, j] < 0) Output[i, j] = 0;
                }
            }
        }

        public void Backward(double[,] dValues)
        {
            DInputs = dValues.Copy();

            // Clip gradients to 0
            for (int i = 0; i < DInputs.Rows(); i++)
            {
                for (int j = 0; j < DInputs.Columns(); j++)
                {
                    if (Inputs[i, j] < 0) DInputs[i, j] = 0;
                }
            }
        }
    }
}
