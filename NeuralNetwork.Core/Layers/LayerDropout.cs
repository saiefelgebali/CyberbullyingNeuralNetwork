using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using Accord.Statistics;

namespace NeuralNetwork.Core.Layers
{
    public class LayerDropout
    {
        public double[,] Inputs { get; set; }
        public double[,] Output { get; set; }
        public double[,] DInputs { get; set; }
        public int[,] BinaryMask { get; set; }

        public double Rate { get; set; }

        public LayerDropout(double rate)
        {
            // Invert rate
            Rate = 1 - rate;
        }

        public void Forward(double[,] inputs)
        {
            Inputs = inputs;

            // Generate and save scaled mask
            BinaryMask = CreateBinaryMask(Rate, inputs.Rows(), inputs.Columns());

            // Apply mask to output values
            Output = Inputs.Multiply(BinaryMask);
            
        }

        public void Backward(double[,] dValues)
        {
            // Gradient on values
            DInputs = dValues.Multiply(BinaryMask);
        }

        // Helper function to create binomial binary mask
        private int[,] CreateBinaryMask(double rate, int rows, int columns)
        {
            var rand = new Random();

            // Return value
            int[,] result = new int[rows, columns];

            // Generate binary mask based on rate
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    bool success = rand.NextDouble() < rate;
                    result[i, j] = success ? 1 : 0;
                }
            }

            return result;
        }
    }
}
