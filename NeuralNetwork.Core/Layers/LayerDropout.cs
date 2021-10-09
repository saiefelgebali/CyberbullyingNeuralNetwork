using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using Accord.Statistics;

namespace NeuralNetwork.Core.Layers
{
    public class LayerDropout : NetworkLayer
    {
        public int[][] BinaryMask { get; set; }

        public double Rate { get; set; }

        public LayerDropout(double rate)
        {
            // Invert rate
            Rate = 1 - rate;
        }

        public override void Forward(double[][] inputs, bool training = false)
        {
            Inputs = inputs;

            // Only use while training
            if (!training)
            {
                Output = Inputs.Copy();
                return;
            }

            // Generate and save scaled mask
            BinaryMask = CreateBinaryMask(Rate, inputs.Rows(), inputs.Columns());

            // Apply mask to output values
            Output = Inputs.Multiply(BinaryMask);
            
        }

        public override void Backward(double[][] dValues)
        {
            // Gradient on values
            DInputs = dValues.Multiply(BinaryMask);
        }

        // Helper function to create binomial binary mask
        private static int[][] CreateBinaryMask(double rate, int rows, int columns)
        {
            var rand = new Random();

            // Return value
            int[][] result = new int[rows][];

            // Generate binary mask based on rate
            for (int i = 0; i < rows; i++)
            {
                result[i] = new int[columns];
                for (int j = 0; j < columns; j++)
                {
                    bool success = rand.NextDouble() < rate;
                    result[i][j] = success ? 1 : 0;
                }
            }

            return result;
        }
    }
}
