using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace NeuralNetwork.Prototype
{
    public class LayerDense
    {
        public double[,]? Inputs{ get; private set; }
        public double[,] Weights { get; private set; }
        public double[] Biases { get; private set; }
        
        public double[,]? DInputs{ get; private set; }
        public double[,]? DWeights { get; private set; }
        public double[]? DBiases { get; private set; }


        public LayerDense(int numInputs, int numOutputs, double[,]? initialWeights = null)
        {
            // init weights and biases
            if (initialWeights == null)
            {
                Weights = Matrix.Random(numInputs, numOutputs);
            } 
            else
            {
                // allow passing in initial weight values for testing
                Weights = initialWeights;
            }
            Biases = Vector.Zeros(numOutputs);

            // init derivatives to zeros
            DWeights = Matrix.Zeros(numInputs, numOutputs);
            DInputs = Matrix.Zeros(1, numInputs);
            DBiases = Vector.Zeros(numOutputs);
        }

        public double[,] Forward(double[,] X)
        {
            return X.Dot(Weights).Add(Biases, VectorType.RowVector);
        }

        public double[,] Backward(double[,] dValues)
        {
            DWeights = dValues.Dot(Inputs);
            DInputs = dValues.Dot(Weights.Transpose());
            DBiases = dValues.Sum(dimension: 1);

            return DInputs;
        }
    }
}
