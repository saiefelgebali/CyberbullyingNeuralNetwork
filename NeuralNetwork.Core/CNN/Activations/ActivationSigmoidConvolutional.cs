using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.CNN.Activations
{
    public class ActivationSigmoidConvolutional : LayerCNN
    {
        public override int OutputDepth { get; protected set; }
        public override int OutputRows { get; protected set; }
        public override int OutputColumns { get; protected set; }

        public ActivationSigmoidConvolutional(int depth, int rows, int columns)
        {
            OutputDepth = depth;
            OutputRows = rows;
            OutputColumns = columns;
        }

        public override void Forward(double[][][][] input)
        {
            Inputs = input;

            Output = new double[input.Length][][][];
            for (int sample = 0; sample < input.Length; sample++)
            {
                var depth = input[sample].Length;
                Output[sample] = new double[depth][][];
                for (int i = 0; i < depth; i++)
                {
                    // 1 / (1 + e^-input)
                    var currentInput = input[sample][i];
                    Output[sample][i] = Jagged.Ones(currentInput.Rows(), currentInput.Columns()).Divide(currentInput.Multiply(-1).Exp().Add(1));
                }
            }
        }

        public override void Backward(double[][][][] dValues)
        {
            DInputs = new double[dValues.Length][][][];
            for (int sample = 0; sample < dValues.Length; sample++)
            {
                var depth = dValues[sample].Length;
                DInputs[sample] = new double[depth][][];
                for (int i = 0; i < depth; i++)
                {
                    DInputs[sample][i] = dValues[sample][i].Multiply(Output[sample][i].Multiply(-1).Add(1)).Multiply(Output[sample][i]);
                }
            }
        }
    }
}
